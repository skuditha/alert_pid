#!/usr/bin/env python3

import argparse
import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import yaml


# ============================================================
# Frozen feature contract v1
# Hardcoded intentionally because the H5 arrays do not
# reliably carry embedded feature-name metadata.
# ============================================================

FEATURE_NAMES = [
    "px",
    "py",
    "pz",
    "p",
    "pt",
    "theta",
    "phi",
    "vx",
    "vy",
    "vz",
    "vr",
    "v3",
    "n_hits",
    "sum_adc",
    "path",
    "dEdx",
    "dedx_recomputed",
    "p_drift",
    "sum_residuals",
    "residual_per_hit",
    "adc_per_hit",
    "tof_time",
    "pathlength",
    "cluster_x",
    "cluster_y",
    "cluster_z",
    "cluster_energy",
    "n_bar",
    "n_wedge",
    "beta",
    "m2",
    "log_p",
    "log_pt",
    "log_sum_adc",
    "log_path",
    "log_dEdx",
    "log_dedx_recomputed",
    "log_cluster_energy",
]

CLASS_NAMES = ["proton", "deuteron", "triton", "helium3", "helium4"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
DIAGNOSTIC_VARIABLES = ["p", "n_hits", "tof_time", "cluster_energy"]


# ============================================================
# Data containers
# ============================================================

@dataclass
class PreparedSplit:
    X_model: np.ndarray
    X_raw_filled: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    used_feature_names: List[str]
    n_rows_before: int
    n_rows_after: int
    n_masked_rows_before: int
    n_masked_rows_after: int


@dataclass
class SeedRunSummary:
    seed: int
    best_epoch: int
    stopped_epoch: int
    val_best_log_loss: float
    val_accuracy_at_best: float
    test_accuracy: float
    test_log_loss: float
    checkpoint_path: str


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# H5 I/O
# ============================================================

def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        x = f["features/values"][:]
        mask = f["features/masks"][:]
        y = f["labels/class_index"][:]

    x = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    y = np.asarray(y, dtype=np.int64)

    if x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features from frozen contract, "
            f"but found {x.shape[1]} in {path}."
        )

    return x, mask, y


# ============================================================
# Mask handling / feature assembly
# ============================================================

def resolve_mask_config(config: Dict) -> Tuple[str, bool]:
    mode = config.get("masking_mode", "numeric_plus_mask_indicators")

    if mode == "numeric_plus_mask_indicators":
        return mode, True
    if mode == "numeric_only":
        return mode, False
    if mode == "exclude_masked_rows":
        return mode, bool(config.get("append_mask_indicators_when_excluding", False))

    raise ValueError(
        "masking_mode must be one of: "
        "'numeric_plus_mask_indicators', 'numeric_only', 'exclude_masked_rows'"
    )


def compute_fill_values_from_train(
    x_train: np.ndarray,
    mask_train: np.ndarray,
    strategy: str,
) -> np.ndarray:
    if strategy == "zero":
        return np.zeros(x_train.shape[1], dtype=np.float64)

    fill_values = np.zeros(x_train.shape[1], dtype=np.float64)

    for j in range(x_train.shape[1]):
        valid = mask_train[:, j]
        if np.any(valid):
            col = x_train[valid, j]
            if strategy == "median":
                fill_values[j] = float(np.median(col))
            elif strategy == "mean":
                fill_values[j] = float(np.mean(col))
            else:
                raise ValueError(f"Unsupported fill strategy: {strategy}")
        else:
            fill_values[j] = 0.0

    return fill_values


def prepare_split(
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    mode: str,
    append_mask_indicators: bool,
    fill_values: np.ndarray,
) -> PreparedSplit:
    n_rows_before = int(x.shape[0])
    row_has_any_mask = ~mask.all(axis=1)
    n_masked_rows_before = int(row_has_any_mask.sum())

    if mode == "exclude_masked_rows":
        keep = mask.all(axis=1)
        x = x[keep]
        mask = mask[keep]
        y = y[keep]
    elif mode in {"numeric_only", "numeric_plus_mask_indicators"}:
        pass
    else:
        raise ValueError(f"Unsupported masking mode: {mode}")

    n_rows_after = int(x.shape[0])
    n_masked_rows_after = int((~mask.all(axis=1)).sum())

    x_filled = x.copy()
    invalid = ~mask
    if np.any(invalid):
        x_filled[invalid] = np.take(fill_values, np.where(invalid)[1])

    x_model = x_filled
    used_feature_names = list(FEATURE_NAMES)

    if append_mask_indicators:
        mask_indicators = invalid.astype(np.float64)
        indicator_names = [f"mask__{name}" for name in FEATURE_NAMES]
        x_model = np.concatenate([x_model, mask_indicators], axis=1)
        used_feature_names.extend(indicator_names)

    return PreparedSplit(
        X_model=x_model,
        X_raw_filled=x_filled,
        mask=mask,
        y=y,
        used_feature_names=used_feature_names,
        n_rows_before=n_rows_before,
        n_rows_after=n_rows_after,
        n_masked_rows_before=n_masked_rows_before,
        n_masked_rows_after=n_masked_rows_after,
    )


# ============================================================
# Scaling
# ============================================================

def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def transform_with_scaler(
    scaler: StandardScaler,
    x: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    x_out = x.copy()
    numeric_dim = len(FEATURE_NAMES)
    x_out[:, :numeric_dim] = scaler.transform(x_out[:, :numeric_dim])

    # Do not scale appended mask indicators.
    if len(feature_names) > numeric_dim:
        pass

    return x_out


# ============================================================
# Torch dataset / model
# ============================================================

class NumpyClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SmallMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()

        activation_layer = self._get_activation(activation)

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_layer())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    @staticmethod
    def _get_activation(name: str):
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "gelu":
            return nn.GELU
        if name == "elu":
            return nn.ELU
        if name == "leaky_relu":
            return nn.LeakyReLU
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# ============================================================
# Evaluation / metrics
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    labels = np.arange(NUM_CLASSES)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=labels)),
    }

    precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    metrics["precision_per_class"] = {
        CLASS_NAMES[i]: float(precision[i]) for i in range(NUM_CLASSES)
    }
    metrics["recall_per_class"] = {
        CLASS_NAMES[i]: float(recall[i]) for i in range(NUM_CLASSES)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )
    return metrics, cm, cm_norm


def hard_region_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    d_idx = CLASS_TO_INDEX["deuteron"]
    he4_idx = CLASS_TO_INDEX["helium4"]

    subset = np.isin(y_true, [d_idx, he4_idx])
    y_true_sub = y_true[subset]
    y_pred_sub = y_pred[subset]

    cm = confusion_matrix(y_true_sub, y_pred_sub, labels=[d_idx, he4_idx])

    recall_d = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() > 0 else 0.0
    recall_he4 = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() > 0 else 0.0
    misclass_rate = float((cm[0, 1] + cm[1, 0]) / cm.sum()) if cm.sum() > 0 else 0.0

    return {
        "confusion_matrix": cm.tolist(),
        "recall_deuteron": recall_d,
        "recall_helium4": recall_he4,
        "mutual_misclassification_rate": misclass_rate,
        "subset_size": int(subset.sum()),
    }


def save_confusion_matrices(
    cm: np.ndarray,
    cm_norm: np.ndarray,
    output_dir: str,
    prefix: str = "",
) -> None:
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(output_dir, f"{prefix}confusion_matrix_counts.csv")
    )
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(output_dir, f"{prefix}confusion_matrix_normalized.csv")
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(cm, aspect="auto")
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.title("Confusion Matrix (Counts)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}confusion_matrix_counts.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.imshow(cm_norm, aspect="auto", vmin=0.0, vmax=1.0)
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{prefix}confusion_matrix_normalized.png"), dpi=150
    )
    plt.close()


def performance_vs_variable(
    x_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_name: str,
    output_dir: str,
    bins: int = 10,
) -> pd.DataFrame:
    idx = FEATURE_NAMES.index(feature_name)
    values = x_raw[:, idx]

    finite = np.isfinite(values)
    values = values[finite]
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if values.size == 0:
        return pd.DataFrame()

    edges = np.linspace(values.min(), values.max(), bins + 1)
    rows = []

    for i in range(bins):
        if i == bins - 1:
            sel = (values >= edges[i]) & (values <= edges[i + 1])
        else:
            sel = (values >= edges[i]) & (values < edges[i + 1])

        n = int(sel.sum())
        if n == 0:
            continue

        rows.append(
            {
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "bin_center": float((edges[i] + edges[i + 1]) / 2.0),
                "count": n,
                "accuracy": float(accuracy_score(y_true[sel], y_pred[sel])),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(output_dir, f"perf_vs_{feature_name}.csv"), index=False)

        plt.figure(figsize=(6, 4))
        plt.plot(df["bin_center"], df["accuracy"], marker="o")
        plt.xlabel(feature_name)
        plt.ylabel("Accuracy")
        plt.title(f"Performance vs {feature_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"perf_vs_{feature_name}.png"), dpi=150)
        plt.close()

    return df


def probability_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
) -> Dict:
    confidence = y_prob.max(axis=1)
    correct = (y_true == y_pred).astype(int)

    plt.figure(figsize=(6, 4))
    plt.hist(confidence, bins=40)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Count")
    plt.title("Confidence Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_histogram.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(confidence, correct, alpha=0.25, s=10)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Correct (0/1)")
    plt.title("Confidence vs Correctness")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_vs_correctness.png"), dpi=150)
    plt.close()

    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = []
    bin_acc = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            sel = (confidence >= bins[i]) & (confidence <= bins[i + 1])
        else:
            sel = (confidence >= bins[i]) & (confidence < bins[i + 1])

        if sel.sum() == 0:
            continue

        bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
        bin_acc.append(float(correct[sel].mean()))

    if bin_centers:
        plt.figure(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(bin_centers, bin_acc, marker="o")
        plt.xlabel("Confidence bin")
        plt.ylabel("Observed accuracy")
        plt.title("Reliability-style Plot")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reliability_style_plot.png"), dpi=150)
        plt.close()

    prob_summary = []
    for i, cls in enumerate(CLASS_NAMES):
        cls_probs = y_prob[:, i]
        prob_summary.append(
            {
                "class": cls,
                "mean_probability": float(np.mean(cls_probs)),
                "std_probability": float(np.std(cls_probs)),
                "p05": float(np.quantile(cls_probs, 0.05)),
                "p50": float(np.quantile(cls_probs, 0.50)),
                "p95": float(np.quantile(cls_probs, 0.95)),
            }
        )

        plt.figure(figsize=(6, 4))
        plt.hist(cls_probs, bins=40)
        plt.xlabel(f"P({cls})")
        plt.ylabel("Count")
        plt.title(f"Probability Distribution: {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"probability_distribution_{cls}.png"), dpi=150)
        plt.close()

    pd.DataFrame(prob_summary).to_csv(
        os.path.join(output_dir, "probability_summary.csv"),
        index=False,
    )

    return {
        "mean_confidence": float(np.mean(confidence)),
        "accuracy": float(np.mean(correct)),
        "probability_summary": prob_summary,
    }


# ============================================================
# MLP inference helpers
# ============================================================

@torch.no_grad()
def predict_logits_prob_pred(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    logits_list = []
    prob_list = []
    pred_list = []

    for start in range(0, x.shape[0], batch_size):
        xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        logits_list.append(logits.cpu().numpy())
        prob_list.append(probs.cpu().numpy())
        pred_list.append(preds.cpu().numpy())

    logits_np = np.concatenate(logits_list, axis=0)
    prob_np = np.concatenate(prob_list, axis=0)
    pred_np = np.concatenate(pred_list, axis=0)

    return logits_np, prob_np, pred_np


# ============================================================
# Training
# ============================================================

def get_class_weights_tensor(
    y_train: np.ndarray,
    mode: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mode in {None, "none"}:
        return None

    if mode == "balanced":
        classes = np.arange(NUM_CLASSES)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    raise ValueError("class_weight_mode must be 'none' or 'balanced'")


def make_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    dataset = NumpyClassificationDataset(x, y)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def train_one_seed(
    train_split: PreparedSplit,
    val_split: PreparedSplit,
    test_split: PreparedSplit,
    config: Dict,
    feature_names_used: List[str],
    seed: int,
    run_output_dir: str,
) -> Tuple[SeedRunSummary, pd.DataFrame, Dict]:
    ensure_dir(run_output_dir)
    set_deterministic_seed(seed)

    device_name = config.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    batch_size = int(config["batch_size"])

    train_loader = make_dataloader(
        train_split.X_model, train_split.y, batch_size, shuffle=True, seed=seed
    )

    model = SmallMLP(
        input_dim=train_split.X_model.shape[1],
        hidden_dims=list(config["hidden_dims"]),
        output_dim=NUM_CLASSES,
        activation=config.get("activation", "relu"),
    ).to(device)

    class_weights = get_class_weights_tensor(
        train_split.y,
        config.get("class_weight_mode", "none"),
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    max_epochs = int(config["max_epochs"])
    patience = int(config["early_stopping_patience"])
    min_delta = float(config.get("early_stopping_min_delta", 0.0))

    history_rows = []
    best_state_dict = None
    best_epoch = 0
    best_val_log_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_n = xb.shape[0]
            running_loss += float(loss.item()) * batch_n
            n_seen += batch_n

        train_loss_epoch = running_loss / max(n_seen, 1)

        _, train_prob, train_pred = predict_logits_prob_pred(
            model, train_split.X_model, batch_size=batch_size, device=device
        )
        _, val_prob, val_pred = predict_logits_prob_pred(
            model, val_split.X_model, batch_size=batch_size, device=device
        )

        train_acc = float(accuracy_score(train_split.y, train_pred))
        val_acc = float(accuracy_score(val_split.y, val_pred))
        train_logloss = float(log_loss(train_split.y, train_prob, labels=np.arange(NUM_CLASSES)))
        val_logloss = float(log_loss(val_split.y, val_prob, labels=np.arange(NUM_CLASSES)))

        history_rows.append(
            {
                "epoch": epoch,
                "train_ce_loss_epoch": train_loss_epoch,
                "train_log_loss_eval": train_logloss,
                "val_log_loss": val_logloss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

        improved = val_logloss < (best_val_log_loss - min_delta)
        if improved:
            best_val_log_loss = val_logloss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    stopped_epoch = int(history_rows[-1]["epoch"])
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = stopped_epoch

    model.load_state_dict(best_state_dict)

    checkpoint_path = os.path.join(run_output_dir, "best_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train_split.X_model.shape[1],
            "hidden_dims": list(config["hidden_dims"]),
            "activation": config.get("activation", "relu"),
            "feature_names_used": feature_names_used,
            "seed": seed,
            "class_names": CLASS_NAMES,
        },
        checkpoint_path,
    )

    _, val_prob_best, val_pred_best = predict_logits_prob_pred(
        model, val_split.X_model, batch_size=batch_size, device=device
    )
    _, test_prob, test_pred = predict_logits_prob_pred(
        model, test_split.X_model, batch_size=batch_size, device=device
    )

    val_best_logloss = float(log_loss(val_split.y, val_prob_best, labels=np.arange(NUM_CLASSES)))
    val_best_accuracy = float(accuracy_score(val_split.y, val_pred_best))
    test_metrics, test_cm, test_cm_norm = compute_metrics(test_split.y, test_pred, test_prob)
    hard_region = hard_region_analysis(test_split.y, test_pred)

    save_confusion_matrices(test_cm, test_cm_norm, run_output_dir)
    save_json(os.path.join(run_output_dir, "metrics.json"), test_metrics)
    save_json(os.path.join(run_output_dir, "hard_region.json"), hard_region)

    pd.DataFrame(history_rows).to_csv(
        os.path.join(run_output_dir, "training_history.csv"),
        index=False,
    )

    # Curves
    hist_df = pd.DataFrame(history_rows)

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist_df["epoch"], hist_df["train_ce_loss_epoch"], label="train_ce_loss_epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training Loss vs Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, "training_loss_vs_epoch.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist_df["epoch"], hist_df["val_log_loss"], label="val_log_loss")
    plt.axvline(best_epoch, linestyle="--", label="best_epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation log loss")
    plt.title("Validation Log Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, "validation_log_loss_vs_epoch.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist_df["epoch"], hist_df["train_accuracy"], label="train_accuracy")
    plt.plot(hist_df["epoch"], hist_df["val_accuracy"], label="val_accuracy")
    plt.axvline(best_epoch, linestyle="--", label="best_epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, "accuracy_vs_epoch.png"), dpi=150)
    plt.close()

    prob_diag = probability_diagnostics(
        y_true=test_split.y,
        y_pred=test_pred,
        y_prob=test_prob,
        output_dir=run_output_dir,
    )
    save_json(os.path.join(run_output_dir, "probability_diagnostics.json"), prob_diag)

    summary = SeedRunSummary(
        seed=int(seed),
        best_epoch=int(best_epoch),
        stopped_epoch=int(stopped_epoch),
        val_best_log_loss=float(val_best_logloss),
        val_accuracy_at_best=float(val_best_accuracy),
        test_accuracy=float(test_metrics["accuracy"]),
        test_log_loss=float(test_metrics["log_loss"]),
        checkpoint_path=checkpoint_path,
    )

    detailed_payload = {
        "test_metrics": test_metrics,
        "hard_region": hard_region,
        "probability_diagnostics": prob_diag,
    }

    return summary, hist_df, detailed_payload


# ============================================================
# Lightweight feature/input inspection
# ============================================================

class SklearnMLPWrapper:
    """
    Lightweight sklearn-compatible wrapper around the trained PyTorch model
    so permutation_importance can call predict / predict_proba.
    """

    def __init__(self, model: nn.Module, batch_size: int, device: torch.device):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        _, prob, _ = predict_logits_prob_pred(
            self.model,
            np.asarray(X, dtype=np.float32),
            batch_size=self.batch_size,
            device=self.device,
        )
        return prob

    def predict(self, X):
        _, _, pred = predict_logits_prob_pred(
            self.model,
            np.asarray(X, dtype=np.float32),
            batch_size=self.batch_size,
            device=self.device,
        )
        return pred


def save_permutation_importance_for_best_model(
    checkpoint_path: str,
    input_dim: int,
    hidden_dims: List[int],
    activation: str,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    batch_size: int,
    device_name: str,
    random_state: int,
    n_repeats: int,
    output_dir: str,
):
    device_name = device_name if (device_name != "cuda" or torch.cuda.is_available()) else "cpu"
    device = torch.device(device_name)

    model = SmallMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=NUM_CLASSES,
        activation=activation,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    wrapper = SklearnMLPWrapper(model=model, batch_size=batch_size, device=device)

    result = permutation_importance(
        wrapper,
        x_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_log_loss",
        n_jobs=1,
    )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    df.to_csv(os.path.join(output_dir, "permutation_importance_validation.csv"), index=False)

    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, max(5, 0.25 * len(top))))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.title("Validation Permutation Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "permutation_importance_validation.png"), dpi=150)
    plt.close()

    return df


# ============================================================
# Summary writing
# ============================================================

def write_summary_markdown(
    output_dir: str,
    config: Dict,
    mask_mode: str,
    append_mask_indicators: bool,
    fill_strategy: str,
    train_split: PreparedSplit,
    val_split: PreparedSplit,
    test_split: PreparedSplit,
    selected_seed: int,
    selected_checkpoint: str,
    selected_summary: SeedRunSummary,
    seed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
):
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write("# Phase G — Small MLP Summary\n\n")

        f.write("## Run mode\n\n")
        f.write(f"- masking_mode: `{mask_mode}`\n")
        f.write(f"- append_mask_indicators: `{append_mask_indicators}`\n")
        f.write(f"- masked_value_fill_strategy: `{fill_strategy}`\n")
        f.write(
            "- note: this implementation hardcodes the frozen canonical feature order "
            "from the project contract because the H5 arrays do not reliably contain "
            "embedded feature names.\n\n"
        )

        f.write("## Dataset sizes\n\n")
        for name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
            f.write(f"### {name}\n")
            f.write(f"- rows before filtering: {split.n_rows_before}\n")
            f.write(f"- rows after filtering: {split.n_rows_after}\n")
            f.write(f"- rows with any masked feature before filtering: {split.n_masked_rows_before}\n")
            f.write(f"- rows with any masked feature after filtering: {split.n_masked_rows_after}\n\n")

        f.write("## Input / preprocessing\n\n")
        f.write(f"- numeric feature count: {len(FEATURE_NAMES)}\n")
        f.write(f"- total feature count used: {len(train_split.used_feature_names)}\n")
        f.write("- numeric features are standardized using a scaler fit on the training split only\n")
        f.write("- appended mask indicators, if present, are not standardized\n\n")

        f.write("## Architecture / training setup\n\n")
        f.write(f"- hidden_dims: {config['hidden_dims']}\n")
        f.write(f"- activation: {config.get('activation', 'relu')}\n")
        f.write(f"- learning_rate: {config['learning_rate']}\n")
        f.write(f"- batch_size: {config['batch_size']}\n")
        f.write(f"- weight_decay: {config.get('weight_decay', 0.0)}\n")
        f.write(f"- max_epochs: {config['max_epochs']}\n")
        f.write(f"- early_stopping_patience: {config['early_stopping_patience']}\n")
        f.write(f"- class_weight_mode: {config.get('class_weight_mode', 'none')}\n")
        f.write(f"- device: {config.get('device', 'cpu')}\n\n")

        f.write("## Seed stability\n\n")
        f.write(f"- seed list: {config['seed_list']}\n")
        f.write(f"- selected seed for best checkpoint: {selected_seed}\n")
        f.write(f"- selected checkpoint: `{selected_checkpoint}`\n\n")

        if not seed_df.empty:
            key_metrics = ["val_best_log_loss", "val_accuracy_at_best", "test_accuracy", "test_log_loss"]
            stats = seed_df[key_metrics].agg(["mean", "std"]).T
            f.write("### Per-seed summary statistics\n\n")
            for metric, row in stats.iterrows():
                f.write(f"- {metric}: mean={row['mean']:.6f}, std={row['std']:.6f}\n")
            f.write("\n")

        f.write("## Selected model\n\n")
        for k, v in asdict(selected_summary).items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Interpretation prompts for later comparison\n\n")
        f.write(
            "- Compare this run against Phase D / E / F after those baselines are run.\n"
        )
        f.write(
            "- If gains over simpler models are small, the compact MLP may not justify extra deployment complexity.\n"
        )
        f.write(
            "- If confidence plots look overly sharp relative to observed accuracy, calibration may be needed later.\n"
        )
        f.write(
            "- If per-seed variability is high, the model family may need tighter regularization or more data.\n"
        )
        f.write(
            "- Check whether mask-indicator features appear among the top permutation-importance features.\n\n"
        )

        if not permutation_df.empty:
            f.write("## Top validation permutation-importance features\n\n")
            for _, row in permutation_df.head(15).iterrows():
                f.write(
                    f"- {row['feature']}: mean={row['importance_mean']:.6f}, "
                    f"std={row['importance_std']:.6f}\n"
                )
            f.write("\n")

        f.write("## Limitations / assumptions\n\n")
        f.write("- No probability calibration is applied yet.\n")
        f.write("- No architecture sweep is performed; this is intentionally a compact baseline.\n")
        f.write("- Permutation importance is lightweight and approximate, not a full explanation method.\n")
        f.write("- GPU is optional; CPU is fully supported.\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    mask_mode, append_mask_indicators = resolve_mask_config(config)
    fill_strategy = config.get("masked_value_fill_strategy", "zero")

    x_train_raw, mask_train, y_train = load_h5_dataset(args.train)
    x_val_raw, mask_val, y_val = load_h5_dataset(args.val)
    x_test_raw, mask_test, y_test = load_h5_dataset(args.test)

    fill_values = compute_fill_values_from_train(
        x_train=x_train_raw,
        mask_train=mask_train,
        strategy=fill_strategy,
    )

    train_split = prepare_split(
        x=x_train_raw,
        mask=mask_train,
        y=y_train,
        mode=mask_mode,
        append_mask_indicators=append_mask_indicators,
        fill_values=fill_values,
    )
    val_split = prepare_split(
        x=x_val_raw,
        mask=mask_val,
        y=y_val,
        mode=mask_mode,
        append_mask_indicators=append_mask_indicators,
        fill_values=fill_values,
    )
    test_split = prepare_split(
        x=x_test_raw,
        mask=mask_test,
        y=y_test,
        mode=mask_mode,
        append_mask_indicators=append_mask_indicators,
        fill_values=fill_values,
    )

    scaler = fit_scaler(train_split.X_model[:, :len(FEATURE_NAMES)])

    train_split.X_model = transform_with_scaler(scaler, train_split.X_model, train_split.used_feature_names)
    val_split.X_model = transform_with_scaler(scaler, val_split.X_model, val_split.used_feature_names)
    test_split.X_model = transform_with_scaler(scaler, test_split.X_model, test_split.used_feature_names)

    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    with open(os.path.join(args.output_dir, "feature_names_used.json"), "w") as f:
        json.dump(train_split.used_feature_names, f, indent=2)

    seed_list = list(config.get("seed_list", [int(config.get("random_seed", 42))]))
    run_summaries = []
    detailed_by_seed = {}

    for seed in seed_list:
        run_dir = os.path.join(args.output_dir, f"seed_{seed}")
        ensure_dir(run_dir)

        summary, history_df, detailed = train_one_seed(
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            config=config,
            feature_names_used=train_split.used_feature_names,
            seed=int(seed),
            run_output_dir=run_dir,
        )

        run_summaries.append(asdict(summary))
        detailed_by_seed[str(seed)] = detailed

    seed_df = pd.DataFrame(run_summaries)
    seed_df.to_csv(os.path.join(args.output_dir, "seed_summary.csv"), index=False)
    save_json(os.path.join(args.output_dir, "seed_detailed_metrics.json"), detailed_by_seed)

    # Select best seed by validation log loss, then validation accuracy.
    seed_df = seed_df.sort_values(
        ["val_best_log_loss", "val_accuracy_at_best"],
        ascending=[True, False],
    ).reset_index(drop=True)

    best_row = seed_df.iloc[0].to_dict()
    selected_seed = int(best_row["seed"])
    selected_checkpoint = str(best_row["checkpoint_path"])

    permutation_df = save_permutation_importance_for_best_model(
        checkpoint_path=selected_checkpoint,
        input_dim=train_split.X_model.shape[1],
        hidden_dims=list(config["hidden_dims"]),
        activation=config.get("activation", "relu"),
        x_val=val_split.X_model,
        y_val=val_split.y,
        feature_names=train_split.used_feature_names,
        batch_size=int(config["batch_size"]),
        device_name=config.get("device", "cpu"),
        random_state=selected_seed,
        n_repeats=int(config.get("permutation_importance_repeats", 3)),
        output_dir=args.output_dir,
    )

    # Copy selected seed artifacts to top level summary context by regenerating diagnostics from saved files.
    selected_run_dir = os.path.join(args.output_dir, f"seed_{selected_seed}")

    # Performance-vs-variable from selected seed predictions
    device_name = config.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = SmallMLP(
        input_dim=train_split.X_model.shape[1],
        hidden_dims=list(config["hidden_dims"]),
        output_dim=NUM_CLASSES,
        activation=config.get("activation", "relu"),
    ).to(device)

    ckpt = torch.load(selected_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, test_prob, test_pred = predict_logits_prob_pred(
        model, test_split.X_model, batch_size=int(config["batch_size"]), device=device
    )
    test_metrics, test_cm, test_cm_norm = compute_metrics(test_split.y, test_pred, test_prob)
    hard_region = hard_region_analysis(test_split.y, test_pred)

    save_json(os.path.join(args.output_dir, "selected_seed_metrics.json"), test_metrics)
    save_json(os.path.join(args.output_dir, "selected_seed_hard_region.json"), hard_region)
    save_confusion_matrices(test_cm, test_cm_norm, args.output_dir, prefix="selected_seed_")
    probability_summary = probability_diagnostics(
        y_true=test_split.y,
        y_pred=test_pred,
        y_prob=test_prob,
        output_dir=args.output_dir,
    )
    save_json(os.path.join(args.output_dir, "selected_seed_probability_diagnostics.json"), probability_summary)

    for variable in DIAGNOSTIC_VARIABLES:
        performance_vs_variable(
            x_raw=test_split.X_raw_filled,
            y_true=test_split.y,
            y_pred=test_pred,
            feature_name=variable,
            output_dir=args.output_dir,
            bins=int(config.get("performance_bins", 10)),
        )

    selected_summary = SeedRunSummary(
        seed=selected_seed,
        best_epoch=int(best_row["best_epoch"]),
        stopped_epoch=int(best_row["stopped_epoch"]),
        val_best_log_loss=float(best_row["val_best_log_loss"]),
        val_accuracy_at_best=float(best_row["val_accuracy_at_best"]),
        test_accuracy=float(best_row["test_accuracy"]),
        test_log_loss=float(best_row["test_log_loss"]),
        checkpoint_path=selected_checkpoint,
    )

    write_summary_markdown(
        output_dir=args.output_dir,
        config=config,
        mask_mode=mask_mode,
        append_mask_indicators=append_mask_indicators,
        fill_strategy=fill_strategy,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        selected_seed=selected_seed,
        selected_checkpoint=selected_checkpoint,
        selected_summary=selected_summary,
        seed_df=seed_df,
        permutation_df=permutation_df,
    )


if __name__ == "__main__":
    main()