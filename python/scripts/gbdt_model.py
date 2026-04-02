#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import pickle

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# Frozen feature contract v1
# Hardcoded by design because the H5 arrays do not reliably
# carry embedded feature-name metadata.
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
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

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
class SearchResult:
    learning_rate: float
    max_depth: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    best_iteration: int
    best_val_log_loss: float
    best_val_accuracy: float
    final_train_log_loss_at_best: float
    early_stopped: bool


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

    used_feature_names = list(FEATURE_NAMES)
    x_model = x_filled

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


def resolve_mask_config(config: Dict) -> Tuple[str, bool]:
    mode = config.get("masking_mode", "exclude_masked_rows")

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


# ============================================================
# Metrics and analyses
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    labels = np.arange(len(CLASS_NAMES))
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
        CLASS_NAMES[i]: float(precision[i]) for i in range(len(CLASS_NAMES))
    }
    metrics["recall_per_class"] = {
        CLASS_NAMES[i]: float(recall[i]) for i in range(len(CLASS_NAMES))
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
    d_idx = CLASS_INDEX["deuteron"]
    he4_idx = CLASS_INDEX["helium4"]

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
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title("Confusion Matrix (Counts)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}confusion_matrix_counts.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.imshow(cm_norm, aspect="auto", vmin=0.0, vmax=1.0)
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
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

        acc = float(accuracy_score(y_true[sel], y_pred[sel]))

        rows.append(
            {
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "bin_center": float((edges[i] + edges[i + 1]) / 2.0),
                "count": n,
                "accuracy": acc,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

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

    # Lightweight reliability-style plot on max confidence
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

    df = pd.DataFrame(prob_summary)
    df.to_csv(os.path.join(output_dir, "probability_summary.csv"), index=False)

    return {
        "mean_confidence": float(np.mean(confidence)),
        "accuracy": float(np.mean(correct)),
        "probability_summary": prob_summary,
    }


# ============================================================
# Model search / training
# ============================================================

def get_sample_weight(y: np.ndarray, class_weight_mode: str):
    if class_weight_mode in {None, "none"}:
        return None
    if class_weight_mode == "balanced":
        return compute_sample_weight(class_weight="balanced", y=y)
    raise ValueError("class_weight_mode must be 'none' or 'balanced'")


def build_hgbt(config: Dict, params: Dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_depth=params["max_depth"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        early_stopping=False,   # custom validation-based early stopping
        random_state=config["random_seed"],
        warm_start=True,
        verbose=int(config.get("verbose", 0)),
    )


def iterate_hyperparameter_grid(config: Dict) -> List[Dict]:
    grid = []
    for learning_rate in config["learning_rate_grid"]:
        for max_depth in config["max_depth_grid"]:
            for max_leaf_nodes in config["max_leaf_nodes_grid"]:
                for min_samples_leaf in config["min_samples_leaf_grid"]:
                    for l2_regularization in config["l2_regularization_grid"]:
                        grid.append(
                            {
                                "learning_rate": float(learning_rate),
                                "max_depth": int(max_depth),
                                "max_leaf_nodes": int(max_leaf_nodes),
                                "min_samples_leaf": int(min_samples_leaf),
                                "l2_regularization": float(l2_regularization),
                                "max_iter": int(config["max_iter"]),
                            }
                        )
    return grid


def train_with_custom_early_stopping(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    params: Dict,
    sample_weight=None,
) -> Tuple[HistGradientBoostingClassifier, SearchResult, pd.DataFrame]:
    patience = int(config["early_stopping_patience"])
    min_delta = float(config.get("early_stopping_min_delta", 0.0))
    min_iter_before_stop = int(config.get("min_iter_before_stop", 20))

    model = build_hgbt(config, params)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = np.inf
    best_iteration = 0
    best_snapshot_bytes = None
    epochs_without_improvement = 0

    for iteration in range(1, int(params["max_iter"]) + 1):
        model.max_iter = iteration
        model.fit(x_train, y_train, sample_weight=sample_weight)

        train_prob = model.predict_proba(x_train)
        val_prob = model.predict_proba(x_val)
        val_pred = np.argmax(val_prob, axis=1)

        train_loss = float(log_loss(y_train, train_prob, labels=np.arange(len(CLASS_NAMES))))
        val_loss = float(log_loss(y_val, val_prob, labels=np.arange(len(CLASS_NAMES))))
        val_acc = float(accuracy_score(y_val, val_pred))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_iteration = iteration
            best_snapshot_bytes = pickle.dumps(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            iteration >= min_iter_before_stop
            and epochs_without_improvement >= patience
        ):
            break

    if best_snapshot_bytes is None:
        best_snapshot_bytes = pickle.dumps(model)
        best_iteration = len(val_losses)
        best_val_loss = val_losses[-1]

    best_model = pickle.loads(best_snapshot_bytes)

    history_df = pd.DataFrame(
        {
            "iteration": np.arange(1, len(train_losses) + 1),
            "train_log_loss": train_losses,
            "val_log_loss": val_losses,
            "val_accuracy": val_accuracies,
        }
    )

    early_stopped = len(history_df) < int(params["max_iter"])

    result = SearchResult(
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        best_iteration=int(best_iteration),
        best_val_log_loss=float(best_val_loss),
        best_val_accuracy=float(history_df.loc[best_iteration - 1, "val_accuracy"]),
        final_train_log_loss_at_best=float(
            history_df.loc[best_iteration - 1, "train_log_loss"]
        ),
        early_stopped=bool(early_stopped),
    )

    return best_model, result, history_df


# ============================================================
# Importance / difficult-region inspection
# ============================================================

def save_permutation_importance(
    model,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_csv_path: str,
    random_state: int,
    n_repeats: int,
    scoring: str = "neg_log_loss",
    max_samples=None,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
        max_samples=max_samples,
    )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    df.to_csv(output_csv_path, index=False)
    return df


def plot_top_importances(df: pd.DataFrame, output_path: str, title: str, top_n: int = 20):
    top = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(5, 0.25 * len(top))))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def difficult_region_importance(
    model,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    random_state: int,
    n_repeats: int,
):
    d_idx = CLASS_INDEX["deuteron"]
    he4_idx = CLASS_INDEX["helium4"]

    subset = np.isin(y, [d_idx, he4_idx])
    if subset.sum() < 5:
        return pd.DataFrame()

    y_bin = (y[subset] == he4_idx).astype(int)
    x_sub = x[subset]

    df = save_permutation_importance(
        model=model,
        x=x_sub,
        y=y_bin,
        feature_names=feature_names,
        output_csv_path=os.path.join(output_dir, "difficult_region_permutation_importance.csv"),
        random_state=random_state,
        n_repeats=n_repeats,
        scoring="accuracy",
        max_samples=None,
    )
    plot_top_importances(
        df,
        os.path.join(output_dir, "difficult_region_permutation_importance.png"),
        "Permutation Importance: deuteron vs helium4 subset",
    )
    return df


# ============================================================
# Reporting
# ============================================================

def save_feature_names(feature_names: List[str], output_dir: str):
    with open(os.path.join(output_dir, "feature_names_used.json"), "w") as f:
        json.dump(feature_names, f, indent=2)


def write_summary_markdown(
    output_dir: str,
    config: Dict,
    mask_mode: str,
    append_mask_indicators: bool,
    fill_strategy: str,
    train_split: PreparedSplit,
    val_split: PreparedSplit,
    test_split: PreparedSplit,
    best_result: SearchResult,
    metrics: Dict,
    hard_region: Dict,
    top_importance_df: pd.DataFrame,
    difficult_df: pd.DataFrame,
):
    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Phase F — Gradient-Boosted Trees Summary\n\n")

        f.write("## Run mode\n\n")
        f.write(f"- masking_mode: `{mask_mode}`\n")
        f.write(f"- append_mask_indicators: `{append_mask_indicators}`\n")
        f.write(f"- fill_strategy_for_masked_numeric_values: `{fill_strategy}`\n")
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

        f.write("## Feature count used\n\n")
        f.write(f"- numeric feature count: {len(FEATURE_NAMES)}\n")
        f.write(f"- total feature count used by model: {len(train_split.used_feature_names)}\n\n")

        f.write("## Selected model configuration\n\n")
        for k, v in asdict(best_result).items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Metrics on test set\n\n")
        f.write(f"- accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"- log_loss: {metrics['log_loss']:.6f}\n\n")

        f.write("### Per-class precision\n\n")
        for cls, value in metrics["precision_per_class"].items():
            f.write(f"- {cls}: {value:.6f}\n")
        f.write("\n### Per-class recall\n\n")
        for cls, value in metrics["recall_per_class"].items():
            f.write(f"- {cls}: {value:.6f}\n")
        f.write("\n")

        f.write("## Hard-region analysis: deuteron vs helium4\n\n")
        f.write(f"- subset_size: {hard_region['subset_size']}\n")
        f.write(f"- recall_deuteron: {hard_region['recall_deuteron']:.6f}\n")
        f.write(f"- recall_helium4: {hard_region['recall_helium4']:.6f}\n")
        f.write(
            f"- mutual_misclassification_rate: "
            f"{hard_region['mutual_misclassification_rate']:.6f}\n\n"
        )

        f.write("## Feature-importance summary\n\n")
        f.write(
            "- global importance is estimated with permutation importance on the test set.\n"
        )
        f.write(
            "- difficult-region importance is estimated with permutation importance on the "
            "deuteron/helium4 subset.\n\n"
        )

        if not top_importance_df.empty:
            f.write("### Top global features\n\n")
            for _, row in top_importance_df.head(15).iterrows():
                f.write(
                    f"- {row['feature']}: mean={row['importance_mean']:.6f}, "
                    f"std={row['importance_std']:.6f}\n"
                )
            f.write("\n")

        if not difficult_df.empty:
            f.write("### Top difficult-region features\n\n")
            for _, row in difficult_df.head(10).iterrows():
                f.write(
                    f"- {row['feature']}: mean={row['importance_mean']:.6f}, "
                    f"std={row['importance_std']:.6f}\n"
                )
            f.write("\n")

        f.write("## Interpretation prompts for later comparison\n\n")
        f.write(
            "- Check whether `m2`, `beta`, and `dEdx` appear among the top global or "
            "difficult-region features.\n"
        )
        f.write(
            "- Check whether mask-indicator features (`mask__...`) appear near the top; "
            "if so, maskedness is contributing useful nonlinear signal.\n"
        )
        f.write(
            "- Compare this run against Phase D and Phase E after those baselines are run. "
            "Do not infer superiority from code structure alone.\n"
        )
        f.write(
            "- If this model outperforms the linear baseline, the likely explanation is "
            "nonlinear interactions and/or useful handling of masked features.\n\n"
        )

        f.write("## Limitations / assumptions\n\n")
        f.write(
            "- Uses scikit-learn HistGradientBoostingClassifier to avoid extra dependencies.\n"
        )
        f.write(
            "- Uses custom validation-based early stopping with warm-start refits, because "
            "the baseline requires using the explicit validation split rather than an internal split.\n"
        )
        f.write(
            "- No probability calibration is applied yet.\n"
        )
        f.write(
            "- SHAP is intentionally omitted to keep dependencies light; difficult-region "
            "inspection falls back to subset permutation importance.\n"
        )


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

    os.makedirs(args.output_dir, exist_ok=True)

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

    save_feature_names(train_split.used_feature_names, args.output_dir)

    sample_weight = get_sample_weight(
        train_split.y,
        config.get("class_weight_mode", "none"),
    )

    grid = iterate_hyperparameter_grid(config)
    if not grid:
        raise ValueError("Hyperparameter grid is empty.")

    search_rows = []
    best_model = None
    best_result = None
    best_history = None
    best_score = (np.inf, -np.inf)  # minimize val log loss, then maximize val accuracy

    for params in grid:
        model, result, history_df = train_with_custom_early_stopping(
            x_train=train_split.X_model,
            y_train=train_split.y,
            x_val=val_split.X_model,
            y_val=val_split.y,
            config=config,
            params=params,
            sample_weight=sample_weight,
        )

        row = asdict(result)
        search_rows.append(row)

        score = (result.best_val_log_loss, -result.best_val_accuracy)
        if score < best_score:
            best_score = score
            best_model = model
            best_result = result
            best_history = history_df

    if best_model is None or best_result is None or best_history is None:
        raise RuntimeError("No model was successfully selected.")

    pd.DataFrame(search_rows).sort_values(
        ["best_val_log_loss", "best_val_accuracy"],
        ascending=[True, False],
    ).to_csv(os.path.join(args.output_dir, "hyperparameter_search.csv"), index=False)

    best_history.to_csv(os.path.join(args.output_dir, "validation_tracking.csv"), index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(best_history["iteration"], best_history["train_log_loss"], label="train")
    plt.plot(best_history["iteration"], best_history["val_log_loss"], label="val")
    plt.axvline(best_result.best_iteration, linestyle="--", label="best_iteration")
    plt.xlabel("Boosting iteration")
    plt.ylabel("Log loss")
    plt.title("Training vs Validation Log Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(best_history["iteration"], best_history["val_log_loss"], label="val_log_loss")
    plt.axvline(best_result.best_iteration, linestyle="--", label="best_iteration")
    plt.xlabel("Boosting iteration")
    plt.ylabel("Validation log loss")
    plt.title("Validation Log Loss vs Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "validation_log_loss_vs_iteration.png"), dpi=150)
    plt.close()

    y_pred = best_model.predict(test_split.X_model)
    y_prob = best_model.predict_proba(test_split.X_model)

    metrics, cm, cm_norm = compute_metrics(test_split.y, y_pred, y_prob)
    hard_region = hard_region_analysis(test_split.y, y_pred)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        [
            {"metric": "accuracy", "value": metrics["accuracy"]},
            {"metric": "log_loss", "value": metrics["log_loss"]},
        ]
    ).to_csv(os.path.join(args.output_dir, "metrics_overall.csv"), index=False)

    save_confusion_matrices(cm, cm_norm, args.output_dir)

    with open(os.path.join(args.output_dir, "hard_region.json"), "w") as f:
        json.dump(hard_region, f, indent=2)

    hard_cm = np.array(hard_region["confusion_matrix"], dtype=int)
    pd.DataFrame(
        hard_cm,
        index=["deuteron", "helium4"],
        columns=["pred_deuteron", "pred_helium4"],
    ).to_csv(os.path.join(args.output_dir, "hard_region_confusion_matrix.csv"))

    for variable in DIAGNOSTIC_VARIABLES:
        performance_vs_variable(
            x_raw=test_split.X_raw_filled,
            y_true=test_split.y,
            y_pred=y_pred,
            feature_name=variable,
            output_dir=args.output_dir,
            bins=int(config.get("performance_bins", 10)),
        )

    probability_summary = probability_diagnostics(
        y_true=test_split.y,
        y_pred=y_pred,
        y_prob=y_prob,
        output_dir=args.output_dir,
    )
    with open(os.path.join(args.output_dir, "probability_diagnostics.json"), "w") as f:
        json.dump(probability_summary, f, indent=2)

    global_importance_df = save_permutation_importance(
        model=best_model,
        x=test_split.X_model,
        y=test_split.y,
        feature_names=test_split.used_feature_names,
        output_csv_path=os.path.join(args.output_dir, "global_permutation_importance.csv"),
        random_state=int(config["random_seed"]),
        n_repeats=int(config.get("permutation_importance_repeats", 5)),
        scoring="neg_log_loss",
        max_samples=None,
    )
    plot_top_importances(
        global_importance_df,
        os.path.join(args.output_dir, "global_permutation_importance.png"),
        "Global Permutation Importance",
    )

    difficult_df = difficult_region_importance(
        model=best_model,
        x=test_split.X_model,
        y=test_split.y,
        feature_names=test_split.used_feature_names,
        output_dir=args.output_dir,
        random_state=int(config["random_seed"]),
        n_repeats=int(config.get("permutation_importance_repeats", 5)),
    )

    joblib.dump(best_model, os.path.join(args.output_dir, "gbdt_model.joblib"))

    write_summary_markdown(
        output_dir=args.output_dir,
        config=config,
        mask_mode=mask_mode,
        append_mask_indicators=append_mask_indicators,
        fill_strategy=fill_strategy,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        best_result=best_result,
        metrics=metrics,
        hard_region=hard_region,
        top_importance_df=global_importance_df,
        difficult_df=difficult_df,
    )


if __name__ == "__main__":
    main()