
#!/usr/bin/env python3
"""
Phase D baseline 1: physics-inspired hierarchical cut model for ALERT post-PID.

This script:
- loads ALERT post-PID HDF5 splits
- reconstructs named features from the frozen v1 feature contract
- excludes rows with any masked feature for this baseline
- derives a hierarchical cut model from train
- refines tunable thresholds on validation data
- evaluates once on test
- saves metrics, plots, and human-readable summaries

Important implementation note:
The H5 files only contain /features/values and /features/masks arrays. This script therefore
hardcodes the canonical v1 feature order from postpid_feature_contract_v1.md and maps array
columns to names accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


CANONICAL_FEATURE_ORDER: List[str] = [
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

# Approximate rest masses in MeV/c^2 for physics ordering and reference only.
CLASS_MASSES_MEV: Dict[str, float] = {
    "proton": 938.2720813,
    "deuteron": 1875.61294257,
    "helium3": 2808.39160743,
    "triton": 2808.92113298,
    "helium4": 3727.3794066,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "random_seed": 42,
    "downsample_for_plots_per_class": 12000,
    "coarse": {
        "primary_variable": "m2",
        "use_beta_fallback": True,
        "ambiguity_margin_scale": 0.12,
        "beta_guard_margin": 0.03,
    },
    "refinement": {
        "variable": "dedx_score",
        "use_cluster_energy_tiebreaker": True,
        "dedx_unknown_margin": 0.03,
        "energy_unknown_margin": 0.03,
    },
    "plotting": {
        "scatter_alpha": 0.18,
        "marker_size": 6,
        "max_points_per_class": 12000,
        "dpi": 150,
    },
    "output_filenames": {
        "metrics_json": "metrics_test.json",
        "metrics_markdown": "metrics_test.md",
        "cut_summary": "cut_definition_summary.md",
        "ambiguity_json": "ambiguity_analysis.json",
        "val_tuning_json": "validation_tuning.json",
        "feature_summary_csv": "feature_summary_by_class.csv",
    },
}


@dataclass
class DatasetBundle:
    name: str
    features: pd.DataFrame
    labels: np.ndarray
    raw_masks: np.ndarray
    row_meta: Dict[str, np.ndarray]
    kept_mask: np.ndarray
    dropped_mask: np.ndarray


@dataclass
class PairRule:
    low_class: str
    high_class: str
    variable: str
    threshold: float
    polarity: str  # "greater_is_high" or "greater_is_low"
    deadband: float
    energy_threshold: Optional[float] = None
    energy_polarity: Optional[str] = None
    energy_deadband: float = 0.0


@dataclass
class CutModel:
    label_map: Dict[str, Any]
    classes: List[str]
    class_to_index: Dict[str, int]
    index_to_class: Dict[int, str]
    physical_order: List[str]
    coarse_variable: str
    coarse_thresholds: List[float]
    beta_thresholds: Optional[List[float]]
    ambiguity_margin_scale: float
    beta_guard_margin: float
    pair_rules: Dict[Tuple[str, str], PairRule]
    tuning_summary: Dict[str, Any]


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) if cfg_path.suffix.lower() in {".yaml", ".yml"} else json.load(f)

    def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                deep_update(dst[key], value)
            else:
                dst[key] = value
        return dst

    return deep_update(config, user_cfg or {})


def load_label_map() -> Dict[str, Any]:
    # Keep local fallback aligned with uploaded label_map.json.
    return {
        "classes": ["proton", "deuteron", "triton", "helium3", "helium4"],
        "class_to_index": {"proton": 0, "deuteron": 1, "triton": 2, "helium3": 3, "helium4": 4},
        "index_to_class": {"0": "proton", "1": "deuteron", "2": "triton", "3": "helium3", "4": "helium4"},
        "class_to_pid": {"proton": 2212, "deuteron": 45, "triton": 46, "helium3": 49, "helium4": 47},
        "pid_to_class": {"2212": "proton", "45": "deuteron", "46": "triton", "49": "helium3", "47": "helium4"},
    }


def load_h5_dataset(path: str, name: str) -> DatasetBundle:
    with h5py.File(path, "r") as f:
        values = f["features/values"][:]
        masks = f["features/masks"][:].astype(bool)
        labels = f["labels/class_index"][:].astype(int)
        row_meta = {key: f[f"row_meta/{key}"][:] for key in f["row_meta"].keys()}

    if values.shape[1] != len(CANONICAL_FEATURE_ORDER):
        raise ValueError(
            f"{path}: expected {len(CANONICAL_FEATURE_ORDER)} features from frozen contract, "
            f"found {values.shape[1]}."
        )

    features = pd.DataFrame(values, columns=CANONICAL_FEATURE_ORDER)
    kept_mask = masks.all(axis=1)
    dropped_mask = ~kept_mask

    return DatasetBundle(
        name=name,
        features=features.loc[kept_mask].reset_index(drop=True),
        labels=labels[kept_mask],
        raw_masks=masks,
        row_meta={k: v[kept_mask] for k, v in row_meta.items()},
        kept_mask=kept_mask,
        dropped_mask=dropped_mask,
    )


def class_name_array(labels: np.ndarray, index_to_class: Dict[int, str]) -> np.ndarray:
    return np.array([index_to_class[int(i)] for i in labels], dtype=object)


def physical_class_order(classes: Sequence[str], feature_df: pd.DataFrame, labels: np.ndarray, index_to_class: Dict[int, str]) -> List[str]:
    medians = {}
    for cls in classes:
        cls_idx = [i for i, name in index_to_class.items() if name == cls][0]
        vals = feature_df.loc[labels == cls_idx, "m2"].to_numpy()
        medians[cls] = float(np.nanmedian(vals))
    return [cls for cls, _ in sorted(medians.items(), key=lambda kv: kv[1])]


def beta_from_mass_and_p(mass_mev: float, p_mev: np.ndarray) -> np.ndarray:
    energy = np.sqrt(np.maximum(p_mev, 0.0) ** 2 + mass_mev ** 2)
    return np.divide(p_mev, energy, out=np.zeros_like(p_mev, dtype=float), where=energy > 0.0)


def build_coarse_thresholds(train_df: pd.DataFrame, train_y: np.ndarray, physical_order: Sequence[str], class_to_index: Mapping[str, int]) -> Tuple[List[float], List[float], Dict[str, float]]:
    medians_m2: Dict[str, float] = {}
    medians_beta: Dict[str, float] = {}
    for cls in physical_order:
        idx = class_to_index[cls]
        medians_m2[cls] = float(np.nanmedian(train_df.loc[train_y == idx, "m2"]))
        medians_beta[cls] = float(np.nanmedian(train_df.loc[train_y == idx, "beta"]))

    coarse_thresholds = []
    beta_thresholds = []
    for low, high in zip(physical_order[:-1], physical_order[1:]):
        coarse_thresholds.append(0.5 * (medians_m2[low] + medians_m2[high]))
        beta_thresholds.append(0.5 * (medians_beta[low] + medians_beta[high]))

    return coarse_thresholds, beta_thresholds, medians_m2


def find_best_threshold_for_pair(
    x: np.ndarray,
    y_binary: np.ndarray,
    high_when_greater: bool = True,
) -> Tuple[float, str, float]:
    """
    y_binary: 0 for low class, 1 for high class.
    Returns threshold, polarity, balanced accuracy.
    """
    x = np.asarray(x, dtype=float)
    y_binary = np.asarray(y_binary, dtype=int)

    valid = np.isfinite(x)
    x = x[valid]
    y_binary = y_binary[valid]
    if x.size == 0 or len(np.unique(y_binary)) < 2:
        return 0.0, "greater_is_high", 0.5

    candidates = np.unique(np.quantile(x, np.linspace(0.05, 0.95, 91)))
    best = (float(candidates[0]), "greater_is_high" if high_when_greater else "greater_is_low", -1.0)

    for threshold in candidates:
        pred_high = x > threshold
        for polarity in ("greater_is_high", "greater_is_low"):
            pred = pred_high.astype(int) if polarity == "greater_is_high" else (~pred_high).astype(int)
            tp = ((pred == 1) & (y_binary == 1)).sum()
            tn = ((pred == 0) & (y_binary == 0)).sum()
            p = (y_binary == 1).sum()
            n = (y_binary == 0).sum()
            tpr = tp / p if p else 0.0
            tnr = tn / n if n else 0.0
            bal_acc = 0.5 * (tpr + tnr)
            if bal_acc > best[2]:
                best = (float(threshold), polarity, float(bal_acc))
    return best


def derive_pair_rules(
    train_df: pd.DataFrame,
    train_y: np.ndarray,
    val_df: pd.DataFrame,
    val_y: np.ndarray,
    physical_order: Sequence[str],
    class_to_index: Mapping[str, int],
    config: Dict[str, Any],
) -> Tuple[Dict[Tuple[str, str], PairRule], Dict[str, Any]]:
    rules: Dict[Tuple[str, str], PairRule] = {}
    tuning_summary: Dict[str, Any] = {"pair_rules": {}}

    dedx_variable = config["refinement"]["variable"]
    dedx_deadband = float(config["refinement"]["dedx_unknown_margin"])
    energy_deadband = float(config["refinement"]["energy_unknown_margin"])
    use_energy = bool(config["refinement"]["use_cluster_energy_tiebreaker"])

    for low_cls, high_cls in zip(physical_order[:-1], physical_order[1:]):
        low_idx = class_to_index[low_cls]
        high_idx = class_to_index[high_cls]

        pair_train = train_df[(train_y == low_idx) | (train_y == high_idx)].copy()
        pair_val = val_df[(val_y == low_idx) | (val_y == high_idx)].copy()
        train_binary = np.where(train_y[(train_y == low_idx) | (train_y == high_idx)] == high_idx, 1, 0)
        val_binary = np.where(val_y[(val_y == low_idx) | (val_y == high_idx)] == high_idx, 1, 0)

        if dedx_variable == "dedx_score":
            x_train = pair_train["log_dEdx"].to_numpy() - pair_train["log_p"].to_numpy()
            x_val = pair_val["log_dEdx"].to_numpy() - pair_val["log_p"].to_numpy()
        else:
            x_train = pair_train[dedx_variable].to_numpy()
            x_val = pair_val[dedx_variable].to_numpy()

        thr_train, pol_train, train_score = find_best_threshold_for_pair(x_train, train_binary)
        # Validation refinement in a local window around train threshold.
        val_candidates = np.unique(
            np.concatenate(
                [
                    [thr_train],
                    np.quantile(x_val[np.isfinite(x_val)], np.linspace(0.10, 0.90, 41)) if np.isfinite(x_val).any() else np.array([thr_train]),
                    np.linspace(thr_train - 0.25, thr_train + 0.25, 25),
                ]
            )
        )
        best_val = (thr_train, pol_train, -1.0)
        for threshold in val_candidates:
            pred_high = x_val > threshold
            for polarity in ("greater_is_high", "greater_is_low"):
                pred = pred_high.astype(int) if polarity == "greater_is_high" else (~pred_high).astype(int)
                tp = ((pred == 1) & (val_binary == 1)).sum()
                tn = ((pred == 0) & (val_binary == 0)).sum()
                p = (val_binary == 1).sum()
                n = (val_binary == 0).sum()
                bal_acc = 0.5 * ((tp / p) if p else 0.0 + (tn / n) if n else 0.0)
                if bal_acc > best_val[2]:
                    best_val = (float(threshold), polarity, float(bal_acc))

        energy_threshold = None
        energy_polarity = None
        if use_energy:
            e_train = pair_train["log_cluster_energy"].to_numpy()
            e_binary = train_binary
            energy_threshold, energy_polarity, _ = find_best_threshold_for_pair(e_train, e_binary)

        rule = PairRule(
            low_class=low_cls,
            high_class=high_cls,
            variable=dedx_variable,
            threshold=float(best_val[0]),
            polarity=best_val[1],
            deadband=dedx_deadband,
            energy_threshold=float(energy_threshold) if energy_threshold is not None else None,
            energy_polarity=energy_polarity,
            energy_deadband=energy_deadband,
        )
        rules[(low_cls, high_cls)] = rule
        tuning_summary["pair_rules"][f"{low_cls}__{high_cls}"] = {
            "train_balanced_accuracy": train_score,
            "val_balanced_accuracy": best_val[2],
            "threshold": rule.threshold,
            "polarity": rule.polarity,
            "deadband": rule.deadband,
            "energy_threshold": rule.energy_threshold,
            "energy_polarity": rule.energy_polarity,
        }

    return rules, tuning_summary


def classify_coarse(
    df: pd.DataFrame,
    physical_order: Sequence[str],
    coarse_thresholds: Sequence[float],
    beta_thresholds: Optional[Sequence[float]],
    ambiguity_margin_scale: float,
    beta_guard_margin: float,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[Tuple[str, str]]]]:
    m2 = df["m2"].to_numpy()
    beta = df["beta"].to_numpy()

    n_classes = len(physical_order)
    coarse_idx = np.searchsorted(np.asarray(coarse_thresholds), m2, side="right")
    coarse_idx = np.clip(coarse_idx, 0, n_classes - 1)

    coarse_classes = np.array([physical_order[i] for i in coarse_idx], dtype=object)
    ambiguous = np.zeros(len(df), dtype=bool)
    pair_context: List[Optional[Tuple[str, str]]] = [None] * len(df)

    for i in range(len(df)):
        cls_pos = coarse_idx[i]
        distances = []
        candidate_pairs: List[Tuple[float, Tuple[str, str]]] = []

        if cls_pos > 0:
            thr = coarse_thresholds[cls_pos - 1]
            gap = abs(coarse_thresholds[cls_pos - 1] - (coarse_thresholds[cls_pos - 2] if cls_pos - 2 >= 0 else thr))
            gap = gap if gap > 0 else abs(thr) if thr != 0 else 1.0
            margin = ambiguity_margin_scale * gap
            dist = abs(m2[i] - thr)
            distances.append(dist / max(margin, 1e-12))
            candidate_pairs.append((dist, (physical_order[cls_pos - 1], physical_order[cls_pos])))

        if cls_pos < n_classes - 1:
            thr = coarse_thresholds[cls_pos]
            gap = abs((coarse_thresholds[cls_pos + 1] if cls_pos + 1 < len(coarse_thresholds) else thr) - thr)
            gap = gap if gap > 0 else abs(thr) if thr != 0 else 1.0
            margin = ambiguity_margin_scale * gap
            dist = abs(m2[i] - thr)
            distances.append(dist / max(margin, 1e-12))
            candidate_pairs.append((dist, (physical_order[cls_pos], physical_order[cls_pos + 1])))

        beta_flag = False
        if beta_thresholds is not None and 0 < cls_pos < n_classes:
            if cls_pos - 1 < len(beta_thresholds):
                low_beta_thr = beta_thresholds[cls_pos - 1]
                beta_flag = beta[i] > low_beta_thr + beta_guard_margin if np.isfinite(beta[i]) else False
            if cls_pos < len(beta_thresholds):
                high_beta_thr = beta_thresholds[cls_pos]
                beta_flag = beta_flag or (beta[i] < high_beta_thr - beta_guard_margin if np.isfinite(beta[i]) else False)

        if any(x <= 1.0 for x in distances) or beta_flag:
            ambiguous[i] = True
            if candidate_pairs:
                pair_context[i] = min(candidate_pairs, key=lambda t: t[0])[1]

    return coarse_classes, ambiguous, pair_context


def apply_pair_rule(row: pd.Series, rule: PairRule) -> Tuple[Optional[str], Dict[str, Any]]:
    if rule.variable == "dedx_score":
        score = float(row["log_dEdx"] - row["log_p"])
    else:
        score = float(row[rule.variable])

    decision: Optional[str]
    info = {"primary_score": score, "energy_score": None, "used_tiebreaker": False}

    lower = rule.threshold - rule.deadband
    upper = rule.threshold + rule.deadband

    if rule.polarity == "greater_is_high":
        if score < lower:
            decision = rule.low_class
        elif score > upper:
            decision = rule.high_class
        else:
            decision = None
    else:
        if score > upper:
            decision = rule.low_class
        elif score < lower:
            decision = rule.high_class
        else:
            decision = None

    if decision is not None:
        return decision, info

    if rule.energy_threshold is None:
        return None, info

    e_score = float(row["log_cluster_energy"])
    info["energy_score"] = e_score
    info["used_tiebreaker"] = True
    e_lower = rule.energy_threshold - rule.energy_deadband
    e_upper = rule.energy_threshold + rule.energy_deadband

    if rule.energy_polarity == "greater_is_high":
        if e_score < e_lower:
            return rule.low_class, info
        if e_score > e_upper:
            return rule.high_class, info
    else:
        if e_score > e_upper:
            return rule.low_class, info
        if e_score < e_lower:
            return rule.high_class, info

    return None, info


def predict_with_cut_model(model: CutModel, df: pd.DataFrame) -> Dict[str, Any]:
    coarse_classes, ambiguous_initial, pair_context = classify_coarse(
        df=df,
        physical_order=model.physical_order,
        coarse_thresholds=model.coarse_thresholds,
        beta_thresholds=model.beta_thresholds,
        ambiguity_margin_scale=model.ambiguity_margin_scale,
        beta_guard_margin=model.beta_guard_margin,
    )

    refined_classes = coarse_classes.copy()
    ambiguous_after_dedx = np.zeros(len(df), dtype=bool)
    unknown_pair_names: List[Optional[str]] = [None] * len(df)

    for i, is_amb in enumerate(ambiguous_initial):
        if not is_amb:
            continue
        pair = pair_context[i]
        if pair is None:
            ambiguous_after_dedx[i] = True
            continue
        rule = model.pair_rules[pair]
        decision, _ = apply_pair_rule(df.iloc[i], rule)
        if decision is None:
            ambiguous_after_dedx[i] = True
            unknown_pair_names[i] = f"{pair[0]}__{pair[1]}"
        else:
            refined_classes[i] = decision

    forced_classes = refined_classes.copy()
    for i, is_unknown in enumerate(ambiguous_after_dedx):
        if is_unknown:
            # Forced 5-way assignment rule: fall back to coarse class nearest the coarse m2 region.
            forced_classes[i] = coarse_classes[i]

    y_pred = np.array([model.class_to_index[c] for c in forced_classes], dtype=int)

    return {
        "coarse_classes": coarse_classes,
        "refined_classes": refined_classes,
        "forced_classes": forced_classes,
        "y_pred": y_pred,
        "ambiguous_initial": ambiguous_initial,
        "ambiguous_after_dedx": ambiguous_after_dedx,
        "unknown_pair_names": np.array(unknown_pair_names, dtype=object),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    class_to_index: Mapping[str, int],
    index_to_class: Mapping[int, str],
) -> Dict[str, Any]:
    labels = [class_to_index[c] for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": None,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class": per_class,
    }


def compute_hard_region_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_to_index: Mapping[str, int],
) -> Dict[str, Any]:
    d_idx = class_to_index["deuteron"]
    he4_idx = class_to_index["helium4"]
    mask = np.isin(y_true, [d_idx, he4_idx])

    yt = y_true[mask]
    yp = y_pred[mask]
    cm = confusion_matrix(yt, yp, labels=[d_idx, he4_idx])

    recall_d = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    recall_he4 = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
    misclassification_rate = ((cm[0, 1] + cm[1, 0]) / cm.sum()) if cm.sum() else 0.0

    return {
        "n_rows": int(mask.sum()),
        "confusion_matrix": cm.tolist(),
        "recall_deuteron": float(recall_d),
        "recall_helium4": float(recall_he4),
        "mutual_misclassification_rate": float(misclassification_rate),
    }


def compute_ambiguity_analysis(
    y_true: np.ndarray,
    pred_bundle: Dict[str, Any],
    classes: Sequence[str],
    class_to_index: Mapping[str, int],
) -> Dict[str, Any]:
    initial = pred_bundle["ambiguous_initial"]
    after = pred_bundle["ambiguous_after_dedx"]
    result = {
        "n_total": int(len(y_true)),
        "n_ambiguous_initial": int(initial.sum()),
        "fraction_ambiguous_initial": float(initial.mean()),
        "n_ambiguous_after_dedx": int(after.sum()),
        "fraction_ambiguous_after_dedx": float(after.mean()),
        "ambiguity_reduction_absolute": int(initial.sum() - after.sum()),
        "ambiguity_reduction_fraction_of_initial": float(
            ((initial.sum() - after.sum()) / initial.sum()) if initial.sum() else 0.0
        ),
        "by_true_class": {},
    }
    for cls in classes:
        idx = class_to_index[cls]
        cls_mask = y_true == idx
        result["by_true_class"][cls] = {
            "n_rows": int(cls_mask.sum()),
            "initial_ambiguous_fraction": float(initial[cls_mask].mean()) if cls_mask.sum() else 0.0,
            "after_dedx_ambiguous_fraction": float(after[cls_mask].mean()) if cls_mask.sum() else 0.0,
        }
    return result


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_features_by_class(df: pd.DataFrame, y: np.ndarray, classes: Sequence[str], class_to_index: Mapping[str, int], out_path: Path) -> None:
    rows = []
    for cls in classes:
        sub = df.loc[y == class_to_index[cls], ["p", "beta", "m2", "dEdx", "dedx_recomputed", "cluster_energy"]]
        row = {"class": cls, "n_rows": int(len(sub))}
        for col in sub.columns:
            row[f"{col}_median"] = float(np.nanmedian(sub[col]))
            row[f"{col}_q16"] = float(np.nanquantile(sub[col], 0.16))
            row[f"{col}_q84"] = float(np.nanquantile(sub[col], 0.84))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def format_confusion_markdown(cm: List[List[float]], classes: Sequence[str], float_fmt: str = ".4f") -> str:
    header = "| true \\ pred | " + " | ".join(classes) + " |\n"
    header += "|---|" + "|".join(["---"] * len(classes)) + "|\n"
    lines = []
    for cls, row in zip(classes, cm):
        vals = " | ".join(f"{float(v):{float_fmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in row)
        lines.append(f"| {cls} | {vals} |")
    return header + "\n".join(lines)


def save_metrics_markdown(
    out_path: Path,
    metrics: Dict[str, Any],
    hard_region: Dict[str, Any],
    ambiguity: Dict[str, Any],
    classes: Sequence[str],
) -> None:
    lines = [
        "# Phase D Cut Model Test Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        "- Log loss: not reported for this explicit cut model.",
        "",
        "## Per-class precision / recall",
        "",
        "| class | precision | recall | support |",
        "|---|---:|---:|---:|",
    ]
    for cls in classes:
        m = metrics["per_class"][cls]
        lines.append(f"| {cls} | {m['precision']:.6f} | {m['recall']:.6f} | {m['support']} |")

    lines.extend(
        [
            "",
            "## Confusion matrix (counts)",
            "",
            format_confusion_markdown(metrics["confusion_matrix"], classes, ".0f"),
            "",
            "## Confusion matrix (row-normalized)",
            "",
            format_confusion_markdown(metrics["confusion_matrix_normalized"], classes, ".4f"),
            "",
            "## Hard region: deuteron vs helium4",
            "",
            f"- Rows in subset: {hard_region['n_rows']}",
            f"- Recall (deuteron): {hard_region['recall_deuteron']:.6f}",
            f"- Recall (helium4): {hard_region['recall_helium4']:.6f}",
            f"- Mutual misclassification rate: {hard_region['mutual_misclassification_rate']:.6f}",
            "",
            "2x2 confusion matrix order: [deuteron, helium4]",
            "",
            str(hard_region["confusion_matrix"]),
            "",
            "## Ambiguity analysis",
            "",
            f"- Initial ambiguous rows: {ambiguity['n_ambiguous_initial']} ({ambiguity['fraction_ambiguous_initial']:.6f})",
            f"- Ambiguous after dEdx refinement: {ambiguity['n_ambiguous_after_dedx']} ({ambiguity['fraction_ambiguous_after_dedx']:.6f})",
            f"- Ambiguity reduction (absolute): {ambiguity['ambiguity_reduction_absolute']}",
            f"- Ambiguity reduction / initial: {ambiguity['ambiguity_reduction_fraction_of_initial']:.6f}",
        ]
    )
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_cut_summary(
    out_path: Path,
    model: CutModel,
    config: Dict[str, Any],
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    train_pred: Dict[str, Any],
    val_pred: Dict[str, Any],
    test_pred: Dict[str, Any],
) -> None:
    lines = [
        "# Phase D — baseline 1: physics-inspired cut model",
        "",
        "## Scope",
        "",
        "This is an explicit, hand-built hierarchical ruleset. It is not a trained classifier.",
        "",
        "## Canonical inputs used",
        "",
        "- Primary coarse variables: `m2` with optional `beta` consistency guard.",
        "- Refinement variables: `dEdx` via `dedx_score = log_dEdx - log_p`.",
        "- Optional tie-breaker: `cluster_energy` via `log_cluster_energy`.",
        "",
        "## Dataset filtering for this baseline",
        "",
        "- Rows with any invalid/masked feature are excluded entirely from train / val / test.",
        f"- Train rows kept: {len(train_bundle.labels)}",
        f"- Val rows kept: {len(val_bundle.labels)}",
        f"- Test rows kept: {len(test_bundle.labels)}",
        "",
        "## Decision hierarchy",
        "",
        "1. Coarse separation by `m2` into ordered species bands.",
        "2. Mark rows near class boundaries or inconsistent with beta guard as internally ambiguous.",
        "3. For ambiguous rows, resolve the neighboring class pair using explicit dEdx-based thresholds.",
        "4. If still unresolved after dEdx (and optional cluster-energy tie-breaker), mark as internal unknown.",
        "5. For final required 5-way scoring, force unresolved rows back to the nearest coarse class.",
        "",
        "## Coarse thresholds",
        "",
        f"- Physical order used internally: {model.physical_order}",
        f"- Coarse variable: `{model.coarse_variable}`",
        f"- m2 thresholds: {json.dumps(model.coarse_thresholds)}",
        f"- beta thresholds (diagnostic guard): {json.dumps(model.beta_thresholds)}",
        f"- Ambiguity margin scale: {model.ambiguity_margin_scale}",
        f"- Beta guard margin: {model.beta_guard_margin}",
        "",
        "## Pair refinement rules",
        "",
    ]
    for pair, rule in model.pair_rules.items():
        lines.extend(
            [
                f"### {pair[0]} vs {pair[1]}",
                "",
                f"- Primary variable: `{rule.variable}`",
                f"- Threshold: {rule.threshold:.6f}",
                f"- Polarity: `{rule.polarity}`",
                f"- Deadband: ±{rule.deadband:.6f}",
                f"- Cluster-energy tie-breaker threshold: {rule.energy_threshold if rule.energy_threshold is not None else 'disabled'}",
                f"- Cluster-energy tie-breaker polarity: {rule.energy_polarity if rule.energy_polarity is not None else 'disabled'}",
                f"- Cluster-energy deadband: ±{rule.energy_deadband:.6f}",
                "",
            ]
        )

    def add_ambiguity_block(name: str, pred: Dict[str, Any]) -> List[str]:
        return [
            f"## Ambiguity summary: {name}",
            "",
            f"- Initial ambiguous rows: {int(pred['ambiguous_initial'].sum())}",
            f"- Initial ambiguous fraction: {float(pred['ambiguous_initial'].mean()):.6f}",
            f"- Ambiguous after dEdx: {int(pred['ambiguous_after_dedx'].sum())}",
            f"- Ambiguous after dEdx fraction: {float(pred['ambiguous_after_dedx'].mean()):.6f}",
            "",
        ]

    lines.extend(add_ambiguity_block("train", train_pred))
    lines.extend(add_ambiguity_block("val", val_pred))
    lines.extend(add_ambiguity_block("test", test_pred))
    lines.extend(
        [
            "## Log loss note",
            "",
            "Log loss is not reported. This baseline is an explicit cut model and does not emit calibrated or probabilistic outputs.",
            "",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sample_per_class(df: pd.DataFrame, y: np.ndarray, classes: Sequence[str], class_to_index: Mapping[str, int], max_points: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for cls in classes:
        idx = np.flatnonzero(y == class_to_index[cls])
        if len(idx) > max_points:
            idx = rng.choice(idx, size=max_points, replace=False)
        part = df.iloc[idx].copy()
        part["true_class"] = cls
        parts.append(part)
    return pd.concat(parts, axis=0, ignore_index=True)


def add_coarse_boundaries(ax: plt.Axes, model: CutModel, xvar: str, yvar: str, xlim: Tuple[float, float]) -> None:
    if yvar == "m2" and xvar == "p":
        for thr in model.coarse_thresholds:
            ax.axhline(thr, linestyle="--", linewidth=1.0, alpha=0.8)
    elif yvar == "beta" and xvar == "p" and model.beta_thresholds is not None:
        for thr in model.beta_thresholds:
            ax.axhline(thr, linestyle="--", linewidth=1.0, alpha=0.8)
        # Optional physics guide curves from nominal masses
        p_grid = np.linspace(max(1.0, xlim[0]), xlim[1], 500)
        for cls in model.physical_order:
            beta_curve = beta_from_mass_and_p(CLASS_MASSES_MEV[cls], p_grid)
            ax.plot(p_grid, beta_curve, linewidth=1.0, alpha=0.75, label=f"{cls} ref")
    elif yvar == "dEdx" and xvar == "p":
        # Show approximate pair thresholds in dedx_score space as text note only; direct overlay in dEdx-vs-p is not exact.
        txt = "Pair refinement uses dedx_score = log_dEdx - log_p"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8)


def scatter_by_class(
    df: pd.DataFrame,
    y: np.ndarray,
    classes: Sequence[str],
    class_to_index: Mapping[str, int],
    xvar: str,
    yvar: str,
    out_path: Path,
    model: Optional[CutModel],
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> None:
    plot_df = sample_per_class(
        df, y, classes, class_to_index, config["plotting"]["max_points_per_class"], rng
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in classes:
        sub = plot_df[plot_df["true_class"] == cls]
        ax.scatter(
            sub[xvar],
            sub[yvar],
            s=config["plotting"]["marker_size"],
            alpha=config["plotting"]["scatter_alpha"],
            label=cls,
        )

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(f"{xvar} vs {yvar} by true class")
    if model is not None:
        xlim = (float(np.nanmin(plot_df[xvar])), float(np.nanmax(plot_df[xvar])))
        add_coarse_boundaries(ax, model, xvar, yvar, xlim)
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config["plotting"]["dpi"])
    plt.close(fig)


def build_model(
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    config: Dict[str, Any],
    label_map: Dict[str, Any],
) -> CutModel:
    classes = label_map["classes"]
    class_to_index = {k: int(v) for k, v in label_map["class_to_index"].items()}
    index_to_class = {int(k): v for k, v in label_map["index_to_class"].items()}

    physical_order = physical_class_order(classes, train_bundle.features, train_bundle.labels, index_to_class)
    coarse_thresholds, beta_thresholds, _ = build_coarse_thresholds(
        train_bundle.features, train_bundle.labels, physical_order, class_to_index
    )
    pair_rules, tuning_summary = derive_pair_rules(
        train_bundle.features,
        train_bundle.labels,
        val_bundle.features,
        val_bundle.labels,
        physical_order,
        class_to_index,
        config,
    )

    # Tune ambiguity margin scale on validation by explicit scan.
    coarse_cfg = config["coarse"]
    margin_candidates = np.unique(
        np.concatenate(
            [
                np.array([float(coarse_cfg["ambiguity_margin_scale"])]),
                np.linspace(0.04, 0.24, 11),
            ]
        )
    )
    best_margin = float(coarse_cfg["ambiguity_margin_scale"])
    best_score = -1.0
    best_summary = {}
    for margin in margin_candidates:
        tmp_model = CutModel(
            label_map=label_map,
            classes=classes,
            class_to_index=class_to_index,
            index_to_class=index_to_class,
            physical_order=physical_order,
            coarse_variable=str(coarse_cfg["primary_variable"]),
            coarse_thresholds=coarse_thresholds,
            beta_thresholds=beta_thresholds if coarse_cfg["use_beta_fallback"] else None,
            ambiguity_margin_scale=float(margin),
            beta_guard_margin=float(coarse_cfg["beta_guard_margin"]),
            pair_rules=pair_rules,
            tuning_summary={},
        )
        pred = predict_with_cut_model(tmp_model, val_bundle.features)
        overall_acc = accuracy_score(val_bundle.labels, pred["y_pred"])
        hard = compute_hard_region_metrics(val_bundle.labels, pred["y_pred"], class_to_index)
        score = 2.0 * hard["recall_deuteron"] + 2.0 * hard["recall_helium4"] + overall_acc
        if score > best_score:
            best_score = score
            best_margin = float(margin)
            best_summary = {
                "validation_selection_score": score,
                "validation_accuracy": float(overall_acc),
                "validation_hard_region": hard,
            }

    tuning_summary["best_ambiguity_margin_scale"] = best_margin
    tuning_summary["validation_model_selection"] = best_summary

    return CutModel(
        label_map=label_map,
        classes=classes,
        class_to_index=class_to_index,
        index_to_class=index_to_class,
        physical_order=physical_order,
        coarse_variable=str(coarse_cfg["primary_variable"]),
        coarse_thresholds=coarse_thresholds,
        beta_thresholds=beta_thresholds if coarse_cfg["use_beta_fallback"] else None,
        ambiguity_margin_scale=best_margin,
        beta_guard_margin=float(coarse_cfg["beta_guard_margin"]),
        pair_rules=pair_rules,
        tuning_summary=tuning_summary,
    )


def make_output_dirs(base: Path) -> Dict[str, Path]:
    subdirs = {
        "base": base,
        "plots": base / "plots",
        "metrics": base / "metrics",
        "summaries": base / "summaries",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def main() -> None:
    parser = argparse.ArgumentParser(description="ALERT Phase D physics-inspired cut model")
    parser.add_argument("--train", required=True, help="Path to train.h5")
    parser.add_argument("--val", required=True, help="Path to val.h5")
    parser.add_argument("--test", required=True, help="Path to test.h5")
    parser.add_argument("--config", default=None, help="Path to YAML or JSON config")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(int(config["random_seed"]))
    label_map = load_label_map()
    outdirs = make_output_dirs(Path(args.output_dir))

    train_bundle = load_h5_dataset(args.train, "train")
    val_bundle = load_h5_dataset(args.val, "val")
    test_bundle = load_h5_dataset(args.test, "test")

    model = build_model(train_bundle, val_bundle, config, label_map)

    train_pred = predict_with_cut_model(model, train_bundle.features)
    val_pred = predict_with_cut_model(model, val_bundle.features)
    test_pred = predict_with_cut_model(model, test_bundle.features)

    metrics = compute_metrics(
        y_true=test_bundle.labels,
        y_pred=test_pred["y_pred"],
        classes=model.classes,
        class_to_index=model.class_to_index,
        index_to_class=model.index_to_class,
    )
    hard_region = compute_hard_region_metrics(test_bundle.labels, test_pred["y_pred"], model.class_to_index)
    ambiguity = compute_ambiguity_analysis(test_bundle.labels, test_pred, model.classes, model.class_to_index)

    full_metrics = {
        "model_type": "physics_inspired_cut_model",
        "contract_feature_order": CANONICAL_FEATURE_ORDER,
        "dataset_sizes_after_mask_filter": {
            "train": int(len(train_bundle.labels)),
            "val": int(len(val_bundle.labels)),
            "test": int(len(test_bundle.labels)),
        },
        "filtered_out_rows_due_to_masks": {
            "train": int(train_bundle.dropped_mask.sum()),
            "val": int(val_bundle.dropped_mask.sum()),
            "test": int(test_bundle.dropped_mask.sum()),
        },
        "cut_model": {
            "physical_order": model.physical_order,
            "coarse_variable": model.coarse_variable,
            "coarse_thresholds": model.coarse_thresholds,
            "beta_thresholds": model.beta_thresholds,
            "ambiguity_margin_scale": model.ambiguity_margin_scale,
            "beta_guard_margin": model.beta_guard_margin,
            "pair_rules": {
                f"{k[0]}__{k[1]}": {
                    "variable": v.variable,
                    "threshold": v.threshold,
                    "polarity": v.polarity,
                    "deadband": v.deadband,
                    "energy_threshold": v.energy_threshold,
                    "energy_polarity": v.energy_polarity,
                    "energy_deadband": v.energy_deadband,
                }
                for k, v in model.pair_rules.items()
            },
            "tuning_summary": model.tuning_summary,
        },
        "metrics_test": metrics,
        "hard_region_test": hard_region,
        "ambiguity_test": ambiguity,
        "notes": {
            "log_loss": "Not reported. This explicit cut model does not define meaningful calibrated probabilities.",
            "forced_assignment_rule": "Rows unresolved after dEdx refinement are forced to the nearest coarse class from the m2 hierarchy.",
        },
    }

    save_json(outdirs["metrics"] / config["output_filenames"]["metrics_json"], full_metrics)
    save_json(outdirs["summaries"] / config["output_filenames"]["ambiguity_json"], ambiguity)
    save_json(outdirs["summaries"] / config["output_filenames"]["val_tuning_json"], model.tuning_summary)
    save_metrics_markdown(
        outdirs["metrics"] / config["output_filenames"]["metrics_markdown"],
        metrics,
        hard_region,
        ambiguity,
        model.classes,
    )
    save_cut_summary(
        outdirs["summaries"] / config["output_filenames"]["cut_summary"],
        model,
        config,
        train_bundle,
        val_bundle,
        test_bundle,
        train_pred,
        val_pred,
        test_pred,
    )
    summarize_features_by_class(
        train_bundle.features,
        train_bundle.labels,
        model.classes,
        model.class_to_index,
        outdirs["summaries"] / config["output_filenames"]["feature_summary_csv"],
    )

    # Diagnostic plots on test set.
    scatter_by_class(
        test_bundle.features,
        test_bundle.labels,
        model.classes,
        model.class_to_index,
        xvar="p",
        yvar="beta",
        out_path=outdirs["plots"] / "p_vs_beta_by_class.png",
        model=None,
        config=config,
        rng=rng,
    )
    scatter_by_class(
        test_bundle.features,
        test_bundle.labels,
        model.classes,
        model.class_to_index,
        xvar="p",
        yvar="beta",
        out_path=outdirs["plots"] / "p_vs_beta_with_boundaries.png",
        model=model,
        config=config,
        rng=rng,
    )
    scatter_by_class(
        test_bundle.features,
        test_bundle.labels,
        model.classes,
        model.class_to_index,
        xvar="p",
        yvar="m2",
        out_path=outdirs["plots"] / "p_vs_m2_by_class.png",
        model=None,
        config=config,
        rng=rng,
    )
    scatter_by_class(
        test_bundle.features,
        test_bundle.labels,
        model.classes,
        model.class_to_index,
        xvar="p",
        yvar="m2",
        out_path=outdirs["plots"] / "p_vs_m2_with_boundaries.png",
        model=model,
        config=config,
        rng=rng,
    )
    scatter_by_class(
        test_bundle.features,
        test_bundle.labels,
        model.classes,
        model.class_to_index,
        xvar="p",
        yvar="dEdx",
        out_path=outdirs["plots"] / "p_vs_dEdx_by_class.png",
        model=None,
        config=config,
        rng=rng,
    )
    scatter_by_class(
        test_bundle.features,
        test_bundle.labels,
        model.classes,
        model.class_to_index,
        xvar="p",
        yvar="dEdx",
        out_path=outdirs["plots"] / "p_vs_dEdx_with_rule_note.png",
        model=model,
        config=config,
        rng=rng,
    )

    print("=== Phase D cut model complete ===")
    print(f"Output directory: {outdirs['base']}")
    print(f"Rows kept after mask filtering: train={len(train_bundle.labels)}, val={len(val_bundle.labels)}, test={len(test_bundle.labels)}")
    print(f"Test accuracy: {metrics['accuracy']:.6f}")
    print("Per-class recall:")
    for cls in model.classes:
        print(f"  {cls:>9s}: {metrics['per_class'][cls]['recall']:.6f}")
    print("Hard region (deuteron vs helium4):")
    print(f"  recall_deuteron = {hard_region['recall_deuteron']:.6f}")
    print(f"  recall_helium4  = {hard_region['recall_helium4']:.6f}")
    print(f"  mutual_misclassification_rate = {hard_region['mutual_misclassification_rate']:.6f}")
    print("Ambiguity:")
    print(f"  initial ambiguous fraction = {ambiguity['fraction_ambiguous_initial']:.6f}")
    print(f"  after dEdx ambiguous fraction = {ambiguity['fraction_ambiguous_after_dedx']:.6f}")


if __name__ == "__main__":
    main()