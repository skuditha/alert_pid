
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_MM_PER_NS = 299.792458
DEFAULT_FEATURE_NAMES = [
    "px", "py", "pz", "p", "pt", "theta", "phi", "vx", "vy", "vz", "vr", "v3",
    "n_hits", "sum_adc", "path", "dEdx", "dedx_recomputed", "p_drift", "sum_residuals",
    "residual_per_hit", "adc_per_hit", "tof_time", "pathlength", "cluster_x", "cluster_y",
    "cluster_z", "cluster_energy", "n_bar", "n_wedge", "beta", "m2", "log_p", "log_pt",
    "log_sum_adc", "log_path", "log_dEdx", "log_dedx_recomputed", "log_cluster_energy",
]

KEY_PHYSICS_FEATURES = [
    "p", "pt", "tof_time", "pathlength", "dEdx", "dedx_recomputed", "cluster_energy",
    "beta", "m2", "n_hits",
]

PAIR_FOCUS = ("deuteron", "helium4")


@dataclass
class AuditDataset:
    paths: List[Path]
    feature_names: List[str]
    values: np.ndarray
    masks: np.ndarray
    class_index: np.ndarray
    truth_pid: np.ndarray
    row_meta: Dict[str, np.ndarray]
    attrs: Dict[str, str]
    class_names: List[str]

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)

    def valid_feature_array(self, name: str) -> np.ndarray:
        idx = self.feature_index(name)
        arr = self.values[:, idx].astype(np.float64, copy=True)
        valid = self.masks[:, idx].astype(bool)
        arr[~valid] = np.nan
        return arr

    def dataframe(self, features: Optional[Sequence[str]] = None, include_labels: bool = True) -> pd.DataFrame:
        features = list(features) if features is not None else list(self.feature_names)
        data = {name: self.valid_feature_array(name) for name in features}
        df = pd.DataFrame(data)
        if include_labels:
            df["class_index"] = self.class_index
            df["class_name"] = [self.class_names[i] if 0 <= i < len(self.class_names) else f"class_{i}" for i in self.class_index]
            df["truth_pid"] = self.truth_pid
        for key, values in self.row_meta.items():
            df[key] = values
        return df


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_feature_names(group: h5py.Group) -> List[str]:
    if "feature_names_csv" in group.attrs:
        return [x.strip() for x in _decode_attr(group.attrs["feature_names_csv"]).split(",") if x.strip()]
    return list(DEFAULT_FEATURE_NAMES)


def load_label_map(label_map_path: str | Path) -> List[str]:
    with open(label_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    index_to_class = payload.get("index_to_class", {})
    return [index_to_class[str(i)] for i in sorted(int(k) for k in index_to_class)]


def load_audit_dataset(h5_paths: Sequence[str | Path], label_map_path: str | Path) -> AuditDataset:
    paths = [Path(p) for p in h5_paths]
    if not paths:
        raise ValueError("At least one HDF5 file is required.")

    feature_names: Optional[List[str]] = None
    values_list, masks_list = [], []
    class_index_list, truth_pid_list = [], []
    row_meta_cols: Dict[str, List[np.ndarray]] = {}
    attrs: Dict[str, str] = {}

    for path in paths:
        with h5py.File(path, "r") as h5:
            current_names = _read_feature_names(h5["/features"])
            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise ValueError(f"Feature name mismatch in {path}")

            values_list.append(h5["/features/values"][:])
            masks_list.append(h5["/features/masks"][:].astype(np.uint8))
            class_index_list.append(h5["/labels/class_index"][:].astype(np.int32))
            truth_pid_list.append(h5["/labels/truth_pid"][:].astype(np.int32))

            for key, ds in h5["/row_meta"].items():
                row_meta_cols.setdefault(key, []).append(ds[:])

            for key, value in h5["/dataset_meta"].attrs.items():
                attrs.setdefault(key, _decode_attr(value))

    class_names = load_label_map(label_map_path)
    row_meta = {k: np.concatenate(v) for k, v in row_meta_cols.items()}
    return AuditDataset(
        paths=paths,
        feature_names=feature_names or list(DEFAULT_FEATURE_NAMES),
        values=np.concatenate(values_list, axis=0).astype(np.float32),
        masks=np.concatenate(masks_list, axis=0).astype(np.uint8),
        class_index=np.concatenate(class_index_list, axis=0),
        truth_pid=np.concatenate(truth_pid_list, axis=0),
        row_meta=row_meta,
        attrs=attrs,
        class_names=class_names,
    )


def compute_feature_summary(ds: AuditDataset) -> pd.DataFrame:
    rows = []
    n_rows = ds.n_rows
    for j, name in enumerate(ds.feature_names):
        vals = ds.values[:, j].astype(np.float64)
        valid = ds.masks[:, j].astype(bool)
        valid_vals = vals[valid]
        row = {
            "feature": name,
            "valid_count": int(valid.sum()),
            "invalid_count": int(n_rows - valid.sum()),
            "valid_fraction": float(valid.mean()),
            "raw_min": float(np.nanmin(vals)) if vals.size else np.nan,
            "raw_max": float(np.nanmax(vals)) if vals.size else np.nan,
            "valid_min": float(np.nanmin(valid_vals)) if valid_vals.size else np.nan,
            "valid_max": float(np.nanmax(valid_vals)) if valid_vals.size else np.nan,
            "valid_mean": float(np.nanmean(valid_vals)) if valid_vals.size else np.nan,
            "valid_std": float(np.nanstd(valid_vals)) if valid_vals.size else np.nan,
            "zeros_in_stored_values": int(np.count_nonzero(vals == 0.0)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["invalid_count", "feature"], ascending=[False, True]).reset_index(drop=True)


def compute_class_balance(ds: AuditDataset) -> pd.DataFrame:
    counts = pd.Series(ds.class_index).value_counts().sort_index()
    rows = []
    total = ds.n_rows
    for idx, count in counts.items():
        name = ds.class_names[idx] if 0 <= idx < len(ds.class_names) else f"class_{idx}"
        rows.append({"class_index": int(idx), "class_name": name, "count": int(count), "fraction": float(count / total)})
    return pd.DataFrame(rows)


def compute_mask_summary(ds: AuditDataset) -> pd.DataFrame:
    per_row_invalid = (ds.masks == 0).sum(axis=1)
    return pd.DataFrame({
        "metric": [
            "rows_with_any_masked_feature",
            "rows_with_no_masked_feature",
            "mean_invalid_features_per_row",
            "max_invalid_features_in_row",
        ],
        "value": [
            int(np.count_nonzero(per_row_invalid > 0)),
            int(np.count_nonzero(per_row_invalid == 0)),
            float(per_row_invalid.mean()),
            int(per_row_invalid.max()),
        ],
    })


def fisher_score(ds: AuditDataset, feature: str) -> float:
    x = ds.valid_feature_array(feature)
    overall_mean = np.nanmean(x)
    numerator = 0.0
    denominator = 0.0
    for c in np.unique(ds.class_index):
        xc = x[ds.class_index == c]
        xc = xc[np.isfinite(xc)]
        if xc.size < 2:
            continue
        mean_c = float(np.mean(xc))
        var_c = float(np.var(xc))
        numerator += xc.size * (mean_c - overall_mean) ** 2
        denominator += xc.size * var_c
    if denominator <= 0:
        return np.nan
    return numerator / denominator


def compute_separation_table(ds: AuditDataset) -> pd.DataFrame:
    rows = []
    for feature in ds.feature_names:
        rows.append({"feature": feature, "fisher_score": fisher_score(ds, feature)})
    return pd.DataFrame(rows).sort_values("fisher_score", ascending=False, na_position="last").reset_index(drop=True)


def compute_correlation_matrix(ds: AuditDataset, features: Optional[Sequence[str]] = None) -> pd.DataFrame:
    features = list(features) if features is not None else list(ds.feature_names)
    df = ds.dataframe(features=features, include_labels=False)[features]
    return df.corr(numeric_only=True)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_a": cols[i], "feature_b": cols[j], "corr": float(value)})
    return pd.DataFrame(rows).sort_values("corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def infer_unit_sanity(ds: AuditDataset) -> pd.DataFrame:
    p = ds.valid_feature_array("p")
    time = ds.valid_feature_array("tof_time")
    pathlength = ds.valid_feature_array("pathlength")
    beta = ds.valid_feature_array("beta")
    m2 = ds.valid_feature_array("m2")

    beta_reco = pathlength / (C_MM_PER_NS * time)
    rows = [
        {"check": "p_median", "value": float(np.nanmedian(p)), "comment": "Large O(1) suggests GeV/c, O(100-1000) suggests MeV/c."},
        {"check": "tof_time_median_ns", "value": float(np.nanmedian(time)), "comment": "Expected ns-scale positive cluster timing."},
        {"check": "pathlength_median", "value": float(np.nanmedian(pathlength)), "comment": "Check whether pathlength looks mm-scale rather than cm-scale."},
        {"check": "beta_median_stored", "value": float(np.nanmedian(beta)), "comment": "Should be comfortably below 1 for most rows."},
        {"check": "beta_median_recomputed_mm_ns", "value": float(np.nanmedian(beta_reco)), "comment": "Recomputed with c = 299.792458 mm/ns."},
        {"check": "frac_beta_gt_1p0_stored", "value": float(np.nanmean(beta > 1.0)), "comment": "Diagnostic only; no row cuts in audit."},
        {"check": "frac_beta_gt_1p0_recomputed", "value": float(np.nanmean(beta_reco > 1.0)), "comment": "High value flags a unit mismatch or timing pathology."},
        {"check": "frac_m2_negative", "value": float(np.nanmean(m2 < 0.0)), "comment": "Negative m2 can occur, but large fractions deserve inspection."},
    ]
    return pd.DataFrame(rows)


def detect_pathologies(ds: AuditDataset) -> pd.DataFrame:
    p = ds.valid_feature_array("p")
    time = ds.valid_feature_array("tof_time")
    pathlength = ds.valid_feature_array("pathlength")
    dedx = ds.valid_feature_array("dEdx")
    energy = ds.valid_feature_array("cluster_energy")
    beta = ds.valid_feature_array("beta")

    pathology_rows = [
        ("nonfinite_stored_values", int(np.count_nonzero(~np.isfinite(ds.values)))),
        ("rows_with_any_nonfinite_stored_value", int(np.count_nonzero(np.any(~np.isfinite(ds.values), axis=1)))),
        ("valid_time_le_zero", int(np.count_nonzero(np.isfinite(time) & (time <= 0)))),
        ("valid_pathlength_le_zero", int(np.count_nonzero(np.isfinite(pathlength) & (pathlength <= 0)))),
        ("valid_p_le_zero", int(np.count_nonzero(np.isfinite(p) & (p <= 0)))),
        ("valid_dEdx_le_zero", int(np.count_nonzero(np.isfinite(dedx) & (dedx <= 0)))),
        ("valid_cluster_energy_le_zero", int(np.count_nonzero(np.isfinite(energy) & (energy <= 0)))),
        ("valid_beta_le_zero", int(np.count_nonzero(np.isfinite(beta) & (beta <= 0)))),
        ("valid_beta_gt_1p2", int(np.count_nonzero(np.isfinite(beta) & (beta > 1.2)))),
    ]
    return pd.DataFrame(pathology_rows, columns=["pathology", "count"])


def pair_focus_summary(ds: AuditDataset, features: Sequence[str] = ("p", "tof_time", "pathlength", "dEdx", "cluster_energy", "beta", "m2")) -> pd.DataFrame:
    classes = {name: idx for idx, name in enumerate(ds.class_names)}
    wanted = [classes[name] for name in PAIR_FOCUS if name in classes]
    rows = []
    for feature in features:
        x = ds.valid_feature_array(feature)
        for class_name in PAIR_FOCUS:
            if class_name not in classes:
                continue
            idx = classes[class_name]
            vals = x[ds.class_index == idx]
            vals = vals[np.isfinite(vals)]
            rows.append({
                "feature": feature,
                "class_name": class_name,
                "count": int(vals.size),
                "median": float(np.median(vals)) if vals.size else np.nan,
                "p16": float(np.percentile(vals, 16)) if vals.size else np.nan,
                "p84": float(np.percentile(vals, 84)) if vals.size else np.nan,
            })
    return pd.DataFrame(rows)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_feature_histograms(ds: AuditDataset, features: Sequence[str], outdir: str | Path, bins: int = 80, logy: bool = False) -> None:
    outdir = _ensure_dir(outdir)
    class_to_name = {i: name for i, name in enumerate(ds.class_names)}
    for feature in features:
        plt.figure(figsize=(8, 5))
        for c in sorted(np.unique(ds.class_index)):
            vals = ds.valid_feature_array(feature)
            vals = vals[ds.class_index == c]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            plt.hist(vals, bins=bins, histtype="step", density=True, label=class_to_name.get(int(c), str(c)))
        plt.xlabel(feature)
        plt.ylabel("density")
        plt.title(f"Per-class distribution: {feature}")
        if logy:
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"hist_{feature}.png", dpi=150)
        plt.close()


def plot_scatter_by_class(ds: AuditDataset, x_feature: str, y_feature: str, outpath: str | Path, max_points: int = 200_000) -> None:
    rng = np.random.default_rng(12345)
    x = ds.valid_feature_array(x_feature)
    y = ds.valid_feature_array(y_feature)
    valid = np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(valid)
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    plt.figure(figsize=(8, 6))
    for c in sorted(np.unique(ds.class_index)):
        class_idx = idx[ds.class_index[idx] == c]
        if class_idx.size == 0:
            continue
        plt.scatter(x[class_idx], y[class_idx], s=3, alpha=0.35, label=ds.class_names[int(c)])
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"{y_feature} vs {x_feature}")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_correlation_heatmap(corr: pd.DataFrame, outpath: str | Path) -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="correlation")
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def run_full_feature_audit(
    h5_paths: Sequence[str | Path],
    label_map_path: str | Path,
    outdir: str | Path,
    histogram_features: Optional[Sequence[str]] = None,
    correlation_features: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    outdir = _ensure_dir(outdir)
    ds = load_audit_dataset(h5_paths, label_map_path)

    feature_summary = compute_feature_summary(ds)
    class_balance = compute_class_balance(ds)
    mask_summary = compute_mask_summary(ds)
    separation = compute_separation_table(ds)
    unit_sanity = infer_unit_sanity(ds)
    pathologies = detect_pathologies(ds)
    pair_focus = pair_focus_summary(ds)

    corr = compute_correlation_matrix(ds, features=correlation_features or ds.feature_names)
    high_corr = high_correlation_pairs(corr)

    tables = {
        "feature_summary": feature_summary,
        "class_balance": class_balance,
        "mask_summary": mask_summary,
        "separation": separation,
        "unit_sanity": unit_sanity,
        "pathologies": pathologies,
        "pair_focus": pair_focus,
        "high_correlation_pairs": high_corr,
    }

    for name, table in tables.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    corr.to_csv(outdir / "correlation_matrix.csv")

    histogram_features = list(histogram_features or KEY_PHYSICS_FEATURES)
    plot_feature_histograms(ds, histogram_features, outdir / "plots" / "histograms")
    plot_scatter_by_class(ds, "p", "beta", outdir / "plots" / "beta_vs_p.png")
    plot_scatter_by_class(ds, "p", "m2", outdir / "plots" / "m2_vs_p.png")
    plot_scatter_by_class(ds, "p", "dEdx", outdir / "plots" / "dEdx_vs_p.png")
    plot_scatter_by_class(ds, "pathlength", "cluster_energy", outdir / "plots" / "cluster_energy_vs_pathlength.png")
    plot_correlation_heatmap(corr, outdir / "plots" / "correlation_heatmap.png")

    return tables
