#!/usr/bin/env python3
"""
Build train/val/test splits for ALERT post-PID data by SOURCE FILE, not by row.

This version is designed for the current HDF5 schema, for example:

  /features/values                  shape (N, F)
  /features/masks                   shape (N, F)
  /labels/class_index               shape (N,)
  /labels/truth_pid                 shape (N,)
  /row_meta/source_file_index       shape (N,)
  /dataset_meta/source_files        shape (n_files,)

Key behavior:
- Reads HDF5 with h5py
- Supports nested dataset paths like "labels/class_index"
- Groups rows by row_meta/source_file_index
- Computes file-level class histograms
- Performs file-level greedy stratified train/val/test split
- Writes row-index arrays for each split
- Writes a JSON summary with sanity checks
- Detects leakage if any source file appears in multiple splits

If a dataset_meta/source_files mapping is present, it is included in logs/summary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more input HDF5 files.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write split outputs.",
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Target train fraction.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Target val fraction.",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Target test fraction.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed.",
    )

    p.add_argument(
        "--label-col",
        default="labels/class_index",
        help="Nested HDF5 dataset path for labels. Default: labels/class_index",
    )
    p.add_argument(
        "--source-file-col",
        default="row_meta/source_file_index",
        help="Nested HDF5 dataset path for per-row source file index. Default: row_meta/source_file_index",
    )
    p.add_argument(
        "--source-files-dset",
        default="dataset_meta/source_files",
        help="Optional nested HDF5 dataset path for source file index -> filename mapping.",
    )

    p.add_argument(
        "--max-rows-per-input",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    p.add_argument(
        "--summary-json",
        default="split_summary.json",
        help="Name of summary JSON file to write in output-dir.",
    )
    p.add_argument(
        "--print-file-map",
        action="store_true",
        help="Print source_file_index -> filename mapping when available.",
    )

    return p.parse_args()


def decode_if_bytes(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return x


def read_dataset_exact_or_leaf(f: h5py.File, dataset_path: str) -> h5py.Dataset:
    """
    Resolve dataset by exact nested path first.
    If that fails and the provided name is a leaf name, recursively search by leaf.
    """
    if dataset_path in f:
        obj = f[dataset_path]
        if isinstance(obj, h5py.Dataset):
            return obj
        raise KeyError(f"Path exists but is not a dataset: {dataset_path}")

    # Fallback: recursive leaf-name search
    matches = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            if name == dataset_path or name.split("/")[-1] == dataset_path:
                matches.append(name)

    f.visititems(visitor)

    if not matches:
        raise KeyError(f"Could not find dataset '{dataset_path}'")
    if len(matches) > 1:
        raise KeyError(
            f"Dataset '{dataset_path}' is ambiguous. Matches: {matches}. "
            f"Please pass the full nested path."
        )

    return f[matches[0]]


def read_1d_dataset(
    f: h5py.File,
    dataset_path: str,
    max_rows: Optional[int] = None,
) -> np.ndarray:
    ds = read_dataset_exact_or_leaf(f, dataset_path)
    arr = ds[:]
    if arr.ndim != 1:
        raise ValueError(f"Dataset '{dataset_path}' must be 1D, got shape {arr.shape}")
    if max_rows is not None:
        arr = arr[:max_rows]
    return arr


def maybe_read_source_file_names(
    f: h5py.File,
    dataset_path: Optional[str],
) -> Optional[List[str]]:
    if not dataset_path:
        return None
    try:
        ds = read_dataset_exact_or_leaf(f, dataset_path)
    except KeyError:
        return None

    arr = ds[:]
    if arr.ndim != 1:
        raise ValueError(
            f"Source files dataset '{dataset_path}' must be 1D, got shape {arr.shape}"
        )
    return [str(decode_if_bytes(x)) for x in arr]


def load_input_rows(
    path: Path,
    input_id: int,
    label_col: str,
    source_file_col: str,
    source_files_dset: Optional[str],
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    with h5py.File(path, "r") as f:
        labels = read_1d_dataset(f, label_col, max_rows=max_rows)
        source_file_index = read_1d_dataset(f, source_file_col, max_rows=max_rows)
        source_files = maybe_read_source_file_names(f, source_files_dset)

    if len(labels) != len(source_file_index):
        raise ValueError(
            f"Mismatched lengths in {path}: "
            f"{label_col}={len(labels)} vs {source_file_col}={len(source_file_index)}"
        )

    labels = np.asarray(labels)
    source_file_index = np.asarray(source_file_index)

    # Normalize types
    if labels.dtype.kind not in ("i", "u"):
        labels = labels.astype(np.int64)
    if source_file_index.dtype.kind not in ("i", "u"):
        source_file_index = source_file_index.astype(np.int64)

    row_index = np.arange(len(labels), dtype=np.int64)

    df = pd.DataFrame(
        {
            "input_id": input_id,
            "input_path": str(path),
            "row_index": row_index,
            "label": labels.astype(np.int64),
            "source_file_index": source_file_index.astype(np.int64),
        }
    )

    # Namespace the grouping key by input file so multiple H5 inputs do not collide.
    df["group_key"] = df["input_id"].astype(str) + ":" + df["source_file_index"].astype(str)

    return df, source_files


def aggregate_file_stats(
    inputs: List[Path],
    label_col: str,
    source_file_col: str,
    source_files_dset: Optional[str],
    max_rows_per_input: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Optional[List[str]]], List[int]]:
    row_chunks = []
    source_files_by_input: Dict[int, Optional[List[str]]] = {}

    for input_id, path in enumerate(inputs):
        rows_df, source_files = load_input_rows(
            path=path,
            input_id=input_id,
            label_col=label_col,
            source_file_col=source_file_col,
            source_files_dset=source_files_dset,
            max_rows=max_rows_per_input,
        )
        row_chunks.append(rows_df)
        source_files_by_input[input_id] = source_files

    if not row_chunks:
        raise RuntimeError("No rows loaded.")

    all_rows = pd.concat(row_chunks, ignore_index=True)

    unique_labels = sorted(int(x) for x in all_rows["label"].unique().tolist())

    grouped = (
        all_rows.groupby(["group_key", "input_id", "input_path", "source_file_index"])["label"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=unique_labels, fill_value=0)
        .reset_index()
    )

    grouped["total_rows"] = grouped[unique_labels].sum(axis=1)

    # Add optional file-name mapping
    source_file_name: List[Optional[str]] = []
    for _, row in grouped.iterrows():
        input_id = int(row["input_id"])
        source_idx = int(row["source_file_index"])
        names = source_files_by_input.get(input_id)
        if names is not None and 0 <= source_idx < len(names):
            source_file_name.append(names[source_idx])
        else:
            source_file_name.append(None)
    grouped["source_file_name"] = source_file_name

    # Stable/reproducible ordering
    grouped = grouped.sort_values(
        ["total_rows", "input_id", "source_file_index"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return grouped, all_rows, source_files_by_input, unique_labels


def compute_global_distribution(stats_df: pd.DataFrame, class_cols: List[int]) -> Dict[int, float]:
    totals = stats_df[class_cols].sum(axis=0)
    grand_total = float(totals.sum())
    if grand_total <= 0:
        raise ValueError("No rows found for any class.")
    return {cls: float(totals[cls]) / grand_total for cls in class_cols}


def split_score(split_counts: Dict[int, float], global_dist: Dict[int, float]) -> float:
    total = sum(split_counts.values())
    if total <= 0:
        return 0.0
    score = 0.0
    for cls, g in global_dist.items():
        frac = split_counts[cls] / total
        score += abs(frac - g)
    return score


def greedy_stratified_split(
    stats_df: pd.DataFrame,
    class_cols: List[int],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, List[str]]:
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/val/test fractions must sum to 1.")

    rng = random.Random(seed)

    files = stats_df.to_dict(orient="records")
    rng.shuffle(files)

    # Prioritize rare-class-heavy files and larger files.
    class_totals = {cls: float(stats_df[cls].sum()) for cls in class_cols}

    def rarity_key(row):
        present = []
        for cls in class_cols:
            if row[cls] > 0 and class_totals[cls] > 0:
                present.append(row[cls] / class_totals[cls])
        if not present:
            return 0.0
        return max(present)

    files.sort(
        key=lambda r: (
            rarity_key(r),
            r["total_rows"],
        ),
        reverse=True,
    )

    n_files = len(files)
    target_n = {
        "train": round(train_frac * n_files),
        "val": round(val_frac * n_files),
        "test": n_files - round(train_frac * n_files) - round(val_frac * n_files),
    }

    total_rows_all = float(stats_df["total_rows"].sum())
    target_rows = {
        "train": train_frac * total_rows_all,
        "val": val_frac * total_rows_all,
        "test": test_frac * total_rows_all,
    }

    global_dist = compute_global_distribution(stats_df, class_cols)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    split_counts: Dict[str, Dict[int, float]] = {
        "train": {cls: 0.0 for cls in class_cols},
        "val": {cls: 0.0 for cls in class_cols},
        "test": {cls: 0.0 for cls in class_cols},
    }
    split_row_totals = {"train": 0.0, "val": 0.0, "test": 0.0}

    for row in files:
        best_split = None
        best_obj = None

        for split_name in ["train", "val", "test"]:
            projected_n = len(splits[split_name]) + 1

            cap_penalty = 0.0
            if projected_n > target_n[split_name]:
                cap_penalty = 1000.0 * (projected_n - target_n[split_name])

            tmp_counts = split_counts[split_name].copy()
            for cls in class_cols:
                tmp_counts[cls] += row[cls]

            dist_penalty = split_score(tmp_counts, global_dist)

            projected_rows = split_row_totals[split_name] + row["total_rows"]
            row_penalty = abs(projected_rows - target_rows[split_name]) / max(total_rows_all, 1.0)

            obj = dist_penalty + row_penalty + cap_penalty

            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_split = split_name

        assert best_split is not None
        splits[best_split].append(row["group_key"])
        for cls in class_cols:
            split_counts[best_split][cls] += row[cls]
        split_row_totals[best_split] += row["total_rows"]

    return splits


def build_split_row_indices(
    all_rows: pd.DataFrame,
    splits: Dict[str, List[str]],
) -> Dict[str, Dict[int, np.ndarray]]:
    out: Dict[str, Dict[int, np.ndarray]] = {"train": {}, "val": {}, "test": {}}

    for split_name, group_keys in splits.items():
        sub = all_rows[all_rows["group_key"].isin(group_keys)].copy()
        for input_id, part in sub.groupby("input_id"):
            arr = np.sort(part["row_index"].to_numpy(dtype=np.int64))
            out[split_name][int(input_id)] = arr

    return out


def summarize_splits(
    stats_df: pd.DataFrame,
    splits: Dict[str, List[str]],
    class_cols: List[int],
) -> Dict:
    total_rows_all = int(stats_df["total_rows"].sum())
    total_files_all = int(len(stats_df))

    summary = {
        "global": {
            "n_files": total_files_all,
            "rows": total_rows_all,
            "class_counts": {str(cls): int(stats_df[cls].sum()) for cls in class_cols},
            "class_fractions": {},
        },
        "splits": {},
    }

    for cls in class_cols:
        cnt = summary["global"]["class_counts"][str(cls)]
        summary["global"]["class_fractions"][str(cls)] = cnt / total_rows_all if total_rows_all else 0.0

    for split_name, group_keys in splits.items():
        sub = stats_df[stats_df["group_key"].isin(group_keys)].copy()

        rows = int(sub["total_rows"].sum())
        n_files = int(len(sub))

        class_counts = {str(cls): int(sub[cls].sum()) for cls in class_cols}
        class_fractions = {
            str(cls): (class_counts[str(cls)] / rows if rows else 0.0) for cls in class_cols
        }

        summary["splits"][split_name] = {
            "n_files": n_files,
            "rows": rows,
            "row_fraction": (rows / total_rows_all if total_rows_all else 0.0),
            "file_fraction": (n_files / total_files_all if total_files_all else 0.0),
            "class_counts": class_counts,
            "class_fractions": class_fractions,
            "source_files": [
                {
                    "group_key": row["group_key"],
                    "input_id": int(row["input_id"]),
                    "input_path": row["input_path"],
                    "source_file_index": int(row["source_file_index"]),
                    "source_file_name": row["source_file_name"],
                    "rows": int(row["total_rows"]),
                    "class_histogram": {str(cls): int(row[cls]) for cls in class_cols},
                }
                for _, row in sub.sort_values(["input_id", "source_file_index"]).iterrows()
            ],
        }

    return summary


def validate_no_leakage(splits: Dict[str, List[str]]) -> None:
    seen = {}
    for split_name, keys in splits.items():
        for key in keys:
            if key in seen:
                raise RuntimeError(
                    f"Leakage detected: group_key '{key}' appears in both "
                    f"'{seen[key]}' and '{split_name}'."
                )
            seen[key] = split_name


def validate_assignment_complete(stats_df: pd.DataFrame, splits: Dict[str, List[str]]) -> None:
    assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    expected = set(stats_df["group_key"].tolist())

    missing = expected - assigned
    extra = assigned - expected

    if missing:
        raise RuntimeError(f"Some source-file groups were not assigned: {sorted(missing)[:10]}")
    if extra:
        raise RuntimeError(f"Unexpected assigned groups found: {sorted(extra)[:10]}")

    overlap_tv = set(splits["train"]) & set(splits["val"])
    overlap_tt = set(splits["train"]) & set(splits["test"])
    overlap_vt = set(splits["val"]) & set(splits["test"])

    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError("Leakage detected: one or more file groups appear in multiple splits.")


def write_outputs(
    out_dir: Path,
    inputs: List[Path],
    stats_df: pd.DataFrame,
    all_rows: pd.DataFrame,
    splits: Dict[str, List[str]],
    split_row_indices: Dict[str, Dict[int, np.ndarray]],
    summary: Dict,
    class_cols: List[int],
    summary_json_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # File-level stats table
    stats_df.to_csv(out_dir / "file_level_stats.csv", index=False)

    # Row index outputs
    # If only one input, also write train_indices.npy / val_indices.npy / test_indices.npy
    if len(inputs) == 1:
        for split_name in ["train", "val", "test"]:
            arr = split_row_indices[split_name].get(0, np.array([], dtype=np.int64))
            np.save(out_dir / f"{split_name}_indices.npy", arr)

    # For multiple inputs, or additionally for traceability, write a compressed NPZ
    npz_payload = {}
    for split_name in ["train", "val", "test"]:
        for input_id in range(len(inputs)):
            key = f"{split_name}_input{input_id}"
            npz_payload[key] = split_row_indices[split_name].get(input_id, np.array([], dtype=np.int64))
    np.savez_compressed(out_dir / "split_indices_by_input.npz", **npz_payload)

    # Per-split row manifests
    for split_name, group_keys in splits.items():
        sub = all_rows[all_rows["group_key"].isin(group_keys)].copy()
        sub = sub.sort_values(["input_id", "row_index"]).reset_index(drop=True)
        sub.to_csv(out_dir / f"{split_name}_rows.csv", index=False)

    # Per-split file manifests
    for split_name, group_keys in splits.items():
        sub = stats_df[stats_df["group_key"].isin(group_keys)].copy()
        sub = sub.sort_values(["input_id", "source_file_index"]).reset_index(drop=True)
        sub.to_csv(out_dir / f"{split_name}_source_files.csv", index=False)

    # Summary JSON
    with open(out_dir / summary_json_name, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Human-readable manifest
    manifest = {
        "inputs": [str(p.resolve()) for p in inputs],
        "file_level_stats_csv": str((out_dir / "file_level_stats.csv").resolve()),
        "split_indices_npz": str((out_dir / "split_indices_by_input.npz").resolve()),
        "summary_json": str((out_dir / summary_json_name).resolve()),
        "single_input_indices": {
            "train": str((out_dir / "train_indices.npy").resolve()) if len(inputs) == 1 else None,
            "val": str((out_dir / "val_indices.npy").resolve()) if len(inputs) == 1 else None,
            "test": str((out_dir / "test_indices.npy").resolve()) if len(inputs) == 1 else None,
        },
        "per_split_row_manifests": {
            split_name: str((out_dir / f"{split_name}_rows.csv").resolve())
            for split_name in ["train", "val", "test"]
        },
        "per_split_source_file_manifests": {
            split_name: str((out_dir / f"{split_name}_source_files.csv").resolve())
            for split_name in ["train", "val", "test"]
        },
        "class_labels_seen": class_cols,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def print_summary(summary: Dict) -> None:
    print("\nSplit complete.\n")
    print(f"Files total: {summary['global']['n_files']}")
    print(f"Rows total:  {summary['global']['rows']}")
    print(f"Global class counts: {summary['global']['class_counts']}")

    for split_name in ["train", "val", "test"]:
        info = summary["splits"][split_name]
        print(
            f"{split_name:>5}: "
            f"{info['n_files']:>4} files, "
            f"{info['rows']:>10} rows, "
            f"file_frac={info['file_fraction']:.3f}, "
            f"row_frac={info['row_fraction']:.3f}"
        )
        print(f"       class counts: {info['class_counts']}")


def print_file_mapping(
    stats_df: pd.DataFrame,
    max_print: int = 200,
) -> None:
    with_names = stats_df[stats_df["source_file_name"].notna()].copy()
    if with_names.empty:
        print("\nNo dataset_meta/source_files mapping found.")
        return

    print("\nSource file mapping (showing up to first %d rows):" % max_print)
    cols = ["input_id", "source_file_index", "source_file_name"]
    dedup = with_names[cols].drop_duplicates().sort_values(["input_id", "source_file_index"])
    for _, row in dedup.head(max_print).iterrows():
        print(
            f"  input_id={int(row['input_id'])} "
            f"source_file_index={int(row['source_file_index'])} "
            f"-> {row['source_file_name']}"
        )
    if len(dedup) > max_print:
        print(f"  ... ({len(dedup) - max_print} more)")


def main() -> None:
    args = parse_args()

    inputs = [Path(x) for x in args.inputs]
    out_dir = Path(args.output_dir)

    stats_df, all_rows, source_files_by_input, class_cols = aggregate_file_stats(
        inputs=inputs,
        label_col=args.label_col,
        source_file_col=args.source_file_col,
        source_files_dset=args.source_files_dset,
        max_rows_per_input=args.max_rows_per_input,
    )

    if stats_df.empty:
        raise RuntimeError("No rows loaded after reading HDF5 inputs.")

    splits = greedy_stratified_split(
        stats_df=stats_df,
        class_cols=class_cols,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    validate_assignment_complete(stats_df, splits)
    validate_no_leakage(splits)

    split_row_indices = build_split_row_indices(all_rows, splits)
    summary = summarize_splits(stats_df, splits, class_cols)

    write_outputs(
        out_dir=out_dir,
        inputs=inputs,
        stats_df=stats_df,
        all_rows=all_rows,
        splits=splits,
        split_row_indices=split_row_indices,
        summary=summary,
        class_cols=class_cols,
        summary_json_name=args.summary_json,
    )

    print_summary(summary)

    if args.print_file_map:
        print_file_mapping(stats_df)

    print("\nLeakage check passed: no source_file_index group appears in more than one split.")


if __name__ == "__main__":
    main()