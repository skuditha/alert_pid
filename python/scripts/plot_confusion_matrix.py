#!/usr/bin/env python3
"""
Plot a normalized confusion matrix from CSV.

Expected CSV format:
,proton,deuteron,triton,helium3,helium4
proton,0.859...,0.129...,...
deuteron,...

Usage examples:
    python plot_confusion_matrix.py reports/gbdt_model/confusion_matrix_normalized.csv
    python plot_confusion_matrix.py reports/gbdt_model/confusion_matrix_normalized.csv \
        --output reports/gbdt_model/confusion_matrix_pretty.png \
        --title "GBDT PID Confusion Matrix" \
        --cmap Blues \
        --figsize 8 6 \
        --percent-decimals 1 \
        --annot-fontsize 11 \
        --label-fontsize 12 \
        --title-fontsize 14 \
        --dpi 300
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a normalized confusion matrix from CSV.")
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to normalized confusion matrix CSV file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output image path (default: same name as input, with .png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Normalized Confusion Matrix",
        help="Plot title",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Predicted label",
        help="X-axis label",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="True label",
        help="Y-axis label",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        help="Matplotlib colormap name",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[8.0, 6.5],
        metavar=("W", "H"),
        help="Figure size in inches",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI",
    )
    parser.add_argument(
        "--percent-decimals",
        type=int,
        default=1,
        help="Number of decimals for percentage annotations",
    )
    parser.add_argument(
        "--annot-fontsize",
        type=float,
        default=11,
        help="Font size for cell annotations",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=12,
        help="Font size for axis labels and tick labels",
    )
    parser.add_argument(
        "--title-fontsize",
        type=float,
        default=14,
        help="Font size for title",
    )
    parser.add_argument(
        "--colorbar-label",
        type=str,
        default="Fraction",
        help="Colorbar label",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable colorbar",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Minimum color scale value",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.0,
        help="Maximum color scale value",
    )
    parser.add_argument(
        "--rotate-x",
        type=float,
        default=0.0,
        help="Rotation angle for x tick labels",
    )
    parser.add_argument(
        "--rotate-y",
        type=float,
        default=0.0,
        help="Rotation angle for y tick labels",
    )
    parser.add_argument(
        "--use-seaborn",
        action="store_true",
        help="Use seaborn heatmap if seaborn is installed",
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help="Use tight_layout() before saving",
    )
    return parser.parse_args()


def load_matrix(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, index_col=0)

    # Strip accidental whitespace from labels
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    return df


def make_annotation_strings(values: np.ndarray, decimals: int) -> np.ndarray:
    fmt = f"{{:.{decimals}f}}%"
    ann = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ann[i, j] = fmt.format(values[i, j] * 100.0)
    return ann


def plot_with_matplotlib(df: pd.DataFrame, args):
    values = df.values
    ann = make_annotation_strings(values, args.percent_decimals)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    im = ax.imshow(values, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, aspect="auto")

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, fontsize=args.label_fontsize, rotation=args.rotate_x)
    ax.set_yticklabels(df.index, fontsize=args.label_fontsize, rotation=args.rotate_y)

    ax.set_xlabel(args.xlabel, fontsize=args.label_fontsize)
    ax.set_ylabel(args.ylabel, fontsize=args.label_fontsize)
    ax.set_title(args.title, fontsize=args.title_fontsize)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate cells
    threshold = (args.vmin + args.vmax) / 2.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] > threshold else "black"
            ax.text(
                j, i, ann[i, j],
                ha="center", va="center",
                color=text_color,
                fontsize=args.annot_fontsize,
            )

    if not args.no_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(args.colorbar_label, fontsize=args.label_fontsize)
        cbar.ax.tick_params(labelsize=args.label_fontsize - 1)

    if args.tight:
        plt.tight_layout()

    return fig


def plot_with_seaborn(df: pd.DataFrame, args):
    if not HAVE_SEABORN:
        raise ImportError("seaborn is not installed, but --use-seaborn was requested.")

    values = df.values
    ann = make_annotation_strings(values, args.percent_decimals)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    sns.heatmap(
        df,
        annot=ann,
        fmt="",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        cbar=not args.no_colorbar,
        linewidths=1.0,
        linecolor="white",
        square=False,
        annot_kws={"fontsize": args.annot_fontsize},
        ax=ax,
    )

    ax.set_title(args.title, fontsize=args.title_fontsize)
    ax.set_xlabel(args.xlabel, fontsize=args.label_fontsize)
    ax.set_ylabel(args.ylabel, fontsize=args.label_fontsize)
    ax.tick_params(axis="x", labelrotation=args.rotate_x, labelsize=args.label_fontsize)
    ax.tick_params(axis="y", labelrotation=args.rotate_y, labelsize=args.label_fontsize)

    if not args.no_colorbar:
        cbar = ax.collections[0].colorbar
        cbar.set_label(args.colorbar_label, fontsize=args.label_fontsize)
        cbar.ax.tick_params(labelsize=args.label_fontsize - 1)

    if args.tight:
        plt.tight_layout()

    return fig


def main():
    args = parse_args()
    df = load_matrix(args.csv_file)

    if args.output is None:
        args.output = str(Path(args.csv_file).with_suffix(".png"))

    if args.use_seaborn:
        fig = plot_with_seaborn(df, args)
    else:
        fig = plot_with_matplotlib(df, args)

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()