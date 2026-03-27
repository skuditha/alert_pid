
from __future__ import annotations

import argparse
from pathlib import Path

from feature_audit import run_full_feature_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALERT post-PID Task 2.1 feature audit on one or more HDF5 files.")
    parser.add_argument("h5", nargs="+", help="Input HDF5 training files")
    parser.add_argument("--label-map", required=True, help="Path to label_map.json")
    parser.add_argument("--outdir", required=True, help="Directory for CSV tables and plots")
    args = parser.parse_args()

    run_full_feature_audit(args.h5, args.label_map, args.outdir)
    print(f"Wrote feature audit outputs to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
