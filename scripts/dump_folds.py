"""
Dump per-fold train/val/test CSVs so non-Python baselines (R: GAM, laGP)
consume the *same* fold slicing the PINN uses.

This guarantees the GAM/laGP comparison is apples-to-apples with the PINN
benchmark — same preprocessing pipeline, same fold boundaries, same val_months.

Outputs (one dir per fold):
  data/folds/<fold_name>/train.csv     # everything before val
  data/folds/<fold_name>/val.csv       # last `val_months` months before test
  data/folds/<fold_name>/test.csv      # the test month
  data/folds/<fold_name>/meta.json     # fold metadata + per-fold sigma_fixed, r_fixed

Usage:
  python scripts/dump_folds.py
  python scripts/dump_folds.py --val_months 1 --output_dir data/folds
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_and_preprocess, make_fold, compute_constants

# Single source of truth — must match run_walk_forward.py FOLDS exactly
FOLDS = [
    {"name": "Apr2020", "train_end": "2020-04-01", "test_start": "2020-04-01", "test_end": "2020-05-01"},
    {"name": "May2020", "train_end": "2020-05-01", "test_start": "2020-05-01", "test_end": "2020-06-01"},
    {"name": "Jun2020", "train_end": "2020-06-01", "test_start": "2020-06-01", "test_end": "2020-07-01"},
    {"name": "Jul2020", "train_end": "2020-07-01", "test_start": "2020-07-01", "test_end": "2020-08-01"},
    {"name": "Aug2020", "train_end": "2020-08-01", "test_start": "2020-08-01", "test_end": "2020-09-01"},
    {"name": "Sep2020", "train_end": "2020-09-01", "test_start": "2020-09-01", "test_end": "2020-10-01"},
    {"name": "Oct2020", "train_end": "2020-10-01", "test_start": "2020-10-01", "test_end": "2020-11-01"},
    {"name": "Nov2020", "train_end": "2020-11-01", "test_start": "2020-11-01", "test_end": "2020-12-01"},
    {"name": "Dec2020", "train_end": "2020-12-01", "test_start": "2020-12-01", "test_end": "2021-01-01"},
]

# Columns the R baselines need (drops everything else to keep CSVs lean)
R_COLS = [
    "date", "exdate",
    "forward_price", "strike_price", "time_to_exp",
    "interest_rate", "dividend_yield",
    "best_bid", "best_offer", "mid_price",
    "impl_volatility", "volume", "open_interest",
    "moneyness",
]


def main():
    ap = argparse.ArgumentParser(description="Dump per-fold CSVs for R baselines")
    ap.add_argument("--data", default="data/TSLA_2020_Split_Adjusted.csv")
    ap.add_argument("--val_months", type=int, default=1)
    ap.add_argument("--output_dir", default="data/folds")
    args = ap.parse_args()

    df = load_and_preprocess(args.data)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows from {args.data}")
    print(f"Writing folds to {out_root}/  (val_months={args.val_months})")

    for f in FOLDS:
        train_df, val_df, test_df = make_fold(
            df,
            train_end=f["train_end"],
            test_start=f["test_start"],
            test_end=f["test_end"],
            val_months=args.val_months,
        )
        # Per-fold constants (exact same call BS_A1 and PINN make)
        constants = compute_constants(train_df)

        fold_dir = out_root / f["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Write CSVs (only columns the R scripts need)
        train_df[R_COLS].to_csv(fold_dir / "train.csv", index=False)
        val_df[R_COLS].to_csv(fold_dir / "val.csv", index=False)
        test_df[R_COLS].to_csv(fold_dir / "test.csv", index=False)

        meta = {
            "fold":         f["name"],
            "train_end":    f["train_end"],
            "test_start":   f["test_start"],
            "test_end":     f["test_end"],
            "val_months":   args.val_months,
            "n_train":      int(len(train_df)),
            "n_val":        int(len(val_df)),
            "n_test":       int(len(test_df)),
            # Per-fold constants — R reuses these so BS baseline is identical to BS_A1
            "sigma_fixed":  float(constants["sigma_fixed"]),
            "r_fixed":      float(constants["r_fixed"]),
            "n_atm":        int(constants["n_atm"]),
        }
        (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"  {f['name']:>8s}  train={len(train_df):>5d}  val={len(val_df):>5d}  "
              f"test={len(test_df):>5d}  σ={constants['sigma_fixed']:.4f}  r={constants['r_fixed']:.5f}")

    print(f"\nDone. {len(FOLDS)} folds written to {out_root}/")
    print("Next: Rscript scripts/run_gam_baseline.R")
    print("      Rscript scripts/run_lagp_baseline.R")


if __name__ == "__main__":
    main()
