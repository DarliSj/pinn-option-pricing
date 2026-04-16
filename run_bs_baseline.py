"""
Black-Scholes baseline (A1 in BENCHMARKING_PLAN.md) on the 9 walk-forward folds.

Reuses the FOLDS definition from run_walk_forward.py so BS and PINN runs are
guaranteed to share fold boundaries. Runs in seconds; populates the BS column
of the master benchmark table.

Usage:
    python run_bs_baseline.py
    python run_bs_baseline.py --data data/TSLA_2020_Split_Adjusted.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.baselines import bs_baseline_fold
from src.data import load_and_preprocess, make_fold
from run_walk_forward import FOLDS


def main():
    parser = argparse.ArgumentParser(description="BS baseline on walk-forward folds")
    parser.add_argument("--data", default="data/TSLA_2020_Split_Adjusted.csv")
    parser.add_argument("--output_dir", default="results/bs_baseline")
    args = parser.parse_args()

    print("Loading and preprocessing data...")
    df = load_and_preprocess(args.data)
    print(f"Full dataset: {len(df):,} obs, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, fold in enumerate(FOLDS):
        train_df, test_df = make_fold(
            df, train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"],
        )
        if len(test_df) == 0:
            print(f"  WARNING: no test data for {fold['name']}, skipping")
            continue

        res = bs_baseline_fold(train_df, test_df)
        res["fold"] = fold["name"]
        results.append(res)
        print(f"  fold {i+1}/{len(FOLDS)} {fold['name']:<8}  "
              f"σ={res['sigma_fixed']:.3f}  "
              f"RMSE=${res['rmse_mkt']:>7.2f}  MAE=${res['mae_mkt']:>6.2f}  "
              f"(n_train={res['n_train']:,}, n_test={res['n_test']:,})")

    # ── Summary table ───────────────────────────────────────────────
    rmses = np.array([r["rmse_mkt"] for r in results])
    maes = np.array([r["mae_mkt"] for r in results])

    # Pooled = concatenate residuals across all folds, compute one RMSE/MAE.
    # Equivalent to sqrt(Σ sum_sq_err / Σ n_test). This is the statistic
    # comparable to a single 2-month split RMSE.
    total_sse = sum(r["sum_sq_err"] for r in results)
    total_sae = sum(r["sum_abs_err"] for r in results)
    total_n   = sum(r["n_test"]     for r in results)
    pooled_rmse = float(np.sqrt(total_sse / total_n))
    pooled_mae  = float(total_sae / total_n)

    print(f"\n{'='*70}")
    print(f"BS BASELINE — WALK-FORWARD ({len(results)} folds)")
    print(f"{'='*70}")
    print(f"{'Fold':<10} | {'σ_fix':>6} | {'r_fix':>7} | {'RMSE($)':>8} | "
          f"{'MAE($)':>7} | {'n_test':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['fold']:<10} | {r['sigma_fixed']:>6.3f} | "
              f"{r['r_fixed']:>7.4f} | {r['rmse_mkt']:>8.2f} | "
              f"{r['mae_mkt']:>7.2f} | {r['n_test']:>7,}")
    print("-" * 70)
    print(f"{'Mean(folds)':<10} | {'':>6} | {'':>7} | {rmses.mean():>8.2f} | "
          f"{maes.mean():>7.2f} | {'':>7}")
    print(f"{'Std(folds)':<10} | {'':>6} | {'':>7} | {rmses.std():>8.2f} | "
          f"{maes.std():>7.2f} | {'':>7}")
    print(f"{'Pooled':<10} | {'':>6} | {'':>7} | {pooled_rmse:>8.2f} | "
          f"{pooled_mae:>7.2f} | {total_n:>7,}")
    print(f"\n  mean-of-folds: arithmetic mean of 9 per-fold RMSEs "
          f"(under-weights large folds)")
    print(f"  pooled:        single RMSE over all {total_n:,} concatenated "
          f"residuals (apples-to-apples\n                 with the old "
          f"single-split $17.23 figure)")

    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "black_scholes",
            "folds": results,
            "summary": {
                "mean_rmse_mkt": float(rmses.mean()),
                "std_rmse_mkt":  float(rmses.std()),
                "mean_mae_mkt":  float(maes.mean()),
                "std_mae_mkt":   float(maes.std()),
                "pooled_rmse_mkt": pooled_rmse,
                "pooled_mae_mkt":  pooled_mae,
                "total_n_test":    total_n,
            },
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
