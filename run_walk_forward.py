"""
Walk-Forward Backtesting for PINN Option Pricing.

Expanding-window temporal CV with monthly test folds.
Uses the modular src/ package for data, model, and training.

Usage:
    python run_walk_forward.py --mode physics --epochs 15000
    python run_walk_forward.py --mode hybrid --epochs 15000
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import (load_and_preprocess, make_fold, compute_constants,
                      df_to_arrays, build_boundary_terminal)
from src.training import run_training


# ── Walk-forward fold definitions ───────────────────────────────────
# Expanding window: train on all data up to train_end, test on one month
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


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Backtesting")
    parser.add_argument("--data", default="data/TSLA_2020_Split_Adjusted.csv")
    parser.add_argument("--mode", choices=["physics", "hybrid"], default="physics")
    parser.add_argument("--arch", choices=["standard", "modified"], default="modified",
                        help="Network architecture: 'modified' (Modified MLP + RWF) "
                             "or 'standard' (plain MLP, naive PINN baseline B1). "
                             "When 'standard', --rwf_mu is ignored.")
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rwf_mu", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs/walk_forward")
    parser.add_argument("--folds", default=None,
                        help="Comma-separated subset of fold names to run "
                             "(e.g. 'Nov2020' for a single-fold smoke test). "
                             "Default: all 9 folds.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load full dataset once ──────────────────────────────────────
    print("Loading and preprocessing data...")
    df = load_and_preprocess(args.data)
    print(f"Full dataset: {len(df):,} obs, {df['date'].min().date()} → {df['date'].max().date()}")

    # Output directory naming reflects arch + mode + (μ if applicable)
    if args.arch == "modified":
        run_tag = f"{args.arch}_{args.mode}_mu{args.rwf_mu}"
    else:
        run_tag = f"{args.arch}_{args.mode}"
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional fold subset for smoke tests
    if args.folds is not None:
        wanted = {name.strip() for name in args.folds.split(",")}
        folds_to_run = [f for f in FOLDS if f["name"] in wanted]
        missing = wanted - {f["name"] for f in folds_to_run}
        if missing:
            print(f"WARNING: requested folds not found: {sorted(missing)}")
    else:
        folds_to_run = FOLDS

    # ── Run each fold ───────────────────────────────────────────────
    results = []

    for i, fold in enumerate(folds_to_run):
        print(f"\n{'='*60}")
        print(f"FOLD {i+1}/{len(folds_to_run)}: {fold['name']}")
        print(f"  Train: start → {fold['train_end']}  |  Test: {fold['test_start']} → {fold['test_end']}")
        print(f"{'='*60}")

        # Build fold data
        train_df, test_df = make_fold(
            df, train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"]
        )

        if len(test_df) == 0:
            print(f"  WARNING: No test data for {fold['name']}, skipping")
            continue

        config = compute_constants(train_df)
        train_arrays = df_to_arrays(train_df)
        test_arrays = df_to_arrays(test_df)
        boundary_data = build_boundary_terminal(
            config, config["sigma_fixed"], config["r_fixed"]
        )

        print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")
        print(f"  σ_fixed: {config['sigma_fixed']:.4f}")

        # Train
        model, history, run_info = run_training(
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            boundary_data=boundary_data,
            config=config,
            mode=args.mode,
            arch=args.arch,
            epochs=args.epochs,
            lr=args.lr,
            rwf_mu=args.rwf_mu,
            seed=args.seed,
            device=device,
            verbose=True,
            log_every=2000,
            val_every=1000,
        )

        # Record results — final-epoch metrics enter the table; best/drift
        # are stability diagnostics only. sum_sq_err_mkt and sum_abs_err_mkt
        # are kept per-fold so the summary can compute a pooled RMSE/MAE
        # (the headline metric in BENCHMARKING_PLAN.md §C1).
        fold_result = {
            "fold": fold["name"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "sigma_fixed": config["sigma_fixed"],
            "rmse_bs_norm": run_info["final_rmse_bs_norm"],
            "rmse_mkt": run_info["final_rmse_mkt"],
            "mae_mkt": run_info["final_mae_mkt"],
            "sum_sq_err_mkt":  run_info["final_sum_sq_err_mkt"],
            "sum_abs_err_mkt": run_info["final_sum_abs_err_mkt"],
            "best_epoch": run_info["best_epoch"],
            "best_rmse_bs_norm": run_info["best_rmse_bs_norm"],
            "best_rmse_mkt": run_info["best_rmse_mkt"],
            "drift_gap": run_info["drift_gap"],
            "elapsed": run_info["elapsed_seconds"],
            # Validation trajectory — for drift/convergence inspection
            "val_epoch":         [int(x)   for x in history["val_epoch"]],
            "val_rmse_mkt":      [float(x) for x in history["rmse_mkt"]],
            "val_rmse_bs_norm":  [float(x) for x in history["rmse_bs_norm"]],
        }
        results.append(fold_result)

        # Save fold checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "run_info": run_info,
        }, output_dir / f"fold_{fold['name']}.pt")

        print(f"  → RMSE vs Market: ${run_info['final_rmse_mkt']:.2f}")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n\n{'='*86}")
    if args.arch == "modified":
        print(f"WALK-FORWARD RESULTS: arch={args.arch}, mode={args.mode}, μ={args.rwf_mu}")
    else:
        print(f"WALK-FORWARD RESULTS: arch={args.arch}, mode={args.mode}")
    print(f"  Reporting final-epoch RMSE; best/drift columns are stability diagnostics only")
    print(f"{'='*86}")
    print(f"{'Fold':<10} | {'σ_fix':>6} | {'final($)':>9} | {'best($)':>8} | "
          f"{'best@ep':>7} | {'drift':>7} | {'time(s)':>7}")
    print("-" * 86)

    rmses = []
    maes = []
    drifts = []
    for r in results:
        best_str = f"{r['best_rmse_mkt']:.2f}" if r['best_rmse_mkt'] is not None else "—"
        print(f"{r['fold']:<10} | {r['sigma_fixed']:>6.3f} | "
              f"{r['rmse_mkt']:>9.2f} | {best_str:>8} | "
              f"{r['best_epoch']:>7d} | {r['drift_gap']:>+7.2f} | "
              f"{r['elapsed']:>7.0f}")
        rmses.append(r["rmse_mkt"])
        maes.append(r["mae_mkt"])
        drifts.append(r["drift_gap"])

    print("-" * 86)

    # Pooled = single RMSE over all concatenated residuals across folds.
    # This is the headline metric per BENCHMARKING_PLAN.md "Reporting Protocol".
    total_sse = sum(r["sum_sq_err_mkt"] for r in results)
    total_sae = sum(r["sum_abs_err_mkt"] for r in results)
    total_n   = sum(r["n_test"]         for r in results)
    pooled_rmse = float((total_sse / total_n) ** 0.5)
    pooled_mae  = float(total_sae / total_n)
    worst_rmse  = float(max(rmses))
    worst_fold  = results[int(np.argmax(rmses))]["fold"]

    print(f"{'Pooled':<10} | {'':>6} | {pooled_rmse:>9.2f} | "
          f"{'':>8} | {'':>7} | {'':>7} | "
          f"{sum(r['elapsed'] for r in results):>7.0f}   ← headline")
    print(f"{'Mean(fold)':<10} | {'':>6} | {np.mean(rmses):>9.2f} | "
          f"{'':>8} | {'':>7} | {np.mean(drifts):>+7.2f} |")
    print(f"{'Std(fold)':<10} | {'':>6} | {np.std(rmses):>9.2f} | "
          f"{'':>8} | {'':>7} | {np.std(drifts):>7.2f} |")
    print(f"{'Worst':<10} | {'':>6} | {worst_rmse:>9.2f} | "
          f"{'':>8} | {'':>7} | {'':>7} |    ({worst_fold})")

    # Save results — folds list + aggregate summary block.
    summary = {
        "pooled_rmse_mkt":  pooled_rmse,
        "pooled_mae_mkt":   pooled_mae,
        "mean_rmse_mkt":    float(np.mean(rmses)),
        "std_rmse_mkt":     float(np.std(rmses)),
        "worst_rmse_mkt":   worst_rmse,
        "worst_fold":       worst_fold,
        "mean_drift_gap":   float(np.mean(drifts)),
        "max_drift_gap":    float(np.max(drifts)) if len(drifts) else 0.0,
        "total_n_test":     total_n,
        "n_folds":          len(results),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
