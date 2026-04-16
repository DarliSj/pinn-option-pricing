"""
Stage 2: Learnable Volatility Surface — Walk-Forward Backtesting.

Warm-starts pricing net from Stage 1 per-fold checkpoints and trains a
volatility surface network alongside it. Both C-Vol (multiplicative,
NSM-inspired) and A-Vol (direct) parameterizations are supported.

Usage:
    python run_stage2.py --vol_type cvol \
        --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75 \
        --pricing_lr 1e-4 --vol_lr 1e-3 \
        --epochs 10000 --rwf_mu 0.75

    python run_stage2.py --vol_type avol \
        --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75 \
        --pricing_lr 1e-4 --vol_lr 1e-3 \
        --epochs 10000 --rwf_mu 0.75
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

from src.data import (load_and_preprocess, make_fold, compute_constants,
                      df_to_arrays, build_boundary_terminal)
from src.model import VolatilityNet, CVolWrapper, AVolWrapper
from src.training import run_training
from src.diagnostics import (plot_training_summary, plot_full_diagnostics,
                              plot_solution_surface, plot_test_scatter,
                              plot_vol_surface, plot_vol_slices)

# Reuse fold definitions from Stage 1
from run_walk_forward import FOLDS


def build_vol_model(vol_type, sigma_fixed, vol_hidden=None,
                    fourier_features=64, fourier_scale=0.5,
                    rwf_mu=1.0, rwf_sigma=0.1):
    """Construct the volatility network + wrapper."""
    vol_net = VolatilityNet(
        hidden_dims=vol_hidden,
        fourier_features=fourier_features,
        fourier_scale=fourier_scale,
        rwf_mu=rwf_mu,
        rwf_sigma=rwf_sigma,
    )
    if vol_type == "cvol":
        return CVolWrapper(vol_net, sigma_0=sigma_fixed)
    elif vol_type == "avol":
        return AVolWrapper(vol_net, sigma_init=sigma_fixed)
    else:
        raise ValueError(f"vol_type must be 'cvol' or 'avol', got {vol_type!r}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Learnable Vol Surface")
    parser.add_argument("--data", default="data/TSLA_2020_Split_Adjusted.csv")
    parser.add_argument("--vol_type", choices=["cvol", "avol"], required=True,
                        help="Volatility parameterization: 'cvol' (multiplicative, "
                             "NSM-inspired) or 'avol' (direct, standard baseline)")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing Stage 1 per-fold checkpoints "
                             "(e.g. runs/walk_forward/modified_hybrid_mu0.75)")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--pricing_lr", type=float, default=1e-4,
                        help="LR for pricing net (fine-tuning, default 1e-4)")
    parser.add_argument("--vol_lr", type=float, default=1e-3,
                        help="LR for vol net (learning from scratch, default 1e-3)")
    parser.add_argument("--rwf_mu", type=float, default=0.75,
                        help="RWF μ for pricing net (must match Stage 1 checkpoint)")
    parser.add_argument("--rwf_sigma", type=float, default=0.1)
    parser.add_argument("--vol_hidden", default=None,
                        help="Vol net hidden dims as comma-separated ints "
                             "(default: '32,32')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs/stage2")
    parser.add_argument("--folds", default=None,
                        help="Comma-separated subset of fold names to run "
                             "(e.g. 'Nov2020' for smoke test). Default: all 9.")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse vol hidden dims
    vol_hidden = None
    if args.vol_hidden is not None:
        vol_hidden = [int(x) for x in args.vol_hidden.split(",")]

    # ── Load full dataset once ──────────────────────────────────────
    print("Loading and preprocessing data...")
    df = load_and_preprocess(args.data)
    print(f"Full dataset: {len(df):,} obs, {df['date'].min().date()} -> {df['date'].max().date()}")

    # Output directory
    run_tag = args.vol_type
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Optional fold subset
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
        print(f"FOLD {i+1}/{len(folds_to_run)}: {fold['name']} — Stage 2 ({args.vol_type})")
        print(f"  Train: start -> {fold['train_end']}  |  Test: {fold['test_start']} -> {fold['test_end']}")
        print(f"{'='*60}")

        # Locate Stage 1 checkpoint for this fold
        ckpt_path = ckpt_dir / f"fold_{fold['name']}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: Checkpoint not found: {ckpt_path}, skipping fold")
            continue

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
        print(f"  sigma_fixed: {config['sigma_fixed']:.4f}")

        # Construct fresh vol model for this fold
        vol_model = build_vol_model(
            vol_type=args.vol_type,
            sigma_fixed=config["sigma_fixed"],
            vol_hidden=vol_hidden,
            rwf_mu=args.rwf_mu,
            rwf_sigma=args.rwf_sigma,
        )

        # Train
        model, history, run_info = run_training(
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            boundary_data=boundary_data,
            config=config,
            mode="hybrid",  # Stage 2 is always hybrid
            arch="modified",
            epochs=args.epochs,
            lr=args.pricing_lr,  # base LR (used for scheduler eta_min)
            rwf_mu=args.rwf_mu,
            rwf_sigma=args.rwf_sigma,
            seed=args.seed,
            device=device,
            verbose=True,
            log_every=2000,
            val_every=1000,
            # Stage 2 extensions
            vol_model=vol_model,
            vol_type=args.vol_type,
            pricing_lr=args.pricing_lr,
            vol_lr=args.vol_lr,
            checkpoint_path=str(ckpt_path),
        )

        # Record results
        fold_result = {
            "fold": fold["name"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "sigma_fixed": config["sigma_fixed"],
            "rmse_mkt": run_info["final_rmse_mkt"],
            "mae_mkt": run_info["final_mae_mkt"],
            "sum_sq_err_mkt":  run_info["final_sum_sq_err_mkt"],
            "sum_abs_err_mkt": run_info["final_sum_abs_err_mkt"],
            "best_epoch": run_info["best_epoch"],
            "best_rmse_mkt": run_info["best_rmse_mkt"],
            "drift_gap": run_info["drift_gap"],
            "elapsed": run_info["elapsed_seconds"],
            # Validation trajectory — for drift/convergence inspection
            "val_epoch":         [int(x)   for x in history["val_epoch"]],
            "val_rmse_mkt":      [float(x) for x in history["rmse_mkt"]],
            "val_rmse_bs_norm":  [float(x) for x in history["rmse_bs_norm"]],
        }
        results.append(fold_result)

        # Save fold checkpoint (pricing + vol model)
        torch.save({
            "model_state_dict": model.state_dict(),
            "vol_model_state_dict": vol_model.state_dict(),
            "config": config,
            "run_info": run_info,
        }, output_dir / f"fold_{fold['name']}.pt")

        print(f"  -> RMSE vs Market: ${run_info['final_rmse_mkt']:.2f}")

        # Per-fold diagnostics
        if not args.no_plots:
            suffix = f"({args.vol_type}, {fold['name']})"

            # Training summary
            vol_loss_names = ["pde", "tc", "bc", "data", "reg"]
            fig1 = plot_training_summary(history, vol_loss_names, suffix)
            fig1.savefig(output_dir / f"fold_{fold['name']}_training.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig1)

            # Vol surface
            is_cvol = (args.vol_type == "cvol")
            fig2 = plot_vol_surface(vol_model, config, device,
                                    config["sigma_fixed"], suffix, is_cvol=is_cvol)
            fig2.savefig(output_dir / f"fold_{fold['name']}_vol_surface.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig2)

            # Vol slices
            fig3 = plot_vol_slices(vol_model, config, device,
                                   config["sigma_fixed"], suffix)
            fig3.savefig(output_dir / f"fold_{fold['name']}_vol_slices.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig3)

            # Test scatter
            fig4, _ = plot_test_scatter(model, test_arrays, "hybrid", device, suffix)
            fig4.savefig(output_dir / f"fold_{fold['name']}_test_scatter.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig4)

    # ── Summary table ───────────────────────────────────────────────
    if not results:
        print("\nNo folds completed. Check checkpoint directory.")
        return

    print(f"\n\n{'='*86}")
    print(f"STAGE 2 WALK-FORWARD RESULTS: vol_type={args.vol_type}")
    print(f"  Reporting final-epoch RMSE; best/drift = stability diagnostics")
    print(f"{'='*86}")
    print(f"{'Fold':<10} | {'sigma':>6} | {'final($)':>9} | {'best($)':>8} | "
          f"{'best@ep':>7} | {'drift':>7} | {'time(s)':>7}")
    print("-" * 86)

    rmses = []
    maes = []
    drifts = []
    for r in results:
        best_str = f"{r['best_rmse_mkt']:.2f}" if r['best_rmse_mkt'] is not None else "-"
        print(f"{r['fold']:<10} | {r['sigma_fixed']:>6.3f} | "
              f"{r['rmse_mkt']:>9.2f} | {best_str:>8} | "
              f"{r['best_epoch']:>7d} | {r['drift_gap']:>+7.2f} | "
              f"{r['elapsed']:>7.0f}")
        rmses.append(r["rmse_mkt"])
        maes.append(r["mae_mkt"])
        drifts.append(r["drift_gap"])

    print("-" * 86)

    # Pooled RMSE
    total_sse = sum(r["sum_sq_err_mkt"] for r in results)
    total_sae = sum(r["sum_abs_err_mkt"] for r in results)
    total_n   = sum(r["n_test"]         for r in results)
    pooled_rmse = float((total_sse / total_n) ** 0.5)
    pooled_mae  = float(total_sae / total_n)
    worst_rmse  = float(max(rmses))
    worst_fold  = results[int(np.argmax(rmses))]["fold"]

    print(f"{'Pooled':<10} | {'':>6} | {pooled_rmse:>9.2f} | "
          f"{'':>8} | {'':>7} | {'':>7} | "
          f"{sum(r['elapsed'] for r in results):>7.0f}   <- headline")
    print(f"{'Mean(fold)':<10} | {'':>6} | {np.mean(rmses):>9.2f} | "
          f"{'':>8} | {'':>7} | {np.mean(drifts):>+7.2f} |")
    print(f"{'Std(fold)':<10} | {'':>6} | {np.std(rmses):>9.2f} | "
          f"{'':>8} | {'':>7} | {np.std(drifts):>7.2f} |")
    print(f"{'Worst':<10} | {'':>6} | {worst_rmse:>9.2f} | "
          f"{'':>8} | {'':>7} | {'':>7} |    ({worst_fold})")

    # Save results
    summary = {
        "vol_type":         args.vol_type,
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
        "pricing_lr":       args.pricing_lr,
        "vol_lr":           args.vol_lr,
        "epochs":           args.epochs,
        "rwf_mu":           args.rwf_mu,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
