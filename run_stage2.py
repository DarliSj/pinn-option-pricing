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
                              plot_vol_surface, plot_vol_slices,
                              compute_arbitrage_ratios)

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
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory containing Stage 1 per-fold checkpoints "
                             "(e.g. runs/walk_forward/modified_hybrid_mu0.75). "
                             "Required unless --from_scratch is set.")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train Stage 2 (pricing + vol) jointly from scratch, "
                             "ignoring Stage 1 warm-start. Useful as an ablation "
                             "to show warm-start is a compute optimization, not a "
                             "methodological requirement. Implies pricing_lr=lr "
                             "(no fine-tune) and longer training is recommended.")
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
    parser.add_argument("--val_months", type=int, default=1,
                        help="Months of training tail held out as the "
                             "validation window. Drives val-best snapshot "
                             "selection; never part of training.")
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

    # Output directory — tag distinguishes warm-start vs from-scratch
    run_tag = args.vol_type + ("_scratch" if args.from_scratch else "")
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory (only required when warm-starting)
    if not args.from_scratch:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint_dir is required unless --from_scratch is set")
        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    else:
        ckpt_dir = None
        print("WARNING: --from_scratch enabled — Stage 2 will train pricing+vol "
              "jointly from random init (no Stage 1 warm-start).")

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

        # Locate Stage 1 checkpoint for this fold (None when --from_scratch)
        if ckpt_dir is not None:
            ckpt_path = ckpt_dir / f"fold_{fold['name']}.pt"
        else:
            ckpt_path = None
        if ckpt_path is not None and not ckpt_path.exists():
            print(f"  WARNING: Checkpoint not found: {ckpt_path}, skipping fold")
            continue

        # Build fold data — train / val / test
        train_df, val_df, test_df = make_fold(
            df, train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"],
            val_months=args.val_months,
        )

        if len(test_df) == 0:
            print(f"  WARNING: No test data for {fold['name']}, skipping")
            continue

        config = compute_constants(train_df)
        train_arrays = df_to_arrays(train_df)
        test_arrays  = df_to_arrays(test_df)
        val_arrays   = df_to_arrays(val_df) if len(val_df) > 0 else None
        boundary_data = build_boundary_terminal(
            config, config["sigma_fixed"], config["r_fixed"]
        )

        print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
        print(f"  sigma_fixed: {config['sigma_fixed']:.4f}")

        # Fresh vol model bound to this fold's σ_fixed
        vol_model = build_vol_model(
            vol_type=args.vol_type,
            sigma_fixed=config["sigma_fixed"],
            vol_hidden=vol_hidden,
            rwf_mu=args.rwf_mu,
            rwf_sigma=args.rwf_sigma,
        )

        # Single training run. Pricing net is warm-started from Stage 1
        # fold checkpoint (which under the val-best scheme also never
        # saw val_window, so this val is genuinely held-out for Stage 2).
        model, history, run_info = run_training(
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            val_arrays=val_arrays,
            boundary_data=boundary_data,
            config=config,
            mode="hybrid",   # Stage 2 is always hybrid
            arch="modified",
            epochs=args.epochs,
            lr=args.pricing_lr,   # base LR (used for scheduler eta_min)
            rwf_mu=args.rwf_mu,
            rwf_sigma=args.rwf_sigma,
            seed=args.seed,
            device=device,
            verbose=True,
            log_every=2000,
            val_every=250,
            vol_model=vol_model,
            vol_type=args.vol_type,
            pricing_lr=args.pricing_lr,
            vol_lr=args.vol_lr,
            checkpoint_path=(str(ckpt_path) if ckpt_path is not None else None),
        )

        # Model + vol_model are now restored to val-best (or final-epoch
        # if val=None). No-arbitrage diagnostic on the reported model.
        arb = compute_arbitrage_ratios(model, config, device,
                                       vol_model=vol_model)

        fold_result = {
            "fold": fold["name"],
            "n_train": len(train_df),
            "n_val":   len(val_df),
            "n_test":  len(test_df),
            "sigma_fixed": config["sigma_fixed"],
            "used_val":         run_info["used_val"],
            "restored_to_best": run_info["restored_to_best"],
            "best_val_epoch":   run_info["best_val_epoch"],
            "best_val_metric":  run_info["best_val_metric"],
            "rmse_mkt":     run_info["final_rmse_mkt"],
            "mae_mkt":      run_info["final_mae_mkt"],
            "rmse_spread":  run_info["final_rmse_spread"],
            "rmse_otm":     run_info["final_rmse_otm"],
            "rmse_atm":     run_info["final_rmse_atm"],
            "rmse_itm":     run_info["final_rmse_itm"],
            "n_otm":        run_info["n_otm"],
            "n_atm":        run_info["n_atm"],
            "n_itm":        run_info["n_itm"],
            "sum_sq_err_mkt":    run_info["final_sum_sq_err_mkt"],
            "sum_abs_err_mkt":   run_info["final_sum_abs_err_mkt"],
            "sum_sq_err_spread": run_info["final_sum_sq_err_spread"],
            "sum_sq_err_otm":    run_info["final_sum_sq_err_otm"],
            "sum_sq_err_atm":    run_info["final_sum_sq_err_atm"],
            "sum_sq_err_itm":    run_info["final_sum_sq_err_itm"],
            "arb_butterfly_ratio":   arb["butterfly_ratio"],
            "arb_calendar_ratio":    arb["calendar_ratio"],
            "arb_butterfly_max_neg": arb["butterfly_max_neg"],
            "arb_calendar_max_neg":  arb["calendar_max_neg"],
            "elapsed": run_info["elapsed_seconds"],
            "val_epoch":        [int(x)   for x in history["val_epoch"]],
            "val_rmse_mkt":     [float(x) for x in history["val_rmse_mkt"]],
            "val_rmse_bs_norm": [float(x) for x in history["val_rmse_bs_norm"]],
        }
        results.append(fold_result)

        # Saved checkpoint = val-best snapshot (post-restore state)
        torch.save({
            "model_state_dict": model.state_dict(),
            "vol_model_state_dict": vol_model.state_dict(),
            "config": config,
            "run_info": run_info,
        }, output_dir / f"fold_{fold['name']}.pt")

        ep = run_info["best_val_epoch"] or run_info["epochs"]
        print(f"  -> REPORTED test RMSE: ${run_info['final_rmse_mkt']:.2f} "
              f"(val-best epoch: {ep})")

        # Per-fold diagnostics — built on the val-best model
        if not args.no_plots:
            suffix = f"({args.vol_type}, {fold['name']}, ep*={ep})"

            vol_loss_names = ["pde", "tc", "bc", "data", "reg"]
            fig1 = plot_training_summary(history, vol_loss_names, suffix)
            fig1.savefig(output_dir / f"fold_{fold['name']}_training.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig1)

            is_cvol = (args.vol_type == "cvol")
            fig2 = plot_vol_surface(vol_model, config, device,
                                    config["sigma_fixed"], suffix,
                                    is_cvol=is_cvol)
            fig2.savefig(output_dir / f"fold_{fold['name']}_vol_surface.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig2)

            fig3 = plot_vol_slices(vol_model, config, device,
                                   config["sigma_fixed"], suffix)
            fig3.savefig(output_dir / f"fold_{fold['name']}_vol_slices.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig3)

            fig4, _ = plot_test_scatter(model, test_arrays, "hybrid", device, suffix)
            fig4.savefig(output_dir / f"fold_{fold['name']}_test_scatter.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig4)

    # ── Summary table ───────────────────────────────────────────────
    if not results:
        print("\nNo folds completed. Check checkpoint directory.")
        return

    print(f"\n\n{'='*100}")
    print(f"STAGE 2 WALK-FORWARD RESULTS: vol_type={args.vol_type}")
    if results[0]["used_val"]:
        print(f"  Single-phase val-best reporting (val held out, test eval'd once on snapshot)")
    else:
        print(f"  Single-phase final-epoch reporting (no val window)")
    print(f"{'='*100}")
    print(f"{'Fold':<10} | {'σ_fix':>6} | {'E*':>6} | {'rep($)':>7} | "
          f"{'spread':>6} | {'OTM/ATM/ITM':>16} | {'arb%':>6} | {'time':>6}")
    print("-" * 100)

    rmses = []
    maes = []
    e_stars = []
    for r in results:
        strat = f"{r['rmse_otm']:.2f}/{r['rmse_atm']:.2f}/{r['rmse_itm']:.2f}"
        arb_pct = 100.0 * (r['arb_butterfly_ratio'] + r['arb_calendar_ratio'])
        e_star = r["best_val_epoch"] if r["best_val_epoch"] is not None else 0
        print(f"{r['fold']:<10} | {r['sigma_fixed']:>6.3f} | "
              f"{e_star:>6d} | {r['rmse_mkt']:>7.2f} | "
              f"{r['rmse_spread']:>6.2f} | {strat:>16} | "
              f"{arb_pct:>5.1f}% | {r['elapsed']:>6.0f}")
        rmses.append(r["rmse_mkt"])
        maes.append(r["mae_mkt"])
        if r["best_val_epoch"] is not None:
            e_stars.append(r["best_val_epoch"])

    print("-" * 100)

    # Pooled metrics
    total_sse        = sum(r["sum_sq_err_mkt"]    for r in results)
    total_sae        = sum(r["sum_abs_err_mkt"]   for r in results)
    total_sse_spread = sum(r["sum_sq_err_spread"] for r in results)
    total_sse_otm    = sum(r["sum_sq_err_otm"]    for r in results)
    total_sse_atm    = sum(r["sum_sq_err_atm"]    for r in results)
    total_sse_itm    = sum(r["sum_sq_err_itm"]    for r in results)
    total_n          = sum(r["n_test"]            for r in results)
    total_n_otm      = sum(r["n_otm"]             for r in results)
    total_n_atm      = sum(r["n_atm"]             for r in results)
    total_n_itm      = sum(r["n_itm"]             for r in results)

    pooled_rmse        = float((total_sse / total_n) ** 0.5)
    pooled_mae         = float(total_sae / total_n)
    pooled_rmse_spread = float((total_sse_spread / total_n) ** 0.5) if total_sse_spread == total_sse_spread else float("nan")
    pooled_rmse_otm    = float((total_sse_otm / total_n_otm) ** 0.5) if total_n_otm > 0 else float("nan")
    pooled_rmse_atm    = float((total_sse_atm / total_n_atm) ** 0.5) if total_n_atm > 0 else float("nan")
    pooled_rmse_itm    = float((total_sse_itm / total_n_itm) ** 0.5) if total_n_itm > 0 else float("nan")
    worst_rmse  = float(max(rmses))
    worst_fold  = results[int(np.argmax(rmses))]["fold"]
    pooled_arb_butterfly = float(np.mean([r["arb_butterfly_ratio"] for r in results]))
    pooled_arb_calendar  = float(np.mean([r["arb_calendar_ratio"]  for r in results]))

    strat_pool = f"{pooled_rmse_otm:.2f}/{pooled_rmse_atm:.2f}/{pooled_rmse_itm:.2f}"
    print(f"{'Pooled':<10} | {'':>6} | {'':>6} | {pooled_rmse:>7.2f} | "
          f"{pooled_rmse_spread:>6.2f} | {strat_pool:>16} | "
          f"{100*(pooled_arb_butterfly+pooled_arb_calendar):>5.1f}% |   <- headline")
    e_star_mean = float(np.mean(e_stars)) if e_stars else 0.0
    e_star_std  = float(np.std(e_stars))  if e_stars else 0.0
    if e_stars:
        print(f"{'Mean(fold)':<10} | {'':>6} | {e_star_mean:>6.0f} | "
              f"{np.mean(rmses):>7.2f} |  E*: mean={e_star_mean:.0f}  std={e_star_std:.0f}  "
              f"min={min(e_stars)} max={max(e_stars)}")

    # Save results
    summary = {
        "vol_type":           args.vol_type,
        "pooled_rmse_mkt":    pooled_rmse,
        "pooled_mae_mkt":     pooled_mae,
        "pooled_rmse_spread": pooled_rmse_spread,
        "pooled_rmse_otm":    pooled_rmse_otm,
        "pooled_rmse_atm":    pooled_rmse_atm,
        "pooled_rmse_itm":    pooled_rmse_itm,
        "mean_rmse_mkt":      float(np.mean(rmses)),
        "std_rmse_mkt":       float(np.std(rmses)),
        "worst_rmse_mkt":     worst_rmse,
        "worst_fold":         worst_fold,
        "e_star_mean":        e_star_mean,
        "e_star_std":         e_star_std,
        "e_star_min":         int(min(e_stars)) if e_stars else None,
        "e_star_max":         int(max(e_stars)) if e_stars else None,
        "mean_arb_butterfly_ratio": pooled_arb_butterfly,
        "mean_arb_calendar_ratio":  pooled_arb_calendar,
        "total_n_test":       total_n,
        "n_folds":            len(results),
        "pricing_lr":         args.pricing_lr,
        "vol_lr":             args.vol_lr,
        "epochs":             args.epochs,
        "rwf_mu":             args.rwf_mu,
        "val_months":         args.val_months,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
