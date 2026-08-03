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
import numpy as np
from pathlib import Path

from src.data import (load_and_preprocess, make_fold, compute_constants,
                      df_to_arrays, build_boundary_terminal)
from src.training import run_training
from src.diagnostics import compute_arbitrage_ratios


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
    parser.add_argument("--val_months", type=int, default=1,
                        help="Months of training tail held out as the "
                             "validation window (val = month immediately "
                             "before test). Used to drive val-best "
                             "snapshot selection; never part of training.")
    parser.add_argument("--fixed_data_weight", type=float, default=None,
                        help="Pin λ_data to this value and exclude it from "
                             "Wang grad-norm balancing. Decouples data "
                             "fitting from constraint balancing — useful "
                             "when grad-norm runs away in hybrid+RWF setups. "
                             "Typical value: 1.0 to 10.0. Default: None "
                             "(data is balanced like everything else).")
    parser.add_argument("--data_loss_warmup", type=int, default=0,
                        help="Number of warmup epochs to train as physics-only "
                             "before turning on L_data. Lets PDE/TC/BC find "
                             "a stable basin first. Default: 0 (no warmup).")
    parser.add_argument("--balancer", default="gradnorm",
                        choices=["gradnorm", "gradnorm_renorm", "relobralo", "fixed"],
                        help="Loss-balancing scheme (W1). 'gradnorm' = v1 "
                             "default (Wang, UNBOUNDED — reproduces completed "
                             "runs); 'gradnorm_renorm' = bounded Σλ control; "
                             "'relobralo' = bounded loss-statistics scheme; "
                             "'fixed' = no adaptation (robustness baseline). "
                             "Non-default choices are suffixed onto the "
                             "output dir name.")
    parser.add_argument("--relobralo_temperature", type=float, default=0.1,
                        help="ReLoBRaLo softmax temperature T (only used "
                             "with --balancer relobralo).")
    parser.add_argument("--relobralo_rho", type=float, default=0.99,
                        help="ReLoBRaLo Bernoulli lookback probability ρ, "
                             "drawn once per balancer update (only used "
                             "with --balancer relobralo).")
    parser.add_argument("--regime_input", default="none",
                        choices=["none", "atm_iv_lag"],
                        help="R1 regime conditioning. 'atm_iv_lag' adds a "
                             "third network input ν = per-quote 1-day-lagged "
                             "ATM IV (strictly no-look-ahead). Physics stays "
                             "at σ_fixed; ν makes the DATA term identifiable "
                             "across vol regimes. Default 'none' = v1 "
                             "2-input net, byte-identical.")
    parser.add_argument("--track_test_curve", action="store_true",
                        help="DIAGNOSTIC ONLY: also evaluate the test month "
                             "at every val cadence and record the curve in "
                             "history. Never used for selection — enables "
                             "post-hoc val-best vs final-epoch vs oracle "
                             "comparison (W0/W3).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load full dataset once ──────────────────────────────────────
    print("Loading and preprocessing data...")
    df = load_and_preprocess(args.data)
    print(f"Full dataset: {len(df):,} obs, {df['date'].min().date()} → {df['date'].max().date()}")

    # Output directory naming reflects arch + mode + (μ if applicable)
    # plus loss-balancing ablation suffixes (so B6 vs B9 vs B10 land in
    # distinct directories).
    if args.arch == "modified":
        run_tag = f"{args.arch}_{args.mode}_mu{args.rwf_mu}"
    else:
        run_tag = f"{args.arch}_{args.mode}"
    if args.fixed_data_weight is not None:
        run_tag += f"_fixdata{args.fixed_data_weight}"
    if args.data_loss_warmup > 0:
        run_tag += f"_warmup{args.data_loss_warmup}"
    if args.balancer != "gradnorm":
        run_tag += f"_{args.balancer}"
    if args.regime_input != "none":
        run_tag += "_nuatm"
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

        # Build fold data — train / val / test
        train_df, val_df, test_df = make_fold(
            df, train_end=fold["train_end"],
            test_start=fold["test_start"], test_end=fold["test_end"],
            val_months=args.val_months,
        )

        if len(test_df) == 0:
            print(f"  WARNING: No test data for {fold['name']}, skipping")
            continue

        # Train: model never sees val or test. Val drives val-best
        # snapshot; test is evaluated ONCE after restore (inside run_training).
        config = compute_constants(train_df)
        train_arrays = df_to_arrays(train_df)
        test_arrays  = df_to_arrays(test_df)
        val_arrays   = df_to_arrays(val_df) if len(val_df) > 0 else None
        boundary_data = build_boundary_terminal(
            config, config["sigma_fixed"], config["r_fixed"]
        )

        print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
        print(f"  σ_fixed: {config['sigma_fixed']:.4f}")
        if val_arrays is None and args.val_months > 0:
            print(f"  WARNING: val_months={args.val_months} requested but val window is empty")

        model, history, run_info = run_training(
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            val_arrays=val_arrays,
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
            val_every=500,
            fixed_data_weight=args.fixed_data_weight,
            data_loss_warmup=args.data_loss_warmup,
            balancer=args.balancer,
            relobralo_temperature=args.relobralo_temperature,
            relobralo_rho=args.relobralo_rho,
            track_test_curve=args.track_test_curve,
            regime_input=(None if args.regime_input == "none"
                          else args.regime_input),
        )

        # Model is now restored to val-best (or final-epoch if val=None).
        # No-arbitrage diagnostic on the reported model. With regime
        # conditioning, check the ν-slice this fold actually prices with
        # (the test month's median ν).
        nu_slice = (float(np.median(test_arrays["nu"]))
                    if args.regime_input != "none" else None)
        arb = compute_arbitrage_ratios(model, config, device, nu=nu_slice)

        fold_result = {
            "fold": fold["name"],
            "n_train": len(train_df),
            "n_val":   len(val_df),
            "n_test":  len(test_df),
            "sigma_fixed": config["sigma_fixed"],
            # Selection / reporting
            "used_val":         run_info["used_val"],
            "restored_to_best": run_info["restored_to_best"],
            "best_val_epoch":   run_info["best_val_epoch"],
            "best_val_metric":  run_info["best_val_metric"],
            # Reported test metrics (val-best snapshot, single eval)
            "rmse_bs_norm": run_info["final_rmse_bs_norm"],
            "rmse_mkt":     run_info["final_rmse_mkt"],
            "mae_mkt":      run_info["final_mae_mkt"],
            "medape_mkt":   run_info.get("final_medape_mkt"),
            "rmse_spread":  run_info["final_rmse_spread"],
            "rmse_otm":     run_info["final_rmse_otm"],
            "rmse_atm":     run_info["final_rmse_atm"],
            "rmse_itm":     run_info["final_rmse_itm"],
            "n_otm":        run_info["n_otm"],
            "n_atm":        run_info["n_atm"],
            "n_itm":        run_info["n_itm"],
            # Pooling ingredients
            "sum_sq_err_mkt":    run_info["final_sum_sq_err_mkt"],
            "sum_abs_err_mkt":   run_info["final_sum_abs_err_mkt"],
            "sum_sq_err_spread": run_info["final_sum_sq_err_spread"],
            "sum_sq_err_otm":    run_info["final_sum_sq_err_otm"],
            "sum_sq_err_atm":    run_info["final_sum_sq_err_atm"],
            "sum_sq_err_itm":    run_info["final_sum_sq_err_itm"],
            # Arbitrage diagnostic — rate AND severity (int_neg = integrated
            # negative part; rates alone hide the local blow-ups)
            "arb_butterfly_ratio":   arb["butterfly_ratio"],
            "arb_calendar_ratio":    arb["calendar_ratio"],
            "arb_butterfly_max_neg": arb["butterfly_max_neg"],
            "arb_calendar_max_neg":  arb["calendar_max_neg"],
            "arb_butterfly_int_neg": arb["butterfly_int_neg"],
            "arb_calendar_int_neg":  arb["calendar_int_neg"],
            "elapsed": run_info["elapsed_seconds"],
            # Val curve trajectory (for diagnostic plotting)
            "val_epoch":        [int(x)   for x in history["val_epoch"]],
            "val_rmse_mkt":     [float(x) for x in history["val_rmse_mkt"]],
            "val_rmse_bs_norm": [float(x) for x in history["val_rmse_bs_norm"]],
        }
        results.append(fold_result)

        # Save fold checkpoint — this IS the val-best snapshot (the
        # model has already been restored). Stage 2 warm-starts from here.
        # `history` carries per-epoch loss + adaptive-weight series; needed
        # for stability/PDE-dominance plots in the paper.
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "run_info": run_info,
            "history": history,
        }, output_dir / f"fold_{fold['name']}.pt")

        ep = run_info["best_val_epoch"] or run_info["epochs"]
        print(f"  → REPORTED test RMSE vs Market: ${run_info['final_rmse_mkt']:.2f} "
              f"(val-best epoch: {ep})")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n\n{'='*86}")
    if args.arch == "modified":
        print(f"WALK-FORWARD RESULTS: arch={args.arch}, mode={args.mode}, μ={args.rwf_mu}")
    else:
        print(f"WALK-FORWARD RESULTS: arch={args.arch}, mode={args.mode}")
    if results and results[0]["used_val"]:
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

    # Pooled metrics — single RMSE over all concatenated residuals across folds.
    # Headline metric per BENCHMARKING_PLAN.md "Reporting Protocol".
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
          f"{100*(pooled_arb_butterfly+pooled_arb_calendar):>5.1f}% |   ← headline")
    e_star_mean = float(np.mean(e_stars)) if e_stars else 0.0
    e_star_std  = float(np.std(e_stars))  if e_stars else 0.0
    if e_stars:
        print(f"{'Mean(fold)':<10} | {'':>6} | {e_star_mean:>6.0f} | "
              f"{np.mean(rmses):>7.2f} |  E*: mean={e_star_mean:.0f}  std={e_star_std:.0f}  "
              f"min={min(e_stars)} max={max(e_stars)}")

    # Save results — folds list + aggregate summary block.
    summary = {
        "pooled_rmse_mkt":     pooled_rmse,
        "pooled_mae_mkt":      pooled_mae,
        "pooled_rmse_spread":  pooled_rmse_spread,
        "pooled_rmse_otm":     pooled_rmse_otm,
        "pooled_rmse_atm":     pooled_rmse_atm,
        "pooled_rmse_itm":     pooled_rmse_itm,
        "mean_rmse_mkt":       float(np.mean(rmses)),
        "std_rmse_mkt":        float(np.std(rmses)),
        "mean_mae_mkt":        float(np.mean(maes)),
        "std_mae_mkt":         float(np.std(maes)),
        "worst_rmse_mkt":      worst_rmse,
        "worst_fold":          worst_fold,
        "e_star_mean":         e_star_mean,
        "e_star_std":          e_star_std,
        "e_star_min":          int(min(e_stars)) if e_stars else None,
        "e_star_max":          int(max(e_stars)) if e_stars else None,
        "mean_arb_butterfly_ratio": pooled_arb_butterfly,
        "mean_arb_calendar_ratio":  pooled_arb_calendar,
        # Severity (integrated negative part) — mean across folds
        "mean_arb_butterfly_int_neg": float(np.mean(
            [r["arb_butterfly_int_neg"] for r in results])),
        "mean_arb_calendar_int_neg":  float(np.mean(
            [r["arb_calendar_int_neg"] for r in results])),
        "total_n_test":        total_n,
        "n_folds":             len(results),
        "val_months":          args.val_months,
        "balancer":            args.balancer,
        "regime_input":        args.regime_input,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
