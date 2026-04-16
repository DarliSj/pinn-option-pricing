"""
Example: Run Stage 0 training using the modular code.

Usage:
    python run_stage0.py --mode physics --epochs 15000 --rwf_mu 0.75
    python run_stage0.py --mode hybrid --epochs 15000 --rwf_mu 1.0
"""

import argparse
import json
import torch
from pathlib import Path

from src.data import (load_and_preprocess, make_temporal_split, compute_constants,
                      df_to_arrays, build_boundary_terminal)
from src.training import run_training
from src.diagnostics import (plot_training_summary, plot_full_diagnostics,
                              plot_solution_surface, plot_test_scatter,
                              print_epoch_summary)


def main():
    parser = argparse.ArgumentParser(description="Stage 0 PINN Training")
    parser.add_argument("--data", default="data/TSLA_2020_Split_Adjusted.csv")
    parser.add_argument("--mode", choices=["physics", "hybrid"], default="physics")
    parser.add_argument("--arch", choices=["standard", "modified"], default="modified",
                        help="Network architecture: 'modified' (Modified MLP + RWF, default) "
                             "or 'standard' (plain MLP, naive PINN baseline). "
                             "When 'standard', --rwf_mu and --rwf_sigma are ignored.")
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rwf_mu", type=float, default=0.75)
    parser.add_argument("--rwf_sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────
    print("Loading and preprocessing data...")
    df = load_and_preprocess(args.data)
    train_df, test_df = make_temporal_split(df)
    config = compute_constants(train_df)
    train_arrays = df_to_arrays(train_df)
    test_arrays = df_to_arrays(test_df)
    boundary_data = build_boundary_terminal(config, config["sigma_fixed"], config["r_fixed"])

    print(f"Train: {config['n_train']:,}  Test: {len(test_df):,}")
    print(f"σ_fixed: {config['sigma_fixed']:.4f}  r_fixed: {config['r_fixed']:.4f}")

    # ── Train ───────────────────────────────────────────────────────
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
        rwf_sigma=args.rwf_sigma,
        seed=args.seed,
        device=device,
    )

    # ── Save ────────────────────────────────────────────────────────
    if args.arch == "modified":
        run_tag = f"stage0_{args.arch}_{args.mode}_mu{args.rwf_mu}"
    else:
        run_tag = f"stage0_{args.arch}_{args.mode}"
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "run_info": run_info,
    }, output_dir / "checkpoint.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f)

    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    print(f"\nSaved to {output_dir}/")

    # ── Summary ─────────────────────────────────────────────────────
    loss_names = ["pde", "tc", "bc"] + (["data"] if args.mode == "hybrid" else [])
    arch_label = f"arch={args.arch}, mode={args.mode}"
    if args.arch == "modified":
        arch_label += f", μ={args.rwf_mu}"
    print(f"\n{'='*60}")
    print(f"STAGE 0 RESULTS ({arch_label})")
    print(f"  Reporting final-epoch metrics; best/drift = stability diagnostics only")
    print(f"{'='*60}")
    print(f"Final RMSE vs BS (norm):  {run_info['final_rmse_bs_norm']:.6f}")
    print(f"Final RMSE vs Market ($): ${run_info['final_rmse_mkt']:.2f}")
    print(f"Final MAE  vs Market ($): ${run_info['final_mae_mkt']:.2f}")
    if run_info.get("best_rmse_mkt") is not None:
        print(f"  [diag] best epoch:      {run_info['best_epoch']}")
        print(f"  [diag] best RMSE_mkt:   ${run_info['best_rmse_mkt']:.2f}")
        print(f"  [diag] drift gap:       {run_info['drift_gap']:+.4f} "
              f"({run_info['selection_key']})")
    print(f"Training time:            {run_info['elapsed_seconds']:.0f}s")
    print(f"{'='*60}")

    print_epoch_summary(history)

    # ── Plots ───────────────────────────────────────────────────────
    if not args.no_plots:
        suffix = f"({args.mode}, μ={args.rwf_mu})"

        fig1 = plot_training_summary(history, loss_names, suffix)
        fig1.savefig(output_dir / "training_summary.png", dpi=150, bbox_inches="tight")

        fig2 = plot_full_diagnostics(history, loss_names, suffix)
        fig2.savefig(output_dir / "diagnostics.png", dpi=150, bbox_inches="tight")

        fig3 = plot_solution_surface(model, config, config["sigma_fixed"],
                                     config["r_fixed"], device, suffix)
        fig3.savefig(output_dir / "solution_surface.png", dpi=150, bbox_inches="tight")

        fig4, _ = plot_test_scatter(model, test_arrays, args.mode, device, suffix)
        fig4.savefig(output_dir / "test_scatter.png", dpi=150, bbox_inches="tight")

        print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
