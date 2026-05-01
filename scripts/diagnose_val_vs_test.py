"""
Diagnostic: train modified_hybrid and standard_hybrid on ONE fold,
tracking BOTH val and test rmse curves. Plot them side by side.

This tells us whether val-best snapshot selection is matching test-best
or whether they diverge — which would explain why modified hybrid
underperforms its expected behavior in walk-forward.

Usage:
  python scripts/diagnose_val_vs_test.py --fold Nov2020
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (load_and_preprocess, make_fold, df_to_arrays,
                      compute_constants, build_boundary_terminal)
from src.training import run_training


FOLD_DEFS = {
    "Apr2020": ("2020-04-01", "2020-04-01", "2020-05-01"),
    "Sep2020": ("2020-09-01", "2020-09-01", "2020-10-01"),
    "Nov2020": ("2020-11-01", "2020-11-01", "2020-12-01"),
    "Dec2020": ("2020-12-01", "2020-12-01", "2021-01-01"),
}


def run_one(arch, mode, rwf_mu, fold_name, train_end, test_start, test_end,
            data_csv, epochs, device):
    df = load_and_preprocess(data_csv)
    train_df, val_df, test_df = make_fold(df, train_end, test_start, test_end,
                                          val_months=1)
    config = compute_constants(train_df)
    train_arrays = df_to_arrays(train_df)
    test_arrays  = df_to_arrays(test_df)
    val_arrays   = df_to_arrays(val_df)
    boundary = build_boundary_terminal(config, config["sigma_fixed"], config["r_fixed"])

    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"  σ_fixed: {config['sigma_fixed']:.4f}")

    _model, history, run_info = run_training(
        train_arrays=train_arrays,
        test_arrays=test_arrays,
        val_arrays=val_arrays,
        boundary_data=boundary,
        config=config,
        mode=mode, arch=arch,
        epochs=epochs, lr=1e-3,
        rwf_mu=rwf_mu, seed=42, device=device,
        verbose=False, log_every=10**9, val_every=500,
        track_test_curve=True,                # ← diagnostic ON
    )

    return {
        "arch": arch, "mode": mode, "rwf_mu": rwf_mu,
        "val_epoch": list(history["val_epoch"]),
        "val_rmse_mkt": list(history["val_rmse_mkt"]),
        "test_rmse_mkt": list(history["test_rmse_mkt_diagnostic"]),
        "best_val_epoch": run_info["best_val_epoch"],
        "best_val_metric": run_info["best_val_metric"],
        "reported_test_rmse": run_info["final_rmse_mkt"],   # at val-best
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="Nov2020", choices=list(FOLD_DEFS.keys()))
    ap.add_argument("--epochs", type=int, default=15000)
    ap.add_argument("--data", default="data/TSLA_2020_Split_Adjusted.csv")
    ap.add_argument("--output_dir", default="reports/diagnostic")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_end, test_start, test_end = FOLD_DEFS[args.fold]

    runs = []
    for cfg in [
        ("standard", "hybrid",  None),
        ("modified", "hybrid",  0.75),
        ("modified", "hybrid",  1.0),
    ]:
        arch, mode, mu = cfg
        tag = f"{arch}_{mode}" + (f"_mu{mu}" if mu is not None else "")
        print(f"\n=== {tag}  (fold {args.fold}, {args.epochs} epochs) ===")
        r = run_one(arch, mode, mu or 1.0, args.fold, train_end, test_start, test_end,
                    args.data, args.epochs, device)
        r["tag"] = tag
        runs.append(r)

        # Quick summary
        v = np.array(r["val_rmse_mkt"])
        t = np.array(r["test_rmse_mkt"])
        eps = np.array(r["val_epoch"])
        i_val_best = int(np.argmin(v))
        i_test_best = int(np.argmin(t))
        print(f"  val-best : ep {eps[i_val_best]:>5}  val={v[i_val_best]:.3f}  test={t[i_val_best]:.3f}  ← REPORTED")
        print(f"  test-best: ep {eps[i_test_best]:>5}  val={v[i_test_best]:.3f}  test={t[i_test_best]:.3f}  (oracle)")
        print(f"  final-ep : ep {eps[-1]:>5}  val={v[-1]:.3f}  test={t[-1]:.3f}")
        print(f"  reported test (= val-best snapshot post-restore): {r['reported_test_rmse']:.3f}")

    # Save raw curves
    json_path = out_dir / f"diagnostic_{args.fold}.json"
    with open(json_path, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"\nWrote {json_path}")

    # Plot
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), sharey=False)
    if len(runs) == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        eps = np.array(r["val_epoch"])
        v = np.array(r["val_rmse_mkt"])
        t = np.array(r["test_rmse_mkt"])
        ax.plot(eps, v, "b-", label="val", lw=1.5)
        ax.plot(eps, t, "r-", label="test (diagnostic)", lw=1.5)
        i_v = int(np.argmin(v))
        i_t = int(np.argmin(t))
        ax.axvline(eps[i_v], color="blue", linestyle="--", alpha=0.4,
                   label=f"val-best ep {eps[i_v]}")
        ax.axvline(eps[i_t], color="red",  linestyle="--", alpha=0.4,
                   label=f"test-best ep {eps[i_t]}")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (log $)")
        ax.set_title(f"{r['tag']}\n"
                     f"val-best→ test={t[i_v]:.2f},  "
                     f"test-best→ {t[i_t]:.2f},  "
                     f"final→ {t[-1]:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Val vs test trajectories — fold {args.fold}", fontsize=13)
    plt.tight_layout()
    fig_path = out_dir / f"diagnostic_{args.fold}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
