"""
W0 σ_θ-vs-market-IV overlay (analysis-only; run on DCC in pinn_env).

The decisive scientific check for Stage 2: is the learned volatility surface
σ_θ(m, τ) tracking the EMPIRICAL implied-vol smile/term-structure, or is it
just extra fitting freedom? If σ_θ correlates with market IV, Stage 2 learns
real volatility (meaningful even at flat RMSE); if not, the "pressure valve"
story is mechanical, not economic.

For each Stage 2 run dir × fold this script:
  1. rebuilds the vol model (wrapper + net) from fold_<F>.pt
  2. pulls the fold's TEST-month quotes (moneyness, τ, impl_volatility)
  3. scores σ_θ(m_i, τ_i) against iv_i — correlation + vol-space RMSE — and
     compares against the constant σ_fixed baseline (does σ_θ beat a flat
     line at all?)
  4. plots smile overlays in three maturity buckets + writes a CSV summary

Usage (from project root):
    python scripts/overlay_vol_surface.py                       # all 4 cells, Nov+Dec
    python scripts/overlay_vol_surface.py \
        --run_dirs runs/stage2_B10/cvol --folds Nov2020
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_and_preprocess                           # noqa: E402
from src.model import VolatilityNet, CVolWrapper, AVolWrapper      # noqa: E402
from run_walk_forward import FOLDS                                 # noqa: E402
from scripts._csv_util import merge_csv_rows, write_csv_rows       # noqa: E402

DEFAULT_RUN_DIRS = [
    "runs/stage2_B10/cvol",
    "runs/stage2_B10/avol",
    "runs/stage2_B12/cvol",
    "runs/stage2_B12/avol",
]
TAU_BUCKETS = [(0.0, 30 / 365), (30 / 365, 90 / 365), (90 / 365, 10.0)]
BUCKET_LABELS = ["τ ≤ 30d", "30d < τ ≤ 90d", "τ > 90d"]


def load_vol_model(ckpt_path, vol_type, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["vol_model_state_dict"]
    cfg = ckpt["config"]
    vol_net = VolatilityNet()          # defaults match all v1 Stage 2 runs
    if vol_type == "cvol":
        # sigma_0 is a registered buffer → restored by load_state_dict;
        # the constructor value is a placeholder.
        wrapper = CVolWrapper(vol_net, sigma_0=cfg["sigma_fixed"])
    else:
        wrapper = AVolWrapper(vol_net, sigma_init=cfg["sigma_fixed"])
    wrapper.load_state_dict(state)
    wrapper.to(device).eval()
    return wrapper, cfg


def main():
    ap = argparse.ArgumentParser(description="W0 σ_θ vs market-IV overlay")
    ap.add_argument("--run_dirs", default=",".join(DEFAULT_RUN_DIRS))
    ap.add_argument("--folds", default="Nov2020,Dec2020")
    ap.add_argument("--data", default="data/TSLA_2020_Split_Adjusted.csv")
    ap.add_argument("--out_dir", default="reports/diagnostic")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess(args.data)
    fold_by_name = {f["name"]: f for f in FOLDS}
    rows = []

    for run_dir in args.run_dirs.split(","):
        run_dir = Path(run_dir.strip())
        vol_type = ("cvol" if "cvol" in run_dir.name
                    else "avol" if "avol" in run_dir.name else None)
        if vol_type is None:
            print(f"SKIP (cannot infer vol_type from name): {run_dir}")
            continue
        tag = f"{run_dir.parent.name}_{run_dir.name}"   # e.g. stage2_B10_cvol

        for fold_name in args.folds.split(","):
            fold_name = fold_name.strip()
            ckpt = run_dir / f"fold_{fold_name}.pt"
            if not ckpt.exists():
                print(f"SKIP (no checkpoint): {ckpt}")
                continue
            fold = fold_by_name[fold_name]

            wrapper, cfg = load_vol_model(ckpt, vol_type, device)
            sigma_fixed = float(cfg["sigma_fixed"])

            test = df[(df["date"] >= pd.Timestamp(fold["test_start"]))
                      & (df["date"] < pd.Timestamp(fold["test_end"]))]
            if len(test) == 0:
                print(f"SKIP (no test rows): {fold_name}")
                continue

            m_i = test["moneyness"].values.astype(np.float32)
            tau_i = test["time_to_exp"].values.astype(np.float32)
            iv_i = test["impl_volatility"].values.astype(np.float64)

            with torch.no_grad():
                sig_i = wrapper(torch.tensor(m_i).to(device),
                                torch.tensor(tau_i).to(device)
                                ).squeeze().cpu().numpy().astype(np.float64)

            corr = float(np.corrcoef(sig_i, iv_i)[0, 1])
            rmse_sig = float(np.sqrt(((sig_i - iv_i) ** 2).mean()))
            rmse_flat = float(np.sqrt(((sigma_fixed - iv_i) ** 2).mean()))
            rows.append({
                "run": tag, "fold": fold_name, "vol_type": vol_type,
                "sigma_fixed": sigma_fixed,
                "corr_sigma_iv": corr,
                "rmse_sigma_vs_iv": rmse_sig,
                "rmse_flat_vs_iv": rmse_flat,
                "beats_flat": rmse_sig < rmse_flat,
                "sigma_min": float(sig_i.min()),
                "sigma_max": float(sig_i.max()),
                "n_test": len(test),
            })
            verdict = "BEATS flat σ" if rmse_sig < rmse_flat else "does NOT beat flat σ"
            print(f"{tag:<22} {fold_name}: corr(σ_θ, IV)={corr:+.3f}  "
                  f"vol-RMSE {rmse_sig:.4f} vs flat {rmse_flat:.4f} → {verdict}")

            # ── Smile overlay, three maturity buckets ────────────────
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
            m_grid = np.linspace(m_i.min(), m_i.max(), 200).astype(np.float32)
            for ax, (lo, hi), label in zip(axes, TAU_BUCKETS, BUCKET_LABELS):
                mask = (tau_i > lo) & (tau_i <= hi)
                if mask.sum() == 0:
                    ax.set_title(f"{label} (no quotes)")
                    continue
                ax.scatter(m_i[mask], iv_i[mask], s=4, alpha=0.25,
                           color="grey", label="market IV")
                tau_med = float(np.median(tau_i[mask]))
                with torch.no_grad():
                    sig_curve = wrapper(
                        torch.tensor(m_grid).to(device),
                        torch.tensor(np.full_like(m_grid, tau_med)).to(device)
                    ).squeeze().cpu().numpy()
                ax.plot(m_grid, sig_curve, lw=2, color="crimson",
                        label=f"σ_θ(m, τ={tau_med:.2f})")
                ax.axhline(sigma_fixed, ls="--", lw=1.2, color="steelblue",
                           label=f"σ_fixed={sigma_fixed:.2f}")
                ax.set_title(f"{label}  (n={int(mask.sum())})")
                ax.set_xlabel("m = F/K")
                ax.legend(fontsize=8)
            axes[0].set_ylabel("volatility")
            fig.suptitle(f"{tag} — {fold_name}: learned σ_θ vs market IV  "
                         f"(corr {corr:+.2f}; vol-RMSE {rmse_sig:.3f} "
                         f"vs flat {rmse_flat:.3f})")
            fig.tight_layout()
            fig.savefig(out_dir / f"vol_overlay_{tag}_{fold_name}.png", dpi=140)
            plt.close(fig)

    if rows:
        csv_path = out_dir / "vol_overlay_summary.csv"
        # Merge (not truncate): incremental re-runs accumulate.
        merged = merge_csv_rows(csv_path, rows, key=("run", "fold"))
        write_csv_rows(csv_path, merged)
        print(f"\nWrote {csv_path} ({len(rows)} new/updated rows, "
              f"{len(merged)} total) and PNGs to {out_dir}/")
    else:
        print("No checkpoints found — nothing to overlay.")


if __name__ == "__main__":
    main()
