"""
W0 physics-fidelity audit (analysis-only; run on DCC in pinn_env).

Question it answers: even PHYSICS-ONLY PINNs violate no-arbitrage (B4: 38%
butterfly / 21% calendar) although the analytic BS surface they approximate is
arbitrage-free by construction. How much of that violation budget is removable
optimization error (the net simply isn't at the BS solution) vs. structural?

For each physics config × fold checkpoint this script:
  1. rebuilds the reported (val-best) model from runs/walk_forward/<cfg>/fold_<F>.pt
  2. evaluates PINN vs analytic BS on a dense (m, τ) grid → pointwise error map
  3. finite-differences butterfly (∂²v̂/∂m²) and calendar (∂v̂/∂τ) → rate,
     integrated severity, worst case — for BOTH the PINN and the analytic
     surface (the analytic numbers are the finite-difference noise floor)
  4. writes reports/diagnostic/physics_fidelity.csv + one PNG per (cfg, fold)

Interpretation: if PINN error-vs-BS is large exactly where violations sit,
the violations are optimization slop (fixable by training longer / better);
if the PINN matches BS closely yet still violates, the slop is in the
sub-grid curvature — an argument that a pointwise penalty (W4) is needed
even in physics mode.

Usage (from project root):
    python scripts/audit_physics_fidelity.py                       # defaults
    python scripts/audit_physics_fidelity.py --folds Nov2020,Sep2020
    python scripts/audit_physics_fidelity.py --runs_dir runs/walk_forward \
        --configs standard_physics,modified_physics_mu1.0
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as `python scripts/audit_physics_fidelity.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bs_formulas import bs_call_normalized                     # noqa: E402
from src.model import PricingNet, StandardPricingNet               # noqa: E402
from scripts._csv_util import merge_csv_rows, write_csv_rows       # noqa: E402

DEFAULT_CONFIGS = [
    "standard_physics",           # B0
    "modified_physics_mu0.5",     # B2
    "modified_physics_mu0.75",    # B3
    "modified_physics_mu1.0",     # B4
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"]
    cfg = ckpt["config"]
    # arch from the state dict itself: Modified MLP has encoder_U/V keys
    is_modified = any(k.startswith("encoder_U") for k in state)
    model = (PricingNet() if is_modified else StandardPricingNet()).to(device)
    # state_dict load restores ALL params + buffers (incl. the Fourier B
    # matrix), so construction defaults don't matter beyond shapes.
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def surface_stats(v_fn, cfg, n_m=200, n_tau=100, eps_m=1e-3, eps_tau=1e-3,
                  tol=1e-6):
    """Butterfly/calendar rate + severity for any callable v(m_flat, tau_flat).

    Same finite-difference scheme as src.diagnostics.compute_arbitrage_ratios
    (denser grid) so numbers are directly comparable.
    """
    m = np.linspace(cfg["m_min"], cfg["m_max"], n_m).astype(np.float32)
    tau = np.linspace(max(cfg["tau_min"], eps_tau), cfg["tau_max"],
                      n_tau).astype(np.float32)
    M, T = np.meshgrid(m, tau)
    mf, tf = M.flatten(), T.flatten()

    v = v_fn(mf, tf)
    v_mp = v_fn((mf + eps_m).clip(cfg["m_min"], cfg["m_max"]), tf)
    v_mm = v_fn((mf - eps_m).clip(cfg["m_min"], cfg["m_max"]), tf)
    v_tp = v_fn(mf, (tf + eps_tau).clip(max=cfg["tau_max"]))

    d2v = (v_mp - 2.0 * v + v_mm) / eps_m ** 2
    dvt = (v_tp - v) / eps_tau

    return {
        "v": v.reshape(M.shape), "M": M, "T": T,
        "bfly_map": np.maximum(-d2v, 0.0).reshape(M.shape),
        "cal_map":  np.maximum(-dvt, 0.0).reshape(M.shape),
        "bfly_rate": float((d2v < -tol).mean()),
        "bfly_int_neg": float(np.maximum(-d2v, 0.0).mean()),
        "bfly_max_neg": float(d2v.min()),
        "cal_rate": float((dvt < -tol).mean()),
        "cal_int_neg": float(np.maximum(-dvt, 0.0).mean()),
        "cal_max_neg": float(dvt.min()),
    }


def main():
    ap = argparse.ArgumentParser(description="W0 physics-fidelity audit")
    ap.add_argument("--runs_dir", default="runs/walk_forward")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--folds", default="Nov2020",
                    help="Comma-separated fold names (e.g. Nov2020,Sep2020), "
                         "or 'all' for the full 9-fold set. NOTE: the script "
                         "works on ANY walk-forward checkpoint (arch is "
                         "detected from the state dict), so it doubles as a "
                         "severity backfill for v1 hybrid configs — for "
                         "those, ignore the err-vs-BS columns (a hybrid model "
                         "SHOULD deviate from analytic BS) and read only the "
                         "arbitrage rate/severity columns.")
    ap.add_argument("--out_dir", default="reports/diagnostic")
    args = ap.parse_args()

    if args.folds.strip() == "all":
        args.folds = ("Apr2020,May2020,Jun2020,Jul2020,Aug2020,"
                      "Sep2020,Oct2020,Nov2020,Dec2020")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for cfg_name in args.configs.split(","):
        cfg_name = cfg_name.strip()
        for fold in args.folds.split(","):
            fold = fold.strip()
            ckpt = Path(args.runs_dir) / cfg_name / f"fold_{fold}.pt"
            if not ckpt.exists():
                print(f"SKIP (no checkpoint): {ckpt}")
                continue

            model, cfg = load_model(ckpt, device)
            sigma, r = cfg["sigma_fixed"], cfg["r_fixed"]

            def v_pinn(mf, tf):
                with torch.no_grad():
                    return model(torch.tensor(mf).to(device),
                                 torch.tensor(tf).to(device)
                                 ).squeeze().cpu().numpy()

            def v_bs(mf, tf):
                return bs_call_normalized(mf, tf, r, sigma).astype(np.float64)

            sp = surface_stats(v_pinn, cfg)
            sb = surface_stats(v_bs, cfg)   # finite-diff noise floor

            err = sp["v"] - sb["v"]
            row = {
                "config": cfg_name, "fold": fold,
                "err_rmse_norm": float(np.sqrt((err ** 2).mean())),
                "err_max_abs":   float(np.abs(err).max()),
                # PINN violation stats
                "pinn_bfly_rate": sp["bfly_rate"],
                "pinn_bfly_int_neg": sp["bfly_int_neg"],
                "pinn_bfly_max_neg": sp["bfly_max_neg"],
                "pinn_cal_rate": sp["cal_rate"],
                "pinn_cal_int_neg": sp["cal_int_neg"],
                "pinn_cal_max_neg": sp["cal_max_neg"],
                # Analytic-BS floor (finite-diff noise; ~0 expected)
                "bs_bfly_rate": sb["bfly_rate"],
                "bs_bfly_int_neg": sb["bfly_int_neg"],
                "bs_cal_rate": sb["cal_rate"],
                "bs_cal_int_neg": sb["cal_int_neg"],
            }
            rows.append(row)
            print(f"{cfg_name:<32} {fold}: err_rmse={row['err_rmse_norm']:.5f} "
                  f"bfly {100*sp['bfly_rate']:.1f}% (floor "
                  f"{100*sb['bfly_rate']:.1f}%)  sev {sp['bfly_int_neg']:.4g} "
                  f"| cal {100*sp['cal_rate']:.1f}% sev {sp['cal_int_neg']:.4g}")

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
            im0 = axes[0].pcolormesh(sp["M"], sp["T"], np.abs(err),
                                     shading="auto", cmap="viridis")
            axes[0].set_title(f"|PINN − analytic BS|  (rmse "
                              f"{row['err_rmse_norm']:.4f})")
            fig.colorbar(im0, ax=axes[0])
            im1 = axes[1].pcolormesh(sp["M"], sp["T"], sp["bfly_map"],
                                     shading="auto", cmap="magma")
            axes[1].set_title(f"butterfly severity relu(−v̂_mm)  "
                              f"rate {100*sp['bfly_rate']:.1f}%")
            fig.colorbar(im1, ax=axes[1])
            im2 = axes[2].pcolormesh(sp["M"], sp["T"], sp["cal_map"],
                                     shading="auto", cmap="magma")
            axes[2].set_title(f"calendar severity relu(−v̂_τ)  "
                              f"rate {100*sp['cal_rate']:.1f}%")
            fig.colorbar(im2, ax=axes[2])
            for ax in axes:
                ax.set_xlabel("m = F/K")
                ax.set_ylabel("τ")
            fig.suptitle(f"{cfg_name} — {fold} (val-best model)")
            fig.tight_layout()
            fig.savefig(out_dir / f"fidelity_{cfg_name}_{fold}.png", dpi=140)
            plt.close(fig)

    if rows:
        csv_path = out_dir / "physics_fidelity.csv"
        # Merge (not truncate): incremental re-runs over different
        # --configs/--folds subsets accumulate; same (config, fold) is updated.
        merged = merge_csv_rows(csv_path, rows, key=("config", "fold"))
        write_csv_rows(csv_path, merged)
        print(f"\nWrote {csv_path} ({len(rows)} new/updated rows, "
              f"{len(merged)} total) and PNGs to {out_dir}/")
    else:
        print("No checkpoints found — nothing audited.")


if __name__ == "__main__":
    main()
