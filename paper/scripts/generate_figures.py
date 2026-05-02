"""
Generate all matplotlib figures referenced by paper/paper.qmd.

Reads:
  runs/walk_forward_history/<config>/fold_Nov2020.pt   ← stability figs
                                                        (post-patch reruns;
                                                         skipped if absent)
  runs/walk_forward/<config>/fold_Nov2020.pt           ← val curves
                                                        (skipped if absent
                                                         OR if history not in
                                                         saved checkpoint —
                                                         old runs lack it)
  reports/master_table.csv                             ← Pareto + bar chart

Writes:
  paper/figures/fig-loss-curves-stability.png
  paper/figures/fig-weight-traj-stability.png
  paper/figures/fig-pde-dominance.png
  paper/figures/fig-arb-rmse-pareto.png
  paper/figures/fig-stratified-rmse.png
  paper/figures/fig-walkforward-timeline.png
  paper/figures/fig-val-curve-mu-warmup.png   (if B11/B12 history available)

Each figure is wrapped in a try/except — a missing source file emits a
SKIP message and the rest of the figures still render. This means you
can run this script BEFORE the DCC history-capture rerun lands and
still get all the master-table-driven figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib hygiene — publication-friendly defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                                  # project root
HIST_DIR  = ROOT / "runs" / "walk_forward_history"
WF_DIR    = ROOT / "runs" / "walk_forward"
REPORTS   = ROOT / "reports"
OUT       = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Run-tag mapping for figure source lookup
B6_TAG  = "modified_hybrid_mu0.75"               # no warmup → B6
B10_TAG = "modified_hybrid_mu0.75_warmup5000"
B11_TAG = "modified_hybrid_mu0.5_warmup5000"
B12_TAG = "modified_hybrid_mu0.25_warmup5000"


def load_history(parent_dir: Path, run_tag: str, fold: str = "Nov2020"):
    """Load `history` dict from a saved fold .pt. Returns None if file
    missing or .pt was saved before the history-persistence patch."""
    pt = parent_dir / run_tag / f"fold_{fold}.pt"
    if not pt.exists():
        return None
    # weights_only=False because the dict has Python lists (not just tensors).
    # Safe — only our own checkpoints get loaded.
    import torch
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    return ckpt.get("history")


# ──────────────────────────────────────────────────────────────────
# Tier 1 — stability narrative (B6 vs B10)
# ──────────────────────────────────────────────────────────────────

LOSS_NAMES = ["pde", "tc", "bc", "data"]      # reg often has different scale → not in stability triptych
LOSS_COLORS = {
    "pde":  "#1f77b4",
    "tc":   "#ff7f0e",
    "bc":   "#2ca02c",
    "data": "#d62728",
    "reg":  "#9467bd",
}


def fig_loss_curves_stability():
    """Per-loss curves on log scale, B6 vs B10, single fold (Nov2020).

    Two side-by-side panels: left = B6 (failure mode — L_data explodes,
    PDE drops), right = B10 (warmup fix — losses balanced).
    """
    h_b6  = load_history(HIST_DIR, B6_TAG)
    h_b10 = load_history(HIST_DIR, B10_TAG)
    if h_b6 is None or h_b10 is None:
        raise FileNotFoundError(
            f"history missing — run `sbatch slurm/run_history_capture.sh` first.\n"
            f"  expected: {HIST_DIR/B6_TAG/'fold_Nov2020.pt'}\n"
            f"            {HIST_DIR/B10_TAG/'fold_Nov2020.pt'}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, h, title in zip(axes, [h_b6, h_b10],
                            ["B6 — modified hybrid, μ=0.75 (no warmup)",
                             "B10 — modified hybrid, μ=0.75, warmup=5000"]):
        ep = h["epoch"]
        for name in LOSS_NAMES:
            ax.semilogy(ep, h[f"L_{name}"], color=LOSS_COLORS[name],
                        label=fr"$\mathcal{{L}}_{{\mathrm{{{name}}}}}$",
                        alpha=0.85, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
    axes[0].set_ylabel("Loss (log)")
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle("Per-loss training curves: failure mode vs. warmup fix", y=1.02)
    fig.savefig(OUT / "fig-loss-curves-stability.png")
    plt.close(fig)


def fig_weight_traj_stability():
    """Adaptive λ trajectories (grad-norm balanced) over training, B6 vs B10.

    The B6 panel shows λ_data running to 10⁵+ — the diagnosed failure mode;
    B10's warmup keeps it bounded.
    """
    h_b6  = load_history(HIST_DIR, B6_TAG)
    h_b10 = load_history(HIST_DIR, B10_TAG)
    if h_b6 is None or h_b10 is None:
        raise FileNotFoundError("history missing — see fig_loss_curves_stability.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    weight_names = ["pde", "tc", "bc", "data", "reg"]
    for ax, h, title in zip(axes, [h_b6, h_b10],
                            ["B6 (no warmup) — $\\lambda_\\mathrm{data}$ runaway",
                             "B10 (warmup=5000) — bounded weights"]):
        ep = h["epoch"]
        for name in weight_names:
            ax.semilogy(ep, h[f"w_{name}"], color=LOSS_COLORS[name],
                        label=fr"$\lambda_{{\mathrm{{{name}}}}}$",
                        alpha=0.85, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
    axes[0].set_ylabel("Adaptive weight $\\lambda_i$ (log)")
    axes[1].legend(loc="best", fontsize=9, ncol=2)
    fig.suptitle("Gradient-norm-balanced loss weights: failure mode vs. warmup fix",
                 y=1.02)
    fig.savefig(OUT / "fig-weight-traj-stability.png")
    plt.close(fig)


def fig_pde_dominance():
    """PDE dominance ratio over training: λ_pde·L_pde / (λ_tc·L_tc + λ_bc·L_bc).

    A ratio of 1 is balanced; ≪ 1 means PDE constraint is starved of
    influence (the failure mode in B5–B7).
    """
    h_b6  = load_history(HIST_DIR, B6_TAG)
    h_b10 = load_history(HIST_DIR, B10_TAG)
    if h_b6 is None or h_b10 is None:
        raise FileNotFoundError("history missing — see fig_loss_curves_stability.")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for h, label, color in [
        (h_b6,  "B6 (no warmup)",       "#d62728"),
        (h_b10, "B10 (warmup=5000)",    "#1f77b4"),
    ]:
        ep = h["epoch"]
        pde_w = np.array(h["w_pde"])  * np.array(h["L_pde"])
        tc_w  = np.array(h["w_tc"])   * np.array(h["L_tc"])
        bc_w  = np.array(h["w_bc"])   * np.array(h["L_bc"])
        ratio = pde_w / np.maximum(tc_w + bc_w, 1e-12)
        ax.semilogy(ep, ratio, label=label, color=color, linewidth=1.4)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="Balanced (ratio = 1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\lambda_\mathrm{pde}\,\mathcal{L}_\mathrm{pde}$ / "
                  r"$(\lambda_\mathrm{tc}\,\mathcal{L}_\mathrm{tc} "
                  r"+ \lambda_\mathrm{bc}\,\mathcal{L}_\mathrm{bc})$")
    ax.set_title("PDE dominance ratio over training")
    ax.legend(loc="best")
    fig.savefig(OUT / "fig-pde-dominance.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Tier 1 — headline visualizations (no history needed)
# ──────────────────────────────────────────────────────────────────

def _load_master_table():
    p = REPORTS / "master_table.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run `python scripts/build_master_table.py` first."
        )
    return pd.read_csv(p)


def fig_arb_rmse_pareto():
    """Scatter: total arb% (butterfly + calendar) vs pooled RMSE.

    Each B-config is one point; baselines (BS_A1, GAM_A2) plotted with
    distinct markers. Lower-left is the Pareto frontier — B10 should
    sit there.
    """
    mt = _load_master_table()
    # Only configs with arb numbers (B*); baselines have NaN arb
    pinns = mt[mt["config"].str.startswith("B")].copy()
    pinns["arb_total"] = pinns["arb_butterfly%"] + pinns["arb_calendar%"]

    bs   = mt[mt["config"] == "BS_A1"]
    gam  = mt[mt["config"] == "GAM_A2"]
    lagp = mt[mt["config"] == "LAGP_A3"]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))

    # PINN configs — labelled by short tag (B0..B12)
    short_tag = pinns["config"].str.extract(r"(B\d+)")[0]
    is_b10  = short_tag == "B10"
    is_warm = pinns["config"].str.contains("warmup")
    for sel, color, lbl in [
        (~is_warm,            "#888888", "Stage 1 (no warmup)"),
        (is_warm & ~is_b10,   "#1f77b4", "Stage 1 (warmup family)"),
        (is_b10,              "#d62728", "B10 (best Pareto)"),
    ]:
        ax.scatter(pinns.loc[sel, "arb_total"], pinns.loc[sel, "pooled_rmse"],
                   c=color, s=70, label=lbl, alpha=0.9, edgecolor="white",
                   linewidth=0.8, zorder=3)

    for _, row in pinns.iterrows():
        tag = row["config"].split("_")[0]
        ax.annotate(tag,
                    xy=(row["arb_total"], row["pooled_rmse"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color="#444")

    # Baselines — horizontal lines (no arb metric for them)
    if not bs.empty:
        y = bs["pooled_rmse"].iloc[0]
        ax.axhline(y, color="black", ls=":", lw=1)
        ax.text(ax.get_xlim()[1] * 0.98, y, f"  BS_A1 ($\\${y:.2f})",
                ha="right", va="bottom", fontsize=9, color="black")
    if not gam.empty:
        y = gam["pooled_rmse"].iloc[0]
        ax.axhline(y, color="#9467bd", ls=":", lw=1)
        ax.text(ax.get_xlim()[1] * 0.98, y, f"  GAM_A2 (\\${y:.2f})",
                ha="right", va="bottom", fontsize=9, color="#9467bd")

    ax.set_xlabel("Total arbitrage violations (butterfly + calendar) %")
    ax.set_ylabel("Pooled RMSE vs market price (\\$)")
    ax.set_title("Stage 1 ablation: pricing accuracy vs. arbitrage consistency")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(OUT / "fig-arb-rmse-pareto.png")
    plt.close(fig)


def fig_stratified_rmse():
    """Grouped bar chart of stratified RMSE (OTM / ATM / ITM) per config.

    Visualizes the wing-inversion finding: B10 is the only bar where ATM > OTM.
    """
    mt = _load_master_table()
    keep = (mt["config"].str.startswith("B")
            | mt["config"].isin(["BS_A1", "GAM_A2"]))
    sub = mt[keep].copy()
    # Drop rows missing OTM/ATM/ITM (e.g. BS_A1 has only pooled, no strata)
    sub = sub.dropna(subset=["pooled_rmse_otm", "pooled_rmse_atm", "pooled_rmse_itm"])
    sub = sub.reset_index(drop=True)
    # Tighter labels — drop the verbose suffix
    sub["short"] = sub["config"].apply(lambda s: (
        s.split("_")[0] if s.startswith("B") else s.split("_")[0]
    ))

    cats   = ["pooled_rmse_otm", "pooled_rmse_atm", "pooled_rmse_itm"]
    labels = ["OTM ($m<0.97$)", "ATM ($0.97 \\leq m \\leq 1.03$)", "ITM ($m>1.03$)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    n = len(sub)
    x = np.arange(n)
    w = 0.27

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    for i, (col, lbl, col_color) in enumerate(zip(cats, labels, colors)):
        offset = (i - 1) * w
        ax.bar(x + offset, sub[col], width=w, label=lbl, color=col_color,
               alpha=0.9, edgecolor="white", linewidth=0.6)

    # Highlight B10 with a vertical band (after reset_index, position == row idx)
    if "B10" in sub["short"].values:
        pos = int(sub.index[sub["short"] == "B10"][0])
        ax.axvspan(pos - 0.5, pos + 0.5, color="#ffe9e0",
                   alpha=0.5, zorder=0)
        ax.text(pos, sub[cats].max().max() * 1.04,
                "wing inversion", ha="center", fontsize=9,
                fontstyle="italic", color="#a04040")

    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"], rotation=0, fontsize=9)
    ax.set_ylabel("Pooled RMSE (\\$)")
    ax.set_title("Stratified RMSE by moneyness band")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(OUT / "fig-stratified-rmse.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Tier 2 — methodology aids
# ──────────────────────────────────────────────────────────────────

# Single source of truth — must match run_walk_forward.py FOLDS
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


def fig_walkforward_timeline():
    """Horizontal timeline showing train / val / test windows for each fold."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 4.5))
    data_start = pd.Timestamp("2020-01-02")
    val_months = 1

    # matplotlib date floats — NOT proleptic Gregorian ordinals. Use date2num.
    def n(ts):
        return mdates.date2num(ts.to_pydatetime())

    for i, f in enumerate(FOLDS):
        train_end  = pd.Timestamp(f["train_end"])
        val_start  = train_end - pd.DateOffset(months=val_months)
        test_start = pd.Timestamp(f["test_start"])
        test_end   = pd.Timestamp(f["test_end"])

        ax.barh(i, n(val_start)  - n(data_start), left=n(data_start),
                color="#cccccc", height=0.65, edgecolor="white", linewidth=0.5)
        ax.barh(i, n(train_end)  - n(val_start),  left=n(val_start),
                color="#ff9f3a", height=0.65, edgecolor="white", linewidth=0.5)
        ax.barh(i, n(test_end)   - n(test_start), left=n(test_start),
                color="#d62728", height=0.65, edgecolor="white", linewidth=0.5)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    ax.set_yticks(range(len(FOLDS)))
    ax.set_yticklabels([f["name"] for f in FOLDS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Calendar date")
    ax.set_title("Walk-forward fold structure")

    ax.legend(handles=[
        Patch(color="#cccccc", label="Train"),
        Patch(color="#ff9f3a", label="Validation (1 month)"),
        Patch(color="#d62728", label="Test"),
    ], loc="lower right", fontsize=9)

    ax.set_xlim(n(data_start) - 5,
                n(pd.Timestamp("2021-01-01")) + 5)
    fig.savefig(OUT / "fig-walkforward-timeline.png")
    plt.close(fig)


def fig_val_curve_mu_warmup():
    """Validation RMSE vs epoch for B10/B11/B12 on Nov2020.

    Requires history saved in those checkpoints. Existing pre-patch
    runs/walk_forward/B*/fold_Nov2020.pt files won't have history,
    so this figure skips gracefully unless the user re-runs B11/B12
    with --output_dir runs/walk_forward_history.
    """
    runs = [(B10_TAG, "B10 — μ=0.75", "#d62728"),
            (B11_TAG, "B11 — μ=0.50", "#1f77b4"),
            (B12_TAG, "B12 — μ=0.25", "#2ca02c")]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted_any = False

    for run_tag, label, color in runs:
        # Try patched dir first, then canonical dir
        for parent in (HIST_DIR, WF_DIR):
            h = load_history(parent, run_tag)
            if h is not None and "val_epoch" in h and len(h["val_epoch"]) > 0:
                ax.plot(h["val_epoch"], h["val_rmse_mkt"],
                        "o-", label=label, color=color,
                        markersize=4, alpha=0.9, linewidth=1.2)
                # Mark E* (val-best)
                ev = np.argmin(h["val_rmse_mkt"])
                ax.scatter([h["val_epoch"][ev]], [h["val_rmse_mkt"][ev]],
                           s=100, marker="*", color=color, zorder=5,
                           edgecolor="black", linewidth=0.8)
                plotted_any = True
                break

    if not plotted_any:
        raise FileNotFoundError(
            "No history with val_rmse_mkt found for B10/B11/B12 on Nov2020. "
            "Re-run those configs with the patched run_walk_forward.py "
            "(--output_dir runs/walk_forward_history)."
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE vs market (\\$)")
    ax.set_title("Validation curves under warmup, μ ablation (Nov 2020)")
    ax.legend(loc="best", fontsize=9)
    fig.savefig(OUT / "fig-val-curve-mu-warmup.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Tier 2 — architecture / method-evolution diagram
# ──────────────────────────────────────────────────────────────────

def fig_architecture():
    """Pipeline diagram: Stage 0 → 1 → 2 → 3-future.

    Pure matplotlib so the paper renders without a LaTeX install.
    A higher-fidelity TikZ version lives at paper/figures/architecture.tex
    for users who want to swap in a vector PDF.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    stages = [
        dict(x=2,  title="Stage 0", subtitle="baseline PINN",
             header=r"$\hat{v}_\phi(m,\tau)$ MLP",
             bullets=[
                 r"$\bullet$ tanh activations",
                 r"$\bullet$ Black–Scholes PDE residual",
                 r"$\bullet$ TC: $V(m,0)=(m-1)^+$",
                 r"$\bullet$ BC at $m\to 0,\,m\to\infty$",
                 r"$\bullet$ Synthetic / single-fit",
                 r"$\bullet$ Constant $\sigma$",
             ],
             color="#dbe7f7"),
        dict(x=27, title="Stage 1", subtitle="walk-forward",
             header="+ Modified MLP",
             bullets=[
                 r"$\bullet$ Modified-MLP encoders $U,V$",
                 r"$\bullet$ Random weight factorization",
                 r"$\bullet$ Fourier-feature input",
                 r"$\bullet$ Grad-norm $\lambda_i$ balancing",
                 r"$\bullet$ Market loss $\mathcal{L}_\mathrm{data}$",
                 r"$\bullet$ 9 monthly walk-forward folds",
                 r"$\bullet$ Val-best snapshot reporting",
                 r"$\bullet$ PDE-warmup schedule (B10)",
             ],
             color="#deecde"),
        dict(x=52, title="Stage 2", subtitle=r"learnable $\sigma$",
             header=r"+ Vol net $\sigma_\theta(F,\tau)$",
             bullets=[
                 r"$\bullet$ C-Vol: $\sigma_0\cdot\mathrm{softplus}(g_\theta)$",
                 r"$\bullet$ A-Vol: $\sigma_{\rm init}+\mathrm{softplus}(g_\theta)$",
                 r"$\bullet$ Pricing-net warm-start",
                 r"$\quad\,$from B10 / B12 ckpts",
                 r"$\bullet$ Joint training $(\eta_p,\eta_v)$",
                 r"$\bullet$ Same val-best protocol",
             ],
             color="#fce8d4"),
        dict(x=77, title="Stage 3", subtitle="UQ (future)",
             header="+ Anchored ensemble",
             bullets=[
                 r"$\bullet$ $K=10$ PINNs, anchored prior",
                 r"$\bullet$ Per-member val-best select",
                 r"$\bullet$ 95% predictive intervals",
                 r"$\bullet$ Coverage \& width vs.",
                 r"$\quad\,$laGP credible intervals",
                 r"$\mathit{(future\ work)}$",
             ],
             color="#eeeeee", dashed=True),
    ]

    box_w, box_h = 21, 64
    box_y_top = 80
    for s in stages:
        x, y = s["x"], box_y_top - box_h
        # Box
        ls = "--" if s.get("dashed") else "-"
        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            linestyle=ls, edgecolor="#444", facecolor=s["color"],
            linewidth=0.9, zorder=2)
        ax.add_patch(box)

        # Title above box
        ax.text(x + box_w / 2, box_y_top + 4.5, s["title"],
                ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(x + box_w / 2, box_y_top + 1.5, s["subtitle"],
                ha="center", va="bottom", fontsize=9, style="italic",
                color="#666")

        # Header inside box (bold)
        ax.text(x + 1.5, box_y_top - 4.5, s["header"],
                ha="left", va="top", fontsize=10, fontweight="bold")

        # Bullets
        for i, b in enumerate(s["bullets"]):
            ax.text(x + 1.5, box_y_top - 11 - i * 5.5, b,
                    ha="left", va="top", fontsize=8.5)

    # Arrows between stages — placed at the vertical midpoint of the boxes
    arrow_y = box_y_top - box_h / 2
    arrow_labels = ["real options\ndata", "warm-start\nof pricing $\\phi$",
                    "ensemble\n($K$ replicas)"]
    for i, label in enumerate(arrow_labels):
        x_from = stages[i]["x"] + box_w
        x_to   = stages[i + 1]["x"]
        ls_arrow = "--" if i == 2 else "-"
        ax.annotate("",
                    xy=(x_to - 0.3, arrow_y),
                    xytext=(x_from + 0.3, arrow_y),
                    arrowprops=dict(arrowstyle="-|>",
                                    color="#555", lw=0.9,
                                    linestyle=ls_arrow,
                                    mutation_scale=14))
        # Label just ABOVE the arrow, in the gap between the boxes
        ax.text((x_from + x_to) / 2, arrow_y + 3, label,
                ha="center", va="bottom", fontsize=8, style="italic",
                color="#555")

    fig.savefig(OUT / "fig-architecture.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    figures = [
        # Tier 1 — stability narrative (needs history capture)
        fig_loss_curves_stability,
        fig_weight_traj_stability,
        fig_pde_dominance,
        # Tier 1 — master-table-driven (always works once master_table.csv exists)
        fig_arb_rmse_pareto,
        fig_stratified_rmse,
        # Tier 2 — methodology aids
        fig_walkforward_timeline,
        fig_val_curve_mu_warmup,
        fig_architecture,
    ]
    n_ok = n_skip = 0
    for fn in figures:
        try:
            fn()
            print(f"  OK    {fn.__name__}")
            n_ok += 1
        except FileNotFoundError as e:
            print(f"  SKIP  {fn.__name__}: {e}")
            n_skip += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            n_skip += 1
    print(f"\n{n_ok} ok, {n_skip} skipped/errored. Output: {OUT}")
    sys.exit(0 if n_skip == 0 else 1)
