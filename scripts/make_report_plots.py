"""
Generate report-grade figures from the master per-fold table.

Reads:
  reports/per_fold_table.csv          (built by build_master_table.py)
  reports/master_table.csv

Writes (under reports/figures/):
  pooled_rmse_comparison.png          — bar chart, pooled RMSE per config
  per_fold_rmse_lines.png             — line chart, per-fold RMSE per config
  stratified_rmse.png                 — grouped bar chart (OTM/ATM/ITM per config)
  spread_normalized.png               — bar chart, half-spread units, with y=1 line
  e_star_distribution.png             — boxplot of E* across folds, per config
  arbitrage_violations.png            — bar chart, % butterfly + calendar viols

Usage:
  python scripts/make_report_plots.py
  python scripts/make_report_plots.py --reports_dir reports
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Order configs left-to-right in plots
ORDER_HINT = [
    "BS_A1",
    "B0_standard_physics",
    "B1_standard_hybrid",
    "B2_modified_physics_mu0.5",
    "B3_modified_physics_mu0.75",
    "B4_modified_physics_mu1.0",
    "B5_modified_hybrid_mu0.5",
    "B6_modified_hybrid_mu0.75",
    "B7_modified_hybrid_mu1.0",
    "S2_cvol",
    "S2_avol",
]


def short_label(config: str) -> str:
    """Compact x-axis label."""
    if config == "BS_A1":
        return "BS"
    parts = config.split("_")
    if config.startswith("S2_"):
        return "S2 " + parts[1]
    # B0_standard_physics              → B0 std-phy
    # B1_standard_hybrid               → B1 std-hyb
    # B6_modified_hybrid_mu0.75        → B6 mod-hyb μ=0.75
    if len(parts) >= 4:
        return f"{parts[0]} {parts[1][:3]}-{parts[2][:3]} μ={parts[3][2:]}"
    if len(parts) == 3:
        return f"{parts[0]} {parts[1][:3]}-{parts[2][:3]}"
    return config


def order_configs(df: pd.DataFrame, col: str = "config") -> pd.DataFrame:
    df = df.copy()
    df["_o"] = df[col].apply(
        lambda c: ORDER_HINT.index(c) if c in ORDER_HINT else 999
    )
    return df.sort_values("_o").drop(columns=["_o"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# 1. Pooled RMSE bar chart
# ─────────────────────────────────────────────────────────────────
def plot_pooled_rmse(master: pd.DataFrame, out_path: Path):
    df = order_configs(master)
    df = df[df["pooled_rmse"].notna()]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["dimgray" if c.startswith("BS") else
              "lightcoral" if c.startswith("B0") or c.startswith("B1") else
              "steelblue" if c.startswith(("B2", "B3", "B4", "B5", "B6", "B7")) else
              "darkgreen"
              for c in df["config"]]
    bars = ax.bar(range(len(df)), df["pooled_rmse"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, df["pooled_rmse"]):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"${v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([short_label(c) for c in df["config"]], rotation=35, ha="right")
    ax.set_ylabel("Pooled RMSE vs market mid ($)")
    ax.set_title("Pooled walk-forward RMSE across configs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────
# 2. Per-fold RMSE line chart
# ─────────────────────────────────────────────────────────────────
def plot_per_fold_lines(per_fold: pd.DataFrame, out_path: Path):
    pf = per_fold.copy()
    pf = pf[pf["rmse_mkt"].notna()]
    fold_order = sorted(pf["fold"].unique(),
                        key=lambda x: ["Apr", "May", "Jun", "Jul", "Aug",
                                       "Sep", "Oct", "Nov", "Dec"]
                        .index(x[:3]) if x[:3] in
                        ["Apr", "May", "Jun", "Jul", "Aug",
                         "Sep", "Oct", "Nov", "Dec"] else 99)

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot = pf.pivot_table(index="fold", columns="config",
                           values="rmse_mkt", aggfunc="first")
    pivot = pivot.reindex(fold_order)

    cols_in_order = [c for c in ORDER_HINT if c in pivot.columns]
    cmap = plt.cm.tab10
    for i, c in enumerate(cols_in_order):
        marker = "o" if c.startswith("S2") else "s" if c.startswith("BS") else "^"
        ls = "-" if c.startswith("S2") else "--" if c.startswith("BS") else "-"
        ax.plot(pivot.index, pivot[c],
                label=short_label(c), marker=marker, linestyle=ls,
                color=cmap(i % 10), linewidth=1.6, alpha=0.85)

    ax.set_xlabel("Test fold")
    ax.set_ylabel("Test RMSE vs market mid ($)")
    ax.set_title("Per-fold RMSE — all configs")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────
# 3. Stratified RMSE (OTM / ATM / ITM)
# ─────────────────────────────────────────────────────────────────
def plot_stratified(master: pd.DataFrame, out_path: Path):
    df = order_configs(master)
    df = df[df["pooled_rmse_otm"].notna()]
    if df.empty:
        print(f"Skipping {out_path.name}: no stratified data")
        return

    x = np.arange(len(df))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, df["pooled_rmse_otm"], width, label="OTM (m<0.97)", color="tab:blue")
    ax.bar(x,         df["pooled_rmse_atm"], width, label="ATM (0.97≤m≤1.03)", color="tab:orange")
    ax.bar(x + width, df["pooled_rmse_itm"], width, label="ITM (m>1.03)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(c) for c in df["config"]], rotation=35, ha="right")
    ax.set_ylabel("Pooled RMSE vs market mid ($)")
    ax.set_title("Stratified RMSE by moneyness")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────
# 4. Spread-normalized RMSE
# ─────────────────────────────────────────────────────────────────
def plot_spread_normalized(master: pd.DataFrame, out_path: Path):
    df = order_configs(master)
    df = df[df["pooled_rmse_spread"].notna()]
    if df.empty:
        print(f"Skipping {out_path.name}: no spread data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(df)), df["pooled_rmse_spread"],
                  color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="red", linestyle="--", lw=1.5,
               label="Inside the bid-ask spread (= 1 half-spread)")
    for bar, v in zip(bars, df["pooled_rmse_spread"]):
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([short_label(c) for c in df["config"]], rotation=35, ha="right")
    ax.set_ylabel("Pooled RMSE in half-spread units")
    ax.set_title("Spread-normalized RMSE — values > 1 mean errors exceed half-spread")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────
# 5. E* distribution (box plot per config across folds)
# ─────────────────────────────────────────────────────────────────
def plot_e_star_distribution(per_fold: pd.DataFrame, out_path: Path):
    pf = per_fold.copy()
    pf = pf[pf.get("best_val_epoch").notna()] if "best_val_epoch" in pf.columns else pf.iloc[:0]
    if pf.empty:
        print(f"Skipping {out_path.name}: no best_val_epoch column")
        return

    cols = [c for c in ORDER_HINT if c in pf["config"].unique() and not c.startswith("BS")]
    data = [pf[pf["config"] == c]["best_val_epoch"].dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, labels=[short_label(c) for c in cols],
               showmeans=True, meanline=True)
    ax.set_ylabel("E* (val-best epoch)")
    ax.set_title("Distribution of E* across folds — diagnostic for training-time variance")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────
# 6. Arbitrage violation rates
# ─────────────────────────────────────────────────────────────────
def plot_arbitrage(master: pd.DataFrame, out_path: Path):
    df = order_configs(master)
    df = df[df["arb_butterfly%"].notna()]
    if df.empty:
        print(f"Skipping {out_path.name}: no arbitrage data")
        return

    x = np.arange(len(df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, df["arb_butterfly%"], width,
           label="Butterfly (∂²v̂/∂m² < 0)", color="tab:red")
    ax.bar(x + width/2, df["arb_calendar%"], width,
           label="Calendar (∂v̂/∂τ < 0)", color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(c) for c in df["config"]], rotation=35, ha="right")
    ax.set_ylabel("% of grid points violating no-arb")
    ax.set_title("No-arbitrage violations on the (m, τ) grid")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate report figures from master tables")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    rep = Path(args.reports_dir)
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    master_csv = rep / "master_table.csv"
    pf_csv     = rep / "per_fold_table.csv"
    if not master_csv.exists() or not pf_csv.exists():
        print(f"Missing {master_csv} or {pf_csv}. Run build_master_table.py first.")
        return

    master = pd.read_csv(master_csv)
    per_fold = pd.read_csv(pf_csv)

    plot_pooled_rmse(master,    figs / "pooled_rmse_comparison.png")
    plot_per_fold_lines(per_fold, figs / "per_fold_rmse_lines.png")
    plot_stratified(master,     figs / "stratified_rmse.png")
    plot_spread_normalized(master, figs / "spread_normalized.png")
    plot_e_star_distribution(per_fold, figs / "e_star_distribution.png")
    plot_arbitrage(master,      figs / "arbitrage_violations.png")


if __name__ == "__main__":
    main()
