"""
Aggregate all results.json from the runs/ tree into one master benchmark table.

Walks:
  results/bs_baseline/results.json                                 (A1)
  runs/walk_forward/<arch>_<mode>[_mu<X>]/results.json            (B0-B7)
  runs/stage2/<vol_type>/results.json                             (C-Vol, A-Vol)
  runs/stage2/<vol_type>_scratch/results.json                     (ablation)

Outputs:
  reports/master_table.csv          — full schema, one row per config
  reports/master_table_short.md     — Markdown table (paper-ready, key columns)
  reports/per_fold_table.csv        — one row per (config, fold)

Usage:
  python scripts/build_master_table.py
  python scripts/build_master_table.py --runs_dir runs --bs_dir results/bs_baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Display order for the master table ──────────────────────────
CONFIG_ORDER = [
    "BS_A1",
    "B0_standard_physics",
    "B1_standard_hybrid",
    "B2_modified_physics_mu0.5",
    "B3_modified_physics_mu0.75",
    "B4_modified_physics_mu1.0",
    "B5_modified_hybrid_mu0.5",
    "B6_modified_hybrid_mu0.75",
    "B7_modified_hybrid_mu1.0",
    # Loss-balancing ablations
    "B8_modified_hybrid_mu0.0",
    "B9_modified_hybrid_mu0.75_fixdata1000.0",
    "B10_modified_hybrid_mu0.75_warmup5000",
    "B11_modified_hybrid_mu0.5_warmup5000",
    "B12_modified_hybrid_mu0.25_warmup5000",
    "S2_cvol",
    "S2_avol",
    "S2_cvol_scratch",
    "S2_avol_scratch",
]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_pinn_runs(runs_dir: Path):
    """Find every results.json under runs/walk_forward and runs/stage2."""
    found = {}

    # Stage 1: walk_forward/{arch}_{mode}[_mu{X}]/
    wf = runs_dir / "walk_forward"
    if wf.exists():
        for sub in sorted(wf.iterdir()):
            if not sub.is_dir():
                continue
            r = load_json(sub / "results.json")
            if r is None:
                continue
            name = sub.name
            tag = _tag_from_walk_forward_dir(name)
            found[tag] = r

    # Stage 2: stage2/<vol_type>[_scratch]/
    s2 = runs_dir / "stage2"
    if s2.exists():
        for sub in sorted(s2.iterdir()):
            if not sub.is_dir():
                continue
            r = load_json(sub / "results.json")
            if r is None:
                continue
            tag = "S2_" + sub.name
            found[tag] = r

    return found


def _tag_from_walk_forward_dir(dirname: str) -> str:
    """
    Map walk-forward dir name to display tag.
        standard_physics                → B0_standard_physics
        standard_hybrid                 → B1_standard_hybrid
        modified_physics_mu0.5          → B2_modified_physics_mu0.5
        modified_physics_mu0.75         → B3_modified_physics_mu0.75
        modified_physics_mu1.0          → B4_modified_physics_mu1.0
        modified_hybrid_mu0.5           → B5_modified_hybrid_mu0.5
        modified_hybrid_mu0.75          → B6_modified_hybrid_mu0.75
        modified_hybrid_mu1.0           → B7_modified_hybrid_mu1.0
    """
    mapping = {
        "standard_physics":                            "B0",
        "standard_hybrid":                             "B1",
        "modified_physics_mu0.5":                      "B2",
        "modified_physics_mu0.75":                     "B3",
        "modified_physics_mu1.0":                      "B4",
        "modified_hybrid_mu0.5":                       "B5",
        "modified_hybrid_mu0.75":                      "B6",
        "modified_hybrid_mu1.0":                       "B7",
        # Loss-balancing ablations
        "modified_hybrid_mu0.0":                       "B8",
        "modified_hybrid_mu0.75_fixdata1000.0":        "B9",
        "modified_hybrid_mu0.75_warmup5000":           "B10",
        "modified_hybrid_mu0.5_warmup5000":            "B11",
        "modified_hybrid_mu0.25_warmup5000":           "B12",
    }
    prefix = mapping.get(dirname, "B?")
    return f"{prefix}_{dirname}"


def summarize_config(tag: str, results: dict, is_bs: bool = False) -> dict:
    """Extract the row of summary metrics for one config."""
    s = results.get("summary", {})
    folds = results.get("folds", [])

    row = {"config": tag}

    if is_bs:
        # BS baseline schema is leaner — no e_star, no arb metrics
        row.update({
            "arch":             "BS",
            "mode":             "—",
            "mu":               np.nan,
            "vol_type":         "—",
            "pooled_rmse":      s.get("pooled_rmse_mkt"),
            "pooled_mae":       s.get("pooled_mae_mkt"),
            "pooled_rmse_spread": np.nan,
            "pooled_rmse_otm":    np.nan,
            "pooled_rmse_atm":    np.nan,
            "pooled_rmse_itm":    np.nan,
            "mean_fold_rmse":   s.get("mean_rmse_mkt"),
            "std_fold_rmse":    s.get("std_rmse_mkt"),
            "worst_rmse":       np.nan,
            "worst_fold":       "—",
            "e_star_mean":      np.nan,
            "e_star_std":       np.nan,
            "arb_butterfly%":   np.nan,
            "arb_calendar%":    np.nan,
            "n_folds":          len(folds),
            "n_test_total":     s.get("total_n_test"),
        })
        return row

    # PINN configs (Stage 1 / Stage 2): infer arch/mode/μ/vol_type from tag
    arch = mode = vol_type = "—"
    mu = np.nan
    if tag.startswith("S2_"):
        vol_type = tag.split("_")[1]
        arch = "modified"
        mode = "hybrid"
        # μ comes from the run_info's rwf_mu of fold 0 (consistent across folds)
        if folds:
            mu = folds[0].get("sigma_fixed", np.nan)  # placeholder; better: read from run_info
        mu_field = s.get("rwf_mu")
        if mu_field is not None:
            mu = mu_field
    else:
        # B0_standard_physics → standard, physics
        # B6_modified_hybrid_mu0.75 → modified, hybrid, μ=0.75
        parts = tag.split("_")
        # parts: [B6, modified, hybrid, mu0.75]
        if len(parts) >= 3:
            arch = parts[1]
            mode = parts[2]
        for p in parts:
            if p.startswith("mu"):
                try:
                    mu = float(p[2:])
                except ValueError:
                    pass

    row.update({
        "arch":             arch,
        "mode":             mode,
        "mu":               mu,
        "vol_type":         vol_type if tag.startswith("S2") else "—",
        "pooled_rmse":      s.get("pooled_rmse_mkt"),
        "pooled_mae":       s.get("pooled_mae_mkt"),
        "pooled_rmse_spread": s.get("pooled_rmse_spread"),
        "pooled_rmse_otm":    s.get("pooled_rmse_otm"),
        "pooled_rmse_atm":    s.get("pooled_rmse_atm"),
        "pooled_rmse_itm":    s.get("pooled_rmse_itm"),
        "mean_fold_rmse":   s.get("mean_rmse_mkt"),
        "std_fold_rmse":    s.get("std_rmse_mkt"),
        "worst_rmse":       s.get("worst_rmse_mkt"),
        "worst_fold":       s.get("worst_fold"),
        "e_star_mean":      s.get("e_star_mean"),
        "e_star_std":       s.get("e_star_std"),
        "arb_butterfly%":   _pct(s.get("mean_arb_butterfly_ratio")),
        "arb_calendar%":    _pct(s.get("mean_arb_calendar_ratio")),
        "n_folds":          len(folds),
        "n_test_total":     s.get("total_n_test"),
    })
    return row


def _pct(x):
    return None if x is None else round(100.0 * x, 2)


def build_per_fold_rows(tag: str, results: dict, is_bs: bool = False):
    """One row per (config, fold)."""
    rows = []
    for f in results.get("folds", []):
        row = {"config": tag, "fold": f.get("fold")}
        if is_bs:
            row.update({
                "rmse_mkt": f.get("rmse_mkt"),
                "mae_mkt":  f.get("mae_mkt"),
                "n_test":   f.get("n_test"),
                "sigma_fixed": f.get("sigma_fixed"),
            })
        else:
            row.update({
                "rmse_mkt":     f.get("rmse_mkt"),
                "rmse_spread":  f.get("rmse_spread"),
                "rmse_otm":     f.get("rmse_otm"),
                "rmse_atm":     f.get("rmse_atm"),
                "rmse_itm":     f.get("rmse_itm"),
                "n_test":       f.get("n_test"),
                "sigma_fixed":  f.get("sigma_fixed"),
                "best_val_epoch": f.get("best_val_epoch"),
                "arb_butterfly%": _pct(f.get("arb_butterfly_ratio")),
                "arb_calendar%":  _pct(f.get("arb_calendar_ratio")),
                "elapsed_s":    f.get("elapsed"),
            })
        rows.append(row)
    return rows


def to_markdown_short(df: pd.DataFrame) -> str:
    """Slim Markdown table for paper / report inclusion.

    Uses a hand-rolled GFM writer (no `tabulate` dependency).
    """
    cols = ["config", "pooled_rmse", "mean_fold_rmse", "std_fold_rmse",
            "pooled_rmse_otm", "pooled_rmse_atm", "pooled_rmse_itm",
            "pooled_rmse_spread", "arb_butterfly%", "arb_calendar%",
            "e_star_mean"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()

    def _fmt(v):
        if v is None or (isinstance(v, float) and (pd.isna(v))):
            return "—"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |"
            for _, r in sub.iterrows()]
    return "\n".join([header, sep] + rows)


def main():
    ap = argparse.ArgumentParser(description="Aggregate benchmark results into a master table")
    ap.add_argument("--runs_dir", default="runs", help="Root of runs/ tree")
    ap.add_argument("--bs_dir", default="results/bs_baseline",
                    help="BS baseline output directory")
    ap.add_argument("--output_dir", default="reports",
                    help="Where to write master_table.csv / .md / per_fold_table.csv")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    bs_dir   = Path(args.bs_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect ─────────────────────────────────────────────────
    found = {}
    bs_results = load_json(bs_dir / "results.json")
    if bs_results is not None:
        found["BS_A1"] = ("bs", bs_results)
    for tag, r in collect_pinn_runs(runs_dir).items():
        found[tag] = ("pinn", r)

    if not found:
        print("No results.json found anywhere. Run the benchmarks first.")
        return

    # ── Build master table ──────────────────────────────────────
    rows = []
    per_fold_rows = []
    for tag, (kind, r) in found.items():
        is_bs = (kind == "bs")
        rows.append(summarize_config(tag, r, is_bs=is_bs))
        per_fold_rows.extend(build_per_fold_rows(tag, r, is_bs=is_bs))

    df = pd.DataFrame(rows)

    # Ordered display
    df["_order"] = df["config"].apply(
        lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 999
    )
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    # ── Write outputs ───────────────────────────────────────────
    csv_path = out_dir / "master_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} configs)")

    md_path = out_dir / "master_table_short.md"
    md_path.write_text(to_markdown_short(df) + "\n")
    print(f"Wrote {md_path}")

    pf = pd.DataFrame(per_fold_rows)
    pf_path = out_dir / "per_fold_table.csv"
    pf.to_csv(pf_path, index=False)
    print(f"Wrote {pf_path}  ({len(pf)} fold rows)")

    # ── Pretty print to stdout ──────────────────────────────────
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK TABLE")
    print("=" * 80)
    cols_to_show = ["config", "pooled_rmse", "mean_fold_rmse", "std_fold_rmse",
                    "pooled_rmse_spread", "arb_butterfly%", "e_star_mean"]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 30,
                           "display.float_format", "{:.3f}".format):
        print(df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
