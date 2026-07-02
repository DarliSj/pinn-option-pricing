"""
Compare two methodology snapshots (e.g. v1 vs v2) fold-for-fold at the
aggregate level. Reads two master_table.csv files, aligns on `config`, and
emits a side-by-side diff of the headline metrics.

The whole point of the versioned output split (see EXPERIMENTS.md): after we
change the loss balancer / selection protocol / arbitrage handling, we can
answer "did it actually help, and where?" against the frozen baseline.

Usage:
    # Defaults compare the frozen v1 snapshot against the current v2 build:
    python scripts/compare_methodologies.py

    # Or specify explicitly:
    python scripts/compare_methodologies.py \
        --v1 reports/v1/master_table.csv --v2 reports/v2/master_table.csv \
        --out reports/comparison_v1_v2.csv
"""

import argparse
from pathlib import Path

import pandas as pd

# Metrics where lower is better (for the arrow annotation).
LOWER_IS_BETTER = {
    "pooled_rmse", "mean_fold_rmse", "pooled_mae",
    "arb_butterfly%", "arb_calendar%",
}
# Headline metrics to diff, in display order.
METRICS = [
    "mean_fold_rmse", "pooled_rmse", "pooled_mae",
    "arb_butterfly%", "arb_calendar%",
]


def main():
    ap = argparse.ArgumentParser(description="Compare two methodology master tables")
    ap.add_argument("--v1", default="reports/v1/master_table.csv",
                    help="Baseline master table (frozen).")
    ap.add_argument("--v2", default="reports/v2/master_table.csv",
                    help="New-methodology master table.")
    ap.add_argument("--v1_label", default="v1")
    ap.add_argument("--v2_label", default="v2")
    ap.add_argument("--out", default="reports/comparison.csv")
    ap.add_argument("--metrics", nargs="*", default=METRICS,
                    help="Metric columns to diff.")
    args = ap.parse_args()

    v1_path, v2_path = Path(args.v1), Path(args.v2)
    missing = [p for p in (v1_path, v2_path) if not p.exists()]
    if missing:
        print("Missing master table(s):")
        for p in missing:
            print(f"  {p}")
        print("\nBuild them first, e.g.:")
        print("  python scripts/build_master_table.py --label v1   # baseline (or use reports/v1/)")
        print("  python scripts/build_master_table.py --runs_dir runs/v2 "
              "--output_dir reports/v2 --label v2")
        return

    a = pd.read_csv(v1_path).set_index("config")
    b = pd.read_csv(v2_path).set_index("config")

    metrics = [m for m in args.metrics if m in a.columns and m in b.columns]
    if not metrics:
        print("None of the requested metrics are present in both tables.")
        print(f"  v1 columns: {list(a.columns)}")
        return

    configs = [c for c in a.index if c in b.index]
    only_v1 = [c for c in a.index if c not in b.index]
    only_v2 = [c for c in b.index if c not in a.index]

    rows = []
    for c in configs:
        row = {"config": c}
        for m in metrics:
            va, vb = a.loc[c, m], b.loc[c, m]
            row[f"{m}__{args.v1_label}"] = va
            row[f"{m}__{args.v2_label}"] = vb
            try:
                row[f"{m}__delta"] = float(vb) - float(va)
            except (TypeError, ValueError):
                row[f"{m}__delta"] = None
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # ── Readable summary to stdout ──────────────────────────────
    print("=" * 78)
    print(f"METHODOLOGY COMPARISON   {args.v1_label} → {args.v2_label}")
    print(f"  {args.v1_label}: {v1_path}")
    print(f"  {args.v2_label}: {v2_path}")
    print("=" * 78)
    print(f"{len(configs)} shared configs. "
          f"(↓ = {args.v2_label} better; ↑ = worse; for RMSE/MAE/arb lower is better)\n")
    for m in metrics:
        better = m in LOWER_IS_BETTER
        print(f"— {m} —")
        for c in configs:
            va, vb = a.loc[c, m], b.loc[c, m]
            try:
                d = float(vb) - float(va)
                if abs(d) < 1e-9:
                    arrow = "="
                elif (d < 0) == better:
                    arrow = "↓ better"
                else:
                    arrow = "↑ worse"
                print(f"   {c:<26} {float(va):>9.3f} → {float(vb):>9.3f}  "
                      f"({d:+.3f})  {arrow}")
            except (TypeError, ValueError):
                print(f"   {c:<26} {str(va):>9} → {str(vb):>9}  (n/a)")
        print()

    if only_v1:
        print(f"Only in {args.v1_label}: {', '.join(only_v1)}")
    if only_v2:
        print(f"Only in {args.v2_label}: {', '.join(only_v2)}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
