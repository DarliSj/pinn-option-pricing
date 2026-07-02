# reports/v1/ — FROZEN baseline snapshot

**Do not overwrite.** This is the aggregated result of the **v1 methodology**,
snapshotted 2026-06-03, kept as the comparison baseline for all future work.

## What "v1" means

The methodology used for every run under `runs/walk_forward/`, `runs/stage2_B10/`,
`runs/stage2_B12/`:

- **Loss balancing:** Wang gradient-norm scheme (`λ_i = Σg/g_i`, EMA α=0.9,
  TC/BC floor 10) — *unbounded*; the λ_data-runaway defect lives here.
- **Hybrid stabilization:** 5,000-epoch PDE-warmup for the B10–B12 configs.
- **Model selection:** per-fold **val-best snapshot** (argmin validation RMSE on
  the month before test), single test evaluation.
- **Arbitrage:** measured only, never trained against.
- **Reporting:** pooled + mean-fold RMSE.

## Files

`master_table.csv`, `per_fold_table.csv`, `master_table_short.md`,
`results_summary.md` — identical to the top-level `reports/*` at snapshot time.

## Regenerating v1 (should reproduce these exactly)

```bash
python scripts/build_master_table.py --label v1 --output_dir reports/v1
```

The raw v1 runs are committed under `runs/` (git history), so this is fully
reproducible. See `EXPERIMENTS.md` for the v1-vs-v2 comparison workflow.
