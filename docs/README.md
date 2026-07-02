# docs/ — Documentation Index

*Updated 2026-06-03. Read top-to-bottom for orientation; statuses below tell you
what is authoritative vs. kept for provenance.*

## Current / authoritative

| Doc | What it is |
|---|---|
| [`TODO.md`](TODO.md) | **Live status + workstreams (W0–W7).** Where the project is, key findings, actionable checklist. Start here. |
| [`HYBRID_PARETO_PLAN.md`](HYBRID_PARETO_PLAN.md) | **Plan of record** for closing the RMSE-vs-arbitrage Pareto gap. §0 = locked decisions, §2 = problem families A–E, §4 = sequencing (balancer first). |
| [`TRAINING_VALIDATION_DISCUSSION.md`](TRAINING_VALIDATION_DISCUSSION.md) | **Advisor-facing analysis** of the training/validation defects (P1–P5): grad-norm drift, selection regret, arbitrage-as-untrained-objective, Stage 2 interactions. Appendix A = augmented-Lagrangian primer; Appendix B = loss-balancing literature survey. |

## Historical (kept for provenance; do not use for current state)

| Doc | Status |
|---|---|
| [`CLAUDE_CODE_BRIEFING.md`](CLAUDE_CODE_BRIEFING.md) | Apr 2026 migration briefing. ⚠ Its B1–B5 config numbering does **not** match the final B0–B12 grid; its "report final-epoch" rule was superseded twice. |
| [`BENCHMARKING_PLAN.md`](BENCHMARKING_PLAN.md) | Original Stage 1 benchmark design. Executed and extended (B0–B12 + Stage 2 2×2). |
| [`architecture_training_dynamics_summary.md`](architecture_training_dynamics_summary.md) | Stage 0 architecture analysis. Drift diagnosis still valid; μ-selection conclusion superseded (μ=1.0 is the worst arbitrage setting). |

## Output versioning

- [`../EXPERIMENTS.md`](../EXPERIMENTS.md) — **how old vs new runs are kept
  separate.** v1 (frozen baseline) lives in `runs/` + `reports/v1/`; new
  methodology runs go to `runs/v2/` + `reports/v2/`; `scripts/compare_methodologies.py`
  diffs them. Read this before launching any run under the new methodology.

## Other

- `slides/` — presentation material.

## Where results live

- `reports/master_table.csv` — one row per benchmark cell (BS/GAM/laGP, B0–B12,
  Stage 2 ×4); built by `scripts/build_master_table.py`.
- `reports/per_fold_table.csv` — per-fold breakdown.
- `paper/paper.qmd` — the manuscript (Quarto; tables read the master table
  directly, so numbers never go stale).
