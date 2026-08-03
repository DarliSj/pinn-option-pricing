# EXPERIMENTS — methodology versioning & output layout

*Why this exists:* we are about to change core methodology (loss balancer,
selection protocol, arbitrage objective — see `docs/HYBRID_PARETO_PLAN.md`).
To compare fairly, **old results are frozen and new results go to a separate
tree**, so nothing is overwritten and v1-vs-v2 comparison is one command.

## The two methodology versions

| | **v1 (FROZEN baseline)** | **v2 (ACTIVE)** |
|---|---|---|
| Loss balancing | Wang grad-norm (unbounded `Σg/gᵢ`) | ReLoBRaLo / augmented Lagrangian (W1) |
| Selection | per-fold val-best argmin | fixed-budget final-epoch (val dropped, W3) |
| Arbitrage | measured only | trained against (constraint/penalty, W4) |
| Status | complete; matches `paper/paper.qmd` | in progress |

v2's exact recipe is decided in the workstreams (`docs/TODO.md` W1–W4). This file
only fixes **where outputs land** so the comparison stays clean regardless.

## Directory layout

```
runs/
  walk_forward/<config>/            ← v1 Stage 1 (committed)
  stage2_B10|B12/{cvol,avol}/       ← v1 Stage 2 (committed)
  v2/
    walk_forward/<config>/          ← v2 Stage 1
    stage2_<warm>/{cvol,avol}/      ← v2 Stage 2
results/
  {bs,gam,lagp}_baseline/           ← methodology-INDEPENDENT; reused by both
reports/
  master_table.csv, ...             ← live table (currently == v1)
  v1/                               ← FROZEN v1 snapshot (see reports/v1/FROZEN.md)
  v2/                               ← v2 aggregated tables
  comparison.csv                    ← v1-vs-v2 diff (generated)
```

The non-PINN baselines (BS/GAM/laGP) do not depend on the PINN training
methodology, so **v2 reuses the v1 baselines** — no need to re-run them.

## How the split is wired

No path logic changed. The run scripts already take `--output_dir`/`--runs_dir`;
the SLURM leaf scripts read a `RUN_ROOT` env var that defaults to `runs`
(reproducing v1 exactly). Route v2 by setting it:

```bash
# v2 Stage 1 (B0–B12) — same array, different tree
RUN_ROOT=runs/v2 sbatch --array=0-12 slurm/run_benchmark.sh

# v2 Stage 2 (warm-starts off v2 Stage 1 automatically)
RUN_ROOT=runs/v2 sbatch --array=0-3 slurm/run_stage2.sh
```

Locally / single config, pass the dirs directly:

```bash
python run_walk_forward.py --mode hybrid --arch modified --rwf_mu 0.75 \
    --data_loss_warmup 5000 --output_dir runs/v2/walk_forward
```

## Where each step runs (DCC vs local)

Training needs a GPU and stays on DCC; **everything analytical can run locally**
against results pulled via git (`runs/` is tracked — checkpoints are ~156 KB
each, ~62 MB total).

| Step | Where | Environment |
|---|---|---|
| Stage 1 / Stage 2 / W1 training runs | DCC (SLURM, GPU) | `conda activate pinn_env` |
| Aggregation, v1-vs-v2 comparison | either | local `.venv` or `pinn_env` |
| W0 audits (fidelity, σ_θ-vs-IV) | either (CPU is fine) | local `.venv` or `pinn_env` |
| Paper render | local | local `.venv` |

**Every command below needs an environment active** — a bare `python` has no
numpy and will fail with `ModuleNotFoundError`:

```bash
# On DCC:
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

# Locally (one-time setup — see requirements-local.txt):
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-local.txt
./.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# then use ./.venv/bin/python, or `source .venv/bin/activate`
```

## Aggregate & compare

```bash
# (v1 is already frozen in reports/v1/; regenerating reproduces it —
#  verified: max abs diff 0.0 across all shared numeric columns)
python scripts/build_master_table.py --label v1 --output_dir reports/v1

# Build the v2 table from the v2 tree, reusing the shared baselines
python scripts/build_master_table.py \
    --runs_dir runs/v2 --output_dir reports/v2 --label v2

# Side-by-side diff (RMSE / MAE / arbitrage deltas per config)
python scripts/compare_methodologies.py \
    --v1 reports/v1/master_table.csv --v2 reports/v2/master_table.csv \
    --out reports/comparison.csv
```

`--label` stamps a `methodology` column so the two tables can also be
concatenated directly. `compare_methodologies.py` prints per-config deltas with
↓better / ↑worse arrows and writes `reports/comparison.csv`.

## Rules of thumb

- **Never write v2 runs into the default `runs/walk_forward` or `runs/stage2_*`**
  — that would mix methodologies in the default master table. Always set
  `RUN_ROOT=runs/v2` (or an explicit `--output_dir`).
- **Never overwrite `reports/v1/`** — it's the frozen baseline.
- Starting a **v3**? Same pattern: `runs/v3/`, `reports/v3/`, freeze the prior.
- The paper (`paper/paper.qmd`) reads the live `reports/master_table.csv`; leave
  it pointing at whichever methodology is "current for the paper" (v1 today).
