# Walk-Forward Physics-Informed Neural Networks for Option Pricing

This repository contains the code and data pipeline for a study that
benchmarks physics-informed neural networks (PINNs) against classical
methods for European option pricing under a strict walk-forward
protocol. The full methodology, results, and discussion are described
in the paper (`paper/paper.qmd`). This README documents the codebase
itself: what each stage does, how to reproduce the numbers, and where
the artefacts land. For developer-level detail, see `README_dev.md`.

> **Research status:** the benchmark snapshot below is complete and matches the
> paper. Active methodology work (loss-balancer replacement, selection-protocol
> revision, arbitrage-as-objective) is tracked in `docs/TODO.md` and
> `docs/HYBRID_PARETO_PLAN.md`; those results are not yet reflected here.

## Overview

The study evaluates three model classes on TSLA options data
(2020, end-of-day, split-adjusted) across nine monthly walk-forward
folds:

| Stage | What it learns | Volatility | Loss |
|-------|----------------|------------|------|
| **Stage 0** | Price surface $\hat v(m,\tau)$ on a single train/test split | Constant $\sigma_\mathrm{fixed}$ | Physics, optionally + market data |
| **Stage 1** | Same price surface, walk-forward across nine monthly folds | Constant $\sigma_\mathrm{fixed}$ | 13-configuration ablation grid (B0–B12) |
| **Stage 2** | Joint price surface and learnable volatility $\sigma_\theta(F,\tau)$ | Learned, with multiplicative (C-Vol) or additive (A-Vol) parametrization | Physics + market + regularization |

Three non-PINN baselines are evaluated under identical walk-forward
folds and the same information set: per-fold-$\sigma$ Black–Scholes,
a residual GAM (`mgcv`), and a residual local approximate Gaussian
process (`laGP`). Every reported model is additionally evaluated for
butterfly and calendar arbitrage violations on a dense $(F,\tau)$ grid;
violation rates are treated as primary metrics on equal footing with
RMSE.

## Headline results

The reproduced figures and tables live under `reports/` and `paper/figures/`
after running the build pipeline below. Numbers reported in the paper:

- **Best pricing model:** modified-MLP hybrid with $\mu = 0.75$ and
  a 5,000-epoch PDE-warmup schedule (B10), pooled RMSE $8.49 across
  the nine 2020 folds, below the per-fold-$\sigma$ Black–Scholes
  baseline of $8.79.
- **Cleanest Pareto-frontier model:** Stage 2 / A-Vol warm-started
  from B12, pooled RMSE $9.41 with 8.5% combined butterfly + calendar
  arbitrage violations.
- **Pareto trade-off:** RMSE and arbitrage consistency are competing
  objectives; the paper reports the full benchmark as a multi-objective
  frontier rather than a single-metric ranking.

## Repository layout

```
.
├── data/                     # Source dataset and per-fold splits
├── src/                      # Core library (data, model, losses, training, baselines)
├── run_stage0.py             # Single-split development run
├── run_walk_forward.py       # Stage 1 walk-forward benchmark
├── run_bs_baseline.py        # Per-fold Black–Scholes baseline
├── run_stage2.py             # Stage 2 learnable-volatility walk-forward
├── scripts/                  # Aggregation, R baselines, figure generation
├── slurm/                    # SLURM scripts for cluster execution
├── runs/                     # Training output (checkpoints, logs, results.json)
├── results/                  # Non-PINN baseline output
├── reports/                  # Aggregated tables and comparison figures
└── paper/                    # Quarto source for the manuscript
```

## Methodology in brief

**Walk-forward protocol.** Nine monthly test folds spanning April
through December 2020. For each test month $M_f$, training uses every
observation strictly before $M_{f-1}$, the calendar month $M_{f-1}$
serves as the held-out validation set, and the model is evaluated on
$M_f$. Per-fold constants $\sigma_\mathrm{fixed}$ (median of ATM
implied volatilities) and $r_\mathrm{fixed}$ are computed from
training data only.

**Val-best snapshot reporting.** Each fold trains for a fixed budget
of epochs; whenever the validation RMSE improves, a snapshot of the
full model state is taken; after training the val-best snapshot is
restored and the test month is evaluated exactly once. Test data are
never seen during training or hyperparameter selection.

**Loss balancing and PDE-warmup.** The composite loss
(PDE residual + terminal/boundary conditions + market data + regularization)
is balanced via the gradient-norm scheme of Wang et al. (2023). For
hybrid configurations, a 5,000-epoch PDE-warmup phase deactivates the
data loss until the PDE/TC/BC weights have settled; this prevents the
gradient-norm scheme from amplifying $\lambda_\mathrm{data}$ to
runaway values that suppress the PDE-consistency signal.

**Architecture.** The pricing network is a Modified-MLP with Fourier
feature embedding and Random Weight Factorization on every linear
layer. Activations are $\tanh$ throughout to permit second-order
automatic differentiation for the PDE residual. The Stage 2
volatility network is a smaller RWF-MLP that produces $\sigma$ on
the same input domain.

See `paper/paper.qmd` for the full specification, including the
ablation grid (B0–B12) and the Stage 2 $2 \times 2$ design.

## Reproducing the results

### Environment

```bash
conda activate pinn_env       # PyTorch, NumPy, pandas, SciPy, matplotlib
pip install tabulate          # required by paper/paper.qmd table chunks
# R baselines additionally need: mgcv, laGP, jsonlite, dplyr, readr
```

### Non-PINN baselines

Three reference baselines run on the same walk-forward folds as the
PINN configurations, with per-fold $\sigma_\mathrm{fixed} = $ median
ATM IV on training data:

```bash
# A1 — closed-form Black–Scholes (Python)
bash scripts/run_local_baselines.sh

# Per-fold CSV dump for the R baselines
python scripts/dump_folds.py

# A2 — residual GAM (R, mgcv)
Rscript scripts/run_gam_baseline.R

# A3 — residual laGP (R, laGP)
Rscript scripts/run_lagp_baseline.R
```

Output lands under `results/{bs,gam,lagp}_baseline/results.json`.

### Stage 1 walk-forward

A single configuration locally (the paper's headline B10 model):

```bash
python run_walk_forward.py \
    --mode hybrid --arch modified --rwf_mu 0.75 \
    --data_loss_warmup 5000 --epochs 15000
```

The full 13-configuration ablation grid is straightforward to launch
on a SLURM cluster via `slurm/submit_all.sh`; see that script for the
parameter mapping (B0 through B12).

### Stage 2 learnable volatility

Requires a Stage 1 checkpoint as warm-start. The default is the B10
configuration:

```bash
python run_stage2.py --vol_type cvol \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75_warmup5000 \
    --pricing_lr 1e-4 --vol_lr 1e-3 \
    --epochs 10000 --rwf_mu 0.75
```

Pass `--vol_type avol` for the additive parametrization; `--from_scratch`
skips the warm-start as an ablation.

### Aggregating into the master table

After all configurations have run:

```bash
python scripts/build_master_table.py    # → reports/master_table.csv
bash paper/sync_artifacts.sh             # regenerates paper figures
quarto render paper/paper.qmd --to pdf   # builds the manuscript
```

The master table includes one row per benchmark cell (BS_A1, GAM_A2,
LAGP_A3, B0–B12, and the four Stage 2 configurations), with pooled
RMSE, stratified RMSE by moneyness band, butterfly and calendar
arbitrage violation rates, and val-best epoch statistics.

## Citation

If you use this code or the methodology, please cite the accompanying
paper. See `paper/references.bib` for the bibliographic entries used
in the manuscript.

## License

Research code released alongside the paper. See `LICENSE` if present;
otherwise the code is provided as-is for reproducibility.
