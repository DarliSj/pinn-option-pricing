# PINN Option Pricing

Physics-Informed Neural Networks for European call option pricing on TSLA 2020 data. Duke independent study (Spring 2026).

The system evolves through three stages, each adding capability while reusing the same `src/` modules via backward-compatible kwargs:

| Stage | What it learns | Volatility | Data used |
|----|----|----|----|
| **0** | Price surface v(m, tau) | Constant sigma_fixed | Physics only or physics + market |
| **1** | Same, walk-forward validated | Constant sigma_fixed | Same, across 9 temporal folds |
| **2** | Price + volatility surface | Learned sigma(m, tau) | Physics + market + regularization |

**Where to look for results:** `reports/master_table.csv` (one row per config), `reports/master_table_short.md` (paper-ready table), `reports/figures/*.png` (comparison plots), `reports/results_summary.md` (narrative summary + findings + recommendations).

------------------------------------------------------------------------

## Project Structure

```
.
├── data/
│   └── TSLA_2020_Split_Adjusted.csv       # Source data (split-adjusted)
│
├── src/                                     # Shared module library
│   ├── bs_formulas.py                       # Analytical Black-Scholes (dollar + normalized)
│   ├── data.py                              # Loading, filtering, fold construction, arrays
│   ├── model.py                             # All neural network architectures
│   ├── losses.py                            # PDE residual, loss terms, grad-norm balancing
│   ├── training.py                          # Training loop, val-best snapshot, single test eval
│   ├── baselines.py                         # BS constant-sigma baseline evaluator
│   └── diagnostics.py                       # Training plots, solution / vol surfaces, arbitrage check
│
├── run_stage0.py                            # Single train/test split (development)
├── run_walk_forward.py                      # Walk-forward backtest (Stage 1 benchmarking)
├── run_bs_baseline.py                       # BS baseline across all folds
├── run_stage2.py                            # Learnable vol surface walk-forward
│
├── scripts/                                 # Report-time aggregation + plotting
│   ├── run_local_baselines.sh               # Run BS baseline locally (no GPU)
│   ├── build_master_table.py                # Aggregate all results.json → master_table.csv
│   └── make_report_plots.py                 # Generate report figures from master tables
│
├── slurm/                                   # SLURM scripts for Duke DCC
│   ├── run_smoke_test.sh                    # 1-fold smoke test
│   ├── run_benchmark.sh                     # B0–B10 array (11 configs)
│   ├── run_stage2_smoke.sh                  # Stage 2 smoke test
│   ├── run_stage2.sh                        # Stage 2 array (cvol + avol)
│   ├── submit_all.sh                        # Submits smoke → B0–B10 with dependency
│   └── submit_stage2.sh                     # Submits Stage 2 smoke → cvol + avol
│
├── runs/                                    # All training output (checkpoints, logs, plots, results.json)
│   ├── walk_forward/
│   │   ├── standard_physics/                # B0
│   │   ├── standard_hybrid/                 # B1
│   │   ├── modified_physics_mu{0.5,0.75,1.0}/        # B2-B4
│   │   ├── modified_hybrid_mu{0.5,0.75,1.0}/         # B5-B7
│   │   ├── modified_hybrid_mu0.0/                    # B8 — RWF init fix
│   │   ├── modified_hybrid_mu0.75_fixdata1000.0/     # B9 — fixed λ_data ablation
│   │   └── modified_hybrid_mu0.75_warmup5000/        # B10 — physics-warmup ablation
│   └── stage2/
│       ├── cvol/                            # C-Vol (multiplicative)
│       ├── avol/                            # A-Vol (direct softplus)
│       └── *_scratch/                       # --from_scratch ablations (no warm-start)
│
├── results/
│   └── bs_baseline/                         # BS analytical baseline results.json
│
├── reports/                                 # Built by scripts/build_master_table.py + make_report_plots.py
│   ├── master_table.csv                     # One row per config: pooled metrics
│   ├── master_table_short.md                # Markdown table for the writeup
│   ├── per_fold_table.csv                   # One row per (config, fold)
│   └── figures/*.png                        # Comparison + diagnostic plots
│
├── docs/
│   ├── BENCHMARKING_PLAN.md                 # Authoritative spec for the benchmark table
│   └── CLAUDE_CODE_BRIEFING.md              # Full technical context
│
└── references/                              # Research papers
```

------------------------------------------------------------------------

## Non-Dimensionalized Coordinates

All PINN computations use normalized coordinates to improve numerical conditioning:

| Symbol  | Definition                     | Range            |
|---------|--------------------------------|------------------|
| `m`     | Moneyness = S/K                | [0.65, 1.35]     |
| `tau`   | Time to expiry = T - t (years) | [~0.008, ~0.5]  |
| `v_hat` | Normalized price = V/K         | [0, ~0.5]        |

The network learns `v_hat(m, tau)`. Dollar prices are recovered as `V = v_hat * K`.

------------------------------------------------------------------------

## Data Pipeline (`src/data.py`)

```
CSV  ──>  load_and_preprocess()  ──>  Full DataFrame (~59k rows)
                                           │
                    ┌──────────────────────┴──────────────────────┐
              Stage 0 (dev)                              Stage 1 / 2 (walk-forward)
                    │                                            │
          make_temporal_split()                    for each fold in FOLDS:
                                                    make_fold(train_end, test_start, test_end,
                                                              val_months=1)
                                                              │
                                                  ┌───────────┼───────────┐
                                                train_df    val_df      test_df
```

`make_fold` returns three disjoint slices:

```
                test_window  =  M_f                       ← held out
                val_window   =  last `val_months` months  ← held out
                train_window =  everything else before    ← model trains here
```

`val_months` defaults to 1 (val = the calendar month immediately preceding the test month). The model is trained ONLY on `train_window`. Val drives val-best snapshot selection during training. Test is evaluated ONCE after the snapshot is restored.

### Preprocessing filters (applied once)

1. Volume ≥ 50 (liquidity)
2. 1-day lagged daily mean implied volatility
3. Triple Witching week exclusion (abnormal pricing)
4. Moneyness in [0.65, 1.35]
5. Time-to-expiry > 3 days
6. Interest rate: % to decimal

------------------------------------------------------------------------

## Walk-Forward Validation (Stage 1 / 2)

9 monthly test folds from April through December 2020. Expanding window: training always starts from the beginning of the dataset. The val window for fold `M_f` is `M_{f-1}` (carved as the last month of what would otherwise be train).

```
Fold 1 (Apr): Train [Jan─Feb]  Val [Mar]  Test [Apr]
Fold 2 (May): Train [Jan─Mar]  Val [Apr]  Test [May]
Fold 3 (Jun): Train [Jan─Apr]  Val [May]  Test [Jun]
...
Fold 9 (Dec): Train [Jan─Oct]  Val [Nov]  Test [Dec]
```

This mimics real deployment: the model is retrained monthly on history that excludes the most recent month (held out as val) and tested on the next unseen month. Each fold gets its own `sigma_fixed` (from its training window only — no val/test leakage), its own boundary conditions, and its own collocation points.

### Reporting protocol — single-phase val-best snapshot

The reported test number for each fold is produced by:

1. Train on `train_window` only (`val_window` and `test_window` are held out).
2. Every `val_every` epochs (500 for Stage 1, 250 for Stage 2), evaluate val RMSE.
3. Whenever val RMSE improves, snapshot the model state to CPU.
4. After the training loop, restore the val-best snapshot.
5. Evaluate `test_window` exactly ONCE on the restored model.

The reported `final_*` metrics are from that single end-of-training test eval. The val curve is recorded in history for plotting; the test set is touched exactly once per fold. The saved fold checkpoint IS the val-best snapshot — Stage 2 warm-starts from this.

### Headline metrics

- **Pooled RMSE** (headline): `sqrt(sum_all_folds(squared_errors) / sum_all_folds(n_test))` — treats every option equally regardless of fold size.
- **Mean ± Std of per-fold RMSE**: stability across market regimes.
- **Worst-fold RMSE**: robustness check.
- **Pooled RMSE in half-spread units**: residual / (spread/2). Values ≤ 1 mean errors are inside the bid-ask spread (essentially perfect for a market-making sense).
- **Stratified RMSE (OTM / ATM / ITM)**: where in moneyness the surface is most/least accurate. Bands: OTM `m < 0.97`, ATM `0.97 ≤ m ≤ 1.03`, ITM `m > 1.03`.
- **No-arbitrage diagnostic**: % of (m, τ) grid points where `∂²v̂/∂m² < 0` (butterfly violation) or `∂v̂/∂τ < 0` (calendar violation). 0% = arbitrage-free price surface.
- **E\*_f**: the val-best epoch per fold. Distribution across folds tells us how training-time stability varies with the data window.

------------------------------------------------------------------------

## Architectures (`src/model.py`)

### PricingNet (Modified MLP + RWF)

The primary architecture, used for B2–B7. Three enhancements from Wang et al. 2023:

```
Input (m, tau)
    ▼
FourierFeatureEmbedding(scale=1.0)        →  [sin(2πB·x), cos(2πB·x)]  dim=128
    │
    ├──> Encoder U: RWFLinear(128→64) + tanh    (computed once, reused)
    ├──> Encoder V: RWFLinear(128→64) + tanh    (computed once, reused)
    ▼
Modified Hidden Layer 1: RWFLinear(128→64) + tanh
    gating:  g = h ⊙ U + (1-h) ⊙ V             ← keeps input gradients alive
    ▼
Modified Hidden Layers 2-4: same pattern
    ▼
RWFLinear(64→1)  →  v_hat(m, tau)
```

**Why each component matters:**
- **Fourier features**: maps low-dim (m, τ) into spectral space so MLPs can fit high-frequency PDE solutions.
- **RWF (Random Weight Factorization)**: `W = diag(exp(s)) · V`. Gradient on each row gets an `exp(2 s_k)` per-neuron learning-rate factor, which acts as a learned adaptive LR. The `mu` parameter controls *initial* weight magnitude (`Var(W) ∝ exp(2μ)`); the adaptive-LR benefit is present for any μ. Wang recommends μ=1.0 for forward-PDE problems; we use μ=0.0 for hybrid mode (B8) because high μ × high RWF amplification × the data loss term causes runaway adaptive weights — see B5–B7 vs B8 in `reports/results_summary.md`.
- **Modified MLP gating**: parallel `g = h ⊙ U + (1-h) ⊙ V` paths from the input embedding into every hidden layer. Theoretical motivation is to keep input-coordinate gradients alive through the deep `tanh` chain; in our walk-forward results this benefit does not dominate (a plain MLP — B1 — is competitive).

~37,700 parameters.

### StandardPricingNet (Naive Baseline)

Used for B0/B1. Fourier features + plain MLP (no RWF, no gating). ~37,500 parameters. Serves as the naive reference for the Modified MLP comparison.

### VolatilityNet (Stage 2)

Small RWF MLP that learns `sigma_hat(m, tau)`. ~5,000 parameters. No Modified MLP gating — the vol surface has no payoff kink.

### Volatility wrappers

- **CVolWrapper (multiplicative, primary contribution):**
  `sigma_hat = sqrt(softplus(z) · sigma_0²)` where `softplus(z)` is the learned multiplier μ and `sigma_0 = sigma_fixed` (frozen buffer). Initialization: μ ≈ 1 everywhere → `sigma_hat ≈ sigma_0`.
- **AVolWrapper (direct, comparator):**
  `sigma_hat = softplus(z)`. Initialization: `sigma_hat ≈ sigma_fixed` everywhere.

Both wrappers expose `get_sigma_squared()` which the PDE residual uses directly to avoid an unnecessary `sqrt(...)²` round-trip.

------------------------------------------------------------------------

## Loss Functions (`src/losses.py`)

### PDE residual

```
R = ∂v/∂τ - 0.5 · σ² · m² · ∂²v/∂m² - r · m · ∂v/∂m + r · v
```

Computed via PyTorch automatic differentiation (requires `tanh` activation for smooth second derivatives). `σ²` is either constant (Stage 0/1) or `vol_model.get_sigma_squared(m, τ)` (Stage 2, graph-connected so gradients flow into the vol net).

### Loss terms

| Loss | Formula | Present in |
|----|----|----|
| `L_pde` | `mean(R²)` over collocation points | All modes |
| `L_tc` | `mean((v_pred - max(m-1, 0))²)` at τ=0 | All modes |
| `L_bc` | `mean((v_pred - BS(m_boundary))²)` at m_min, m_max | All modes |
| `L_data` | `mean((v_pred - v_hat_market)²)` on training options | Hybrid + Stage 2 |
| `L_reg` | C-Vol: `mean((μ-1)²)`; A-Vol: smoothness penalties | Stage 2 only |

### Grad-norm adaptive balancing

Total loss is `L = Σ λ_i · L_i`. Weights `λ_i` are adapted every 1000 epochs following Wang et al. Algorithm 1 (Eq 5.3):
```
λ̂_i = sum(grad_norms) / grad_norm_i
λ_new_i = α · λ_old_i + (1-α) · λ̂_i
```

Safety floor of 10 on `L_tc` and `L_bc` weights prevents boundary collapse. Pricing and vol gradients are clipped **separately** so a large pricing-grad update doesn't crush the vol-grad signal.

**Important caveat**: Wang's scheme was tested only on `L_pde + L_ic + L_bc` (no data term). Adding a 4th supervised `L_data` term can drive `λ_data` to runaway values when paired with high RWF μ — see `reports/results_summary.md` and the B8/B9/B10 ablations. `update_adaptive_weights` accepts an `excluded_terms` set so individual loss weights can be pinned out of the balancing scheme (used by B9: `--fixed_data_weight`).

------------------------------------------------------------------------

## Training pipeline by stage

### Stage 0: Development (`run_stage0.py`)

Single fixed train/test split. For rapid iteration during architecture development.

### Stage 1: Walk-Forward Benchmarking (`run_walk_forward.py`)

The benchmark table — 11 configs run the same `run_training` per fold:

| Config | Arch | Mode | μ | Extra | Purpose |
|----|----|----|----|----|----|
| **B0** | Standard | Physics | — | — | Naive baseline (no data, no arch improvements) |
| **B1** | Standard | Hybrid | — | — | + market data (does adding data help naive arch?) |
| **B2** | Modified | Physics | 0.50 | — | + arch improvements |
| **B3** | Modified | Physics | 0.75 | — | μ sweep (physics) |
| **B4** | Modified | Physics | 1.00 | — | μ sweep (physics) |
| **B5** | Modified | Hybrid | 0.50 | — | μ sweep (hybrid) |
| **B6** | Modified | Hybrid | 0.75 | — | μ sweep (hybrid) |
| **B7** | Modified | Hybrid | 1.00 | — | μ sweep (hybrid) |
| **B8** | Modified | Hybrid | 0.00 | — | RWF init fix — kills initial gradient blowup |
| **B9** | Modified | Hybrid | 0.75 | `λ_data=1000` fixed | Fix #2 — decouple data from grad-norm balancing |
| **B10** | Modified | Hybrid | 0.75 | `warmup=5000` | Fix #5 — pre-train PDE for 5k epochs, then add data |

### Stage 2: Learnable Volatility (`run_stage2.py`)

Same `run_training` function, with additional kwargs that activate the vol net code path:

```python
vol_model = CVolWrapper(VolatilityNet(), sigma_0=config["sigma_fixed"])

model, history, run_info = run_training(
    ...,
    mode="hybrid",
    val_arrays=val_arrays,                  # drives val-best snapshot
    # ── Stage 2 additions ──
    vol_model=vol_model,
    vol_type="cvol",
    pricing_lr=1e-4,                        # pricing fine-tunes
    vol_lr=1e-3,                            # vol learns from scratch
    checkpoint_path="runs/walk_forward/.../fold_X.pt",   # warm-start
)
```

What changes when `vol_model` is not None:

| Component | Stage 0/1 | Stage 2 |
|----|----|----|
| Pricing net init | Random | Loaded from Stage 1 fold checkpoint (warm-start) |
| Vol net | Doesn't exist | Fresh, bias-init to `sigma_fixed` |
| Optimizer | Adam(pricing, lr) | Adam([{pricing, 1e-4}, {vol, 1e-3}]) |
| LR schedule | Cosine 100x decay | Cosine 100x decay per group (LambdaLR) |
| PDE σ² | Scalar `sigma_fixed²` | `vol_model.get_sigma_squared(m, τ)` |
| Loss terms | pde, tc, bc [, data] | pde, tc, bc, data, **reg** |
| Grad clipping | Pricing only | Pricing AND vol — separately |
| BS comparison | RMSE vs analytical BS | NaN (meaningless with learned σ) |
| Selection metric | rmse_bs_norm (physics) / rmse_mkt (hybrid) | rmse_mkt (always) |

**Why warm-start is safe under val-best**: Stage 1 fold f's checkpoint was trained on `train_window` only (val held out). Stage 2 fold f then warm-starts from that checkpoint and uses the same `val_window` — which the warm-start has never seen. So val is genuinely held-out for Stage 2 too.

Pass `--from_scratch` to `run_stage2.py` to skip warm-start (joint pricing+vol from random init); this is an ablation, not the default.

------------------------------------------------------------------------

## Boundary Conditions

| Boundary | Value | Stage 2 handling |
|----|----|----|
| Terminal (τ → 0) | `max(m - 1, 0)` (call payoff) | Unchanged — σ-independent |
| Lower (m = m_min) | `BS(m_min, τ, r, sigma_fixed)` | Keep `sigma_fixed` (deep OTM, low vega) |
| Upper (m = m_max) | `BS(m_max, τ, r, sigma_fixed)` | Keep `sigma_fixed` (deep ITM, intrinsic) |
| Interior | PDE residual | Uses `vol_model` σ² — this is what we're learning |

The BCs intentionally use constant `sigma_fixed` even in Stage 2. At the domain boundaries, vega is low so the σ mismatch causes negligible BC error. The constant-σ BCs also act as a stabilizing anchor during early training.

------------------------------------------------------------------------

## Running the code

### Environment

```bash
conda activate pinn_env   # PyTorch, numpy, pandas, scipy, matplotlib
```

### Stage 0 (quick development run)

```bash
python run_stage0.py --mode physics --epochs 15000 --rwf_mu 0.75
python run_stage0.py --mode hybrid  --epochs 15000 --rwf_mu 0.75
```

### BS baseline (~10 sec, no GPU)

```bash
bash scripts/run_local_baselines.sh
# Output: results/bs_baseline/results.json
```

### Stage 1 walk-forward (single config locally)

```bash
# Example: modified hybrid μ=0.75 (one of the eleven configs)
python run_walk_forward.py --mode hybrid --arch modified --rwf_mu 0.75 --epochs 15000

# Single-fold smoke test
python run_walk_forward.py --mode hybrid --arch modified --rwf_mu 0.75 \
    --epochs 1500 --folds Nov2020 --output_dir runs/_smoke

# Loss-balancing ablations (B8/B9/B10):
# B8 — kill RWF init blowup
python run_walk_forward.py --mode hybrid --arch modified --rwf_mu 0.0 --epochs 15000

# B9 — pin λ_data, decouple from grad-norm
python run_walk_forward.py --mode hybrid --arch modified --rwf_mu 0.75 \
    --fixed_data_weight 1000.0 --epochs 15000

# B10 — physics-only warmup, then turn on data
python run_walk_forward.py --mode hybrid --arch modified --rwf_mu 0.75 \
    --data_loss_warmup 5000 --epochs 15000
```

### Stage 1 on DCC (all 11 configs in parallel)

```bash
# From your local clone, push first:
git push origin main

# On DCC:
ssh ds555@dcc-login.oit.duke.edu
cd /hpc/group/fisherlab/ds555/pinn_code
git pull
bash slurm/submit_all.sh   # smoke → B0-B10 array (11 tasks, ~3-4h each)

# Or submit one task at a time, e.g. just B8:
sbatch --array=8 slurm/run_benchmark.sh
```

### Stage 2 (learnable volatility) — locally

Requires Stage 1 checkpoints (warm-start from your best Stage 1 config).
Pick `--checkpoint_dir` based on `reports/master_table.csv`; example uses
the originally-planned B6, but in practice the right choice depends on
which Stage 1 config wins:

```bash
python run_stage2.py --vol_type cvol \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75 \
    --pricing_lr 1e-4 --vol_lr 1e-3 \
    --epochs 10000 --rwf_mu 0.75

python run_stage2.py --vol_type avol \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75 \
    --pricing_lr 1e-4 --vol_lr 1e-3 \
    --epochs 10000 --rwf_mu 0.75

# Optional ablation: train pricing+vol jointly from random init
python run_stage2.py --vol_type cvol --from_scratch \
    --pricing_lr 1e-3 --vol_lr 1e-3 --epochs 15000
```

### Stage 2 on DCC

After Stage 1 finishes and you've picked a winning config:

```bash
# Edit STAGE1_MU (and the warm-start dir name if different from
# modified_hybrid_muX) in slurm/run_stage2.sh and run_stage2_smoke.sh.
# Then:
bash slurm/submit_stage2.sh
```

### Build report artifacts

After all benchmarks finish:

```bash
# 1. Pull together the master table (CSV + Markdown + per-fold CSV)
python scripts/build_master_table.py

# 2. Generate report figures (pooled comparison, per-fold lines, stratified, etc.)
python scripts/make_report_plots.py

# Outputs land in reports/  (master_table.csv, master_table_short.md, figures/*.png)
```

------------------------------------------------------------------------

## Key Design Decisions

**Backward compatibility via kwargs.** All Stage 2 additions use `default=None` kwargs in `src/training.py`. When `vol_model=None`, the code path is identical to Stage 0/1. Stage 2 behavior activates only when the caller passes a vol_model.

**Per-group gradient clipping.** Pricing and vol gradients are clipped independently. A single combined `clip_grad_norm_` would let pricing's larger gradients suppress the vol signal entirely.

**Per-group cosine LR schedule with same decay ratio.** `LambdaLR` with a multiplicative cosine factor decays each group's base LR by the same 100x ratio (rather than the asymmetric decay from a single scalar `eta_min`).

**Warm-start as compute optimization, not methodology.** Stage 2 warm-starts pricing from Stage 1 to save ~3x compute, but the val window is never seen by either stage so warm-start is statistically clean. `--from_scratch` flag exists for ablation.
