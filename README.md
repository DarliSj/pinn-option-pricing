# PINN Option Pricing

Physics-Informed Neural Networks for European call option pricing on TSLA 2020 data. Duke independent study (Spring 2026).

The system evolves through three stages, each adding capability while reusing the same `src/` modules via backward-compatible kwargs:

| Stage | What it learns | Volatility | Data used |
|----|----|----|----|
| **0** | Price surface v(m, tau) | Constant sigma_fixed | Physics only or physics + market |
| **1** | Same, walk-forward validated | Constant sigma_fixed | Same, across 9 temporal folds |
| **2** | Price + volatility surface | Learned sigma(m, tau) | Physics + market + regularization |

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
│   ├── training.py                          # Training loop, validation, checkpointing
│   ├── baselines.py                         # BS constant-sigma baseline evaluator
│   └── diagnostics.py                       # Training plots, solution surfaces, vol diagnostics
│
├── run_stage0.py                            # Single train/test split (development)
├── run_walk_forward.py                      # Walk-forward backtest (Stage 1 benchmarking)
├── run_bs_baseline.py                       # BS baseline across all folds
├── run_stage2.py                            # Learnable vol surface walk-forward
│
├── runs/                                    # All output (checkpoints, logs, plots, results.json)
│   ├── walk_forward/
│   │   ├── standard_physics/                # B0: Standard MLP, physics only
│   │   ├── standard_hybrid/                 # B1: Standard MLP, physics + market data
│   │   ├── modified_physics_mu{X}/          # B2: Modified MLP, physics only
│   │   └── modified_hybrid_mu{X}/           # B3-B5: Modified MLP, hybrid, mu sweep
│   └── stage2/
│       ├── cvol/                            # C-Vol (multiplicative) walk-forward
│       └── avol/                            # A-Vol (direct) walk-forward
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
| `tau`   | Time to expiry = T - t (years) | [\~0.008, \~0.5] |
| `v_hat` | Normalized price = V/K         | [0, \~0.5]       |

The network learns `v_hat(m, tau)`. Dollar prices are recovered as `V = v_hat * K`.

------------------------------------------------------------------------

## Data Pipeline (`src/data.py`)

All stages share the same data pipeline:

```         
CSV  ──>  load_and_preprocess()  ──>  Full DataFrame (59k rows)
                                           │
                    ┌──────────────────────┴──────────────────────┐
              Stage 0 (dev)                              Stage 1/2 (walk-forward)
                    │                                            │
          make_temporal_split()                    for each fold in FOLDS:
          (fixed: Nov 2020)                         make_fold(train_end, test_start, test_end)
                    │                                            │
                    └──────────────────────┬──────────────────────┘
                                           │
                              ┌─────────────┴─────────────────┐
                              │                               │
                     compute_constants(train_df)     df_to_arrays(train_df)
                              │                      df_to_arrays(test_df)
                    sigma_fixed = median ATM IV               │
                    r_fixed = median interest rate    Numpy arrays: m, tau,
                    domain bounds: m_min/max, ...     vhat, mid, K
                              │
                    build_boundary_terminal(config, sigma, r)
                              │
                    TC: v_hat = max(m-1, 0) at tau=0
                    BC_lo: BS(m_min, tau) for all tau
                    BC_hi: BS(m_max, tau) for all tau
```

**Key design choice:** `sigma_fixed` and `r_fixed` are computed from training data only (no lookahead). Each walk-forward fold recomputes them from its own expanding training window.

### Preprocessing filters (applied once)

1.  Volume \>= 50 (liquidity)
2.  1-day lagged daily mean implied volatility
3.  Triple Witching week exclusion (abnormal pricing)
4.  Moneyness in [0.65, 1.35]
5.  Time-to-expiry \> 3 days
6.  Interest rate: % to decimal

------------------------------------------------------------------------

## Walk-Forward Validation

The benchmarking uses **expanding-window walk-forward** validation: 9 monthly test folds from April through December 2020. Training always starts from the beginning of the dataset.

```         
Fold 1 (Apr): Train [Jan─Mar]  Test [Apr]
Fold 2 (May): Train [Jan─Apr]  Test [May]
Fold 3 (Jun): Train [Jan─May]  Test [Jun]
...
Fold 9 (Dec): Train [Jan─Nov]  Test [Dec]
```

This mimics real deployment: the model is retrained monthly on all available history and tested on the next unseen month. Each fold gets its own `sigma_fixed` (from its training window), its own boundary conditions, and its own collocation points.

### Reporting

-   **Pooled RMSE** (headline): `sqrt(sum_all_folds(squared_errors) / sum_all_folds(n_test))` -- treats every option equally regardless of fold size
-   **Mean +/- Std of per-fold RMSE**: stability diagnostic -- high std means the model is inconsistent across market regimes
-   **Worst fold**: robustness check
-   **Drift gap**: `final_epoch_metric - best_epoch_metric` -- quantifies late-training degradation. We report final-epoch numbers (not best-of), so drift gap tells us if the model peaked early.

------------------------------------------------------------------------

## Neural Network Architectures (`src/model.py`)

### PricingNet (Modified MLP + RWF)

The primary architecture, used for B2-B5 benchmarks. Three enhancements from Wang et al. 2023:

```         
Input (m, tau)
    │
    ▼
FourierFeatureEmbedding(scale=1.0)     ──>  [sin(2pi*B*x), cos(2pi*B*x)]  dim=128
    │
    ├──> Encoder U: RWFLinear(128→64) + tanh     (computed once, reused)
    ├──> Encoder V: RWFLinear(128→64) + tanh     (computed once, reused)
    │
    ▼
Modified Hidden Layer 1: RWFLinear(128→64) + tanh
    gating:  g = h * U + (1-h) * V                <── keeps input gradients alive
    │
    ▼
Modified Hidden Layer 2-4: same pattern
    │
    ▼
RWFLinear(64→1)  ──>  v_hat(m, tau)
```

**Why each component matters:**

-   **Fourier features**: Maps low-dimensional (m, tau) into high-dimensional spectral space. Without this, MLPs struggle to learn high-frequency PDE solutions.
-   **RWF (Random Weight Factorization)**: W = diag(exp(s)) \@ V gives each neuron an adaptive learning rate. The `mu` parameter controls initial magnitude -- lower mu means smaller initial weights, gentler start.
-   **Modified MLP (U, V encoders with gating)**: The gating `g = h*U + (1-h)*V` at every layer creates skip-connection-like paths from the input embedding. This prevents BC/TC gradient vanishing through deep tanh chains -- the key fix that eliminated the drift problem.

\~37,700 parameters.

### StandardPricingNet (Naive Baseline)

Used for B0/B1 benchmarks. Fourier features + plain MLP (no RWF, no Modified MLP). This is the architecture from the original notebook before drift fixes. Expected to suffer from BC/TC gradient vanishing.

```         
Input (m, tau)  ──>  Fourier  ──>  Linear(128→64)+tanh  ──>  ...  ──>  Linear(64→1)
```

\~37,500 parameters (similar count, different architecture).

### VolatilityNet (Stage 2)

Small RWF MLP that learns the volatility surface sigma(m, tau). Intentionally simple:

```         
Input (m, tau)
    │
    ▼
FourierFeatureEmbedding(scale=0.5)     <── lower scale: vol surface is smoother
    │
    ▼
RWFLinear(128→32) + tanh
    │
RWFLinear(32→32) + tanh
    │
RWFLinear(32→1)  ──>  raw z (pre-activation)
```

\~5,000 parameters. No Modified MLP -- the vol surface has no payoff kink. The raw output `z` is passed to a wrapper that applies the appropriate activation.

### Volatility Wrappers

**CVolWrapper (Multiplicative, primary contribution):**

```         
sigma_hat = sqrt(softplus(z) * sigma_0^2)

where mu(m,tau) = softplus(z) is the learned multiplier
      sigma_0 = sigma_fixed (frozen buffer from training data)
```

The network learns *deviations* from sigma_0, not absolute vol. At initialization, mu = 1 everywhere (output layer weights zeroed, bias set so softplus(bias) = 1). One regularization term: L_reg = mean((mu - 1)\^2).

**AVolWrapper (Direct, standard comparator):**

```         
sigma_hat = softplus(z)
```

The network directly outputs volatility. At initialization, sigma_hat = sigma_fixed everywhere. Three smoothness regularization terms (derivatives of sigma_hat wrt m and tau).

------------------------------------------------------------------------

## Loss Functions (`src/losses.py`)

### PDE Residual (all stages)

The Black-Scholes PDE in normalized coordinates:

```         
R = dv/dtau - 0.5 * sigma^2 * m^2 * d2v/dm2 - r * m * dv/dm + r * v
```

Computed via PyTorch automatic differentiation (requires `tanh` activation for smooth second derivatives). `R = 0` everywhere if the solution satisfies the PDE.

**Stage 0/1:** `sigma` is a scalar constant (`sigma_fixed`). **Stage 2:** `sigma` is `vol_model(m, tau)`, a graph-connected tensor. Gradients from L_pde flow through sigma into the vol net -- this is how the vol network learns what volatility surface satisfies no-arbitrage.

### Loss Terms

| Loss | Formula | Present in |
|----|----|----|
| `L_pde` | `mean(R^2)` over collocation points | All modes |
| `L_tc` | `mean((v_pred - max(m-1,0))^2)` at tau=0 | All modes |
| `L_bc` | `mean((v_pred - BS(m_boundary))^2)` at m_min, m_max | All modes |
| `L_data` | `mean((v_pred - v_hat_market)^2)` on training options | Hybrid + Stage 2 |
| `L_reg` | C-Vol: `mean((mu-1)^2)` / A-Vol: smoothness penalties | Stage 2 only |

### Grad-Norm Adaptive Balancing

The total loss is `L = sum(lambda_i * L_i)` where the weights `lambda_i` are adapted every 1000 epochs using Wang et al. Algorithm 1:

```         
lambda_hat_i = sum(grad_norms) / grad_norm_i
lambda_new_i = alpha * lambda_old_i + (1-alpha) * lambda_hat_i
```

This prevents any single loss term from dominating gradient updates. Safety floor of 10 on TC/BC weights prevents boundary condition collapse.

------------------------------------------------------------------------

## Training Differences by Stage

### Stage 0: Development (`run_stage0.py`)

Single fixed train/test split (Nov 2020 cutoff). For rapid iteration.

``` python
model, history, run_info = run_training(
    ..., mode="physics",   # or "hybrid"
    arch="modified",       # PricingNet
    epochs=15000,
    lr=1e-3,
)
```

**Data flow:**

```         
Full CSV ──> load_and_preprocess ──> make_temporal_split (Nov 2020)
  ──> compute_constants ──> build_boundary_terminal
  ──> run_training(sigma=scalar, mode="physics" or "hybrid")
  ──> validate vs BS + market ──> plots + checkpoint
```

### Stage 1: Walk-Forward Benchmarking (`run_walk_forward.py`)

Same `run_training` call, but looped over 9 folds. Each fold recomputes constants.

``` python
for fold in FOLDS:
    train_df, test_df = make_fold(df, fold["train_end"], ...)
    config = compute_constants(train_df)       # fresh sigma_fixed per fold
    boundary_data = build_boundary_terminal(config, ...)
    model, history, run_info = run_training(
        ..., mode=args.mode, arch=args.arch,   # same function
    )
    torch.save({"model_state_dict": ...}, f"fold_{fold['name']}.pt")
```

The 6 PINN configurations (2x2 ablation grid + mu sweep):

| Config | Arch | Mode | Purpose |
|----|----|----|----|
| B0 | Standard | Physics | Naive baseline (no data, no arch improvements) |
| B1 | Standard | Hybrid | \+ data term (does adding market data help naive arch?) |
| B2 | Modified | Physics | \+ arch improvements (does better arch help?) |
| B3 | Modified | Hybrid, mu=0.5 | Full system, mu sweep |
| B4 | Modified | Hybrid, mu=0.75 | Full system, mu sweep |
| B5 | Modified | Hybrid, mu=1.0 | Full system, mu sweep |

**Data flow is identical to Stage 0** except `make_fold` replaces `make_temporal_split`, and results aggregate into a pooled RMSE across all folds.

### Stage 2: Learnable Volatility (`run_stage2.py`)

Same `run_training` function, but with additional kwargs that activate the vol net code path:

``` python
vol_model = CVolWrapper(VolatilityNet(), sigma_0=config["sigma_fixed"])

model, history, run_info = run_training(
    ...,
    mode="hybrid",                          # always hybrid in Stage 2
    # ── Stage 2 additions (all default to None in Stage 0/1) ──
    vol_model=vol_model,                    # the vol surface network
    vol_type="cvol",                        # selects regularization type
    pricing_lr=1e-4,                        # fine-tune (already converged)
    vol_lr=1e-3,                            # learning from scratch
    checkpoint_path="runs/.../fold_X.pt",   # warm-start pricing net
)
```

**What changes inside `run_training` when `vol_model` is not None:**

| Component | Stage 0/1 | Stage 2 |
|----|----|----|
| Pricing net init | Random | Loaded from Stage 1 checkpoint |
| Vol net | Does not exist | Fresh, bias-initialized to sigma_fixed |
| Optimizer | Adam(pricing_params, lr) | Adam([{pricing, 1e-4}, {vol, 1e-3}]) |
| PDE sigma | Scalar `sigma_fixed` | `vol_model(m, tau)` -- graph-connected |
| Loss terms | pde, tc, bc [, data] | pde, tc, bc, data, **reg** |
| Grad norms | pricing params only | pricing + vol params |
| Grad clipping | pricing params only | pricing + vol params |
| BS comparison | RMSE vs analytical BS | NaN (meaningless with learned sigma) |
| Selection key | rmse_bs_norm (physics) / rmse_mkt (hybrid) | rmse_mkt (always) |
| Checkpoint saves | model_state_dict | model_state_dict + vol_model_state_dict |
| Diagnostics | Solution surface, scatter | \+ vol surface contours, smile/term slices |

**Stage 2 data flow:**

```         
Stage 1 checkpoint (fold_X.pt)
    │
    ▼
Load pricing net weights (warm-start)
Construct fresh VolatilityNet + CVolWrapper (bias init: mu=1)
    │
    ▼
Training loop:
    │
    ├── Collocation batch (m, tau) ──> PDE residual uses vol_model(m,tau) as sigma
    │                                   Gradients flow: L_pde ──> pricing_net AND vol_net
    │
    ├── BC/TC ──> Still use sigma_fixed (boundaries are weakly sigma-dependent)
    │
    ├── Market data batch ──> L_data = MSE(v_pred, v_market)  (pricing net only)
    │
    ├── Regularization ──> C-Vol: L_reg = mean((mu-1)^2)  (vol net only)
    │                       A-Vol: L_smooth = smoothness penalties on sigma_hat
    │
    └── Grad-norm balancing over all 5 loss terms
        Differential LR: pricing @ 1e-4, vol @ 1e-3
```

------------------------------------------------------------------------

## Boundary Conditions

| Boundary | Value | Stage 2 handling |
|----|----|----|
| Terminal (tau -\> 0) | `max(m - 1, 0)` (call payoff) | Unchanged -- sigma-independent |
| Lower (m = m_min) | `BS(m_min, tau, r, sigma_fixed)` | Keep sigma_fixed -- deep OTM, low vega |
| Upper (m = m_max) | `BS(m_max, tau, r, sigma_fixed)` | Keep sigma_fixed -- deep ITM, intrinsic dominates |
| Interior | PDE residual | **Uses vol_model sigma** -- this is what we're learning |

The BCs intentionally use constant sigma_fixed even in Stage 2. At the domain boundaries (m=0.65, m=1.35), vega is low, so the sigma mismatch causes negligible BC error. The constant-sigma BCs also act as a stabilizing anchor during early training when the vol net is still learning.

------------------------------------------------------------------------

## Running the Code

### Environment

``` bash
conda activate pinn_env   # PyTorch, numpy, pandas, scipy, matplotlib
```

### Stage 0 (quick development run)

``` bash
python run_stage0.py --mode physics --epochs 15000 --rwf_mu 0.75
python run_stage0.py --mode hybrid --epochs 15000 --rwf_mu 0.75
```

### BS Baseline

``` bash
python run_bs_baseline.py
# Output: pooled RMSE across 9 folds (~$8.79)
```

### Stage 1 Walk-Forward (full benchmarking)

``` bash
# B0: Standard physics       (~3 hrs)
python run_walk_forward.py --mode physics --epochs 12000 --arch standard

# B1: Standard hybrid        (~3 hrs)
python run_walk_forward.py --mode hybrid --epochs 12000 --arch standard

# B2: Modified physics        (~3 hrs)
python run_walk_forward.py --mode physics --epochs 15000 --arch modified --rwf_mu 1.0

# B5: Modified hybrid mu=1.0  (~3 hrs)
python run_walk_forward.py --mode hybrid --epochs 15000 --arch modified --rwf_mu 1.0

# Single-fold smoke test:
python run_walk_forward.py --mode hybrid --epochs 2000 --folds Nov2020
```

### Stage 2 (learnable volatility)

Requires Stage 1 checkpoints:

``` bash
python run_stage2.py --vol_type cvol \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75 \
    --pricing_lr 1e-4 --vol_lr 1e-3 \
    --epochs 10000 --rwf_mu 0.75

python run_stage2.py --vol_type avol \
    --checkpoint_dir runs/walk_forward/modified_hybrid_mu0.75 \
    --pricing_lr 1e-4 --vol_lr 1e-3 \
    --epochs 10000 --rwf_mu 0.75
```

------------------------------------------------------------------------

## Key Design Decisions

**Backward compatibility via kwargs.** All Stage 2 additions use `default=None` kwargs in `src/` functions. When `vol_model=None`, the code path is identical to Stage 0/1. No separate copies of files, no `if stage == 2` branches scattered around -- the Stage 2 behavior activates only when the caller passes a vol_model.

**Final-epoch reporting.** The benchmark table uses the final-epoch model, not the best-epoch model. The best-epoch metric is tracked as a stability diagnostic only (the "drift gap" = final - best tells us if the model degraded late in training). This avoids implicit early stopping and makes results more reproducible.

**Pooled RMSE as headline metric.** `sqrt(sum_of_squared_errors / total_n_test)` across all folds, rather than the mean of per-fold RMSEs. This weights every option equally regardless of fold size. Mean-of-folds is reported alongside for stability assessment.

**Constant sigma for BCs in Stage 2.** Dynamic BCs (recomputing BC targets with the vol net's sigma each step) create a chicken-and-egg instability. The vol net affects the BCs which affect the loss which affects the vol net. Constant-sigma BCs are stable and approximately correct at the boundaries where vega is low.
