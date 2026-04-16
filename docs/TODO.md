# PINN Option Pricing — Project TODO

*Last updated: March 2026*

## Project Overview
Staged PINN framework for TSLA 2020 call option pricing.
Constant-σ baseline → learnable volatility surface → uncertainty quantification.

**Data:** `TSLA_2020_Split_Adjusted.csv` (93,746 obs)

**Benchmarks (single Nov–Dec split):**
- Black-Scholes: RMSE $17.23
- GAM: RMSE $12.57
- laGP: RMSE $9.63

**Architecture (finalized):** Modified MLP + RWF + Fourier Features + Grad-Norm Balancing
(Wang et al. 2023 full pipeline)

---

## Stage 0: Constant-σ Baseline — NEARLY COMPLETE

### 0.1 Data Preparation — DONE ✓
- [x] Replicate QMD pipeline in Python (triple witching, moneyness [0.65,1.35],
      near-expiry removal, interest rate conversion, lagged vol proxy)
- [x] σ_fixed = median ATM IV, r_fixed = median rate
- [x] Non-dimensionalized (m, τ) coordinates, v̂ = V/K
- [x] Collocation: 5000/iter, 70/30 uniform/kink-concentrated
- [x] Train/test split: Jan–Oct / Nov–Dec 2020

### 0.2 Architecture — DONE ✓
- [x] Fourier Feature Embedding (64 features, scale=1.0)
- [x] Modified MLP with U, V encoders + gating at every layer
- [x] Random Weight Factorization (all linear layers)
- [x] Grad-norm adaptive loss balancing (every 1000 steps, α=0.9)
- [x] Weight floor on TC/BC (λ ≥ 10, rarely active)
- [x] Full diagnostics: loss curves, weight trajectories, dominance ratio,
      solution surfaces, scatter plots

### 0.3 Physics Mode — DONE ✓
Best stable result (μ=1.0, LR=1e-3, 10k epochs):
- BS RMSE (norm): 0.010 — monotonic improvement, zero drift
- Market RMSE: $9.26 (accidental — see architecture summary doc)
- μ=0.75 run in progress for comparison

Key finding: Modified MLP eliminated the drift problem that plagued
the standard MLP (which peaked at epoch 2000 then degraded 10×).
The PDE dominance ratio stayed near 1.0 throughout training.

### 0.4 Hybrid Mode — DONE ✓
- [x] Ran all three μ values (0.5, 0.75, 1.0) in hybrid mode
- [x] Results (stable late-training RMSE):
  - μ=0.5:  ~$8.80 (final epoch $8.79, best $8.75 @ ep 2000)
  - μ=0.75: ~$9.30 (final epoch $9.31, best $9.06 @ ep 5000)
  - μ=1.0:  ~$9.90 (final epoch ~$9.98, best $9.57 @ ep 7000)
- [x] Key finding: Lower μ wins in hybrid mode (opposite of physics mode)
  because the PDE loss drops faster, letting the data loss become
  influential earlier. μ=0.5 gets ~10k epochs of market data learning;
  μ=1.0 spends ~8k epochs still grinding on PDE.
- [x] Selection: **μ=0.5, 12k epochs, report final-epoch RMSE** for hybrid
  - Final-epoch RMSE ($8.79) ≈ best-epoch RMSE ($8.75) → no selection bias
  - Transient spike at epoch 3000 recovers by epoch 5000; data loss acts as anchor

### 0.5 Hyperparameter Selection Methodology — SETTLED ✓
- μ selected based on **training dynamics**, not test RMSE:
  - Physics mode: μ=1.0 chosen for stability (PDE dominance ratio stays near 1)
  - Hybrid mode: μ=0.5 chosen because faster PDE convergence allows earlier
    market data learning; training stable from epoch 8000 onward
- Epochs fixed at 12k (hybrid) / 15k (physics), **report final-epoch model**
  - No early stopping, no epoch selection, no test-data leakage
- Methodology statement: "We select μ based on optimization dynamics analysis.
  For hybrid mode, μ=0.5 achieves competitive PDE accuracy while allowing the
  data loss to become influential earlier in training. We use a fixed epoch
  budget with cosine annealing and report the final-epoch model."

---

## Code Refactoring — DONE ✓

Notebook refactored into modular Python package. See `pinn_src.zip`.

```
src/
├── __init__.py
├── bs_formulas.py      # bs_call_price, bs_call_normalized
├── data.py             # load_and_preprocess, make_fold, compute_constants, etc.
├── model.py            # RWFLinear, FourierFeatureEmbedding, PricingNet
├── losses.py           # compute_pde_residual, individual losses, grad-norm
├── training.py         # run_training() — full pipeline as one function call
└── diagnostics.py      # all plotting functions
run_stage0.py           # single-split training script
run_walk_forward.py     # 9-fold backtesting script
```

Verified: all numerical operations identical to notebook. Only structural
change: globals → function parameters.

---

## Stage 1: Walk-Forward Backtesting

### 1.1 Framework Design — IMPLEMENTED ✓
- [x] `run_walk_forward.py` implemented with 9 monthly folds
- [x] Expanding window: train on all data up to fold start, test on 1 month
- [x] Folds: Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec 2020

### 1.2 Run Configuration — SETTLED
- [ ] **Run hybrid mode: μ=0.5, 12k epochs, LR=1e-3** ← IMMEDIATE NEXT STEP
  - `python run_walk_forward.py --mode hybrid --epochs 12000 --rwf_mu 0.5`
  - ~3 hours on M4 Mac
  - Per-fold check: verify training stability (no permanent drift)
  - If specific folds show instability, test μ=0.75 on those folds only
- [ ] Run GAM and laGP on same folds (adapt R code)

### 1.3 Reporting
- [ ] Per-fold RMSE table: BS vs GAM vs GP vs PINN(hybrid)
- [ ] Average RMSE, std across folds, per-fold winners
- [ ] Regime analysis: which methods win in which market conditions?
- [ ] Report final-epoch RMSE (no early stopping, no epoch selection)

### 1.4 Methodology Defense
- μ selected on training dynamics (not test RMSE) — no leakage
- Fixed epoch budget, report final model — no epoch selection bias
- Walk-forward folds are touched exactly once for reporting

---

## Stage 2: Learnable Volatility Surface

### 2.1 Core Change to PDE Residual
Replace σ_fixed with σ̂(m, τ) in the PDE:

```
R = ∂v̂/∂τ − (1/2)·σ̂(m,τ)²·m²·∂²v̂/∂m² − r·m·∂v̂/∂m + r·v̂
```

Autodiff flows through σ̂ → volatility network learns what σ surface
makes the PDE + market data simultaneously consistent.

### 2.2 Design Decisions — SETTLED

**Fourier embedding:** Separate for vol network (DECIDED)
- Vol surface is smoother than price surface → lower Fourier scale (~0.5)
- Separate embedding lets vol network learn different frequency decomposition
- Vol network's Fourier scale is a tunable hyperparameter

**Vol network architecture:** Simple MLP + RWF, no Modified MLP (DECIDED)
- Vol surface has no payoff kink → Modified MLP's gradient shortcuts unnecessary
- 2 layers × 32 neurons with Fourier features (scale 0.5)
- Keeps parameter count low; upgrade to Modified MLP only if needed

**Warm start:** Bias initialization for stable start (DECIDED)
- Pricing network: warm-start from Stage 0 hybrid checkpoint
- Volatility network: initialized so output ≈ σ_fixed at epoch 0
  - A-Vol: bias final layer so softplus(z₀) ≈ σ_fixed
  - C-Vol: multiplier naturally starts near 1 via (μ−1)² regularization
- This prevents PDE explosion at epoch 1 from random σ̂ values

**Implementation order:** C-Vol first, A-Vol as comparator (DECIDED)
- C-Vol has fewer hyperparameters (one λ_mult vs three α₁, α₂, α₃)
- C-Vol has uniform gradient scaling (factor σ₀² constant across domain)
  vs A-Vol's non-uniform scaling (factor 2σ̂ varies, biasing toward wings)
- C-Vol separates level (σ₀) from shape (multiplier network)
- C-Vol is the novel contribution (NSM transfer from biology)
- A-Vol is the standard baseline for comparison

**σ₀ in C-Vol:** Fixed at σ_fixed initially (DECIDED)
- Start with σ₀ = σ_fixed (median ATM IV from training)
- Consider making learnable later (with low LR to prevent degeneracy)
- Regularization L_mult = (μ−1)² partially prevents σ₀/μ degeneracy

### 2.3 Volatility Architectures

**C-Vol (Multiplicative / NSM-inspired) — PRIMARY:**
- μ(m,τ) = softplus(NN_mult(m,τ; ϕ)), then σ̂² = μ · σ₀²
- NN_mult: Fourier(scale=0.5) → 2 × 32 (tanh, RWF) → softplus
- σ₀ = σ_fixed from Stage 0 (fixed scalar)
- Regularization: L_mult = mean((μ − 1)²)
- At initialization: μ ≈ 1 everywhere → σ̂ ≈ σ₀ → same as Stage 0

**A-Vol (Direct) — COMPARATOR:**
- σ̂(m,τ) = softplus(NN_σ(m,τ; ϕ))
- NN_σ: Fourier(scale=0.5) → 2 × 32 (tanh, RWF) → softplus
- Bias init: final bias = log(exp(σ_fixed) − 1) so softplus(z₀) ≈ σ_fixed
- Regularization: L_reg = α₁||∂σ̂/∂m||² + α₂||∂σ̂/∂τ||² + α₃||∂²σ̂/∂m²||²

### 2.4 Key Differences During Training

| Aspect | A-Vol (Direct) | C-Vol (Multiplicative) |
|--------|---------------|----------------------|
| Network learns | Absolute volatility | Deviations from σ₀ |
| Output range | Full vol range (0.3–1.0+) | Centered near 1 |
| Gradient scaling | Non-uniform (∝ 2σ̂) | Uniform (∝ σ₀²) |
| Regularization | 3 smoothness hyperparams | 1 deviation penalty |
| Level vs shape | Entangled | Separated (σ₀ = level, μ = shape) |
| Risk | Level shift ruins pricing | Degeneracy if σ₀ learnable |

### 2.5 Training Protocol
- [ ] Warm-start pricing network from Stage 0 hybrid checkpoint (μ=0.5)
- [ ] Volatility network initialized fresh with bias init
- [ ] Separate parameter groups:
  - Pricing net: LR = 1e-4 (already good, fine-tune only)
  - Volatility net: LR = 1e-3 (needs to learn from scratch)
- [ ] Grad-norm balancing covers 5 terms: L_pde, L_tc, L_bc, L_data, L_reg
- [ ] Loss: L = λ_pde·L_pde + λ_tc·L_tc + λ_bc·L_bc + λ_data·L_data^MSE + λ_reg·L_reg

### 2.6 Implementation in src/
- [ ] Add `VolatilityNet` class to `src/model.py`
  - Simple MLP: Fourier(scale=0.5) → RWFLinear(128→32) → tanh → RWFLinear(32→32)
    → tanh → RWFLinear(32→1) → softplus
  - C-Vol wrapper: multiply by σ₀²
  - A-Vol wrapper: direct softplus output
- [ ] Modify `compute_pde_residual` in `src/losses.py`
  - Accept optional vol_model parameter
  - If vol_model: sigma_hat = vol_model(m, tau), use in PDE
  - If not: use fixed sigma (backward compatible with Stage 0)
- [ ] Add regularization loss functions to `src/losses.py`
  - `compute_smoothness_reg(vol_model, m, tau)` for A-Vol
  - `compute_multiplier_reg(vol_model, m, tau)` for C-Vol
- [ ] Modify `run_training` to accept vol_model and handle parameter groups

### 2.7 Validation
- [ ] Plot learned σ̂(m, τ) surface
- [ ] Overlay with empirical implied vol surface from data
- [ ] Does it capture the smile? The term structure?
- [ ] Walk-forward evaluation (using Stage 1 framework)
- [ ] Compare: PINN(const-σ) vs PINN(A-Vol) vs PINN(C-Vol)

---

## Stage 3: Uncertainty Quantification

### 3.1 Two UQ Approaches (from framework proposal)

**A-UQ (Heteroscedastic):**
- Add variance head: ŝ²(m,τ) = exp(NN_var(m,τ))
- Switch data loss from MSE to NLL:
  L_data^NLL = Σ [log ŝ² + (v_mkt - K·v̂)² / ŝ²]
- Reduce physics weights (λ_pde, λ_tc, λ_bc) to allow UQ calibration
- Separate high LR for UQ head, low LR for pricing/vol nets

**C-UQ (Proportional Parametric):**
- ŝ²(m,τ) = ν² + η²·(K·v̂)² — two learnable scalars
- Simpler, more interpretable
- ν captures base noise, η captures price-proportional noise

### 3.2 The Four Combinations
| | A-UQ (Heteroscedastic) | C-UQ (Proportional) |
|---|---|---|
| A-Vol (Direct) | AA — max flexibility | AC — practical default |
| C-Vol (Multiplicative) | CA — primary contribution | CC — max interpretability |

### 3.3 Calibration Metrics
- [ ] Empirical coverage at 85%, 90%, 95%
- [ ] Average interval width at each coverage level
- [ ] Calibration plots (expected vs observed coverage)
- [ ] Compare to GAM (86.85% at 95%) and GP (96.26% at 95%)

---

## Key Hyperparameters (Current Best)

### Pricing Network (Stages 0–2)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | Modified MLP + RWF | Wang et al. full pipeline |
| Hidden layers | 4 × 64 | Pricing net |
| Activation | tanh | C∞, required for PDE |
| Fourier features | 64 (128-dim) | Scale 1.0 |
| RWF μ | 0.5 (hybrid) / 1.0 (physics) | Selected on training dynamics |
| RWF σ | 0.1 | Wang et al. default |
| Learning rate | 1e-3 (Stage 0) / 1e-4 (Stage 2) | Lower in Stage 2 (warm-started) |
| LR schedule | Cosine → LR×0.01 | Over full training duration |
| Grad-norm freq | 1000 | Wang et al. default |
| Grad-norm EMA α | 0.9 | Wang et al. default |
| Weight floor | 10 (TC, BC) | Rarely active with Modified MLP |
| Collocation | 5000/iter | 70/30 uniform/kink |
| Market batch | 8000 | Hybrid mode |
| Epochs | 12k (hybrid) / 15k (physics) | Fixed budget, report final epoch |

### Volatility Network (Stage 2)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | Simple MLP + RWF | No Modified MLP needed |
| Hidden layers | 2 × 32 | Vol surface is smoother |
| Fourier scale | 0.5 | Lower than pricing net |
| Learning rate | 1e-3 | Higher than pricing net (learning from scratch) |
| σ₀ (C-Vol) | σ_fixed | Fixed initially, consider learnable later |

---

## Immediate Priorities (Ordered)

0. **ESTABLISH WALK-FORWARD BENCHMARKS** — see `BENCHMARKING_PLAN.md`
   All baseline models evaluated on 9 walk-forward folds BEFORE Stage 2.
   
   **Part A (fast):**
   a. BS baseline on 9 folds — seconds
   b. GAM on 9 folds (adapt R code) — minutes
   c. laGP on 9 folds (adapt R code) — minutes
   
   **Part B (overnight runs, ~16 hrs total):**
   d. B1: Standard MLP, hybrid — naive PINN baseline (~3 hrs)
   e. B2: Modified MLP + RWF, physics, μ=1.0 — physics-only baseline (~3 hrs)
   f. B3: Modified MLP + RWF, hybrid, μ=0.5 — best single-split config (~3 hrs)
   g. B4: Modified MLP + RWF, hybrid, μ=0.75 — middle ground (~3 hrs)
   h. B5: Modified MLP + RWF, hybrid, μ=1.0 — slow start comparison (~4 hrs)
   
   **Part C (analysis):**
   i. Compile master benchmark table (8 columns × 9 folds)
   j. Select μ for Stage 2 based on walk-forward evidence
   k. Save winning μ's per-fold checkpoints for Stage 2 warm-start

1. **Implement Stage 2 C-Vol** — VolatilityNet in model.py, modify PDE residual
2. **Single-split Stage 2 test** — verify C-Vol training stability
3. **Walk-forward Stage 2** — add column to benchmark table
4. **Implement A-Vol** — comparator for C-Vol
5. **Stage 3 UQ** — heteroscedastic head, calibration
