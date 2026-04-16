# PINN Option Pricing — Technical Briefing for Claude Code

## Document Purpose
This is the complete technical context for continuing development in Claude Code CLI.
It summarizes all decisions, code, math, experiments, and findings from the claude.ai sessions.

---

## 0. Migration Checklist & File Manifest

### Project Directory Structure
```
PINN-Option-Pricing/
├── CLAUDE.md                              # Claude Code auto-reads this
├── src/
│   ├── __init__.py
│   ├── bs_formulas.py                     # BS analytical formulas
│   ├── data.py                            # Data loading, preprocessing, fold creation
│   ├── model.py                           # RWFLinear, FourierFeatureEmbedding, PricingNet
│   ├── losses.py                          # PDE residual, individual losses, grad-norm
│   ├── training.py                        # run_training() full pipeline
│   └── diagnostics.py                     # All plotting functions
├── run_stage0.py                          # Single-split training script
├── run_walk_forward.py                    # 9-fold backtesting script
├── data/
│   └── TSLA_2020_Split_Adjusted.csv       # Input data (93,746 obs)
├── docs/
│   ├── CLAUDE_CODE_BRIEFING.md            # ← THIS FILE (full technical context)
│   ├── TODO.md                            # Project roadmap with priorities
│   ├── BENCHMARKING_PLAN.md               # Detailed walk-forward benchmarking plan
│   └── architecture_training_dynamics_summary.md  # Modified MLP/RWF/μ analysis
├── notebooks/
│   └── stage0_pinn_RWF_NLP_diff_mu_1.ipynb  # Original dev notebook (reference only)
├── references/
│   ├── AN_EXPERT_S_GUIDE_TO_TRAINING_PHYSICSINFORMED_NEURAL_NETWORKS.pdf
│   ├── Uncertainity_Aware_Pinn_for_Option_Pricing.pdf
│   ├── Methodology___Framework_proposal2.pdf
│   ├── Wang_et_Al_analysis3.pdf
│   ├── BSM_and_PINNs_Intro2.pdf
│   ├── Pinn_Paper_review__2_2.pdf
│   ├── Tim_De_Ryck__Error_analysis.pdf
│   ├── Pinn_ODE_.pdf
│   ├── Samuel_M_l_equa.pdf
│   ├── s10614024105512.pdf
│   └── s1107102107146z.pdf
├── checkpoints/                           # Saved model states (will populate during benchmarking)
└── results/                               # Walk-forward results (will populate during benchmarking)
```

### Files to Copy from claude.ai Outputs
| File | Copy to | Purpose |
|------|---------|---------|
| `pinn_src.zip` | Unzip → `src/`, `run_stage0.py`, `run_walk_forward.py` | Modular codebase |
| `TODO.md` | `docs/TODO.md` | Project roadmap |
| `BENCHMARKING_PLAN.md` | `docs/BENCHMARKING_PLAN.md` | Walk-forward plan (gate before Stage 2) |
| `CLAUDE_CODE_BRIEFING.md` | `docs/CLAUDE_CODE_BRIEFING.md` | Full technical context |
| `architecture_training_dynamics_summary.md` | `docs/` | Modified MLP / RWF / μ analysis |

### Files Already on Your Machine
| File | Copy to | Purpose |
|------|---------|---------|
| `TSLA_2020_Split_Adjusted.csv` | `data/` | Input dataset |
| `stage0_pinn_RWF_NLP_diff_mu_1.ipynb` | `notebooks/` | Reference notebook |
| All `.pdf` research papers | `references/` | Literature |
| Any saved `.pt` checkpoints | `checkpoints/` | Stage 0 results |

### CLAUDE.md Content
Create `CLAUDE.md` in project root with:
```markdown
# PINN Option Pricing — Project Instructions

## Role & Approach
You are a rigorous research collaborator specializing in computational finance,
derivative pricing, physics-informed neural networks (PINNs), and quantitative
methods. Present options, trade-offs, and considerations rather than prescriptive
solutions. State assumptions explicitly. Distinguish between well-established
knowledge and active research areas.

## Context
Read `docs/CLAUDE_CODE_BRIEFING.md` for complete technical context.
Read `docs/BENCHMARKING_PLAN.md` for the immediate action plan.
Read `docs/TODO.md` for the full project roadmap.

## Current State
- Stage 0 (constant-σ PINN): COMPLETE. Architecture settled.
- IMMEDIATE PRIORITY: Walk-forward benchmarking (8 models × 9 folds)
  See BENCHMARKING_PLAN.md for full details. Summary:
  - B1: Standard MLP hybrid (naive baseline — needs StandardPricingNet in model.py)
  - B2: Modified MLP physics μ=1.0
  - B3: Modified MLP hybrid μ=0.5 (save checkpoints for Stage 2 warm-start)
  - B4: Modified MLP hybrid μ=0.75
  - B5: Modified MLP hybrid μ=1.0
  - Plus BS, GAM, laGP non-PINN baselines
- Stage 2 (learnable volatility): Design settled, implementation AFTER benchmarks.

## Prerequisites Before Running Benchmarks
1. StandardPricingNet class must be added to src/model.py (see BENCHMARKING_PLAN.md)
2. run_walk_forward.py needs --arch flag to switch between standard/modified
3. run_walk_forward.py default data path should be data/TSLA_2020_Split_Adjusted.csv
4. BS baseline function should be added (simple, no training, see BENCHMARKING_PLAN.md)

## Key Reference Documents
- `references/AN_EXPERT_S_GUIDE_TO_TRAINING_PHYSICSINFORMED_NEURAL_NETWORKS.pdf`
  Wang et al. — Modified MLP (§6.4), RWF (§4.3), Fourier features (§4.2),
  grad-norm balancing (§5.2). Our primary architecture reference.
- `references/Methodology___Framework_proposal2.pdf`
  Our framework proposal — 3-stage protocol, 4 combinations (AA/AC/CA/CC),
  volatility network design, UQ approaches.
- `references/Uncertainity_Aware_Pinn_for_Option_Pricing.pdf`
  Kazemian et al. — anchored ensembles for UQ (Stage 3 reference).

## Code Conventions
- PyTorch only. All reusable code in `src/`.
- Always use `RWFLinear` instead of `nn.Linear` for PINN layers.
- Always use Modified MLP (encoders U, V with gating) for pricing network.
- Always use Fourier feature embedding as input layer.
- Always include grad-norm adaptive loss balancing with weight floor of 10 on TC/BC.
- Activation: tanh everywhere (required for PDE second derivatives via autodiff).
- Non-dimensionalized coordinates: m = S/K, τ = T-t, v̂ = V/K.
- Data lives in data/, not project root.

## Diagnostic Requirements
When training any model, always produce:
1. Per-loss curves (log scale) + adaptive weight trajectories
2. PDE dominance ratio: λ_pde·L_pde / (λ_tc·L_tc + λ_bc·L_bc)
3. Validation RMSE vs BS (norm) and vs Market ($) at regular intervals
4. Solution surface contour: PINN vs analytical BS vs error
5. Test set scatter plot with RMSE/MAE annotation

## Methodology Rules
- Report final-epoch RMSE, never best-epoch (avoids selection bias)
- Fixed epoch budget with cosine annealing, no early stopping
- μ selected on training dynamics (PDE dominance ratio), not test RMSE
- Walk-forward folds touched exactly once for reporting
- All paper claims require 9-fold walk-forward evaluation
- Architecture developed on single split (Jan-Oct/Nov-Dec); walk-forward
  folds are the independent evaluation

## Style
- Targeted code chunks over full file regeneration
- Explain reasoning before implementing
- State assumptions explicitly
- Present trade-offs honestly
```

### First Commands in Claude Code
```bash
# 1. Verify structure
ls CLAUDE.md src/__init__.py data/TSLA_2020_Split*.csv docs/BENCHMARKING_PLAN.md

# 2. Test imports
python -c "from src.model import PricingNet; print('imports work')"

# 3. Before running benchmarks, Claude Code needs to:
#    a. Add StandardPricingNet to src/model.py (see BENCHMARKING_PLAN.md)
#    b. Add --arch flag to run_walk_forward.py
#    c. Update default data path to data/TSLA_2020_Split_Adjusted.csv
#    d. Add bs_baseline_fold() function

# 4. Then run benchmarks (overnight):
python run_walk_forward.py --mode hybrid --epochs 12000 --rwf_mu 0.5
```

Read `docs/BENCHMARKING_PLAN.md` for the full 8-model × 9-fold plan and
the `StandardPricingNet` code to add.

---

## 1. Core Goal & Tech Stack

### Research Objective
Build a staged Physics-Informed Neural Network (PINN) framework for pricing TSLA 2020 call options that:
1. Embeds the Black-Scholes PDE as a physics constraint
2. Learns a volatility surface σ̂(m,τ) from market data (replacing constant σ)
3. Produces calibrated uncertainty estimates

The framework benchmarks against prior work: Black-Scholes (RMSE $17.23), GAM (RMSE $12.57, R²=62.2%), and Gaussian Process/laGP (RMSE $9.63, 96.26% coverage).

### Tech Stack
- **Language:** Python 3, PyTorch
- **Data:** TSLA 2020 call options — `TSLA_2020_Split_Adjusted.csv` (93,746 obs after early cleaning)
- **Notebook:** `stage0_pinn.ipynb` (Jupyter, all code in one file)
- **Presentation:** Overleaf / Beamer LaTeX for research proposal
- **Key references:** Wang et al. (2023) Expert Guide, Kazemian et al. (2025), Raissi et al. (2019), Thompson et al. (2025)

### File Structure (user's local machine)
```
Independent Study / PINN Code /
├── stage0_pinn.ipynb          # Main notebook (data prep + model + training)
├── TSLA_2020_Split_Adjusted.csv  # Input data
├── stage0_model_physics.pt    # Saved physics-mode checkpoint
├── TODO.md                    # Project roadmap
└── architecture_training_dynamics_summary.md  # Architecture discussion notes
```

---

## 2. Data Pipeline

### Source
`TSLA_2020_Split_Adjusted.csv` — already has: volume ≥ 50, NA removal, 5-for-1 split adjustment (Aug 31, 2020), calls only.

### Preprocessing (replicated from R QMD pipeline)
1. **1-day lagged daily mean implied volatility** — avoid look-ahead bias
2. **Triple Witching week exclusion** — 4 weeks in 2020 (Mar, Jun, Sep, Dec)
3. **Moneyness** = forward_price / strike_price, restricted to [0.65, 1.35]
4. **Near-expiry removal** — drop options with < 3 days to expiry
5. **Interest rate** — convert from percentage to decimal (/100)
6. **Derived features:** log_volume, v_hat_market (= mid_price/K), v_hat_bs, residual

### Train/Test Split
- **Train:** Jan 2020 – Oct 2020 (~50k obs)
- **Test:** Nov 2020 – Dec 2020 (held out, never seen during training)
- Walk-forward backtesting deferred to Stage 1

### Constants (from training set)
- **σ_fixed:** Median near-ATM implied vol (moneyness ∈ [0.95, 1.05]) — approximately 0.55-0.75
- **r_fixed:** Median risk-free rate from training data

---

## 3. PDE & Non-Dimensionalization

### Black-Scholes in (m, τ) coordinates
Working variables: m = S/K (moneyness), τ = T - t (time to maturity), v̂ = V/K (normalized price).

The BS PDE becomes:
```
∂v̂/∂τ = (1/2)σ²m² ∂²v̂/∂m² + r·m ∂v̂/∂m - r·v̂
```

Residual (should be zero):
```
R = ∂v̂/∂τ - (1/2)σ²m² ∂²v̂/∂m² - r·m ∂v̂/∂m + r·v̂
```

### Boundary/Terminal Conditions
- **Terminal (τ → 0):** v̂(m, 0) = max(m - 1, 0) — the call payoff
- **Lower boundary (m = m_min):** v̂ = BS analytical value (deep OTM ≈ 0)
- **Upper boundary (m = m_max):** v̂ = BS analytical value (deep ITM ≈ intrinsic)

### Analytical BS (for validation)
```python
def bs_call_normalized(m, tau, r, sigma):
    d1 = (log(m) + (r + 0.5*sigma²)*tau) / (sigma*sqrt(tau))
    d2 = d1 - sigma*sqrt(tau)
    return m*N(d1) - exp(-r*tau)*N(d2)
```

---

## 4. Network Architecture (Current)

### Modified MLP + Fourier Features + RWF

This is the Wang et al. (2023) recommended pipeline. Three components:

#### 4a. Fourier Feature Embedding
Maps (m, τ) → [sin(2π·B·x), cos(2π·B·x)] ∈ R^128 where B is a fixed 2×64 random matrix from N(0, scale²), scale=1.0. Mitigates spectral bias — lets the network represent the sharp payoff kink at m=1, τ=0.

```python
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_dim=2, n_features=64, scale=1.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * scale
        self.register_buffer("B", B)  # fixed, not trainable
        self.out_dim = 2 * n_features

    def forward(self, x):
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
```

#### 4b. Random Weight Factorization (RWF)
Replaces nn.Linear. Decomposes W = diag(exp(s)) · V where s ~ N(μ, σ²) is a per-neuron trainable scale. This gives each neuron an effective learning rate of η·(exp(s_k)² + ||v_k||²), enabling automatic specialization.

```python
class RWFLinear(nn.Module):
    def __init__(self, in_features, out_features, mu=1.0, sigma=0.1):
        super().__init__()
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.V)
        self.s = nn.Parameter(torch.normal(mu, sigma, size=(out_features,)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        W = torch.diag(torch.exp(self.s)) @ self.V
        return x @ W.T + self.bias
```

#### 4c. Modified MLP (the gating architecture)
Two encoders U, V transform the Fourier-embedded input once. Each hidden layer gates between them:

```
U = tanh(W_U · embedding + b_U)     # computed once
V = tanh(W_V · embedding + b_V)     # computed once

For each hidden layer l:
  f(l) = W(l) · g(l-1) + b(l)       # standard linear
  h(l) = tanh(f(l))                  # standard activation
  g(l) = h(l) ⊙ U + (1 - h(l)) ⊙ V # gating — INPUT INJECTED AT EVERY LAYER
```

This prevents BC/TC gradient vanishing by providing parallel gradient paths through U and V at every depth, rather than a serial chain of 4 tanh derivatives.

#### Full PricingNet class
```python
class PricingNet(nn.Module):
    def __init__(self, hidden_dims=None, fourier_features=64, fourier_scale=1.0,
                 rwf_mu=1.0, rwf_sigma=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        self.embedding = FourierFeatureEmbedding(in_dim=2, n_features=fourier_features, scale=fourier_scale)
        embed_dim = self.embedding.out_dim  # 128
        hidden = hidden_dims[0]             # 64

        self.encoder_U = RWFLinear(embed_dim, hidden, mu=rwf_mu, sigma=rwf_sigma)
        self.encoder_V = RWFLinear(embed_dim, hidden, mu=rwf_mu, sigma=rwf_sigma)

        self.hidden_layers = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden_dims:
            self.hidden_layers.append(RWFLinear(in_dim, h, mu=rwf_mu, sigma=rwf_sigma))
            in_dim = h

        self.output_layer = RWFLinear(hidden, 1, mu=rwf_mu, sigma=rwf_sigma)
        self.activation = nn.Tanh()

    def forward(self, m, tau):
        if m.dim() == 1: m = m.unsqueeze(1)
        if tau.dim() == 1: tau = tau.unsqueeze(1)
        x = torch.cat([m, tau], dim=1)
        x = self.embedding(x)
        U = self.activation(self.encoder_U(x))
        V = self.activation(self.encoder_V(x))
        for layer in self.hidden_layers:
            f = layer(x if layer is self.hidden_layers[0] else g)
            h = self.activation(f)
            g = h * U + (1 - h) * V
        return self.output_layer(g)
```

---

## 5. Loss Functions

### PDE Residual (via autodiff)
```python
def compute_pde_residual(model, m, tau, sigma, r):
    v = model(m, tau)
    dv_dm, dv_dtau = torch.autograd.grad(v, [m, tau], grad_outputs=torch.ones_like(v),
                                          create_graph=True, retain_graph=True)
    d2v_dm2 = torch.autograd.grad(dv_dm, m, grad_outputs=torch.ones_like(dv_dm),
                                   create_graph=True, retain_graph=True)[0]
    m_col = m.unsqueeze(1) if m.dim() == 1 else m
    return dv_dtau - 0.5*sigma**2*m_col**2*d2v_dm2 - r*m_col*dv_dm + r*v
```

### Composite Loss
**Physics mode:** L = λ_pde·L_pde + λ_tc·L_tc + λ_bc·L_bc
**Hybrid mode:** L = λ_pde·L_pde + λ_tc·L_tc + λ_bc·L_bc + λ_data·L_data

Where:
- L_pde = mean(R²) over collocation points
- L_tc = mean((v̂_pred - max(m-1,0))²) at τ ≈ 0
- L_bc = 0.5 * (MSE at m_min + MSE at m_max) against analytical BS
- L_data = mean((v̂_pred - v̂_market)²) over market training points

### Grad-Norm Adaptive Balancing (Wang et al. Algorithm 1)
Every 1000 steps: compute ||∇_θ L_i|| for each loss, then:
```
λ̂_i = (Σ_j g_j) / g_i
λ_i ← 0.9·λ_i + 0.1·λ̂_i    (EMA smoothing)
```

With a safety floor: λ_tc, λ_bc ≥ 10 (rarely active with Modified MLP).

---

## 6. Collocation Sampling
- 5000 points per iteration, resampled fresh each step
- 70% uniform over [m_min, m_max] × [τ_min, τ_max]
- 30% concentrated near payoff kink: m ~ N(1.0, 0.1), τ ~ Exp(0.1)
- Terminal condition: 2000 points linearly spaced along m at τ = 1e-4
- Boundary conditions: 1000 points each at m_min and m_max

---

## 7. Training Configuration

### Settled Configuration
```python
# Physics mode
MODE = "physics"
EPOCHS = 15_000
LR = 1e-3
rwf_mu = 1.0            # stable, monotonic improvement

# Hybrid mode
MODE = "hybrid"
EPOCHS = 12_000
LR = 1e-3
rwf_mu = 0.5            # faster PDE convergence → more market data learning

# Common
N_COLLOC_BATCH = 5000
N_MARKET_BATCH = 8000
GRAD_NORM_FREQ = 1000
GRAD_NORM_ALPHA = 0.9
rwf_sigma = 0.1
fourier_scale = 1.0
hidden_dims = [64, 64, 64, 64]
```

Optimizer: Adam, cosine annealing LR from LR to LR*0.01.
Gradient clipping: max_norm = 1.0.
**Report final-epoch model** — no early stopping, no epoch selection.

---

## 8. Experimental Results

### Physics Mode (single Nov–Dec split):

| Config | μ | LR | Epochs | BS RMSE (best) | Market RMSE | Drift? |
|--------|-----|------|--------|----------------|-------------|--------|
| Old arch (std MLP) | N/A | 1e-3 | 10000 | 0.007 (ep 2000) | $12.45 | Severe after ep 2000 |
| Mod MLP + RWF | 1.0 | 1e-3 | 10000 | 0.010 (ep 10000) | $9.26 | None — monotonic improvement |
| Mod MLP + RWF | 0.5 | 3e-3 | 15000 | 0.004 (ep 4000) | $11.65 | Yes — LR too high, drift after ep 6500 |

### Hybrid Mode (single Nov–Dec split):

| μ | Stable RMSE (late training) | Final epoch RMSE | Best epoch RMSE | Notes |
|---|----------------------------|-----------------|-----------------|-------|
| 0.5 | ~$8.80 | $8.79 @ ep 12000 | $8.75 @ ep 2000 | Transient spike ep 3000, recovers |
| 0.75 | ~$9.30 | $9.31 @ ep 12000 | $9.06 @ ep 5000 | Stable |
| 1.0 | ~$9.90 | $9.98 @ ep 15000 | $9.57 @ ep 7000 | PDE dominates too long |

### μ Parameter Selection Rationale
- **Physics mode: μ=1.0** — selected for training stability (PDE dominance ratio near 1 throughout)
- **Hybrid mode: μ=0.5** — selected because faster PDE convergence lets data loss become influential earlier
  - μ=0.5 PDE loss at epoch 2000: 0.77 (data loss competitive)
  - μ=1.0 PDE loss at epoch 2000: 123 (data loss irrelevant, still grinding PDE)
  - μ=0.5 gets ~10k epochs of market data learning vs ~7k for μ=1.0
- Selection based on **training dynamics**, not test RMSE — no data leakage
- Final-epoch RMSE reported (no early stopping) — no epoch selection bias

### Key Diagnostic: PDE Dominance Ratio
λ_pde·L_pde / (λ_tc·L_tc + λ_bc·L_bc) — should stay near 1.0.

| Epoch | Old Architecture | Modified MLP (μ=1.0) |
|------:|----------------:|-----------------------:|
| 500 | 2.6 | 0.3 |
| 2000 | 4.3 | 0.4 |
| 5000 | 4.6 | 0.4 |
| 7000 | 13.8 | 1.1 |
| 10000 | 10.8 | 31.0* |

*31.0 at ep 10000 reflects BC/TC at 10⁻⁸ (solved), not gradient failure — RMSE still improving.

### The Drift Problem (Old Architecture)
Standard MLP: input enters only at layer 1, BC/TC gradients chain through 4 tanh layers (attenuation ~0.5⁴ = 6%). By epoch 2000, BC/TC gradient norms collapse, grad-norm balancing can't compensate (large weight × zero gradient = zero). Network drifts to incorrect PDE solution.

Modified MLP fix: U and V encoders inject input at every layer via gating. Gradient has 4 parallel paths instead of 1 serial chain. BC/TC gradients stay healthy through entire training.

---

## 9. Staged Research Plan

### Stage 0: Constant-σ Baseline — COMPLETE ✓
- [x] Data pipeline, architecture (Modified MLP + RWF + Fourier)
- [x] Physics mode: μ=1.0 stable, RMSE 0.010 vs BS
- [x] Hybrid mode: μ=0.5 wins ($8.79 final, $8.75 best)
- [x] μ selection settled on training dynamics (no test leakage)
- [x] Code refactored into src/ modules

### Stage 1: Walk-Forward Backtesting — NEXT
- [ ] Run: `python run_walk_forward.py --mode hybrid --epochs 12000 --rwf_mu 0.5`
- [ ] Re-run GAM and laGP on same 9 folds
- [ ] Report final-epoch RMSE per fold (no epoch selection)

### Stage 2: Learnable Volatility — DESIGN SETTLED
Design decisions (all finalized):
- Separate Fourier embedding for vol net (lower scale ~0.5)
- Simple MLP + RWF for vol net (2×32, no Modified MLP)
- Bias initialization: output ≈ σ_fixed at epoch 0
- C-Vol (multiplicative) first, A-Vol (direct) as comparator
- σ₀ fixed at σ_fixed initially
- Warm-start pricing net from Stage 0 hybrid, vol net from scratch
- Differential LR: pricing 1e-4, vol 1e-3

Implementation:
- [ ] Add VolatilityNet to src/model.py
- [ ] Modify compute_pde_residual to accept optional vol_model
- [ ] Add regularization losses (smoothness for A-Vol, deviation for C-Vol)
- [ ] Modify run_training for parameter groups

### Stage 3: UQ
- [ ] Four combinations: AA, AC, CA, CC
- [ ] Heteroscedastic NLL or proportional parametric variance
- [ ] Calibration metrics vs GAM/GP

### Cross-Cutting
- [ ] All models on identical walk-forward folds (Stage 1+)
- [ ] Comparison table: BS → GAM → GP → PINN(const-σ) → PINN(σ̂) → PINN+UQ

---

## 10. Key Hyperparameters Reference

### Pricing Network
| Parameter | Value | Source/Rationale |
|-----------|-------|-----------------|
| Architecture | Modified MLP + RWF | Wang et al. full pipeline |
| Hidden layers | 4 × 64 | 2D problem; increase if needed |
| Activation | tanh | C∞ smooth, required for ∂²v/∂m² |
| Fourier features | 64 (→128 dim) | Scale 1.0, Wang et al. default |
| RWF μ | **0.5 (hybrid) / 1.0 (physics)** | Selected on training dynamics, not test RMSE |
| RWF σ | 0.1 | Wang et al. default |
| Learning rate | 1e-3 (Stage 0) / 1e-4 (Stage 2, warm-started) | |
| LR schedule | Cosine → LR×0.01 | Over full training duration |
| Grad-norm freq | 1000 steps | Wang et al. default |
| Grad-norm EMA α | 0.9 | Wang et al. default |
| Weight floor | 10 (TC, BC) | Safety net; rarely active |
| Collocation batch | 5000 | 70/30 uniform/kink split |
| Market batch | 8000 | Hybrid mode only |
| Gradient clip | 1.0 | Prevents instability |
| Epochs | **12k (hybrid) / 15k (physics)** | Fixed budget, report final epoch |

### Volatility Network (Stage 2)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | Simple MLP + RWF | No Modified MLP (vol surface is smooth) |
| Hidden layers | 2 × 32 | Smaller than pricing net |
| Fourier scale | 0.5 | Lower — encourages smoother surfaces |
| Learning rate | 1e-3 | Higher than pricing net (learning from scratch) |
| σ₀ (C-Vol) | σ_fixed | Fixed initially |

---

## 11. Known Issues & Decisions

1. **Physics-mode market RMSE is unreliable.** Same BS RMSE can give different market RMSE depending on μ and optimization trajectory. Only trust market RMSE in hybrid mode.

2. **LR > 1e-3 causes drift with Modified MLP + RWF.** μ=0.5/LR=3e-3 drifted after epoch 6500. Stick to LR=1e-3.

3. **μ selection differs by mode.** Physics: μ=1.0 (stability). Hybrid: μ=0.5 (faster PDE convergence → more market data learning). Selected on training dynamics, not test data.

4. **Report final-epoch RMSE, not best-epoch.** Fixed epoch budget eliminates epoch selection bias. For μ=0.5 hybrid, final ($8.79) ≈ best ($8.75) anyway.

5. **The weight floor (λ ≥ 10) was never active** in Modified MLP runs. Keep as cheap insurance.

6. **Hybrid mode differs by 3 lines of code:** loss_names gets "data", compute_individual_losses adds data MSE term, best-model metric switches to rmse_mkt.

7. **Walk-forward backtesting is the next immediate priority.** Single-split results are preliminary; all paper claims need 9-fold evaluation.

8. **Stage 2 design is fully settled.** C-Vol first (NSM-inspired, novel contribution), A-Vol as comparator. Separate Fourier embedding, simple architecture, bias initialization for stability. See TODO.md §2.2 for full rationale.

---

## 12. Immediate Next Steps

**STEP 0: ESTABLISH WALK-FORWARD BENCHMARKS (before any Stage 2 work)**
See `BENCHMARKING_PLAN.md` for the full plan. 8 models × 9 folds.

Part A (fast):
1. BS baseline (seconds)
2. GAM — adapt R code (minutes)
3. laGP — adapt R code (minutes)

Part B (PINN runs, ~16 hrs total, run overnight):
4. B1: Standard MLP, hybrid, 12k epochs — naive PINN baseline
5. B2: Modified MLP + RWF, physics, μ=1.0, 15k epochs
6. B3: Modified MLP + RWF, hybrid, μ=0.5, 12k epochs
7. B4: Modified MLP + RWF, hybrid, μ=0.75, 12k epochs
8. B5: Modified MLP + RWF, hybrid, μ=1.0, 15k epochs

Part C (analysis):
9. Compile master table, select μ for Stage 2 from walk-forward evidence
10. Save winning config's per-fold checkpoints for Stage 2 warm-start

**Only after the benchmark table exists:**
11. Implement Stage 2 C-Vol — add VolatilityNet to src/model.py, modify PDE residual
12. Single-split Stage 2 test — verify C-Vol training stability
13. Walk-forward Stage 2 — add column to benchmark table
14. Implement A-Vol as comparator
15. Stage 3 UQ — heteroscedastic head, calibration metrics

### Stage 2 Implementation Checklist (for Claude Code)
- [ ] `src/model.py`: Add `VolatilityNet` class (Fourier scale=0.5, 2×32, RWF, softplus output)
- [ ] `src/model.py`: Add `CVol` wrapper (μ·σ₀²) and `AVol` wrapper (direct softplus)
- [ ] `src/losses.py`: Modify `compute_pde_residual` to accept optional `vol_model`
- [ ] `src/losses.py`: Add `compute_multiplier_reg(vol_model, m, tau)` for C-Vol
- [ ] `src/losses.py`: Add `compute_smoothness_reg(vol_model, m, tau)` for A-Vol
- [ ] `src/training.py`: Add `run_training_stage2()` with parameter groups (pricing LR=1e-4, vol LR=1e-3)
- [ ] `src/diagnostics.py`: Add `plot_vol_surface(vol_model, config)` for learned σ̂ visualization
- [ ] `run_stage2.py`: Top-level script analogous to run_stage0.py

---

## 13. Key References

All PDFs should be in `references/` directory. The three most important:
- `AN_EXPERT_S_GUIDE_TO_TRAINING_PHYSICSINFORMED_NEURAL_NETWORKS.pdf` — Wang et al. (architecture, training)
- `Methodology___Framework_proposal2.pdf` — Our framework proposal (Stages 2–3 design)
- `Uncertainity_Aware_Pinn_for_Option_Pricing.pdf` — Kazemian et al. (UQ, Stage 3)

See Section 0 file manifest for the complete list.
