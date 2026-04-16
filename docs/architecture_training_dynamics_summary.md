# Architecture & Training Dynamics: Modified MLP, RWF, and the μ Parameter

## PINN Option Pricing Research — Session Summary

---

## 1. The Problem We Were Solving

Our Stage 0 PINN (constant-σ Black-Scholes) exhibited a critical training pathology: validation accuracy peaked early (epoch ~2000) and then degraded severely. The RMSE vs BS went from 0.007 at epoch 2000 to 0.077 by epoch 10000 — a 10× degradation — even though the total training loss kept decreasing.

Diagnostic analysis revealed the root cause: **gradient imbalance between loss terms.** The PDE residual loss, which involves second-order derivatives and operates over thousands of collocation points, produced gradient norms 75× larger than the boundary/terminal condition losses. The adaptive grad-norm balancing scheme tried to compensate by upweighting the BC/TC terms (λ_bc climbed to ~75), but this was multiplying a large weight by a near-zero gradient — the product remained negligible. The optimizer effectively "forgot" the boundary conditions that pin down the unique PDE solution, causing the network to drift toward an incorrect neighboring solution.

---

## 2. The Standard MLP Architecture (Before)

The original architecture processed inputs through a serial chain:

```
(m, τ) → Fourier(128) → Linear(128→64) → Tanh → Linear(64→64) → Tanh
       → Linear(64→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1) → v̂
```

The critical weakness: the input coordinates (m, τ) enter only at the first layer. By the time gradients from the BC/TC losses chain backward through four tanh activations, they are exponentially attenuated. Each tanh derivative satisfies tanh'(z) ≤ 1, and after four sequential multiplications the gradient reaching the early layers is roughly 0.5⁴ ≈ 6% of the output gradient. The PDE loss doesn't suffer as badly because it involves derivatives of the output with respect to the input (computed through the forward graph via autodiff), giving it a structurally different gradient path.

---

## 3. Modified MLP (Wang et al. §6.4)

The Modified MLP injects the input coordinates at every hidden layer through a gating mechanism. Two encoder networks first transform the Fourier-embedded input into two persistent 64-dimensional representations:

```
U = tanh(W_U · embedding + b_U)    — "view 1" of the input
V = tanh(W_V · embedding + b_V)    — "view 2" of the input
```

These are computed once per forward pass and reused at every layer. Each hidden layer then performs:

```
f(l) = W(l) · g(l-1) + b(l)           — standard linear transform
h(l) = tanh(f(l))                      — standard activation
g(l) = h(l) ⊙ U + (1 - h(l)) ⊙ V     — gating (NEW)
```

The gating operation blends the two input encodings at every layer. For each neuron k, when h(l,k) ≈ 1 the output selects U_k; when h(l,k) ≈ 0 it selects V_k. The network learns to blend differently at each neuron and each depth.

### Why This Fixes the Gradient Problem

The gradient of the output with respect to the encoder U now has parallel paths through every layer:

```
∂v̂/∂U = (∂v̂/∂g₄)·h₄ + (∂v̂/∂g₃)·h₃ + (∂v̂/∂g₂)·h₂ + (∂v̂/∂g₁)·h₁
```

Each term involves just one multiplicative factor (h_l) rather than a product of four sequential tanh derivatives. Even if one path is attenuated, the others contribute. The boundary condition gradient, which needs to flow from the output back to the input-dependent parameters, now has four parallel highways instead of one serial chain. This prevents the BC/TC gradient norms from collapsing to near-zero.

### Empirical Verification

The PDE dominance ratio (the key diagnostic) confirms this:

| Epoch | Standard MLP | Modified MLP |
|------:|-------------:|-------------:|
| 500   | 2.6          | 0.3          |
| 1000  | 3.4          | 0.7          |
| 2000  | 4.3          | 0.4          |
| 3000  | 5.1          | 0.3          |
| 5000  | 4.6          | 0.4          |
| 7000  | 13.8         | 1.1          |
| 10000 | 10.8         | 31.0*        |

*The ratio of 31 at epoch 10000 in the Modified MLP reflects BC/TC losses at 10⁻⁸ (genuinely solved), not gradient failure. The validation RMSE was still improving, unlike the standard MLP where ratio > 10 coincided with RMSE degradation.

---

## 4. Random Weight Factorization (Wang et al. §4.3)

RWF replaces each standard weight matrix W with a factorized form:

```
W(l) = diag(exp(s(l))) · V(l)
```

where s(l) is a per-neuron trainable scale vector initialized as s ~ N(μ, σ²), and V(l) is the direction matrix initialized with Xavier. Both s and V are optimized jointly.

### Per-Neuron Adaptive Learning Rate

Wang et al. Theorem B.2 shows the effective gradient update becomes:

```
w_k ← w_k − η · (exp(s_k)² + ||v_k||²) · ∂L/∂w_k
```

The factor (exp(s_k)² + ||v_k||²) acts as a per-neuron learning rate multiplier. Neurons that need to change a lot (e.g., to satisfy boundary conditions) can develop large scale factors, making them more responsive to gradients. Neurons focused on PDE accuracy adapt independently. This is automatic — the optimizer adjusts s alongside V during training.

### Loss Landscape Geometry

Theorem B.1 proves that in the factorized (s, V) parameter space, the distance between any initialization and any target minimum can be made arbitrarily small by choosing large enough scale factors. This means the optimizer has an easier path to the correct solution — it doesn't need to traverse as much of the loss landscape.

### Practical Implementation

RWF is a drop-in replacement for nn.Linear. The only change is storing (s, V) instead of W, with the forward pass computing W = diag(exp(s)) · V at each step. Recommended defaults: μ = 0.5 or 1.0, σ = 0.1. The exponential parameterization ensures scale factors are always positive.

---

## 5. The μ Parameter: Theory and Empirical Findings

The RWF initialization parameter μ controls the initial scale factors via s ~ N(μ, 0.1²), giving initial per-neuron scales of exp(μ). This simultaneously affects three aspects of training:

### 5.1 Initial Output Magnitude

| μ    | exp(μ)  | Initial PDE Loss | Epochs to BS RMSE 0.01 |
|------|---------|-----------------|------------------------|
| 0.5  | 1.65    | ~10⁵            | ~1000                  |
| 0.75 | 2.12    | ~10⁶ (est.)     | ~2000 (est.)           |
| 1.0  | 2.72    | ~10⁷            | ~9000                  |

Higher μ means larger initial weights, larger initial outputs, and much larger initial PDE losses. The network spends more epochs in the "rough sculpting" phase before reaching the accuracy regime.

### 5.2 Per-Neuron Learning Rate Amplification

The effective learning rate multiplier is (exp(s_k)² + ||v_k||²). At initialization:

- μ = 0.5: amplification ≈ 1.65² + ||v||² ≈ 3.7
- μ = 1.0: amplification ≈ 2.72² + ||v||² ≈ 8.4

Higher μ gives each neuron a larger effective learning rate, which sounds beneficial but combines with the larger initial loss to create chaotic early gradients.

### 5.3 Implicit Regularization from Path Length

The most subtle and consequential effect. Higher μ means more total gradient steps before reaching a given accuracy threshold. During these steps, the Adam optimizer's implicit regularization biases the solution toward flatter minima with more structured error patterns. The collocation distribution (30% concentrated near the ATM kink) shapes this structure — the optimizer spends disproportionate effort on the economically important region, pushing residual errors to the domain periphery.

### 5.4 The Market RMSE Paradox

In physics-only mode, the μ=1.0 network at BS RMSE 0.010 achieved market RMSE $9.26, while the μ=0.5 network at the same BS RMSE 0.010 achieved market RMSE $12.71. Same PDE accuracy, different market performance.

This occurs because BS RMSE is a domain-average metric that doesn't capture the spatial distribution of error. Two networks with identical average error can differ in where that error concentrates. The μ=1.0 network's longer optimization path (shaped by the collocation distribution) pushes errors away from the ATM/short-maturity region where BS has the largest systematic mispricing. These residual approximation errors partially cancel BS's volatility smile bias, producing accidentally better market prices.

This effect is an artifact of the optimization trajectory, not a learned market pattern. It varies with initialization, collocation distribution, and the specific structure of the underlying's mispricing. It should not be used as a basis for model selection in physics mode — BS RMSE is the correct metric. Market RMSE becomes meaningful only in hybrid mode where the network explicitly optimizes for it.

### 5.5 Implications for Hybrid Mode

The longer optimization path from higher μ builds more thoroughly developed internal representations before the data loss begins competing with the PDE loss. The Modified MLP encoders U and V, having been shaped by more diverse gradient signals from the PDE, provide richer input-space representations for the data loss to build upon. This suggests that for hybrid mode, the "slow start" from higher μ may produce better final market accuracy — the PDE scaffolding is more robust when the market data corrections begin.

This remains a hypothesis to be tested experimentally by comparing hybrid mode results across μ values.

---

## 6. Interaction Between Components

The three mechanisms (Modified MLP, RWF, grad-norm balancing) operate at different levels and are complementary:

| Component | Level | What it fixes |
|-----------|-------|---------------|
| Modified MLP | Architecture | Gradient flow — keeps BC/TC gradients alive through depth |
| RWF | Parameterization | Per-neuron learning rates — lets neurons specialize |
| Grad-norm balancing | Loss weighting | Cross-term balance — equalizes gradient contributions across loss types |
| Weight floor (λ ≥ 10) | Safety net | Prevents constraint weights from becoming negligible |

The Modified MLP is the most impactful single change — it addresses the root cause of gradient vanishing. RWF provides additional optimization benefits through the factorized parameter space. Grad-norm balancing works properly once the architecture provides healthy gradients. The weight floor is cheap insurance that proved unnecessary once the Modified MLP was in place (the computed weights were always far above 10).

---

## 7. Experimental Configurations Run

| Config | μ | LR | Epochs | BS RMSE (best) | Market RMSE (best) | Drift? |
|--------|---|------|--------|----------------|-------------------|--------|
| Old architecture | N/A | 1e-3 | 10000 | 0.007 (ep 2000) | $12.45 (ep 2000) | Severe after ep 2000 |
| Modified MLP + RWF | 1.0 | 1e-3 | 10000 | 0.010 (ep 10000) | $9.26 (ep 10000) | None |
| Modified MLP + RWF | 0.5 | 3e-3 | 15000 | 0.004 (ep 4000) | $11.65 (ep 4000) | Yes, after ep 6500 (LR too high) |
| Modified MLP + RWF | 0.75 | 1e-3 | 15000 | TBD | TBD | TBD |

The μ=1.0, LR=1e-3 configuration showed zero drift and monotonic improvement. The μ=0.5, LR=3e-3 configuration demonstrated that the drift was caused by the aggressive learning rate, not the architecture. The μ=0.75, LR=1e-3 configuration aims to balance fast early convergence with stable long-term training.

---

## 8. Key Takeaways

1. **Architecture matters more than loss weighting.** The Modified MLP's structural fix to gradient flow was more impactful than any tuning of the adaptive weighting scheme. No amount of weight adjustment can fix near-zero gradients.

2. **The same average accuracy can mean very different things.** BS RMSE is a single number averaging over the domain; the spatial error pattern determines real-world pricing quality. This makes physics-mode market RMSE an unreliable metric.

3. **Optimization trajectory shapes the solution.** Even when two networks converge to similar accuracy, the path taken (determined by initialization, learning rate, and training duration) selects different members from the family of approximate solutions. Longer paths through more gradient steps produce more structured representations.

4. **RWF initialization (μ) is not just a speed knob.** It fundamentally affects the optimization trajectory, the internal representations, and the spatial distribution of approximation errors. The "optimal" μ depends on whether you prioritize fast convergence (lower μ) or representation quality (higher μ).

5. **Conservative learning rates prevent late-training drift.** The Modified MLP + RWF architecture is robust to long training, but only when the learning rate decays sufficiently. A cosine schedule from 1e-3 to 1e-5 over the full training duration is the safe choice.
