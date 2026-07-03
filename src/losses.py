"""
Loss functions for the PINN option pricing model.

PDE residual via autodiff, individual loss terms, grad-norm adaptive balancing.
Stage 2 extensions: learnable volatility surface (vol_model kwarg),
regularization losses (multiplier reg for C-Vol, smoothness reg for A-Vol).
W1 extensions: bounded balancers (Σλ-renormalized grad-norm, ReLoBRaLo).
"""

import math

import numpy as np
import torch


def compute_pde_residual(model, m, tau, sigma, r, vol_model=None):
    """
    BS PDE residual via automatic differentiation.

    R = ∂v̂/∂τ − (1/2)σ²m²·∂²v̂/∂m² − r·m·∂v̂/∂m + r·v̂

    Should be zero everywhere if v̂ satisfies the BS PDE.
    m and tau must have requires_grad=True.

    Args:
        vol_model: Optional learnable volatility surface (CVolWrapper or
            AVolWrapper). When provided, σ = vol_model(m, tau) is a
            graph-connected (N, 1) tensor so gradients flow into the vol
            net's parameters. When None, falls back to the scalar sigma
            (backward compatible with Stage 0/1).
    """
    v = model(m, tau)
    ones = torch.ones_like(v)

    dv_dm, dv_dtau = torch.autograd.grad(
        v, [m, tau], grad_outputs=ones,
        create_graph=True, retain_graph=True
    )

    d2v_dm2 = torch.autograd.grad(
        dv_dm, m, grad_outputs=torch.ones_like(dv_dm),
        create_graph=True, retain_graph=True
    )[0]

    # Ensure all derivative tensors match v's shape [N, 1] to prevent
    # broadcasting bugs. autograd returns the same shape as the input
    # (e.g. [N] if m is 1-D), but v and vol_model output are [N, 1].
    if dv_dm.dim() < v.dim():
        dv_dm = dv_dm.unsqueeze(-1)
        dv_dtau = dv_dtau.unsqueeze(-1)
    if d2v_dm2.dim() < v.dim():
        d2v_dm2 = d2v_dm2.unsqueeze(-1)

    m_col = m.unsqueeze(1) if m.dim() == 1 else m

    # σ²: either learned surface (Stage 2) or constant scalar (Stage 0/1).
    # For Stage 2 we use get_sigma_squared() when available — this avoids
    # computing sqrt(μ·σ₀²) in vol_model.forward() and then squaring it
    # back here, which is both wasteful and introduces a sqrt-gradient
    # singularity as μ → 0 (softplus output isn't bounded away from zero
    # during training). AVolWrapper has no shortcut so it falls through
    # to the explicit square.
    if vol_model is not None:
        if hasattr(vol_model, "get_sigma_squared"):
            sigma_sq = vol_model.get_sigma_squared(m, tau)   # (N, 1)
        else:
            sigma_val = vol_model(m, tau)
            sigma_sq = sigma_val ** 2
    else:
        sigma_sq = sigma ** 2           # scalar, backward compat

    residual = (dv_dtau
                - 0.5 * sigma_sq * m_col**2 * d2v_dm2
                - r * m_col * dv_dm
                + r * v)
    return residual


# ════════════════════════════════════════════════════════════════════
# Volatility regularization (Stage 2)
# ════════════════════════════════════════════════════════════════════


def compute_multiplier_reg(vol_model, m, tau):
    """
    C-Vol regularization: penalize multiplier μ deviations from 1.

    L_reg = E[(μ(m,τ) − 1)²]

    At initialization μ ≈ 1 everywhere (bias init), so L_reg ≈ 0.
    As training progresses the vol net may learn spatially varying μ;
    this term prevents unbounded deviation from σ₀.
    """
    mu = vol_model.get_multiplier(m, tau)
    return torch.mean((mu - 1.0) ** 2)


def compute_smoothness_reg(vol_model, m, tau,
                           alpha1=1.0, alpha2=1.0, alpha3=1.0):
    """
    A-Vol regularization: penalize σ̂ roughness via autodiff derivatives.

    L_smooth = α₁·E[(∂σ̂/∂m)²] + α₂·E[(∂σ̂/∂τ)²] + α₃·E[(∂²σ̂/∂m²)²]

    Encourages a smooth vol surface. The three α coefficients control
    the relative penalty on moneyness slope, term structure slope, and
    moneyness curvature respectively.
    """
    # Detach + re-enable grad so autodiff only tracks vol_model's graph
    m_r = m.detach().requires_grad_(True)
    tau_r = tau.detach().requires_grad_(True)
    sigma_hat = vol_model(m_r, tau_r)

    ones = torch.ones_like(sigma_hat)
    dsig_dm, dsig_dtau = torch.autograd.grad(
        sigma_hat, [m_r, tau_r], grad_outputs=ones,
        create_graph=True, retain_graph=True
    )
    d2sig_dm2 = torch.autograd.grad(
        dsig_dm, m_r, grad_outputs=torch.ones_like(dsig_dm),
        create_graph=True
    )[0]

    return (alpha1 * torch.mean(dsig_dm ** 2)
            + alpha2 * torch.mean(dsig_dtau ** 2)
            + alpha3 * torch.mean(d2sig_dm2 ** 2))


def compute_individual_losses(model, batch, sigma, r, mode="physics",
                              vol_model=None, vol_type=None):
    """
    Compute each loss term separately (needed for grad-norm balancing).

    Returns a dict of {name: loss_tensor} where each tensor retains its
    computation graph for individual gradient computation.

    Stage 2 extensions:
        vol_model: CVolWrapper or AVolWrapper instance (None for Stage 0/1)
        vol_type:  "cvol" or "avol" (selects regularization; None for Stage 0/1)
    """
    losses = {}

    # PDE residual
    m_c = batch["m_colloc"].requires_grad_(True)
    tau_c = batch["tau_colloc"].requires_grad_(True)
    residual = compute_pde_residual(model, m_c, tau_c, sigma, r,
                                    vol_model=vol_model)
    losses["pde"] = torch.mean(residual**2)

    # Terminal condition
    v_tc_pred = model(batch["m_tc"], batch["tau_tc"])
    losses["tc"] = torch.mean((v_tc_pred - batch["v_tc"].unsqueeze(1))**2)

    # Boundary conditions
    v_lo_pred = model(batch["m_bc_lo"], batch["tau_bc_lo"])
    v_hi_pred = model(batch["m_bc_hi"], batch["tau_bc_hi"])
    losses["bc"] = 0.5 * (
        torch.mean((v_lo_pred - batch["v_bc_lo"].unsqueeze(1))**2) +
        torch.mean((v_hi_pred - batch["v_bc_hi"].unsqueeze(1))**2)
    )

    # Data loss (hybrid mode only)
    if mode == "hybrid":
        v_data_pred = model(batch["train_m"], batch["train_tau"])
        losses["data"] = torch.mean((v_data_pred - batch["train_vhat"].unsqueeze(1))**2)

    # Volatility regularization (Stage 2 only)
    if vol_model is not None:
        if vol_type == "cvol":
            losses["reg"] = compute_multiplier_reg(vol_model, m_c, tau_c)
        elif vol_type == "avol":
            losses["reg"] = compute_smoothness_reg(vol_model, m_c, tau_c)

    return losses


def compute_grad_norms(model, losses, extra_params=None, flat_terms=None):
    """
    Compute ||∇_θ L_i|| for each loss term.
    Used by the grad-norm balancing scheme.

    Args:
        extra_params: Additional parameters to include in the gradient norm
            computation (e.g. vol_model.parameters() in Stage 2). This
            ensures the balancing scheme sees gradients through ALL trainable
            networks. None for Stage 0/1 (backward compatible).
        flat_terms: Optional iterable of loss-term NAMES for which to ALSO
            return a flat 1-D grad tensor (zeros where a param is unused by
            that loss), for the gradient-alignment diagnostic. Only the
            requested terms are materialized (the flat vectors are only
            needed pairwise, so building them for every term would be pure
            waste). When given, returns (grad_norms, flat_grads). Default
            None → dict of norms only (backward compatible with all v1
            call sites).
    """
    flat_wanted = set(flat_terms or ())
    grad_norms = {}
    flat_grads = {} if flat_wanted else None
    params = [p for p in model.parameters() if p.requires_grad]
    if extra_params is not None:
        params = params + [p for p in extra_params if p.requires_grad]

    for name, loss in losses.items():
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True
        )
        total_norm = 0.0
        for g in grads:
            if g is not None:
                total_norm += g.detach().pow(2).sum()
        grad_norms[name] = torch.sqrt(total_norm).item()
        if name in flat_wanted:
            flat_grads[name] = torch.cat([
                (g.detach().flatten() if g is not None
                 else torch.zeros(p.numel(), device=p.device))
                for g, p in zip(grads, params)
            ])

    if flat_wanted:
        return grad_norms, flat_grads
    return grad_norms


def grad_cosine(flat_grads, a="data", b="pde"):
    """
    Cosine similarity between two loss terms' full gradient vectors.

    W0 diagnostic for DIRECTIONAL conflict: the balancer corrects magnitude
    imbalance, but cos < 0 means the data and PDE objectives pull the
    parameters in opposing directions — a signature of the constant-σ ⟂
    smile model conflict that no weighting scheme can fix (motivates
    learnable σ, and gradient-surgery methods if persistent).

    Returns float in [−1, 1], or None if either term is absent/degenerate.
    """
    if flat_grads is None or a not in flat_grads or b not in flat_grads:
        return None
    ga, gb = flat_grads[a], flat_grads[b]
    na = torch.linalg.norm(ga)
    nb = torch.linalg.norm(gb)
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return None
    return float((torch.dot(ga, gb) / (na * nb)).item())


def update_adaptive_weights(current_weights, grad_norms, alpha=0.9,
                            min_constraint_weight=10.0,
                            excluded_terms=None,
                            renormalize=False):
    """
    Wang et al. Algorithm 1 (Eq 5.3): grad-norm balancing with EMA.

    λ̂_i = (Σ_j g_j) / g_i
    λ_new = α·λ_old + (1−α)·λ̂

    Includes weight floor on TC/BC as safety net.

    Args:
        excluded_terms: Optional set/list of loss-term names to KEEP at
            their current weight (not adapted, not included in the
            normalization sum). Use this to pin λ_data to a fixed value
            when running the "decouple data from balancing" ablation
            (Wang's Algorithm 1 was tested only with PDE+IC+BC, not with
            an additional supervised data term).
        renormalize: W1 "bounded grad-norm" control. When True, rescale the
            balanced weights after the EMA so Σλ = n_balanced (each averages
            1.0) — the simplex constraint from original GradNorm (Chen et
            al. 2018). No single λ can run away; if λ_data grows the others
            must shrink. The TC/BC floor is SKIPPED in this mode (a floor of
            10 is unsatisfiable when Σλ ≈ 4). Default False → identical to
            the v1 behaviour.
    """
    excluded = set(excluded_terms or ())
    # Sum over the terms that ARE being balanced — Wang's Eq 5.3 enforces
    # equality between weighted gradient norms among the balanced losses.
    # Excluded terms (e.g. fixed λ_data) shouldn't enter the sum because
    # they're not part of the equalization constraint.
    total_norm = sum(g for n, g in grad_norms.items() if n not in excluded)
    new_weights = {}

    for name in current_weights:
        if name in excluded:
            new_weights[name] = current_weights[name]   # frozen
            continue
        if name in grad_norms and grad_norms[name] > 1e-10:
            target = total_norm / grad_norms[name]
        else:
            target = current_weights[name]
        new_weights[name] = alpha * current_weights[name] + (1 - alpha) * target

    if renormalize:
        balanced = [n for n in new_weights if n not in excluded]
        s = sum(new_weights[n] for n in balanced)
        if s > 1e-12:
            scale = float(len(balanced)) / s
            for n in balanced:
                new_weights[n] *= scale
        return new_weights

    # Safety floor on constraint weights (only when adapted)
    for name in ["tc", "bc"]:
        if name in new_weights and name not in excluded:
            new_weights[name] = max(new_weights[name], min_constraint_weight)

    return new_weights


def update_weights_relobralo(current_weights, losses_now, state,
                             alpha=0.9, temperature=0.1, rho_p=0.99,
                             rng=None, excluded_terms=None):
    """
    ReLoBRaLo — Relative Loss Balancing with Random Lookback
    (Bischof & Kraus 2021, arXiv:2110.09813). W1 candidate replacing the
    unbounded Wang grad-norm scheme.

    Uses loss STATISTICS (progress ratios), not gradient norms — no extra
    backward passes — and is bounded by construction: each candidate weight
    vector is m·softmax(L_i(t) / (T·L_i(t'))), so weights live in (0, m)
    and sum to m. The λ_data → 10⁵ runaway is structurally impossible.

        λ̂_i(t; t') = m · softmax_i( L_i(t) / (T · L_i(t')) )
        λ_i(t)     = α·[ ρ·λ_i(t−1) + (1−ρ)·λ̂_i(t; 0) ] + (1−α)·λ̂_i(t; t−1)

    where t' = t−1 uses the previous update's losses (progress since last
    update), t' = 0 uses the losses at each term's FIRST appearance
    ("random lookback" — occasional resets toward a balanced start), and
    ρ ~ Bernoulli(rho_p) is drawn once per update, shared across terms.

    Notes for our setting:
      - Update cadence is every `grad_norm_freq` epochs (1000), not every
        step as in the paper, so α/ρ defaults here are per-update.
      - Terms are registered lazily: when the data loss switches on after
        the PDE-warmup, its init/prev entries are seeded with its first
        observed value (ratio 1 → neutral start).
      - No TC/BC floor (weights are bounded ≤ m; a floor of 10 is
        incompatible). The paper's benchmarks ran without floors.

    Args:
        current_weights: dict {name: λ} (mutated copy is returned).
        losses_now: dict {name: float} — current UNWEIGHTED loss values.
        state: dict carrying {"init": {...}, "prev": {...}} across calls.
            Pass the same dict every update; it is modified in place.
        rng: np.random.Generator for the Bernoulli draw (falls back to a
            fresh draw from numpy's default if None).
        excluded_terms: names kept at their current weight (e.g. pinned
            λ_data via --fixed_data_weight), excluded from the softmax.

    Returns:
        new_weights dict.
    """
    excluded = set(excluded_terms or ())
    init = state.setdefault("init", {})
    prev = state.setdefault("prev", {})

    # Lazily register terms on first appearance (handles warmup→hybrid).
    eps = 1e-12
    active = [n for n in losses_now if n not in excluded]
    for n in active:
        if n not in init:
            init[n] = max(float(losses_now[n]), eps)
            prev[n] = max(float(losses_now[n]), eps)

    m = len(active)
    new_weights = dict(current_weights)
    if m == 0:
        return new_weights

    def _softmax_weights(ref):
        """m · softmax(L_i(t) / (T · ref_i)), max-subtracted for stability."""
        z = [float(losses_now[n]) / (temperature * max(ref[n], eps)) for n in active]
        zmax = max(z)
        exps = [math.exp(v - zmax) for v in z]
        s = sum(exps)
        return {n: m * e / s for n, e in zip(active, exps)}

    lam_prev_upd = _softmax_weights(prev)   # λ̂(t; t−1): progress since last update
    lam_init     = _softmax_weights(init)   # λ̂(t; 0):   lookback to first-seen losses

    if rng is not None:
        rho = 1.0 if rng.random() < rho_p else 0.0
    else:
        rho = 1.0 if np.random.default_rng().random() < rho_p else 0.0

    for n in active:
        hist = rho * current_weights.get(n, 1.0) + (1.0 - rho) * lam_init[n]
        new_weights[n] = alpha * hist + (1.0 - alpha) * lam_prev_upd[n]

    # Record current losses for the next update's progress ratio.
    for n in active:
        prev[n] = max(float(losses_now[n]), eps)

    return new_weights
