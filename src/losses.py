"""
Loss functions for the PINN option pricing model.

PDE residual via autodiff, individual loss terms, grad-norm adaptive balancing.
Stage 2 extensions: learnable volatility surface (vol_model kwarg),
regularization losses (multiplier reg for C-Vol, smoothness reg for A-Vol).
"""

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


def compute_grad_norms(model, losses, extra_params=None):
    """
    Compute ||∇_θ L_i|| for each loss term.
    Used by the grad-norm balancing scheme.

    Args:
        extra_params: Additional parameters to include in the gradient norm
            computation (e.g. vol_model.parameters() in Stage 2). This
            ensures the balancing scheme sees gradients through ALL trainable
            networks. None for Stage 0/1 (backward compatible).
    """
    grad_norms = {}
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

    return grad_norms


def update_adaptive_weights(current_weights, grad_norms, alpha=0.9,
                            min_constraint_weight=10.0,
                            excluded_terms=None):
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

    # Safety floor on constraint weights (only when adapted)
    for name in ["tc", "bc"]:
        if name in new_weights and name not in excluded:
            new_weights[name] = max(new_weights[name], min_constraint_weight)

    return new_weights
