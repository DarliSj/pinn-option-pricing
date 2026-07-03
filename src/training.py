"""
Training loop, batch sampling, validation, and checkpointing.
Encapsulates the full training pipeline as a callable function.
"""

import time
import numpy as np
import torch
from .bs_formulas import bs_call_normalized
from .model import PricingNet, StandardPricingNet
from .losses import (compute_individual_losses, compute_grad_norms,
                     update_adaptive_weights, update_weights_relobralo,
                     grad_cosine)


def make_batch(train_arrays, boundary_data, config, rng, device,
               n_colloc_batch=5000, n_market_batch=8000):
    """
    Sample a fresh mini-batch for one training iteration.
    Collocation points are resampled; TC/BC use all pre-computed points.
    """
    batch = {}
    m_min, m_max = config["m_min"], config["m_max"]
    tau_min, tau_max = config["tau_min"], config["tau_max"]

    # Fresh collocation (70/30 uniform/kink-concentrated)
    n_u = int(0.7 * n_colloc_batch)
    n_k = n_colloc_batch - n_u
    m_u = rng.uniform(m_min, m_max, n_u)
    tau_u = rng.uniform(tau_min, tau_max, n_u)
    m_k = rng.normal(1.0, 0.1, n_k).clip(m_min, m_max)
    tau_k = rng.exponential(0.1, n_k).clip(tau_min, tau_max)
    batch["m_colloc"] = torch.tensor(
        np.concatenate([m_u, m_k]), dtype=torch.float32).to(device)
    batch["tau_colloc"] = torch.tensor(
        np.concatenate([tau_u, tau_k]), dtype=torch.float32).to(device)

    # TC/BC from pre-computed arrays
    bd = boundary_data
    batch["m_tc"] = torch.tensor(bd["m_tc"]).to(device)
    batch["tau_tc"] = torch.tensor(bd["tau_tc"]).to(device)
    batch["v_tc"] = torch.tensor(bd["v_tc"]).to(device)
    batch["m_bc_lo"] = torch.tensor(bd["m_bc_lo"]).to(device)
    batch["tau_bc_lo"] = torch.tensor(bd["tau_bc"]).to(device)
    batch["v_bc_lo"] = torch.tensor(bd["v_bc_lo"]).to(device)
    batch["m_bc_hi"] = torch.tensor(bd["m_bc_hi"]).to(device)
    batch["tau_bc_hi"] = torch.tensor(bd["tau_bc"]).to(device)
    batch["v_bc_hi"] = torch.tensor(bd["v_bc_hi"]).to(device)

    # Market data subsample
    n_total = len(train_arrays["m"])
    n_batch = min(n_market_batch, n_total)
    idx = rng.choice(n_total, size=n_batch, replace=False)
    batch["train_m"] = torch.tensor(train_arrays["m"][idx]).to(device)
    batch["train_tau"] = torch.tensor(train_arrays["tau"][idx]).to(device)
    batch["train_vhat"] = torch.tensor(train_arrays["vhat"][idx]).to(device)

    return batch


@torch.no_grad()
def validate(model, test_arrays, sigma, r, device, vol_model=None,
             atm_band=(0.97, 1.03), spread_floor=0.05):
    """Compute eval-set metrics on the supplied arrays.

    Returns the historical RMSE/MAE block plus three Stage-2-ready add-ons:

      rmse_spread : sqrt(mean( ((p̂ − p_obs)/(spread/2))² )). Residuals
                    expressed in units of the half-spread. A vol surface
                    "inside the spread" everywhere has rmse_spread ≤ 1.
                    Spreads below `spread_floor` ($) are clipped to that
                    floor to avoid divide-by-near-zero blowups from stale
                    quotes.

      rmse_{otm,atm,itm} : RMSE vs market within moneyness bands defined
                    by `atm_band` (default (0.97, 1.03)). Diagnoses where
                    the surface is most inaccurate. Bands with zero rows
                    return NaN.

      sum_sq_err_spread, sum_sq_err_{otm,atm,itm}, n_{otm,atm,itm} :
                    pooling ingredients so the walk-forward summary can
                    compute pooled stratified RMSEs across folds.

    Stage 2: when vol_model is present, rmse_bs_norm is NaN (comparing
    against constant-σ BS is meaningless with a learned σ surface).
    """
    model.eval()
    if vol_model is not None:
        vol_model.eval()

    m_arr = test_arrays["m"]
    m_t   = torch.tensor(m_arr).to(device)
    tau_t = torch.tensor(test_arrays["tau"]).to(device)
    K_t   = torch.tensor(test_arrays["K"]).to(device)
    mid_t = torch.tensor(test_arrays["mid"]).to(device)

    v_pred = model(m_t, tau_t).squeeze()

    # vs analytical BS (normalized) — only meaningful with constant σ
    if vol_model is None:
        v_bs = torch.tensor(
            bs_call_normalized(test_arrays["m"], test_arrays["tau"], r, sigma),
            dtype=torch.float32
        ).to(device)
        rmse_bs = torch.sqrt(torch.mean((v_pred - v_bs)**2)).item()
    else:
        rmse_bs = float("nan")

    # vs market (dollar) — pooled
    price_pred = v_pred * K_t
    residuals_mkt = price_pred - mid_t
    sq = residuals_mkt ** 2
    sum_sq_err_mkt = torch.sum(sq).item()
    sum_abs_err_mkt = torch.sum(torch.abs(residuals_mkt)).item()
    n_test = int(mid_t.numel())
    rmse_mkt = float((sum_sq_err_mkt / n_test) ** 0.5)
    mae_mkt = float(sum_abs_err_mkt / n_test)
    # Median absolute percentage error — relative metric so dollar errors
    # are interpretable across TSLA's huge 2020 price range. Median (not
    # mean) is robust to the near-zero-mid denominator tail.
    medape_mkt = float(torch.median(
        torch.abs(residuals_mkt) / torch.clamp(torch.abs(mid_t), min=1e-6)
    ).item())

    # ── Spread-normalized RMSE ─────────────────────────────────────
    # z_i = (p̂_i − p_obs,i) / (spread_i / 2). Spread floor avoids
    # blowups on stale quotes.
    if "spread" in test_arrays:
        spread_t = torch.tensor(test_arrays["spread"]).to(device)
        half_spread = torch.clamp(spread_t * 0.5, min=float(spread_floor))
        z = residuals_mkt / half_spread
        sum_sq_err_spread = torch.sum(z ** 2).item()
        rmse_spread = float((sum_sq_err_spread / n_test) ** 0.5)
    else:
        sum_sq_err_spread = float("nan")
        rmse_spread = float("nan")

    # ── Moneyness-stratified RMSE ──────────────────────────────────
    lo, hi = atm_band
    otm_mask = m_t < lo
    atm_mask = (m_t >= lo) & (m_t <= hi)
    itm_mask = m_t > hi

    def _strat(mask):
        n = int(mask.sum().item())
        if n == 0:
            return float("nan"), 0.0, 0
        s = float(torch.sum(sq[mask]).item())
        return float((s / n) ** 0.5), s, n

    rmse_otm, ssq_otm, n_otm = _strat(otm_mask)
    rmse_atm, ssq_atm, n_atm = _strat(atm_mask)
    rmse_itm, ssq_itm, n_itm = _strat(itm_mask)

    model.train()
    if vol_model is not None:
        vol_model.train()

    return {
        "rmse_bs_norm":    rmse_bs,
        "rmse_mkt":        rmse_mkt,
        "mae_mkt":         mae_mkt,
        "medape_mkt":      medape_mkt,
        "sum_sq_err_mkt":  sum_sq_err_mkt,
        "sum_abs_err_mkt": sum_abs_err_mkt,
        "n_test":          n_test,
        # Stage 2 add-ons
        "rmse_spread":         rmse_spread,
        "sum_sq_err_spread":   sum_sq_err_spread,
        "rmse_otm":            rmse_otm,
        "rmse_atm":            rmse_atm,
        "rmse_itm":            rmse_itm,
        "sum_sq_err_otm":      ssq_otm,
        "sum_sq_err_atm":      ssq_atm,
        "sum_sq_err_itm":      ssq_itm,
        "n_otm":               n_otm,
        "n_atm":               n_atm,
        "n_itm":               n_itm,
    }




def _snapshot_state(model, vol_model=None):
    """Deep-clone state_dict tensors onto CPU for cheap val-best storage."""
    snap = {"model": {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}}
    if vol_model is not None:
        snap["vol_model"] = {k: v.detach().cpu().clone()
                             for k, v in vol_model.state_dict().items()}
    return snap


def _restore_state(snap, model, vol_model=None, device=None):
    """Restore model (and vol_model) from a snapshot dict."""
    state = snap["model"]
    if device is not None:
        state = {k: v.to(device) for k, v in state.items()}
    model.load_state_dict(state)
    if vol_model is not None and "vol_model" in snap:
        v_state = snap["vol_model"]
        if device is not None:
            v_state = {k: v.to(device) for k, v in v_state.items()}
        vol_model.load_state_dict(v_state)


def run_training(train_arrays, test_arrays, boundary_data, config,
                 mode="physics", epochs=15000, lr=1e-3,
                 arch="modified",
                 hidden_dims=None, fourier_features=64, fourier_scale=1.0,
                 rwf_mu=1.0, rwf_sigma=0.1,
                 n_colloc_batch=5000, n_market_batch=8000,
                 grad_norm_freq=1000, grad_norm_alpha=0.9,
                 log_every=500, val_every=500,
                 seed=42, device=None, verbose=True,
                 vol_model=None, vol_type=None,
                 pricing_lr=None, vol_lr=None,
                 checkpoint_path=None,
                 val_arrays=None,
                 track_test_curve=False,
                 fixed_data_weight=None,
                 data_loss_warmup=0,
                 balancer="gradnorm",
                 relobralo_temperature=0.1,
                 relobralo_rho=0.99):
    """
    Full training pipeline. Returns model, history, run_info.

    Args:
        train_arrays: dict from df_to_arrays(train_df)
        test_arrays: dict from df_to_arrays(test_df)
        boundary_data: dict from build_boundary_terminal()
        config: dict with sigma_fixed, r_fixed, domain bounds
        mode: "physics" or "hybrid"
        arch: "modified" (Modified MLP + RWF, default) or "standard"
              (plain MLP, no RWF — naive PINN baseline B1). When
              arch="standard", rwf_mu / rwf_sigma are ignored.
        ... (all other args are hyperparameters with sensible defaults)

        Stage 2 extensions:
        vol_model: CVolWrapper or AVolWrapper instance (None for Stage 0/1).
            Constructed and bias-initialized by the caller (run_stage2.py).
        vol_type: "cvol" or "avol" — selects regularization loss.
        pricing_lr: Override LR for the pricing net (e.g. 1e-4 for fine-tune).
            When None, falls back to `lr`.
        vol_lr: LR for the vol net (e.g. 1e-3, learning from scratch).
            When None, falls back to `lr`.
        checkpoint_path: Path to a Stage 1 .pt checkpoint for warm-starting
            the pricing net. The vol net is NOT loaded from checkpoint.
        val_arrays: Held-out validation set (dict from df_to_arrays).
            When supplied, the model is snapshotted whenever val rmse
            improves; after the loop the val-best snapshot is restored
            and test_arrays is evaluated ONCE on the restored model.
            That single eval populates the `final_*` fields in run_info
            — those are the official reported numbers. test_arrays is
            NEVER touched during the training loop.

            When None: no snapshotting; final-epoch metrics are reported
            (legacy single-phase behaviour, used by Stage 0).

        fixed_data_weight: When not None and mode=="hybrid", `λ_data` is
            pinned to this value and EXCLUDED from grad-norm balancing.
            Decouples supervised data fitting from the constraint-balancing
            machinery (Wang Algorithm 1 was only tested on PDE+IC+BC; with
            a 4th data term + RWF init the balancer can blow `λ_data` up
            to 10⁵+, swamping training). Typical value: 1.0 to 10.0.

        data_loss_warmup: For epochs ≤ this value, train as if mode=="physics"
            (no L_data term). After warmup, switch to the requested mode.
            Lets PDE/TC/BC find a stable basin before adding the noisy data
            signal. When > 0 and mode=="hybrid", λ_data is initialized to
            1.0 at warmup-end, then either adapted by grad-norm or held
            fixed (per `fixed_data_weight`).

        balancer: Loss-balancing scheme (workstream W1). One of:
            "gradnorm"        — v1 default: Wang et al. Eq 5.3 with EMA +
                                TC/BC floor. UNBOUNDED (the λ_data-runaway
                                defect lives here); kept as-is so completed
                                runs stay reproducible.
            "gradnorm_renorm" — same update renormalized to Σλ = n_terms
                                (bounded 1-line control; no TC/BC floor).
            "relobralo"       — bounded softmax of relative loss progress
                                (Bischof & Kraus 2021). Loss-statistics
                                based: no per-term backward passes.
            "fixed"           — no adaptation; weights stay at their init
                                (1.0 each; data pinnable via
                                fixed_data_weight). Robustness baseline.
        relobralo_temperature: ReLoBRaLo softmax temperature T.
        relobralo_rho: ReLoBRaLo Bernoulli lookback probability (drawn once
            per UPDATE — cadence here is grad_norm_freq epochs, not every
            step as in the paper).

    Returns:
        model: After training, restored to the val-best state (if val
            supplied) or final-epoch state (if not).
        history: dict with per-epoch loss/weight records plus val_epoch
            (eval cadence) and val_rmse_mkt/val_rmse_bs_norm (val curve).
            No test curve is recorded during training.
        run_info: dict with reported test metrics (post-restore) and
            best_val_epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hidden_dims is None:
        hidden_dims = [64, 64, 64, 64]
    valid_balancers = ("gradnorm", "gradnorm_renorm", "relobralo", "fixed")
    if balancer not in valid_balancers:
        raise ValueError(f"balancer must be one of {valid_balancers}, "
                         f"got {balancer!r}")

    # ── Build model ─────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    if arch == "modified":
        model = PricingNet(
            hidden_dims=hidden_dims,
            fourier_features=fourier_features,
            fourier_scale=fourier_scale,
            rwf_mu=rwf_mu,
            rwf_sigma=rwf_sigma,
        ).to(device)
    elif arch == "standard":
        model = StandardPricingNet(
            hidden_dims=hidden_dims,
            fourier_features=fourier_features,
            fourier_scale=fourier_scale,
        ).to(device)
    else:
        raise ValueError(f"arch must be 'modified' or 'standard', got {arch!r}")

    # ── Warm-start from Stage 1 checkpoint (Stage 2 only) ──────────
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if verbose:
            ckpt_mu = ckpt.get("run_info", {}).get("rwf_mu")
            if ckpt_mu is not None and ckpt_mu != rwf_mu:
                print(f"  WARNING: Checkpoint rwf_mu={ckpt_mu} != current rwf_mu={rwf_mu}")
            print(f"  Loaded warm-start pricing net from {checkpoint_path}")

    if vol_model is not None:
        vol_model = vol_model.to(device)

    # ── Optimizer (differential LR for Stage 2) ────────────────────
    pricing_lr_eff = pricing_lr if pricing_lr is not None else lr
    vol_lr_eff     = vol_lr     if vol_lr     is not None else lr
    if vol_model is not None:
        optimizer = torch.optim.Adam([
            {"params": model.parameters(),     "lr": pricing_lr_eff},
            {"params": vol_model.parameters(), "lr": vol_lr_eff},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Cosine annealing with PER-GROUP eta_min. Stock CosineAnnealingLR
    # only accepts a scalar eta_min, which makes the floor asymmetric
    # across groups with different base LRs (e.g. vol_lr=1e-3 and
    # pricing_lr=1e-4 with eta_min=1e-6 → vol decays 1000x while pricing
    # only 100x). Using LambdaLR with a multiplicative factor instead:
    # each group decays to base_lr * 0.01 (uniform 100x ratio).
    floor_ratio = 0.01
    def _cosine_factor(epoch):
        # epoch is 0-indexed by LambdaLR's contract
        if epoch >= epochs:
            return floor_ratio
        return floor_ratio + (1.0 - floor_ratio) * 0.5 * (
            1.0 + np.cos(np.pi * epoch / epochs)
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_factor)

    sigma = config["sigma_fixed"]
    r = config["r_fixed"]

    # ── Adaptive weights ────────────────────────────────────────────
    # `loss_names` covers every term that may appear at any point in
    # training (so history/weights have stable keys). When data_loss_warmup
    # is in effect, the data term is omitted from the loss dict for
    # epochs ≤ warmup (handled in the loop below).
    loss_names = ["pde", "tc", "bc"]
    if mode == "hybrid":
        loss_names.append("data")
    if vol_model is not None:
        loss_names.append("reg")
    adaptive_weights = {name: 1.0 for name in loss_names}
    # Pin λ_data when fixed_data_weight is requested
    if fixed_data_weight is not None and "data" in adaptive_weights:
        adaptive_weights["data"] = float(fixed_data_weight)
    # Names to KEEP at fixed weight (excluded from grad-norm rebalancing)
    excluded_from_balance = set()
    if fixed_data_weight is not None and "data" in adaptive_weights:
        excluded_from_balance.add("data")

    # ── History tracking ────────────────────────────────────────────
    # Only val curves are recorded. Test is NEVER touched in the loop —
    # it is evaluated exactly once after training, on the val-best
    # snapshot (or final-epoch model when val isn't supplied).
    history = {"epoch": [], "L_total": []}
    for name in loss_names:
        history[f"L_{name}"] = []
        history[f"w_{name}"] = []
    history["val_epoch"] = []
    history["val_rmse_mkt"] = []
    history["val_rmse_bs_norm"] = []
    # Diagnostic only: track test curve in training loop. Off by default
    # to preserve "test touched exactly once" invariant. ONLY enable for
    # post-hoc analysis (val-best vs final-epoch comparison).
    history["test_rmse_mkt_diagnostic"] = []
    # W0 diagnostic: gradient alignment cos(∇L_data, ∇L_pde) at the
    # balancer cadence. cos < 0 ⇒ DIRECTIONAL conflict (constant-σ ⟂ smile
    # signature) that no magnitude-balancing scheme can fix.
    history["grad_align_epoch"] = []
    history["grad_align_data_pde"] = []

    # Best-val tracking + snapshot. When val_arrays is supplied, we
    # snapshot the model whenever val rmse improves and restore that
    # snapshot before the final test eval.
    best_val_metric = float("inf")
    best_val_epoch  = 0
    best_state      = None
    use_val = val_arrays is not None
    rng_train = np.random.default_rng(seed)
    # Balancer state: ReLoBRaLo progress ratios + a SEPARATE RNG for its
    # Bernoulli lookback, so the batch-sampling stream (rng_train) is
    # byte-identical across balancer choices.
    relobralo_state = {}
    rng_balancer = np.random.default_rng(seed + 1)

    if verbose:
        n_pricing_params = sum(p.numel() for p in model.parameters())
        print(f"Training: mode={mode}, epochs={epochs}, lr={lr}")
        if arch == "modified":
            print(f"  Architecture: Modified MLP + RWF (μ={rwf_mu})")
        else:
            print(f"  Architecture: Standard MLP (Fourier + plain Linear, no RWF)")
        print(f"  Pricing net params: {n_pricing_params:,}")
        if vol_model is not None:
            n_vol_params = sum(p.numel() for p in vol_model.parameters())
            print(f"  Vol net params:     {n_vol_params:,} ({vol_type})")
            print(f"  Pricing LR: {pricing_lr_eff:.1e}  |  Vol LR: {vol_lr_eff:.1e}")
        print(f"  Loss terms: {loss_names}")
        if fixed_data_weight is not None:
            print(f"  λ_data PINNED at {fixed_data_weight:.2f} "
                  f"(excluded from grad-norm balancing)")
        if data_loss_warmup > 0:
            print(f"  Data loss WARMUP: epochs 1..{data_loss_warmup} run as physics-only")
        if balancer != "gradnorm":
            print(f"  Balancer: {balancer} (v1 default is 'gradnorm')")
        print("=" * 70)

    t0 = time.time()

    # ── Training loop ───────────────────────────────────────────────
    in_warmup_prev = data_loss_warmup > 0   # tracks transition out of warmup
    for epoch in range(1, epochs + 1):
        model.train()
        if vol_model is not None:
            vol_model.train()

        batch = make_batch(train_arrays, boundary_data, config, rng_train,
                           device, n_colloc_batch, n_market_batch)

        # Effective mode for this epoch: hold off on data loss during warmup
        in_warmup = (mode == "hybrid") and (epoch <= data_loss_warmup)
        mode_now = "physics" if in_warmup else mode

        # Detect warmup→hybrid transition. When data loss switches on,
        # reset λ_data to 1.0 (or fixed_data_weight) so grad-norm has a
        # sane starting point — otherwise it stays at the warmup-era value
        # of 1.0 forever (because update_adaptive_weights skips terms not
        # in grad_norms).
        if in_warmup_prev and not in_warmup:
            adaptive_weights["data"] = (float(fixed_data_weight)
                                        if fixed_data_weight is not None
                                        else 1.0)
            if verbose:
                print(f"  [warmup→hybrid] data loss activated at ep {epoch}; "
                      f"λ_data init={adaptive_weights['data']:.2f}")
        in_warmup_prev = in_warmup

        individual_losses = compute_individual_losses(
            model, batch, sigma, r, mode=mode_now,
            vol_model=vol_model, vol_type=vol_type,
        )

        # Balancer weight update (only over terms currently active AND not
        # excluded by fixed_data_weight). All schemes share the cadence.
        if epoch % grad_norm_freq == 0 or epoch == 1:
            extra_params = list(vol_model.parameters()) if vol_model is not None else None
            # Per-term gradients: consumed by the grad-norm family, and
            # DELIBERATELY also computed under relobralo/fixed (which don't
            # need them for weighting) so the W0 gradient-alignment
            # diagnostic cos(∇L_data, ∇L_pde) is logged uniformly across
            # ALL balancer arms — do not "optimize" this away for
            # non-gradnorm balancers. Cost: 4–5 extra backward passes per
            # grad_norm_freq epochs (~15 ticks/run) — negligible. Flat
            # vectors are materialized only for the two terms the
            # diagnostic reads.
            grad_norms, flat_grads = compute_grad_norms(
                model, individual_losses,
                extra_params=extra_params, flat_terms=("data", "pde"),
            )
            align = grad_cosine(flat_grads, a="data", b="pde")
            if align is not None:
                history["grad_align_epoch"].append(epoch)
                history["grad_align_data_pde"].append(align)

            if balancer in ("gradnorm", "gradnorm_renorm"):
                adaptive_weights = update_adaptive_weights(
                    adaptive_weights, grad_norms, alpha=grad_norm_alpha,
                    excluded_terms=excluded_from_balance,
                    renormalize=(balancer == "gradnorm_renorm"),
                )
            elif balancer == "relobralo":
                adaptive_weights = update_weights_relobralo(
                    adaptive_weights,
                    {n: float(l.item()) for n, l in individual_losses.items()},
                    relobralo_state,
                    alpha=grad_norm_alpha,
                    temperature=relobralo_temperature,
                    rho_p=relobralo_rho,
                    rng=rng_balancer,
                    excluded_terms=excluded_from_balance,
                )
            # balancer == "fixed": no adaptation — weights stay at init.

        # Weighted total loss
        loss_total = sum(
            adaptive_weights[name] * individual_losses[name]
            for name in individual_losses
        )

        # Backward + step. Clip pricing and vol gradients SEPARATELY so
        # one network's large gradients don't crush the other's signal.
        # (clip_grad_norm_ over a combined param list scales every group
        # by the same factor; if pricing |g|=100 and vol |g|=1, vol gets
        # multiplied by ~0.01 and effectively can't learn.)
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if vol_model is not None:
            torch.nn.utils.clip_grad_norm_(vol_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging — skip terms that aren't active this epoch (e.g. data
        # during warmup)
        if verbose and (epoch % log_every == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            parts = [f"Ep {epoch:5d}/{epochs}"]
            parts.append(f"L={loss_total.item():.2e}")
            for name in loss_names:
                if name in individual_losses:
                    parts.append(f"{name}={individual_losses[name].item():.2e}")
                    parts.append(f"w={adaptive_weights[name]:.2f}")
            parts.append(f"lr={lr_now:.1e}")
            print(" | ".join(parts))

        # Record history. Use NaN as placeholder when a loss term isn't
        # active for the current epoch (e.g. data during warmup).
        history["epoch"].append(epoch)
        history["L_total"].append(loss_total.item())
        for name in loss_names:
            if name in individual_losses:
                history[f"L_{name}"].append(individual_losses[name].item())
                history[f"w_{name}"].append(adaptive_weights[name])
            else:
                history[f"L_{name}"].append(float("nan"))
                history[f"w_{name}"].append(adaptive_weights[name])

        # Diagnostic: track test in training loop (OFF by default).
        # Used post-hoc to compare val-best vs test-best vs final-epoch.
        if track_test_curve and (epoch % val_every == 0 or epoch == epochs):
            with torch.no_grad():
                t_metrics = validate(model, test_arrays, sigma, r, device,
                                     vol_model=vol_model)
                history["test_rmse_mkt_diagnostic"].append(t_metrics["rmse_mkt"])

        # Validation — val only, NEVER test (unless track_test_curve diagnostic)
        if (epoch % val_every == 0 or epoch == epochs) and use_val:
            # Selection metric: rmse_mkt for hybrid + Stage 2 (the metric
            # that actually matters for benchmarking); rmse_bs_norm only
            # makes sense for pure physics with constant σ.
            key = "rmse_mkt" if vol_model is not None else (
                "rmse_bs_norm" if mode == "physics" else "rmse_mkt"
            )

            val_metrics = validate(model, val_arrays, sigma, r, device,
                                   vol_model=vol_model)
            history["val_epoch"].append(epoch)
            history["val_rmse_mkt"].append(val_metrics["rmse_mkt"])
            history["val_rmse_bs_norm"].append(val_metrics["rmse_bs_norm"])

            sel_metric = val_metrics[key]
            improved = sel_metric < best_val_metric
            if improved:
                best_val_metric = sel_metric
                best_val_epoch  = epoch
                best_state = _snapshot_state(model, vol_model)

            if verbose:
                tag = " ★" if improved else ""
                print(f"  [VAL] ep {epoch:5d}  val RMSE($): {val_metrics['rmse_mkt']:.4f}  "
                      f"| spread: {val_metrics['rmse_spread']:.3f}  "
                      f"| OTM/ATM/ITM: "
                      f"{val_metrics['rmse_otm']:.3f}/"
                      f"{val_metrics['rmse_atm']:.3f}/"
                      f"{val_metrics['rmse_itm']:.3f}{tag}")

    elapsed = time.time() - t0

    selection_key = "rmse_mkt" if vol_model is not None else (
        "rmse_bs_norm" if mode == "physics" else "rmse_mkt"
    )

    # ── Restore val-best snapshot (when val was supplied) ──────────────
    restored_to_best = False
    if use_val and best_state is not None:
        _restore_state(best_state, model, vol_model, device=device)
        restored_to_best = True
        if verbose:
            print(f"\nRestored model to val-best epoch {best_val_epoch} "
                  f"(val {selection_key}={best_val_metric:.6f})")

    # ── Single test eval — touch test_arrays exactly once, here ────────
    final_test = validate(model, test_arrays, sigma, r, device,
                          vol_model=vol_model)

    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")
        report_provenance = ("val-best" if restored_to_best else "final-epoch")
        print(f"REPORTED ({report_provenance}) test RMSE($): "
              f"{final_test['rmse_mkt']:.4f}  "
              f"| spread: {final_test['rmse_spread']:.3f}  "
              f"| OTM/ATM/ITM: "
              f"{final_test['rmse_otm']:.3f}/"
              f"{final_test['rmse_atm']:.3f}/"
              f"{final_test['rmse_itm']:.3f}")

    run_info = {
        "mode": mode,
        "arch": arch,
        "epochs": epochs,
        "lr": lr,
        "rwf_mu": rwf_mu if arch == "modified" else None,
        "rwf_sigma": rwf_sigma if arch == "modified" else None,
        "hidden_dims": hidden_dims,
        "elapsed_seconds": elapsed,
        # Stage 2 fields (None for Stage 0/1)
        "vol_type": vol_type,
        "pricing_lr": pricing_lr,
        "vol_lr": vol_lr,
        "checkpoint_source": str(checkpoint_path) if checkpoint_path else None,
        # Loss-balancing ablations (None / 0 means default Wang Algorithm 1)
        "fixed_data_weight": fixed_data_weight,
        "data_loss_warmup":  data_loss_warmup,
        # ── Selection / reporting policy ───────────────────────────────
        "selection_key":     selection_key,
        "used_val":          use_val,
        "restored_to_best":  restored_to_best,
        "best_val_epoch":    best_val_epoch if use_val else None,
        "best_val_metric":   best_val_metric if use_val else None,
        # ── Reported test metrics (val-best snapshot, single eval) ─────
        "final_rmse_bs_norm": final_test["rmse_bs_norm"],
        "final_rmse_mkt":     final_test["rmse_mkt"],
        "final_mae_mkt":      final_test["mae_mkt"],
        "final_medape_mkt":   final_test["medape_mkt"],
        "final_rmse_spread":  final_test["rmse_spread"],
        "final_rmse_otm":     final_test["rmse_otm"],
        "final_rmse_atm":     final_test["rmse_atm"],
        "final_rmse_itm":     final_test["rmse_itm"],
        "n_otm": final_test["n_otm"],
        "n_atm": final_test["n_atm"],
        "n_itm": final_test["n_itm"],
        # Pooling ingredients across folds (pooled RMSE = sqrt(Σ ssq / Σ n))
        "final_sum_sq_err_mkt":    final_test["sum_sq_err_mkt"],
        "final_sum_abs_err_mkt":   final_test["sum_abs_err_mkt"],
        "final_sum_sq_err_spread": final_test["sum_sq_err_spread"],
        "final_sum_sq_err_otm":    final_test["sum_sq_err_otm"],
        "final_sum_sq_err_atm":    final_test["sum_sq_err_atm"],
        "final_sum_sq_err_itm":    final_test["sum_sq_err_itm"],
        "n_test":                  final_test["n_test"],
        "adaptive_weights": adaptive_weights,
        "config": config,
    }

    return model, history, run_info
