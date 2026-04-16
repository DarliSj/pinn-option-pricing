"""
Training loop, batch sampling, validation, and checkpointing.
Encapsulates the full training pipeline as a callable function.
"""

import time
import numpy as np
import torch
from .bs_formulas import bs_call_normalized
from .model import PricingNet, StandardPricingNet
from .losses import compute_individual_losses, compute_grad_norms, update_adaptive_weights


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
def validate(model, test_arrays, sigma, r, device, vol_model=None):
    """Compute test-set metrics.

    Returns RMSE/MAE plus the per-fold sum-of-squared and sum-of-absolute
    errors needed to compute pooled RMSE/MAE across folds in the walk-
    forward summary (pooled_rmse = sqrt(Σ sum_sq_err / Σ n_test)).

    Stage 2: when vol_model is present, rmse_bs_norm is set to NaN because
    comparing against constant-σ BS is meaningless with a learned σ surface.
    """
    model.eval()
    if vol_model is not None:
        vol_model.eval()

    m_t = torch.tensor(test_arrays["m"]).to(device)
    tau_t = torch.tensor(test_arrays["tau"]).to(device)
    K_t = torch.tensor(test_arrays["K"]).to(device)
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

    # vs market (dollar)
    price_pred = v_pred * K_t
    residuals_mkt = price_pred - mid_t
    sum_sq_err_mkt = torch.sum(residuals_mkt ** 2).item()
    sum_abs_err_mkt = torch.sum(torch.abs(residuals_mkt)).item()
    n_test = int(mid_t.numel())
    rmse_mkt = float((sum_sq_err_mkt / n_test) ** 0.5)
    mae_mkt = float(sum_abs_err_mkt / n_test)

    model.train()
    if vol_model is not None:
        vol_model.train()

    return {
        "rmse_bs_norm":    rmse_bs,
        "rmse_mkt":        rmse_mkt,
        "mae_mkt":         mae_mkt,
        "sum_sq_err_mkt":  sum_sq_err_mkt,
        "sum_abs_err_mkt": sum_abs_err_mkt,
        "n_test":          n_test,
    }


def run_training(train_arrays, test_arrays, boundary_data, config,
                 mode="physics", epochs=15000, lr=1e-3,
                 arch="modified",
                 hidden_dims=None, fourier_features=64, fourier_scale=1.0,
                 rwf_mu=1.0, rwf_sigma=0.1,
                 n_colloc_batch=5000, n_market_batch=8000,
                 grad_norm_freq=1000, grad_norm_alpha=0.9,
                 log_every=500, val_every=1000,
                 seed=42, device=None, verbose=True,
                 vol_model=None, vol_type=None,
                 pricing_lr=None, vol_lr=None,
                 checkpoint_path=None):
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

    Returns:
        model: Trained model (final-epoch state, NOT restored to best)
        history: dict with per-epoch loss/weight/validation records
        run_info: dict with final-epoch metrics + best/drift diagnostics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hidden_dims is None:
        hidden_dims = [64, 64, 64, 64]

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
    if vol_model is not None:
        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": pricing_lr or lr},
            {"params": vol_model.parameters(), "lr": vol_lr or lr},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    sigma = config["sigma_fixed"]
    r = config["r_fixed"]

    # ── Adaptive weights ────────────────────────────────────────────
    loss_names = ["pde", "tc", "bc"]
    if mode == "hybrid":
        loss_names.append("data")
    if vol_model is not None:
        loss_names.append("reg")
    adaptive_weights = {name: 1.0 for name in loss_names}

    # ── History tracking ────────────────────────────────────────────
    history = {"epoch": [], "L_total": []}
    for name in loss_names:
        history[f"L_{name}"] = []
        history[f"w_{name}"] = []
    history["rmse_bs_norm"] = []
    history["rmse_mkt"] = []
    history["val_epoch"] = []

    # Stability diagnostics only — best is NEVER restored. The reported
    # benchmark number is the final-epoch metric. The best/final gap
    # quantifies late-training drift (gap ≈ 0 means stable; large gap
    # means the model peaked early and degraded).
    best_val = float("inf")
    best_epoch = 0
    best_metrics = None
    rng_train = np.random.default_rng(seed)

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
            print(f"  Pricing LR: {pricing_lr or lr:.1e}  |  Vol LR: {vol_lr or lr:.1e}")
        print(f"  Loss terms: {loss_names}")
        print("=" * 70)

    t0 = time.time()

    # ── Training loop ───────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        if vol_model is not None:
            vol_model.train()

        batch = make_batch(train_arrays, boundary_data, config, rng_train,
                           device, n_colloc_batch, n_market_batch)

        individual_losses = compute_individual_losses(
            model, batch, sigma, r, mode=mode,
            vol_model=vol_model, vol_type=vol_type,
        )

        # Grad-norm weight update
        if epoch % grad_norm_freq == 0 or epoch == 1:
            extra_params = list(vol_model.parameters()) if vol_model is not None else None
            grad_norms = compute_grad_norms(model, individual_losses,
                                            extra_params=extra_params)
            adaptive_weights = update_adaptive_weights(
                adaptive_weights, grad_norms, alpha=grad_norm_alpha
            )

        # Weighted total loss
        loss_total = sum(
            adaptive_weights[name] * individual_losses[name]
            for name in individual_losses
        )

        # Backward + step
        optimizer.zero_grad()
        loss_total.backward()
        all_params = list(model.parameters())
        if vol_model is not None:
            all_params += list(vol_model.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if verbose and (epoch % log_every == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            parts = [f"Ep {epoch:5d}/{epochs}"]
            parts.append(f"L={loss_total.item():.2e}")
            for name in loss_names:
                parts.append(f"{name}={individual_losses[name].item():.2e}")
                parts.append(f"w={adaptive_weights[name]:.2f}")
            parts.append(f"lr={lr_now:.1e}")
            print(" | ".join(parts))

        # Record history
        history["epoch"].append(epoch)
        history["L_total"].append(loss_total.item())
        for name in loss_names:
            history[f"L_{name}"].append(individual_losses[name].item())
            history[f"w_{name}"].append(adaptive_weights[name])

        # Validation
        if epoch % val_every == 0 or epoch == epochs:
            metrics = validate(model, test_arrays, sigma, r, device,
                               vol_model=vol_model)
            history["val_epoch"].append(epoch)
            history["rmse_bs_norm"].append(metrics["rmse_bs_norm"])
            history["rmse_mkt"].append(metrics["rmse_mkt"])

            # Stage 2: rmse_bs_norm is NaN → always use rmse_mkt
            key = "rmse_mkt" if vol_model is not None else (
                "rmse_bs_norm" if mode == "physics" else "rmse_mkt"
            )
            val = metrics[key]

            if verbose:
                print(f"  [VAL] RMSE vs BS (norm): {metrics['rmse_bs_norm']:.6f} "
                      f"| RMSE vs Market ($): {metrics['rmse_mkt']:.2f} "
                      f"| MAE vs Market ($): {metrics['mae_mkt']:.2f}")

            # Track best as a diagnostic — model state is NOT snapshotted
            # or restored. Reporting policy is final-epoch only.
            if val < best_val:
                best_val = val
                best_epoch = epoch
                best_metrics = dict(metrics)
                if verbose:
                    print(f"  ** New best {key}: {val:.6f} (diagnostic only)")

    elapsed = time.time() - t0

    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Reporting final-epoch state (best {key}={best_val:.6f} "
              f"@ ep {best_epoch} kept as stability diagnostic only)")

    # ── Collect run info ────────────────────────────────────────────
    # final_*: evaluated on the unmodified final-epoch model — these are
    # the numbers that go into the benchmark table. best_*: stability
    # diagnostic only, NOT used for model selection.
    final_metrics = validate(model, test_arrays, sigma, r, device,
                             vol_model=vol_model)
    selection_key = "rmse_mkt" if vol_model is not None else (
        "rmse_bs_norm" if mode == "physics" else "rmse_mkt"
    )
    drift_gap = final_metrics[selection_key] - best_val if best_metrics is not None else 0.0

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
        # Reporting metrics (final epoch — what enters the benchmark table)
        "final_rmse_bs_norm": final_metrics["rmse_bs_norm"],
        "final_rmse_mkt": final_metrics["rmse_mkt"],
        "final_mae_mkt": final_metrics["mae_mkt"],
        # Pooling ingredients (per-fold sum-of-errors so the walk-forward
        # summary can compute pooled RMSE = sqrt(Σ sum_sq / Σ n_test)).
        "final_sum_sq_err_mkt":  final_metrics["sum_sq_err_mkt"],
        "final_sum_abs_err_mkt": final_metrics["sum_abs_err_mkt"],
        "n_test":                final_metrics["n_test"],
        # Stability diagnostics only (not for selection)
        "selection_key": selection_key,
        "best_epoch": best_epoch,
        "best_rmse_bs_norm": best_metrics["rmse_bs_norm"] if best_metrics else None,
        "best_rmse_mkt": best_metrics["rmse_mkt"] if best_metrics else None,
        "drift_gap": drift_gap,
        "adaptive_weights": adaptive_weights,
        "config": config,
    }

    return model, history, run_info
