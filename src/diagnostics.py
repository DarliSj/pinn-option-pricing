"""
Diagnostic plots and visualization functions for PINN training analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from .bs_formulas import bs_call_normalized


def plot_training_summary(history, loss_names, title_suffix=""):
    """Standard 3-panel training summary: losses, weights, validation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_h = history["epoch"]

    # Loss curves
    for name in loss_names:
        axes[0].semilogy(epochs_h, history[f"L_{name}"], label=f"L_{name}", alpha=0.8)
    axes[0].semilogy(epochs_h, history["L_total"], label="L_total", color="black", ls="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log)")
    axes[0].set_title("Training Losses")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Adaptive weights
    for name in loss_names:
        axes[1].plot(epochs_h, history[f"w_{name}"], label=f"λ_{name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weight")
    axes[1].set_title("Adaptive Loss Weights (Grad-Norm)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Validation
    val_ep = history["val_epoch"]
    axes[2].plot(val_ep, history["rmse_bs_norm"], "b.-", label="RMSE vs BS (norm)")
    ax2 = axes[2].twinx()
    ax2.plot(val_ep, history["rmse_mkt"], "r.-", label="RMSE vs Market ($)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("RMSE vs BS (norm)", color="blue")
    ax2.set_ylabel("RMSE vs Market ($)", color="red")
    axes[2].set_title("Validation Metrics")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Training Summary {title_suffix}", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_full_diagnostics(history, loss_names, title_suffix=""):
    """
    6-panel diagnostic: weighted losses, TC/BC raw, dominance ratio,
    validation, weight small scale, weight large scale.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs_h = history["epoch"]

    # 1. Weighted losses
    for name in loss_names:
        weighted = [w * l for w, l in zip(history[f"w_{name}"], history[f"L_{name}"])]
        axes[0, 0].semilogy(epochs_h, weighted, label=f"λ_{name} × L_{name}", alpha=0.8)
    axes[0, 0].set_title("Effective Loss Contributions")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Weighted Loss (log)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. TC/BC raw
    axes[0, 1].plot(epochs_h, history["L_tc"], label="L_tc", color="tab:orange")
    axes[0, 1].plot(epochs_h, history["L_bc"], label="L_bc", color="tab:green")
    axes[0, 1].set_title("TC & BC Loss (linear scale)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Dominance ratio
    pde_w = [w * l for w, l in zip(history["w_pde"], history["L_pde"])]
    tc_w = [w * l for w, l in zip(history["w_tc"], history["L_tc"])]
    bc_w = [w * l for w, l in zip(history["w_bc"], history["L_bc"])]
    ratio = [p / max(t + b, 1e-12) for p, t, b in zip(pde_w, tc_w, bc_w)]
    axes[0, 2].semilogy(epochs_h, ratio, color="crimson")
    axes[0, 2].axhline(1.0, color="gray", ls="--", lw=1, label="Balanced (ratio=1)")
    axes[0, 2].set_title("PDE Dominance Ratio")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("λ_pde·L_pde / (λ_tc·L_tc + λ_bc·L_bc)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Validation
    val_ep = history["val_epoch"]
    ax_bs = axes[1, 0]
    ax_mkt = ax_bs.twinx()
    ax_bs.plot(val_ep, history["rmse_bs_norm"], "b.-", label="RMSE vs BS (norm)")
    ax_mkt.plot(val_ep, history["rmse_mkt"], "r.-", label="RMSE vs Market ($)")
    ax_bs.set_xlabel("Epoch")
    ax_bs.set_ylabel("RMSE vs BS (norm)", color="blue")
    ax_mkt.set_ylabel("RMSE vs Market ($)", color="red")
    axes[1, 0].set_title("Validation: BS vs Market Accuracy")
    ax_bs.grid(True, alpha=0.3)

    # 5. Weights small scale
    for name in loss_names:
        if name != "bc":
            axes[1, 1].plot(epochs_h, history[f"w_{name}"], label=f"λ_{name}")
    axes[1, 1].set_title("Weights — Small Scale")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Weight")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. BC weight large scale
    axes[1, 2].plot(epochs_h, history[f"w_bc"], label="λ_bc", color="green")
    axes[1, 2].set_title("Weight — BC (Large Scale)")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Weight")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f"Training Diagnostics {title_suffix}", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_solution_surface(model, config, sigma, r, device, title_suffix=""):
    """PINN vs analytical BS contour plots + error."""
    model.eval()
    m_grid = np.linspace(config["m_min"], config["m_max"], 100)
    tau_grid = np.linspace(config["tau_min"], config["tau_max"], 100)
    M_g, TAU_g = np.meshgrid(m_grid, tau_grid)
    m_flat = M_g.flatten().astype(np.float32)
    tau_flat = TAU_g.flatten().astype(np.float32)

    with torch.no_grad():
        v_pred = model(
            torch.tensor(m_flat).to(device),
            torch.tensor(tau_flat).to(device)
        ).squeeze().cpu().numpy()

    v_bs_grid = bs_call_normalized(m_flat, tau_flat, r, sigma)
    v_err = v_pred - v_bs_grid

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    c0 = axes[0].contourf(M_g, TAU_g, v_pred.reshape(M_g.shape), levels=30, cmap="viridis")
    axes[0].set_title("PINN: v̂(m, τ)")
    axes[0].set_xlabel("Moneyness m")
    axes[0].set_ylabel("τ (years)")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(M_g, TAU_g, v_bs_grid.reshape(M_g.shape), levels=30, cmap="viridis")
    axes[1].set_title("Analytical BS: v̂(m, τ)")
    axes[1].set_xlabel("Moneyness m")
    axes[1].set_ylabel("τ (years)")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(M_g, TAU_g, v_err.reshape(M_g.shape), levels=30, cmap="RdBu_r")
    axes[2].set_title("Error: PINN − BS")
    axes[2].set_xlabel("Moneyness m")
    axes[2].set_ylabel("τ (years)")
    plt.colorbar(c2, ax=axes[2])

    plt.suptitle(f"Solution Surface {title_suffix}", fontsize=13, y=1.02)
    plt.tight_layout()

    print(f"Max absolute error (PINN vs BS, normalized): {np.max(np.abs(v_err)):.6f}")
    return fig


def plot_test_scatter(model, test_arrays, mode, device, title_suffix=""):
    """Test set scatter: PINN predicted price vs market mid price."""
    model.eval()
    with torch.no_grad():
        m_t = torch.tensor(test_arrays["m"]).to(device)
        tau_t = torch.tensor(test_arrays["tau"]).to(device)
        v_pred = model(m_t, tau_t).squeeze().cpu().numpy()

    price_pred = v_pred * test_arrays["K"]
    mid_test = test_arrays["mid"]

    rmse = np.sqrt(np.mean((price_pred - mid_test)**2))
    mae = np.mean(np.abs(price_pred - mid_test))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mid_test, price_pred, alpha=0.1, s=5, c="steelblue")
    lims = [0, max(mid_test.max(), price_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1, label="Perfect")
    ax.set_xlabel("Market Mid Price ($)")
    ax.set_ylabel("PINN Predicted Price ($)")
    ax.set_title(f"Test Set: PINN ({mode}) vs Market {title_suffix}")
    ax.text(0.05, 0.92, f"RMSE = ${rmse:.2f}\nMAE = ${mae:.2f}",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()

    return fig, {"rmse": rmse, "mae": mae}


# ════════════════════════════════════════════════════════════════════
# Stage 2: Volatility Surface Diagnostics
# ════════════════════════════════════════════════════════════════════


def plot_vol_surface(vol_model, config, device, sigma_fixed,
                     title_suffix="", is_cvol=False):
    """
    Learned volatility surface: σ̂(m, τ), deviation from σ_fixed,
    and (C-Vol only) multiplier μ(m, τ).

    Panels:
      Left:   contour of σ̂(m, τ)
      Center: contour of σ̂(m, τ) − σ_fixed
      Right:  (C-Vol) contour of μ(m, τ); (A-Vol) blank or |∇σ̂|
    """
    vol_model.eval()
    m_grid = np.linspace(config["m_min"], config["m_max"], 100)
    tau_grid = np.linspace(config["tau_min"], config["tau_max"], 100)
    M_g, TAU_g = np.meshgrid(m_grid, tau_grid)
    m_flat = torch.tensor(M_g.flatten(), dtype=torch.float32).to(device)
    tau_flat = torch.tensor(TAU_g.flatten(), dtype=torch.float32).to(device)

    with torch.no_grad():
        sigma_hat = vol_model(m_flat, tau_flat).squeeze().cpu().numpy()
        if is_cvol and hasattr(vol_model, "get_multiplier"):
            mu = vol_model.get_multiplier(m_flat, tau_flat).squeeze().cpu().numpy()
        else:
            mu = None

    sigma_dev = sigma_hat - sigma_fixed

    n_panels = 3 if is_cvol else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    # Panel 1: σ̂(m, τ)
    c0 = axes[0].contourf(M_g, TAU_g, sigma_hat.reshape(M_g.shape),
                           levels=30, cmap="viridis")
    axes[0].set_title("Learned σ̂(m, τ)")
    axes[0].set_xlabel("Moneyness m")
    axes[0].set_ylabel("τ (years)")
    plt.colorbar(c0, ax=axes[0])

    # Panel 2: deviation from constant
    vmax = max(abs(sigma_dev.min()), abs(sigma_dev.max()))
    c1 = axes[1].contourf(M_g, TAU_g, sigma_dev.reshape(M_g.shape),
                           levels=30, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"σ̂ − σ_fixed ({sigma_fixed:.4f})")
    axes[1].set_xlabel("Moneyness m")
    axes[1].set_ylabel("τ (years)")
    plt.colorbar(c1, ax=axes[1])

    # Panel 3: multiplier (C-Vol only)
    if is_cvol and mu is not None:
        c2 = axes[2].contourf(M_g, TAU_g, mu.reshape(M_g.shape),
                               levels=30, cmap="coolwarm")
        axes[2].set_title("Multiplier μ(m, τ)")
        axes[2].set_xlabel("Moneyness m")
        axes[2].set_ylabel("τ (years)")
        plt.colorbar(c2, ax=axes[2])

    plt.suptitle(f"Volatility Surface {title_suffix}", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_vol_slices(vol_model, config, device, sigma_fixed,
                    title_suffix=""):
    """
    Volatility smile and term structure slices.

    Left:  σ̂(m) for fixed τ values (the smile at different maturities)
    Right: σ̂(τ) for fixed m values (term structure at different moneyness)
    """
    vol_model.eval()
    m_grid = np.linspace(config["m_min"], config["m_max"], 200)
    tau_grid = np.linspace(config["tau_min"], config["tau_max"], 200)

    tau_slices = [0.05, 0.15, 0.30, 0.50]
    m_slices = [0.70, 0.85, 1.00, 1.15, 1.30]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: smile at fixed τ
    for tau_val in tau_slices:
        if tau_val < config["tau_min"] or tau_val > config["tau_max"]:
            continue
        m_t = torch.tensor(m_grid, dtype=torch.float32).to(device)
        tau_t = torch.full_like(m_t, tau_val)
        with torch.no_grad():
            sig = vol_model(m_t, tau_t).squeeze().cpu().numpy()
        axes[0].plot(m_grid, sig, label=f"τ={tau_val:.2f}")
    axes[0].axhline(sigma_fixed, color="gray", ls="--", lw=1, label=f"σ_fixed={sigma_fixed:.4f}")
    axes[0].set_xlabel("Moneyness m")
    axes[0].set_ylabel("σ̂(m, τ)")
    axes[0].set_title("Volatility Smile (fixed τ)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: term structure at fixed m
    for m_val in m_slices:
        if m_val < config["m_min"] or m_val > config["m_max"]:
            continue
        tau_t = torch.tensor(tau_grid, dtype=torch.float32).to(device)
        m_t = torch.full_like(tau_t, m_val)
        with torch.no_grad():
            sig = vol_model(m_t, tau_t).squeeze().cpu().numpy()
        axes[1].plot(tau_grid, sig, label=f"m={m_val:.2f}")
    axes[1].axhline(sigma_fixed, color="gray", ls="--", lw=1, label=f"σ_fixed={sigma_fixed:.4f}")
    axes[1].set_xlabel("τ (years)")
    axes[1].set_ylabel("σ̂(m, τ)")
    axes[1].set_title("Term Structure (fixed m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Volatility Slices {title_suffix}", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def print_epoch_summary(history, target_epochs=None):
    """Print diagnostic table at key epochs."""
    if target_epochs is None:
        target_epochs = [1, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000]

    epochs_h = history["epoch"]
    print(f"\n{'Epoch':>6} | {'L_pde':>10} | {'L_tc':>10} | {'L_bc':>10} | "
          f"{'wL_pde':>10} | {'wL_tc':>10} | {'wL_bc':>10} | {'ratio':>8}")
    print("-" * 90)

    for target_ep in target_epochs:
        if target_ep > len(epochs_h):
            continue
        i = target_ep - 1
        wl_pde = history["w_pde"][i] * history["L_pde"][i]
        wl_tc = history["w_tc"][i] * history["L_tc"][i]
        wl_bc = history["w_bc"][i] * history["L_bc"][i]
        r = wl_pde / max(wl_tc + wl_bc, 1e-12)
        print(f"{target_ep:>6} | {history['L_pde'][i]:>10.2e} | {history['L_tc'][i]:>10.2e} | "
              f"{history['L_bc'][i]:>10.2e} | {wl_pde:>10.2e} | {wl_tc:>10.2e} | "
              f"{wl_bc:>10.2e} | {r:>8.1f}")
