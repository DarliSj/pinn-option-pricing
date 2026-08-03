"""
PINN pricing network architecture + learnable volatility surface.

Pricing networks (Stage 0/1):
  - PricingNet: Modified MLP + RWF + Fourier features (Wang et al. 2023, §4.3 + §6.4)
  - StandardPricingNet: Plain MLP + Fourier features (naive baseline B0/B1)

Volatility networks (Stage 2):
  - VolatilityNet: Simple RWF MLP backbone (no Modified MLP — vol surface is smooth)
  - CVolWrapper: Multiplicative σ̂²(m,τ) = μ(m,τ)·σ₀² (NSM-inspired, primary)
  - AVolWrapper: Direct σ̂(m,τ) = softplus(NN) (standard comparator)
"""

import numpy as np
import torch
import torch.nn as nn


class RWFLinear(nn.Module):
    """
    Random Weight Factorization linear layer (Wang et al. §4.3, Algorithm 2).

    Instead of W, we train s and V where W = diag(exp(s)) · V.
    This gives each neuron its own adaptive learning rate:
      effective_lr_k ∝ (exp(s_k)² + ||v_k||²)
    """
    def __init__(self, in_features, out_features, mu=1.0, sigma=0.1):
        super().__init__()
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.V)
        self.s = nn.Parameter(torch.normal(mu, sigma, size=(out_features,)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        W = torch.diag(torch.exp(self.s)) @ self.V
        return x @ W.T + self.bias


class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier feature embedding for input coordinates.

    Maps x ∈ R^d → [sin(2π·B·x), cos(2π·B·x)] ∈ R^(2·n_features)
    where B is a fixed random matrix ~ N(0, scale²).
    """
    def __init__(self, in_dim=2, n_features=64, scale=1.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features

    def forward(self, x):
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PricingNet(nn.Module):
    """
    Modified MLP with Fourier features and RWF (Wang et al. §6.4 + §4.3).

    Two encoders U, V embed the input at every hidden layer via gating:
      g(l) = σ(f(l)) ⊙ U + (1 - σ(f(l))) ⊙ V

    This keeps input-coordinate gradients alive through all layers,
    preventing the BC/TC gradient vanishing problem.

    Architecture:
      Fourier(m,τ) → Encoder U, Encoder V (persistent)
                    → Modified hidden layers × N → Linear output
    """
    def __init__(self, hidden_dims=None, fourier_features=64, fourier_scale=1.0,
                 rwf_mu=1.0, rwf_sigma=0.1, n_inputs=2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        # n_inputs=3 → regime-conditioned pricing net v̂(m, τ, ν) (R1).
        # ν (lagged ATM IV, [~0.45, 1.4]) is on the same scale as m, so it
        # feeds the Fourier embedding directly. The B-matrix shape differs
        # from the 2-input net, so cross-loading checkpoints fails loudly.
        self.n_inputs = n_inputs
        self.embedding = FourierFeatureEmbedding(
            in_dim=n_inputs, n_features=fourier_features, scale=fourier_scale
        )
        embed_dim = self.embedding.out_dim
        hidden = hidden_dims[0]

        # Two input encoders (Wang et al. Eq. 6.7)
        self.encoder_U = RWFLinear(embed_dim, hidden, mu=rwf_mu, sigma=rwf_sigma)
        self.encoder_V = RWFLinear(embed_dim, hidden, mu=rwf_mu, sigma=rwf_sigma)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden_dims:
            self.hidden_layers.append(RWFLinear(in_dim, h, mu=rwf_mu, sigma=rwf_sigma))
            in_dim = h

        # Output layer
        self.output_layer = RWFLinear(hidden, 1, mu=rwf_mu, sigma=rwf_sigma)
        self.activation = nn.Tanh()

    def forward(self, m, tau, nu=None):
        if m.dim() == 1:
            m = m.unsqueeze(1)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)

        cols = [m, tau]
        if nu is not None:
            if nu.dim() == 1:
                nu = nu.unsqueeze(1)
            cols.append(nu)
        x = torch.cat(cols, dim=1)
        x = self.embedding(x)

        # Compute encoders once (Eq. 6.7)
        U = self.activation(self.encoder_U(x))
        V = self.activation(self.encoder_V(x))

        # Modified MLP forward pass (Eq. 6.8–6.9)
        for layer in self.hidden_layers:
            f = layer(x if layer is self.hidden_layers[0] else g)
            h = self.activation(f)
            g = h * U + (1 - h) * V

        return self.output_layer(g)


class StandardPricingNet(nn.Module):
    """
    Naive PINN baseline (B1 in BENCHMARKING_PLAN.md).

    Fourier feature embedding + plain MLP:
      Fourier(m,τ) → Linear → tanh → ... → Linear → v̂

    No RWF, no Modified MLP encoders. This is the original notebook
    architecture before the drift fix — it's expected to suffer from
    BC/TC gradient vanishing through the serial tanh chain. Used as a
    baseline to demonstrate the value of the architectural changes.
    """
    def __init__(self, hidden_dims=None, fourier_features=64, fourier_scale=1.0,
                 n_inputs=2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]

        self.n_inputs = n_inputs
        self.embedding = FourierFeatureEmbedding(
            in_dim=n_inputs, n_features=fourier_features, scale=fourier_scale
        )

        layers = []
        in_dim = self.embedding.out_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, m, tau, nu=None):
        if m.dim() == 1:
            m = m.unsqueeze(1)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        cols = [m, tau]
        if nu is not None:
            if nu.dim() == 1:
                nu = nu.unsqueeze(1)
            cols.append(nu)
        x = torch.cat(cols, dim=1)
        x = self.embedding(x)
        return self.net(x)


# ════════════════════════════════════════════════════════════════════
# Stage 2: Learnable Volatility Surface
# ════════════════════════════════════════════════════════════════════


class VolatilityNet(nn.Module):
    """
    Volatility surface backbone: simple RWF MLP with separate Fourier embedding.

    Architecture:
      FourierFeatureEmbedding(scale=0.5) → RWFLinear(128→32) → tanh
        → RWFLinear(32→32) → tanh → RWFLinear(32→1)

    Outputs the raw pre-activation z; the wrapper (CVolWrapper / AVolWrapper)
    applies the appropriate output activation (softplus) and scaling.

    Design rationale:
      - Separate Fourier embedding with lower scale (0.5 vs pricing's 1.0)
        because the vol surface is smoother than the price surface.
      - No Modified MLP encoders — the vol surface has no payoff kink, so
        the parallel gradient paths from U/V gating are unnecessary.
      - Small network (2×32 ≈ 5k params) to avoid overfitting the vol
        surface; the PDE residual provides implicit regularization.
    """
    def __init__(self, hidden_dims=None, fourier_features=64, fourier_scale=0.5,
                 rwf_mu=1.0, rwf_sigma=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]

        self.embedding = FourierFeatureEmbedding(
            in_dim=2, n_features=fourier_features, scale=fourier_scale
        )
        embed_dim = self.embedding.out_dim  # 2 * fourier_features = 128

        layers = nn.ModuleList()
        in_dim = embed_dim
        for h in hidden_dims:
            layers.append(RWFLinear(in_dim, h, mu=rwf_mu, sigma=rwf_sigma))
            in_dim = h
        self.hidden_layers = layers
        self.output_layer = RWFLinear(in_dim, 1, mu=rwf_mu, sigma=rwf_sigma)
        self.activation = nn.Tanh()

    def forward(self, m, tau):
        if m.dim() == 1:
            m = m.unsqueeze(1)
        if tau.dim() == 1:
            tau = tau.unsqueeze(1)
        x = torch.cat([m, tau], dim=1)
        x = self.embedding(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)  # raw pre-activation, shape (N, 1)

    def init_output_bias(self, target_pre_activation):
        """
        Initialize so the network output ≈ target_pre_activation everywhere.

        Zeros the output layer weights so the initial output is pure bias,
        regardless of Fourier feature activations. After a few gradient steps,
        the weights become non-zero and the network learns spatial variation.
        """
        with torch.no_grad():
            self.output_layer.bias.fill_(target_pre_activation)
            self.output_layer.V.zero_()
            # s stays at its initialized value; exp(s) @ 0 = 0 anyway


class CVolWrapper(nn.Module):
    """
    Multiplicative (NSM-inspired) volatility parameterization.

    σ̂²(m,τ) = μ(m,τ) · σ₀²

    where μ(m,τ) = softplus(vol_net(m,τ)) is the learned multiplier and
    σ₀ = σ_fixed from training data (frozen). The network learns
    *deviations* from σ₀ rather than absolute volatility levels.

    Advantages over AVolWrapper:
      - Uniform gradient scaling: ∂σ̂²/∂μ = σ₀² (constant across domain)
      - Single regularization hyperparameter: L_mult = mean((μ − 1)²)
      - Level-shape separation: σ₀ captures the level, μ captures the shape
      - Novel contribution (NSM transfer from Thompson et al. 2025)

    The σ₀ value is registered as a buffer (not a parameter) so it is:
      - Frozen during training (not in optimizer)
      - Saved/loaded with state_dict
      - Moved to device automatically
    """
    def __init__(self, vol_net, sigma_0):
        super().__init__()
        self.vol_net = vol_net
        self.register_buffer("sigma_0", torch.tensor(float(sigma_0)))
        # Initialize so μ ≈ 1 at startup → σ̂ ≈ σ₀
        # softplus(z) = 1 when z = log(exp(1) - 1) ≈ 0.5414
        self.vol_net.init_output_bias(float(np.log(np.exp(1.0) - 1.0)))

    def forward(self, m, tau):
        """Return σ̂(m, τ) as a tensor of shape (N, 1)."""
        mu = torch.nn.functional.softplus(self.vol_net(m, tau))
        return torch.sqrt(mu * self.sigma_0 ** 2)

    def get_sigma_squared(self, m, tau):
        """Return σ̂²(m, τ) = μ · σ₀² — avoids the sqrt for the PDE."""
        mu = torch.nn.functional.softplus(self.vol_net(m, tau))
        return mu * self.sigma_0 ** 2

    def get_multiplier(self, m, tau):
        """Return μ(m, τ) = softplus(z) — needed for regularization loss."""
        return torch.nn.functional.softplus(self.vol_net(m, tau))


class AVolWrapper(nn.Module):
    """
    Direct volatility parameterization (standard baseline comparator).

    σ̂(m,τ) = softplus(vol_net(m,τ))

    The network directly outputs the volatility surface. Softplus ensures
    σ̂ > 0 everywhere.

    Compared to CVolWrapper:
      - More flexible (no σ₀ anchor)
      - Non-uniform gradient scaling: ∂σ̂²/∂z ∝ σ̂ (biases learning toward wings)
      - Three regularization hyperparameters (smoothness: α₁, α₂, α₃)
      - Standard approach — serves as the baseline for the C-Vol comparison
    """
    def __init__(self, vol_net, sigma_init):
        super().__init__()
        self.vol_net = vol_net
        # Initialize so σ̂ ≈ σ_fixed at startup
        # softplus(z) = σ_init when z = log(exp(σ_init) - 1)
        target_bias = float(np.log(np.exp(float(sigma_init)) - 1.0))
        self.vol_net.init_output_bias(target_bias)

    def forward(self, m, tau):
        """Return σ̂(m, τ) as a tensor of shape (N, 1)."""
        return torch.nn.functional.softplus(self.vol_net(m, tau))
