"""
Black-Scholes analytical formulas for validation and boundary conditions.
"""

import numpy as np
from scipy.stats import norm


def bs_call_price(S, K, T, r, sigma):
    """
    Vectorized BS European call price in dollar space.
    Used for computing residuals against market data.
    """
    T = np.maximum(T, 1e-4)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_normalized(m, tau, r, sigma):
    """
    BS call in non-dimensionalized (m, tau) coordinates.
    v_hat(m, tau) = V(S,t) / K  where m = S/K, tau = T - t.
    This is what the PINN learns to approximate.
    """
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(m) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return m * norm.cdf(d1) - np.exp(-r * tau) * norm.cdf(d2)
