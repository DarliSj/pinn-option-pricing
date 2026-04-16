"""
Non-PINN baselines evaluated on the same walk-forward folds.

Currently provides:
  - bs_baseline_fold: Black-Scholes with σ_fixed (median ATM IV from train).

GAM and laGP baselines are deferred — they will be re-fit in R against
the same fold definitions and imported as per-fold CSVs.
"""

import numpy as np

from .bs_formulas import bs_call_price
from .data import compute_constants


def bs_baseline_fold(train_df, test_df):
    """
    Black-Scholes baseline for one walk-forward fold.

    Per-fold protocol:
      1. Compute σ_fixed (median ATM IV) and r_fixed from training data only.
      2. Evaluate the BS analytical formula on every option in the test set.
      3. Report RMSE and MAE in dollars vs. market mid prices.

    No training, no fit — this is purely a constant-σ analytical pricer
    used as the lower bound for the benchmark table.

    Args:
        train_df: training-window DataFrame (already filtered/preprocessed)
        test_df:  test-window DataFrame (already filtered/preprocessed)

    Returns:
        dict with rmse_mkt, mae_mkt, sigma_fixed, r_fixed, n_train, n_test,
        plus sum_sq_err and sum_abs_err so the caller can compute a pooled
        RMSE / MAE across all folds (pooled_rmse = sqrt(Σ sum_sq_err / Σ n_test)).
    """
    config = compute_constants(train_df)

    bs_prices = bs_call_price(
        S=test_df["forward_price"].values,
        K=test_df["strike_price"].values,
        T=test_df["time_to_exp"].values,
        r=config["r_fixed"],
        sigma=config["sigma_fixed"],
    )

    market = test_df["mid_price"].values
    residuals = market - bs_prices
    sum_sq = float(np.sum(residuals ** 2))
    sum_abs = float(np.sum(np.abs(residuals)))
    n_test = len(test_df)

    return {
        "rmse_mkt": float(np.sqrt(sum_sq / n_test)),
        "mae_mkt": float(sum_abs / n_test),
        "sum_sq_err": sum_sq,
        "sum_abs_err": sum_abs,
        "sigma_fixed": config["sigma_fixed"],
        "r_fixed": config["r_fixed"],
        "n_train": len(train_df),
        "n_test": n_test,
    }
