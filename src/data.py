"""
Data loading, preprocessing, train/test splitting, and tensor construction.
Replicates the R pipeline from STA-325-option.qmd.
"""

import numpy as np
import pandas as pd
from .bs_formulas import bs_call_price, bs_call_normalized


# ── Triple Witching dates for 2020 ──────────────────────────────────
TRIPLE_WITCHING_2020 = [
    ("2020-03-16", "2020-03-20"),
    ("2020-06-15", "2020-06-19"),
    ("2020-09-14", "2020-09-18"),
    ("2020-12-14", "2020-12-18"),
]


def load_and_preprocess(csv_path, moneyness_range=(0.65, 1.35), min_tte_days=3):
    """
    Load split-adjusted CSV and apply all modeling-specific filters.

    Steps:
      1. Volume >= 50 and NA removal (safety — should already be done)
      2. 1-day lagged daily mean implied volatility
      3. Triple Witching week exclusion
      4. Moneyness filter
      5. Near-expiry removal
      6. Interest rate conversion (% → decimal)
      7. Derived features: log_volume, v_hat_market, v_hat_bs, residual

    Returns:
        df: Filtered and feature-engineered DataFrame
    """
    df = pd.read_csv(csv_path, parse_dates=["date", "exdate"])

    # Step 1: safety filters
    df = df[df["volume"] >= 50].copy()
    df = df.dropna(subset=["impl_volatility"])

    # Step 2: 1-day lagged daily mean IV
    daily_vol = (
        df.groupby("date")["impl_volatility"]
        .mean()
        .reset_index()
        .rename(columns={"impl_volatility": "todays_vol_mean"})
        .sort_values("date")
    )
    daily_vol["daily_vol_proxy"] = daily_vol["todays_vol_mean"].shift(1)
    daily_vol["daily_vol_proxy"] = daily_vol["daily_vol_proxy"].ffill().bfill()
    df = df.merge(daily_vol[["date", "todays_vol_mean", "daily_vol_proxy"]],
                  on="date", how="left")

    # Step 3: Triple Witching exclusion
    for start, end in TRIPLE_WITCHING_2020:
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        df = df[~mask]

    # Step 4: Moneyness
    df["moneyness"] = df["forward_price"] / df["strike_price"]
    df = df[(df["moneyness"] >= moneyness_range[0]) &
            (df["moneyness"] <= moneyness_range[1])].copy()

    # Step 5: Near-expiry removal
    df = df[df["time_to_exp"] > (min_tte_days / 365)].copy()

    # Step 6: Interest rate conversion
    df["interest_rate"] = df["interest_rate"] / 100.0

    # Step 7: Derived features
    df["log_volume"] = np.log(df["volume"])
    df["bs_theoretical"] = bs_call_price(
        S=df["forward_price"].values, K=df["strike_price"].values,
        T=df["time_to_exp"].values, r=df["interest_rate"].values,
        sigma=df["daily_vol_proxy"].values,
    )
    df["residual"] = df["mid_price"] - df["bs_theoretical"]
    df["v_hat_market"] = df["mid_price"] / df["strike_price"]
    df["v_hat_bs"] = df["bs_theoretical"] / df["strike_price"]

    return df


def make_temporal_split(df, split_date="2020-11-01"):
    """Simple temporal train/test split."""
    train_df = df[df["date"] < pd.Timestamp(split_date)].copy()
    test_df = df[df["date"] >= pd.Timestamp(split_date)].copy()
    return train_df, test_df


def make_fold(df, train_end, test_start, test_end, val_months=1, train_start=None):
    """
    Create a single walk-forward fold with a held-out validation window.

    The validation window is carved as the LAST `val_months` months of the
    training window (i.e. the most recent slice of train). This gives an
    out-of-sample-yet-pre-test set for honest best-epoch model selection
    without contaminating the test set.

    Args:
        df: Full preprocessed DataFrame
        train_end: Last date in training window (exclusive)
        test_start: First date in test window (inclusive)
        test_end: Last date in test window (exclusive)
        val_months: Number of months at the tail of train to use as validation
            (default 1). Set to 0 to disable val (val_df returned empty).
        train_start: First date in training window (inclusive), or None for
            expanding window.

    Returns:
        (train_df, val_df, test_df) — three disjoint frames, all copies.
    """
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Full training window
    if train_start is not None:
        full_train_mask = ((df["date"] >= pd.Timestamp(train_start)) &
                           (df["date"] < train_end_ts))
    else:
        full_train_mask = df["date"] < train_end_ts

    # Validation = last `val_months` months of train
    if val_months > 0:
        val_start_ts = train_end_ts - pd.DateOffset(months=val_months)
        val_mask = full_train_mask & (df["date"] >= val_start_ts)
        train_mask = full_train_mask & (df["date"] < val_start_ts)
    else:
        val_mask = pd.Series(False, index=df.index)
        train_mask = full_train_mask

    test_mask = ((df["date"] >= test_start_ts) & (df["date"] < test_end_ts))

    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def compute_constants(train_df, atm_band=(0.95, 1.05)):
    """
    Compute σ_fixed and r_fixed from training data.

    Returns:
        dict with sigma_fixed, r_fixed, and domain bounds
    """
    atm_mask = ((train_df["moneyness"] >= atm_band[0]) &
                (train_df["moneyness"] <= atm_band[1]))
    atm_options = train_df[atm_mask]

    return {
        "sigma_fixed": float(atm_options["impl_volatility"].median()),
        "r_fixed": float(train_df["interest_rate"].median()),
        "m_min": float(train_df["moneyness"].min()),
        "m_max": float(train_df["moneyness"].max()),
        "tau_min": float(train_df["time_to_exp"].min()),
        "tau_max": float(train_df["time_to_exp"].max()),
        "n_train": len(train_df),
        "n_atm": len(atm_options),
    }


def df_to_arrays(frame):
    """Extract the columns the PINN needs from a DataFrame.

    `spread` is the raw bid-ask spread (best_offer - best_bid) in dollars.
    Used downstream for spread-normalized RMSE, where the residual is
    expressed in units of the half-spread:
        z_i = (mid_pred_i - mid_obs_i) / (spread_i / 2)
    """
    spread = (frame["best_offer"] - frame["best_bid"]).values.astype(np.float32)
    return {
        "m":       frame["moneyness"].values.astype(np.float32),
        "tau":     frame["time_to_exp"].values.astype(np.float32),
        "vhat":    frame["v_hat_market"].values.astype(np.float32),
        "vhat_bs": frame["v_hat_bs"].values.astype(np.float32),
        "mid":     frame["mid_price"].values.astype(np.float32),
        "K":       frame["strike_price"].values.astype(np.float32),
        "spread":  spread,
    }


def build_boundary_terminal(config, sigma_fixed, r_fixed, n_tc=2000, n_bc=1000):
    """
    Build terminal condition and boundary condition arrays.

    Returns:
        dict with m_tc, tau_tc, v_tc, m_bc_lo, tau_bc, v_bc_lo, m_bc_hi, v_bc_hi
    """
    m_min, m_max = config["m_min"], config["m_max"]
    tau_min, tau_max = config["tau_min"], config["tau_max"]

    # Terminal condition: τ → 0
    m_tc = np.linspace(m_min, m_max, n_tc).astype(np.float32)
    tau_tc = np.full(n_tc, 1e-4, dtype=np.float32)
    v_tc = np.maximum(m_tc - 1.0, 0.0).astype(np.float32)

    # Boundary conditions
    tau_bc = np.linspace(tau_min, tau_max, n_bc).astype(np.float32)
    m_bc_lo = np.full(n_bc, m_min, dtype=np.float32)
    v_bc_lo = bs_call_normalized(m_bc_lo, tau_bc, r_fixed, sigma_fixed).astype(np.float32)
    m_bc_hi = np.full(n_bc, m_max, dtype=np.float32)
    v_bc_hi = bs_call_normalized(m_bc_hi, tau_bc, r_fixed, sigma_fixed).astype(np.float32)

    return {
        "m_tc": m_tc, "tau_tc": tau_tc, "v_tc": v_tc,
        "m_bc_lo": m_bc_lo, "tau_bc": tau_bc, "v_bc_lo": v_bc_lo,
        "m_bc_hi": m_bc_hi, "v_bc_hi": v_bc_hi,
    }
