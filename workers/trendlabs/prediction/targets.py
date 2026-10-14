"""Actionable 5-day targets under next-session execution.

Execution rule (documented for manifests and reviews):
  - Decision / features at close of day t (no future bars).
  - Enter long at open of session t+1.
  - Exit at close of session t+5.
  - R = ln(close_{t+5} / open_{t+1})
  - z = R / (sigma_20 * sqrt(5)), sigma_20 from past 20 daily log returns at t only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendlabs.prediction import HORIZON_DAYS, VOL_WINDOW


def daily_log_returns(close: pd.Series) -> pd.Series:
    """Close-to-close log returns; first bar NaN."""
    c = close.astype(float)
    return np.log(c / c.shift(1))


def sigma_at_t(log_ret: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Rolling std of past `window` daily log returns ending at t (min_periods=window)."""
    return log_ret.rolling(window=window, min_periods=window).std()


def actionable_return(
    open_: pd.Series,
    close: pd.Series,
    *,
    horizon: int = HORIZON_DAYS,
) -> pd.Series:
    """R_t = ln(close_{t+horizon} / open_{t+1})."""
    future_close = close.shift(-horizon)
    next_open = open_.shift(-1)
    return np.log(future_close.astype(float) / next_open.astype(float))


def volatility_normalized_z(
    R: pd.Series,
    sigma: pd.Series,
    *,
    horizon: int = HORIZON_DAYS,
) -> pd.Series:
    """z = R / (sigma * sqrt(horizon)); NaN when sigma missing or zero."""
    denom = sigma.astype(float) * np.sqrt(float(horizon))
    z = R.astype(float) / denom.replace(0.0, np.nan)
    return z


def attach_targets(df: pd.DataFrame, *, horizon: int = HORIZON_DAYS, vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    """Add log_ret, sigma_20, R, z columns. Preserves row order."""
    out = df.copy()
    out["log_ret"] = daily_log_returns(out["close"])
    out["sigma_20"] = sigma_at_t(out["log_ret"], window=vol_window)
    out["R"] = actionable_return(out["open"], out["close"], horizon=horizon)
    out["z"] = volatility_normalized_z(out["R"], out["sigma_20"], horizon=horizon)
    return out
