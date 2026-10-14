"""Point-in-time feature families for Prediction Engine V1.

Every feature must be computable from bars with timestamp ≤ t.
No shift(-k) on prices that would pull future information into features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendlabs.indicators import atr
from trendlabs.prediction.targets import daily_log_returns, sigma_at_t

FEATURE_COLUMNS = [
    "MOM20",
    "MOM60",
    "MOM12_1",
    "VOL20",
    "VOL120",
    "VOLRatio",
    "ATRNorm",
    "DollarVolume",
    "VolumeZ",
    "RS_SPY_20",
    "SPY_MOM20",
    "SPY_VOL20",
    "XS_Dispersion",
]


def _mom(close: pd.Series, lookback: int) -> pd.Series:
    return close.astype(float) / close.astype(float).shift(lookback) - 1.0


def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol PIT features (excludes market-context / RS columns filled later)."""
    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    log_ret = daily_log_returns(close)

    out["MOM20"] = _mom(close, 20)
    out["MOM60"] = _mom(close, 60)
    # 12-1 momentum: return from t-252 to t-21 (skip most recent ~1 month)
    out["MOM12_1"] = close.shift(21) / close.shift(252) - 1.0

    out["VOL20"] = sigma_at_t(log_ret, window=20)
    out["VOL120"] = sigma_at_t(log_ret, window=120)
    out["VOLRatio"] = out["VOL20"] / out["VOL120"].replace(0.0, np.nan)

    atr14 = atr(out, period=14)
    out["ATRNorm"] = atr14 / close.replace(0.0, np.nan)

    out["DollarVolume"] = close * volume
    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std().replace(0.0, np.nan)
    out["VolumeZ"] = (volume - vol_mean) / vol_std
    return out


def attach_market_context(panel: pd.DataFrame, *, spy_symbol: str = "SPY") -> pd.DataFrame:
    """Broadcast SPY momentum/vol and cross-sectional MOM20 dispersion by date.

    Relative strength RS_SPY_20 = MOM20 - SPY MOM20 on the same date.
    """
    if panel.empty:
        return panel

    out = panel.copy()
    spy = out.loc[out["symbol"] == spy_symbol, ["ts", "MOM20", "VOL20"]].rename(
        columns={"MOM20": "SPY_MOM20", "VOL20": "SPY_VOL20"}
    )
    if spy.empty:
        out["SPY_MOM20"] = np.nan
        out["SPY_VOL20"] = np.nan
        out["RS_SPY_20"] = np.nan
    else:
        out = out.merge(spy, on="ts", how="left")
        out["RS_SPY_20"] = out["MOM20"] - out["SPY_MOM20"]

    # Cross-sectional dispersion of MOM20 across names present that day
    xs = (
        out.groupby("ts", sort=False)["MOM20"]
        .transform(lambda s: float(s.std(ddof=1)) if s.notna().sum() >= 2 else np.nan)
    )
    out["XS_Dispersion"] = xs
    return out


def assert_no_future_in_feature_ops() -> None:
    """Documentation hook — unit tests verify concrete frames; this lists invariants."""
    # Features use only .shift(+lag) with lag > 0 (past) or rolling windows ending at t.
    assert all(isinstance(c, str) for c in FEATURE_COLUMNS)
