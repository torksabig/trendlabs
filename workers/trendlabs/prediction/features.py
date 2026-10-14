"""Point-in-time feature families for Prediction Engine (V1 + V2 Sprint 1).

Every feature must be computable from bars with timestamp ≤ t.
No shift(-k) on prices that would pull future information into features.

Momentum uses log returns consistently: MOM_L = ln(P_t / P_{t-L}).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendlabs.indicators import atr
from trendlabs.prediction.targets import daily_log_returns, sigma_at_t

FEATURE_COLUMNS = [
    # Momentum structure (log returns)
    "MOM5",
    "MOM10",
    "MOM20",
    "MOM60",
    "MOM120",
    "MOM12_1",
    "MOMACCEL_10_20",
    "MOMACCEL_20_60",
    # Trend quality (rolling OLS of price vs time)
    "TREND_SLOPE20",
    "TREND_R2_20",
    "TREND_SLOPE60",
    "TREND_R2_60",
    # Volatility
    "VOL20",
    "VOL120",
    "VOLRatio",
    "ATRNorm",
    # Volume / liquidity
    "DollarVolume",
    "VolumeZ",
    # Relative strength vs SPY
    "RS_SPY_5",
    "RS_SPY_20",
    "RS_SPY_60",
    # Market context (broadcast per date)
    "SPY_MOM20",
    "SPY_VOL20",
    "XS_Dispersion",
    # Market breadth (fraction of universe above MA)
    "BREADTH20",
    "BREADTH50",
    "BREADTH200",
    # Overnight / intraday decomposition
    "OVERNIGHT_RET",
    "INTRADAY_RET",
    "OVERNIGHT_RET5",
    "OVERNIGHT_RET20",
    "INTRADAY_RET5",
    "INTRADAY_RET20",
]

# Internal helpers used to build breadth; dropped before export.
_BREADTH_HELPERS = ("_above_ma20", "_above_ma50", "_above_ma200")


def _mom_log(close: pd.Series, lookback: int) -> pd.Series:
    """Log momentum: ln(P_t / P_{t-lookback})."""
    c = close.astype(float)
    prev = c.shift(lookback)
    return np.log(c / prev.replace(0.0, np.nan))


def _rolling_ols_price_vs_time(close: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS of price on time index 0..window-1 ending at t (PIT).

    Returns (slope, R²). Uses only bars in [t-window+1, t].
    """
    c = close.astype(float).to_numpy(dtype=float, copy=False)
    n = len(c)
    slopes = np.full(n, np.nan, dtype=float)
    r2s = np.full(n, np.nan, dtype=float)
    if window < 2 or n < window:
        return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx <= 0:
        return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)

    for i in range(window - 1, n):
        y = c[i - window + 1 : i + 1]
        if not np.isfinite(y).all():
            continue
        y_mean = y.mean()
        sxy = float(((x - x_mean) * (y - y_mean)).sum())
        slope = sxy / sxx
        syy = float(((y - y_mean) ** 2).sum())
        if syy <= 0:
            slopes[i] = slope
            r2s[i] = np.nan
        else:
            r2 = (slope * slope * sxx) / syy
            slopes[i] = slope
            r2s[i] = float(np.clip(r2, 0.0, 1.0))
    return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)


def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol PIT features (excludes market-context / RS / breadth filled later)."""
    out = df.copy()
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)
    log_ret = daily_log_returns(close)

    # --- Momentum (log) ---
    out["MOM5"] = _mom_log(close, 5)
    out["MOM10"] = _mom_log(close, 10)
    out["MOM20"] = _mom_log(close, 20)
    out["MOM60"] = _mom_log(close, 60)
    out["MOM120"] = _mom_log(close, 120)
    # 12-1 momentum: return from t-252 to t-21 (skip most recent ~1 month)
    c21 = close.shift(21)
    c252 = close.shift(252)
    out["MOM12_1"] = np.log(c21 / c252.replace(0.0, np.nan))

    out["MOMACCEL_10_20"] = out["MOM10"] - out["MOM20"]
    out["MOMACCEL_20_60"] = out["MOM20"] - out["MOM60"]

    # --- Trend quality ---
    slope20, r2_20 = _rolling_ols_price_vs_time(close, 20)
    slope60, r2_60 = _rolling_ols_price_vs_time(close, 60)
    out["TREND_SLOPE20"] = slope20
    out["TREND_R2_20"] = r2_20
    out["TREND_SLOPE60"] = slope60
    out["TREND_R2_60"] = r2_60

    # --- Volatility ---
    out["VOL20"] = sigma_at_t(log_ret, window=20)
    out["VOL120"] = sigma_at_t(log_ret, window=120)
    out["VOLRatio"] = out["VOL20"] / out["VOL120"].replace(0.0, np.nan)

    atr14 = atr(out, period=14)
    out["ATRNorm"] = atr14 / close.replace(0.0, np.nan)

    out["DollarVolume"] = close * volume
    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std().replace(0.0, np.nan)
    out["VolumeZ"] = (volume - vol_mean) / vol_std

    # --- Overnight / intraday (PIT at close of t) ---
    # overnight_t = ln(open_t / close_{t-1}); intraday_t = ln(close_t / open_t)
    prev_close = close.shift(1)
    out["OVERNIGHT_RET"] = np.log(open_ / prev_close.replace(0.0, np.nan))
    out["INTRADAY_RET"] = np.log(close / open_.replace(0.0, np.nan))
    out["OVERNIGHT_RET5"] = out["OVERNIGHT_RET"].rolling(5, min_periods=5).sum()
    out["OVERNIGHT_RET20"] = out["OVERNIGHT_RET"].rolling(20, min_periods=20).sum()
    out["INTRADAY_RET5"] = out["INTRADAY_RET"].rolling(5, min_periods=5).sum()
    out["INTRADAY_RET20"] = out["INTRADAY_RET"].rolling(20, min_periods=20).sum()

    # Helpers for breadth (close > MA) — aggregated in attach_market_context
    ma20 = close.rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    out["_above_ma20"] = (close > ma20).astype(float)
    out["_above_ma50"] = (close > ma50).astype(float)
    out["_above_ma200"] = (close > ma200).astype(float)
    # NaN where MA not yet available
    out.loc[ma20.isna(), "_above_ma20"] = np.nan
    out.loc[ma50.isna(), "_above_ma50"] = np.nan
    out.loc[ma200.isna(), "_above_ma200"] = np.nan
    return out


def attach_market_context(panel: pd.DataFrame, *, spy_symbol: str = "SPY") -> pd.DataFrame:
    """Broadcast SPY momentum/vol, RS, breadth, and XS dispersion by date.

    Relative strength RS_SPY_L = MOM_L − SPY MOM_L on the same date (log moms).
    Breadth_W = fraction of universe with close > MA_W on date t.
    """
    if panel.empty:
        return panel

    out = panel.copy()
    spy_cols = ["ts", "MOM5", "MOM20", "MOM60", "VOL20"]
    spy = out.loc[out["symbol"] == spy_symbol, spy_cols].rename(
        columns={
            "MOM5": "SPY_MOM5",
            "MOM20": "SPY_MOM20",
            "MOM60": "SPY_MOM60",
            "VOL20": "SPY_VOL20",
        }
    )
    if spy.empty:
        out["SPY_MOM5"] = np.nan
        out["SPY_MOM20"] = np.nan
        out["SPY_MOM60"] = np.nan
        out["SPY_VOL20"] = np.nan
        out["RS_SPY_5"] = np.nan
        out["RS_SPY_20"] = np.nan
        out["RS_SPY_60"] = np.nan
    else:
        out = out.merge(spy, on="ts", how="left")
        out["RS_SPY_5"] = out["MOM5"] - out["SPY_MOM5"]
        out["RS_SPY_20"] = out["MOM20"] - out["SPY_MOM20"]
        out["RS_SPY_60"] = out["MOM60"] - out["SPY_MOM60"]

    # Cross-sectional dispersion of MOM20 across names present that day
    xs = out.groupby("ts", sort=False)["MOM20"].transform(
        lambda s: float(s.std(ddof=1)) if s.notna().sum() >= 2 else np.nan
    )
    out["XS_Dispersion"] = xs

    # Market breadth: fraction of names with valid above-MA flag that day
    for w, helper in ((20, "_above_ma20"), (50, "_above_ma50"), (200, "_above_ma200")):
        if helper not in out.columns:
            out[f"BREADTH{w}"] = np.nan
            continue
        out[f"BREADTH{w}"] = out.groupby("ts", sort=False)[helper].transform(
            lambda s: float(s.mean()) if s.notna().any() else np.nan
        )

    # Drop internal helpers and non-feature SPY mom columns used only for RS
    drop_cols = [c for c in (*_BREADTH_HELPERS, "SPY_MOM5", "SPY_MOM60") if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def assert_no_future_in_feature_ops() -> None:
    """Documentation hook — unit tests verify concrete frames; this lists invariants."""
    # Features use only .shift(+lag) with lag > 0 (past) or rolling windows ending at t.
    assert all(isinstance(c, str) for c in FEATURE_COLUMNS)
