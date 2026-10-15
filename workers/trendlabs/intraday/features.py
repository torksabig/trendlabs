"""Point-in-time intraday features for the 15m/1h/4h panels.

Every feature is computable from bars with timestamp <= t (no shift(-k) on prices).
Momentum windows are expressed in HOURS/DAYS and converted to panel bars via TF_MINUTES,
so the same horizon means the same wall-clock move on every panel.

Volume relative to the same time-of-day slot uses only PRIOR sessions (PIT expanding
mean per slot, shifted by one session).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trendlabs.intraday import TF_MINUTES
from trendlabs.prediction.features import _rolling_ols_price_vs_time

_ANN = np.sqrt(6.5 * 252)  # annualization factor for per-bar sigma (6.5h session)

FEATURE_COLUMNS = [
    # Momentum (log returns over wall-clock horizons)
    "MOM_1h",
    "MOM_4h",
    "MOM_1d",
    "MOM_5d",
    "MOMACCEL_1h_1d",
    # Trend quality (rolling OLS of price vs time)
    "TREND_SLOPE_1d",
    "TREND_R2_1d",
    "TREND_SLOPE_5d",
    "TREND_R2_5d",
    # Volatility / range
    "VOL_1h",
    "VOL_1d",
    "VOLRatio",
    "RANGE_1d",
    "GAP",
    "GAP_Z",
    # Volume / liquidity
    "DollarVolume",
    "VolumeZ",
    "RVOL_SLOT",
    # Time-of-day / calendar
    "BAR_FRAC",
    "TOD_SIN",
    "TOD_COS",
    "DOW_SIN",
    "DOW_COS",
    # Market context (broadcast per bar timestamp)
    "SPY_MOM_1h",
    "SPY_MOM_1d",
    "SPY_VOL_1d",
    "RS_SPY_1h",
    "RS_SPY_1d",
    "XS_Dispersion",
    "BREADTH_1d",
]

# Internal helper columns dropped before export.
_HELPERS = ("_above_ma1d", "session_date", "slot")


def bars_for_hours(tf: str, hours: float) -> int:
    """Convert a wall-clock horizon to panel bars; minimum 1."""
    return max(1, round(hours * 60.0 / TF_MINUTES[tf]))


def _mom_log(close: pd.Series, lookback: int) -> pd.Series:
    c = close.astype(float)
    return np.log(c / c.shift(lookback).replace(0.0, np.nan))


def relative_volume_slot(df: pd.DataFrame) -> pd.Series:
    """Volume / mean(volume of the same slot on PRIOR sessions) — PIT-safe.

    Requires 'slot' and a per-(symbol, session) order. Expanding mean shifted by
    one session inside each (symbol, slot) group, so today's bars never leak in.
    """
    vol_mean = (
        df.groupby(["symbol", "slot"], sort=False)["volume"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    return df["volume"].astype(float) / vol_mean.replace(0.0, np.nan)


def symbol_features(df: pd.DataFrame, *, tf: str, bars_per_day: int) -> pd.DataFrame:
    """Per-symbol PIT features on a session-anchored panel (columns: ts, OHLCV, slot)."""
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    volume = out["volume"].astype(float)

    w_1h = bars_for_hours(tf, 1.0)
    w_4h = bars_for_hours(tf, 4.0)
    w_1d = bars_per_day
    w_5d = 5 * bars_per_day

    # --- Momentum ---
    out["MOM_1h"] = _mom_log(close, w_1h)
    out["MOM_4h"] = _mom_log(close, w_4h)
    out["MOM_1d"] = _mom_log(close, w_1d)
    out["MOM_5d"] = _mom_log(close, w_5d)
    out["MOMACCEL_1h_1d"] = out["MOM_1h"] - out["MOM_1h"].shift(w_1d)

    # --- Trend quality ---
    slope1, r2_1 = _rolling_ols_price_vs_time(close, w_1d)
    slope5, r2_5 = _rolling_ols_price_vs_time(close, w_5d)
    out["TREND_SLOPE_1d"] = slope1
    out["TREND_R2_1d"] = r2_1
    out["TREND_SLOPE_5d"] = slope5
    out["TREND_R2_5d"] = r2_5

    # --- Volatility / range ---
    log_ret = np.log(close / close.shift(1).replace(0.0, np.nan))
    # Window-1 std is degenerate (always NaN); a 1-bar short-vol proxy is |log_ret|.
    if w_1h <= 1:
        out["VOL_1h"] = log_ret.abs() * _ANN
    else:
        out["VOL_1h"] = log_ret.rolling(w_1h, min_periods=w_1h).std() * _ANN
    out["VOL_1d"] = log_ret.rolling(w_1d, min_periods=w_1d).std() * _ANN
    out["VOLRatio"] = out["VOL_1h"] / out["VOL_1d"].replace(0.0, np.nan)
    out["RANGE_1d"] = ((high - out["low"].astype(float)) / close.replace(0.0, np.nan)).rolling(
        w_1d, min_periods=w_1d
    ).mean()
    out["GAP"] = np.log(open_ / close.shift(1).replace(0.0, np.nan))
    gap_std = out["GAP"].rolling(bars_per_day, min_periods=bars_per_day).std().replace(0.0, np.nan)
    out["GAP_Z"] = out["GAP"] / gap_std

    # --- Volume ---
    out["DollarVolume"] = close * volume
    vol_mean = volume.rolling(bars_per_day, min_periods=bars_per_day).mean()
    vol_std = volume.rolling(bars_per_day, min_periods=bars_per_day).std().replace(0.0, np.nan)
    out["VolumeZ"] = (volume - vol_mean) / vol_std
    out["RVOL_SLOT"] = relative_volume_slot(out)

    # --- Time of day / calendar (known at t) ---
    slots_per_day = max(bars_per_day - 1, 1)
    frac = out["slot"].astype(float) / slots_per_day
    out["BAR_FRAC"] = frac
    out["TOD_SIN"] = np.sin(2 * np.pi * frac)
    out["TOD_COS"] = np.cos(2 * np.pi * frac)
    dow = out["ts"].dt.tz_convert("America/New_York").dt.weekday.astype(float)
    out["DOW_SIN"] = np.sin(2 * np.pi * dow / 5.0)
    out["DOW_COS"] = np.cos(2 * np.pi * dow / 5.0)

    # --- Breadth helper (aggregated in attach_market_context) ---
    ma1d = close.rolling(w_1d, min_periods=w_1d).mean()
    out["_above_ma1d"] = (close > ma1d).astype(float)
    out.loc[ma1d.isna(), "_above_ma1d"] = np.nan
    return out


def attach_market_context(panel: pd.DataFrame, *, spy_symbol: str = "SPY") -> pd.DataFrame:
    """Broadcast SPY momentum/vol, relative strength, dispersion, breadth per bar ts."""
    if panel.empty:
        return panel
    out = panel.copy()

    spy_cols = ["ts", "MOM_1h", "MOM_1d", "VOL_1d"]
    spy = out.loc[out["symbol"] == spy_symbol, spy_cols].rename(
        columns={"MOM_1h": "SPY_MOM_1h", "MOM_1d": "SPY_MOM_1d", "VOL_1d": "SPY_VOL_1d"}
    )
    if spy.empty:
        for c in ("SPY_MOM_1h", "SPY_MOM_1d", "SPY_VOL_1d", "RS_SPY_1h", "RS_SPY_1d"):
            out[c] = np.nan
    else:
        out = out.merge(spy, on="ts", how="left")
        out["RS_SPY_1h"] = out["MOM_1h"] - out["SPY_MOM_1h"]
        out["RS_SPY_1d"] = out["MOM_1d"] - out["SPY_MOM_1d"]

    out["XS_Dispersion"] = out.groupby("ts", sort=False)["MOM_1d"].transform(
        lambda s: float(s.std(ddof=1)) if s.notna().sum() >= 2 else np.nan
    )
    out["BREADTH_1d"] = out.groupby("ts", sort=False)["_above_ma1d"].transform(
        lambda s: float(s.mean()) if s.notna().any() else np.nan
    )
    drop = [c for c in _HELPERS if c in out.columns]
    return out.drop(columns=drop)
