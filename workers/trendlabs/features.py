"""Feature matrix for TrendLab ML models (no look-ahead labels)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.indicators import enrich


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ema_spread_20_50",
    "ema_spread_50_200",
    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema200",
    "rsi_14",
    "rsi_norm",
    "adx_14",
    "atr_pct",
    "volatility_20",
    "volume_z",
    "volume_ratio",
    "hl_range_pct",
    "close_location",
    "st_direction",
    "momentum_10",
    "regime_bull",
    "regime_bear",
    "regime_side",
    "regime_hivol",
    "regime_riskoff",
]

# Frozen feature-set identity: deterministic function of the column list, so ANY
# change to FEATURE_COLUMNS (add/remove/reorder) bumps the version and invalidates
# old snapshots/models rather than silently mixing feature definitions.
FEATURE_VERSION = "fv1-" + hashlib.sha1("|".join(FEATURE_COLUMNS).encode("utf-8")).hexdigest()[:8]


def build_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Enrich OHLCV and compute model features. Index aligned to input rows."""
    if candles.empty:
        return pd.DataFrame()

    df = enrich(candles).copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)
    df["ema_spread_20_50"] = (df["ema_20"] - df["ema_50"]) / close
    df["ema_spread_50_200"] = (df["ema_50"] - df["ema_200"]) / close
    df["price_vs_ema20"] = close / df["ema_20"] - 1.0
    df["price_vs_ema50"] = close / df["ema_50"] - 1.0
    df["price_vs_ema200"] = close / df["ema_200"] - 1.0
    df["rsi_norm"] = (df["rsi_14"] - 50.0) / 50.0
    df["atr_pct"] = df["atr_14"] / close
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    df["volume_z"] = (volume - vol_mean) / vol_std
    df["volume_ratio"] = volume / vol_mean.replace(0, np.nan)
    df["hl_range_pct"] = (high - low) / close.replace(0, np.nan)
    df["close_location"] = (close - low) / (high - low).replace(0, np.nan)
    df["momentum_10"] = close / close.shift(10) - 1.0

    regime = df.get("market_regime", pd.Series("unknown", index=df.index)).astype(str)
    df["regime_bull"] = regime.str.contains("bull").astype(float)
    df["regime_bear"] = regime.str.contains("bear").astype(float)
    df["regime_side"] = (regime == "sideways").astype(float)
    df["regime_hivol"] = (regime == "high_volatility").astype(float)
    df["regime_riskoff"] = (regime == "risk_off").astype(float)
    df["st_direction"] = df["st_direction"].astype(float)

    return df


def make_labels(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    band: float = 0.001,
) -> pd.Series:
    """Forward return classification: up / down / sideways. Label uses future bars only via shift(-h)."""
    fwd = df["close"].astype(float).shift(-horizon) / df["close"].astype(float) - 1.0
    labels = pd.Series("sideways", index=df.index, dtype=object)
    labels = labels.mask(fwd > band, "up")
    labels = labels.mask(fwd < -band, "down")
    labels[fwd.isna()] = np.nan
    return labels


def dataset_from_candles(
    candles: pd.DataFrame,
    *,
    horizon: int = 1,
    band: float = 0.001,
    min_history: int = 60,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y, and meta frame (ts/close) with NaNs dropped. No shuffle."""
    feat = build_feature_frame(candles)
    if feat.empty:
        return pd.DataFrame(), pd.Series(dtype=object), pd.DataFrame()

    y = make_labels(feat, horizon=horizon, band=band)
    meta = pd.DataFrame(
        {
            "ts": feat["ts"],
            "close": feat["close"].astype(float),
            "symbol": feat["symbol"] if "symbol" in feat.columns else None,
            "timeframe": feat["timeframe"] if "timeframe" in feat.columns else None,
        }
    )
    X = feat[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    # Drop warm-up head for indicator stability
    if min_history > 0:
        idx = np.arange(len(X))
        mask = mask & (idx >= min_history)
    # Drop last `horizon` rows (no label)
    if horizon > 0:
        mask.iloc[-horizon:] = False

    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), meta.loc[mask].reset_index(drop=True)


def default_band(timeframe: str) -> float:
    return {
        "15m": 0.0005,
        "1h": 0.0008,
        "4h": 0.0015,
        "1d": 0.002,
    }.get(timeframe, 0.001)


def feature_importance_frame(booster_or_model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    try:
        importances = booster_or_model.feature_importances_
    except Exception:  # noqa: BLE001
        return []
    pairs = sorted(zip(feature_names, importances, strict=True), key=lambda x: -float(x[1]))
    return [{"feature": n, "importance": float(v)} for n, v in pairs]
