"""Market feature engineering for ML training (no look-ahead).

Features at bar t use only OHLCV available at or before t. Forward labels are
built separately by the training pipeline and must never enter this module.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.config import SYMBOLS
from trendlabs.db import Store, normalize_ts, utc_now
import trendlabs.store_ml  # noqa: F401  # install ML Store methods
from trendlabs.indicators import adx, atr, ema, rsi

FEATURE_VERSION = "v1"
FEATURE_COLUMNS = [
    "rsi_14",
    "ema_20",
    "ema_50",
    "ema_200",
    "sma_20",
    "sma_50",
    "atr_14",
    "ret_1",
    "ret_5",
    "ret_10",
    "volatility_20",
    "relative_volume",
    "dist_ema_20",
    "dist_ema_50",
    "dist_ema_200",
    "dist_sma_20",
    "dist_sma_50",
    "momentum_10",
    "momentum_20",
    "hl_range",
    "trend_strength",
]

# Bars needed before the first fully-defined feature row (EMA200 dominates).
MIN_WARMUP_BARS = 200


def _pct_dist(price: pd.Series, level: pd.Series) -> pd.Series:
    return (price - level) / level.replace(0, np.nan)


def compute_feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute feature columns from OHLCV. Index preserved; NaNs during warmup."""
    if candles.empty:
        return pd.DataFrame(columns=["ts", *FEATURE_COLUMNS])

    df = candles.sort_values("ts").reset_index(drop=True).copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    out = pd.DataFrame({"ts": pd.to_datetime(df["ts"], utc=True)})
    out["rsi_14"] = rsi(close, 14)
    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)
    out["ema_200"] = ema(close, 200)
    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()
    out["atr_14"] = atr(df, 14)
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["volatility_20"] = close.pct_change().rolling(20, min_periods=20).std()
    vol_sma = volume.rolling(20, min_periods=20).mean()
    out["relative_volume"] = volume / vol_sma.replace(0, np.nan)
    out["dist_ema_20"] = _pct_dist(close, out["ema_20"])
    out["dist_ema_50"] = _pct_dist(close, out["ema_50"])
    out["dist_ema_200"] = _pct_dist(close, out["ema_200"])
    out["dist_sma_20"] = _pct_dist(close, out["sma_20"])
    out["dist_sma_50"] = _pct_dist(close, out["sma_50"])
    out["momentum_10"] = close / close.shift(10) - 1.0
    out["momentum_20"] = close / close.shift(20) - 1.0
    out["hl_range"] = (high - low) / close.replace(0, np.nan)
    # ADX as trend strength; finite and causal (uses past bars only).
    out["trend_strength"] = adx(df, 14)
    return out


def assert_no_lookahead(candles: pd.DataFrame, features: pd.DataFrame) -> None:
    """Verify truncating future candles does not change earlier feature rows."""
    if len(candles) < MIN_WARMUP_BARS + 5:
        return
    mid = len(candles) // 2
    full = compute_feature_frame(candles)
    truncated = compute_feature_frame(candles.iloc[:mid].copy())
    cols = FEATURE_COLUMNS
    left = full.iloc[:mid][cols].reset_index(drop=True)
    right = truncated[cols].reset_index(drop=True)
    # Compare only rows where both sides are finite for a stable column.
    mask = left["ema_200"].notna() & right["ema_200"].notna()
    if not mask.any():
        return
    diff = (left.loc[mask, cols] - right.loc[mask, cols]).abs()
    max_diff = float(diff.max().max()) if not diff.empty else 0.0
    if max_diff > 1e-9:
        raise AssertionError(f"Look-ahead leakage detected: max abs diff={max_diff}")


def generate_features(
    symbol: str,
    timeframe: str,
    *,
    store: Store | None = None,
    candles: pd.DataFrame | None = None,
    feature_version: str = FEATURE_VERSION,
    persist: bool = True,
) -> pd.DataFrame:
    """Generate features for one symbol/timeframe and optionally persist them."""
    store = store or Store()
    symbol = symbol.upper()
    frame = candles if candles is not None else store.load_candles(symbol, timeframe)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "timeframe", "ts", "feature_version", *FEATURE_COLUMNS])

    feats = compute_feature_frame(frame)
    assert_no_lookahead(frame, feats)

    ready = feats.dropna(subset=FEATURE_COLUMNS).copy()
    ready.insert(0, "symbol", symbol)
    ready.insert(1, "timeframe", timeframe)
    ready.insert(3, "feature_version", feature_version)

    if persist and not ready.empty:
        store.upsert_market_features(ready, feature_version=feature_version)

    return ready


def generate_features_universe(
    *,
    store: Store | None = None,
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    feature_version: str = FEATURE_VERSION,
    persist: bool = True,
) -> dict[str, Any]:
    store = store or Store()
    targets = [s.upper() for s in (symbols or SYMBOLS)]
    results: list[dict[str, Any]] = []
    for symbol in targets:
        df = generate_features(
            symbol,
            timeframe,
            store=store,
            feature_version=feature_version,
            persist=persist,
        )
        results.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "feature_version": feature_version,
                "rows": int(len(df)),
                "first_ts": normalize_ts(df["ts"].iloc[0]) if len(df) else None,
                "last_ts": normalize_ts(df["ts"].iloc[-1]) if len(df) else None,
            }
        )
        print(f"{symbol} {timeframe}: {len(df)} feature rows ({feature_version})")
    return {
        "feature_version": feature_version,
        "timeframe": timeframe,
        "generated_at": utc_now(),
        "symbols": results,
        "total_rows": sum(r["rows"] for r in results),
    }


def features_to_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return X matrix with FEATURE_COLUMNS only (ordered)."""
    missing = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return feature_df[FEATURE_COLUMNS].astype(float), list(FEATURE_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ML market features")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    payload = generate_features_universe(
        symbols=args.symbols,
        timeframe=args.timeframe,
        feature_version=args.feature_version,
        persist=not args.no_persist,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["total_rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
