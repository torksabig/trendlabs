"""Quantitative market-regime classification for strategy evaluation.

Labels (mutually exclusive, priority order):
  risk_off → high_volatility → trending_bullish → trending_bearish
  → low_volatility → sideways → unknown
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trendlabs.indicators import ema


REGIME_LABELS = (
    "trending_bullish",
    "trending_bearish",
    "high_volatility",
    "low_volatility",
    "sideways",
    "risk_off",
    "unknown",
)


def classify_row(
    *,
    close: float,
    ema_fast: float | None,
    ema_slow: float | None,
    ema_long: float | None,
    adx: float | None,
    vol: float | None,
    ret_20: float | None,
    vol_hi: float,
    vol_lo: float,
    adx_trend: float = 22.0,
    adx_chop: float = 18.0,
    risk_off_ret: float = -0.06,
) -> str:
    if ema_fast is None or ema_slow is None or close != close:
        return "unknown"

    adx_v = float(adx) if adx is not None and adx == adx else 0.0
    vol_v = float(vol) if vol is not None and vol == vol else 0.0
    ret_v = float(ret_20) if ret_20 is not None and ret_20 == ret_20 else 0.0
    below_long = ema_long is not None and close < float(ema_long)

    if ret_v <= risk_off_ret or (below_long and vol_v >= vol_hi and ret_v < -0.02):
        return "risk_off"
    if vol_v >= vol_hi and adx_v < adx_trend:
        return "high_volatility"
    if adx_v >= adx_trend and ema_fast > ema_slow and close > ema_slow:
        return "trending_bullish"
    if adx_v >= adx_trend and ema_fast < ema_slow and close < ema_slow:
        return "trending_bearish"
    if vol_v <= vol_lo and adx_v < adx_chop:
        return "low_volatility"
    if adx_v < adx_chop:
        return "sideways"
    return "sideways"


def classify_series(df: pd.DataFrame) -> pd.Series:
    """Return a regime label per row. Expects OHLCV; uses existing indicator cols when present."""
    out = df.copy()
    if "ema_20" not in out.columns:
        out["ema_20"] = ema(out["close"], 20)
    if "ema_50" not in out.columns:
        out["ema_50"] = ema(out["close"], 50)
    if "ema_200" not in out.columns:
        out["ema_200"] = ema(out["close"], 200)
    if "adx_14" not in out.columns:
        from trendlabs.indicators import adx as adx_fn

        out["adx_14"] = adx_fn(out, 14)
    if "volatility_20" not in out.columns:
        out["volatility_20"] = out["close"].pct_change().rolling(20).std() * np.sqrt(252)

    out["ret_20"] = out["close"].pct_change(20)
    vol = out["volatility_20"]
    # Adaptive thresholds from the series itself (no future peek beyond expanding median early on).
    vol_med = vol.expanding(min_periods=40).median()
    vol_hi = (vol_med * 1.35).fillna(vol.median() if vol.notna().any() else 0.2)
    vol_lo = (vol_med * 0.65).fillna(vol.median() if vol.notna().any() else 0.1)

    labels: list[str] = []
    for i in range(len(out)):
        row = out.iloc[i]
        labels.append(
            classify_row(
                close=float(row["close"]),
                ema_fast=float(row["ema_20"]) if pd.notna(row["ema_20"]) else None,
                ema_slow=float(row["ema_50"]) if pd.notna(row["ema_50"]) else None,
                ema_long=float(row["ema_200"]) if pd.notna(row["ema_200"]) else None,
                adx=float(row["adx_14"]) if pd.notna(row["adx_14"]) else None,
                vol=float(row["volatility_20"]) if pd.notna(row["volatility_20"]) else None,
                ret_20=float(row["ret_20"]) if pd.notna(row["ret_20"]) else None,
                vol_hi=float(vol_hi.iloc[i]) if pd.notna(vol_hi.iloc[i]) else 0.25,
                vol_lo=float(vol_lo.iloc[i]) if pd.notna(vol_lo.iloc[i]) else 0.08,
            )
        )
    return pd.Series(labels, index=df.index, name="market_regime")


def regime_distribution(labels: pd.Series) -> dict[str, float]:
    if labels.empty:
        return {}
    counts = labels.value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def latest_regime(df: pd.DataFrame) -> str:
    if df.empty:
        return "unknown"
    series = df["market_regime"] if "market_regime" in df.columns else classify_series(df)
    return str(series.iloc[-1])


def attribute_trades_by_regime(
    trades: list[dict[str, Any]],
    regime_by_symbol_ts: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Group round-trips by entry-bar regime. Keys are regime labels."""
    open_buys: dict[str, list[dict[str, Any]]] = {}
    buckets: dict[str, list[float]] = {r: [] for r in REGIME_LABELS}

    for trade in sorted(trades, key=lambda t: str(t["ts"])):
        sym = trade["symbol"]
        if trade["side"] == "BUY":
            open_buys.setdefault(sym, []).append(trade)
            continue
        if trade["side"] != "SELL" or not open_buys.get(sym):
            continue
        buy = open_buys[sym].pop(0)
        qty = float(trade["qty"])
        entry = float(buy["price"])
        exit_px = float(trade["price"])
        fees = float(buy.get("fee", 0)) + float(trade.get("fee", 0))
        pnl = (exit_px - entry) * qty - fees
        ts_key = str(buy["ts"])
        regime = regime_by_symbol_ts.get(sym, {}).get(ts_key, "unknown")
        if regime not in buckets:
            buckets[regime] = []
        buckets[regime].append(pnl)

    out: dict[str, dict[str, Any]] = {}
    for regime, pnls in buckets.items():
        if not pnls:
            continue
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        out[regime] = {
            "trade_count": len(pnls),
            "win_rate": len(winners) / len(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "total_pnl": sum(pnls),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        }
    return out


def score_regime_fit(regime_stats: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Human-readable fit labels from per-regime PnL."""
    labels: dict[str, str] = {}
    for regime, stats in regime_stats.items():
        wr = stats.get("win_rate") or 0.0
        avg = stats.get("avg_pnl") or 0.0
        n = stats.get("trade_count") or 0
        if n < 3:
            labels[regime] = "insufficient"
        elif avg > 0 and wr >= 0.55:
            labels[regime] = "excellent"
        elif avg > 0:
            labels[regime] = "good"
        elif avg > -abs(stats.get("total_pnl", 0)) * 0.01 and wr >= 0.45:
            labels[regime] = "mediocre"
        else:
            labels[regime] = "poor"
    return labels
