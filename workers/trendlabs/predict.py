"""Hourly paper predictions from collected candles + indicators (never executes trades)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from trendlabs.config import DEFAULT_TIMEFRAME, SYMBOLS
from trendlabs.db import Store
from trendlabs.indicators import enrich
from trendlabs.snapshot import write_snapshot


def _num(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def next_hourly_slot(now: datetime | None = None) -> datetime:
    """Next top-of-hour + 15 minutes (matches Actions cron :15)."""
    base = now or datetime.now(timezone.utc)
    slot = base.replace(minute=15, second=0, microsecond=0)
    if base >= slot:
        slot = slot + timedelta(hours=1)
    return slot


def predict_symbol(store: Store, symbol: str, timeframe: str) -> dict[str, Any] | None:
    candles = store.load_candles(symbol, timeframe)
    if candles.empty or len(candles) < 30:
        return None

    enriched = enrich(candles)
    last = enriched.iloc[-1]
    prev = enriched.iloc[-2]
    close = float(last["close"])
    ema20 = _num(last.get("ema_20"))
    ema50 = _num(last.get("ema_50"))
    ema200 = _num(last.get("ema_200"))
    rsi14 = _num(last.get("rsi_14"))
    adx14 = _num(last.get("adx_14"))
    atr14 = _num(last.get("atr_14"))
    trend = str(last.get("trend_label") or "neutral")

    score = 0.0
    drivers: dict[str, Any] = {
        "trend_label": trend,
        "ema_20": ema20,
        "ema_50": ema50,
        "ema_200": ema200,
        "rsi_14": rsi14,
        "adx_14": adx14,
        "atr_14": atr14,
        "bar_change_pct": ((close / float(prev["close"])) - 1.0) * 100 if prev["close"] else None,
    }

    if ema20 is not None and ema50 is not None:
        if ema20 > ema50:
            score += 1.0
        elif ema20 < ema50:
            score -= 1.0
    if ema50 is not None and ema200 is not None:
        if ema50 > ema200:
            score += 0.75
        elif ema50 < ema200:
            score -= 0.75
    if rsi14 is not None:
        if rsi14 >= 55:
            score += 0.5
        elif rsi14 <= 45:
            score -= 0.5
        if rsi14 >= 70:
            score -= 0.25  # overbought fade
        if rsi14 <= 30:
            score += 0.25  # oversold bounce
    if trend == "bullish":
        score += 0.75
    elif trend == "bearish":
        score -= 0.75

    if score >= 1.25:
        direction = "up"
    elif score <= -1.25:
        direction = "down"
    else:
        direction = "sideways"

    # Confidence scales with |score| and ADX trend strength
    adx_boost = min(max((adx14 or 15) / 50.0, 0.0), 0.35)
    confidence = min(0.92, 0.35 + min(abs(score) / 4.0, 0.45) + adx_boost)

    created = datetime.now(timezone.utc)
    next_update = next_hourly_slot(created)
    atr_pct = (atr14 / close * 100) if atr14 and close else None
    summary = (
        f"{symbol} {timeframe}: rule-based outlook is {direction} over the next ~1h "
        f"(confidence {confidence:.0%}). Trend={trend}"
        + (f", RSI={rsi14:.1f}" if rsi14 is not None else "")
        + (f", ADX={adx14:.1f}" if adx14 is not None else "")
        + (f", ATR≈{atr_pct:.2f}%" if atr_pct is not None else "")
        + ". Paper research only — does not execute trades."
    )

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "horizon": "1h",
        "direction": direction,
        "confidence": round(confidence, 3),
        "last_close": close,
        "drivers": drivers,
        "summary": summary,
        "model": "indicator_score_v1",
        "source": "hourly_predict",
        "created_at": created.isoformat(),
        "next_update_at": next_update.isoformat(),
        "executes_trades": False,
    }


def run_predictions(
    store: Store,
    *,
    timeframe: str = DEFAULT_TIMEFRAME,
    symbols: list[str] | None = None,
) -> list[dict[str, Any]]:
    targets = symbols or SYMBOLS
    results: list[dict[str, Any]] = []
    for symbol in targets:
        pred = predict_symbol(store, symbol, timeframe)
        if not pred:
            print(f"{symbol}: skipped (insufficient candles)")
            continue
        store.insert_prediction(pred)
        results.append(pred)
        print(
            f"{symbol}: {pred['direction']} conf={pred['confidence']:.2f} "
            f"close={pred['last_close']:.2f}"
        )

    now = datetime.now(timezone.utc).isoformat()
    store.set_meta("last_prediction", now)
    store.set_meta("prediction_timeframe", timeframe)
    store.set_meta("prediction_count", str(len(results)))
    store.set_meta("next_prediction_at", next_hourly_slot().isoformat())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="TrendLab hourly predictions")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    store = Store()
    results = run_predictions(store, timeframe=args.timeframe, symbols=args.symbols)
    write_snapshot(store, timeframe=args.timeframe)
    print(
        json.dumps(
            {
                "count": len(results),
                "timeframe": args.timeframe,
                "last_prediction": store.get_meta("last_prediction"),
                "next_prediction_at": store.get_meta("next_prediction_at"),
                "directions": {r["symbol"]: r["direction"] for r in results},
            },
            indent=2,
        )
    )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
