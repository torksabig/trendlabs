"""Historical trend detection with forward returns filled only after the fact."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_trend_events(symbol: str, timeframe: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect regime shifts using completed bars only; attach forward returns afterwards."""
    if len(df) < 80:
        return []

    events: list[dict[str, Any]] = []
    labels = df["trend_label"].tolist()
    for i in range(60, len(df) - 21):
        prev = labels[i - 1]
        cur = labels[i]
        if cur == prev or cur == "unknown" or cur == "sideways":
            continue
        row = df.iloc[i]
        window = df.iloc[i - 40 : i + 1]
        support = float(window["low"].min())
        resistance = float(window["high"].max())
        future = df.iloc[i + 1 :]
        close = float(row["close"])

        def fwd(n: int) -> float | None:
            if len(future) < n:
                return None
            return (float(future.iloc[n - 1]["close"]) / close - 1) * 100

        future_slice = future.head(20)
        mfe = ((future_slice["high"].max() / close) - 1) * 100 if not future_slice.empty else None
        mae = ((future_slice["low"].min() / close) - 1) * 100 if not future_slice.empty else None

        events.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "detected_at": pd.Timestamp(row["ts"]).isoformat(),
                "direction": cur,
                "strength": float(row["adx_14"]) if pd.notna(row["adx_14"]) else None,
                "volatility": float(row["volatility_20"]) if pd.notna(row["volatility_20"]) else None,
                "support": support,
                "resistance": resistance,
                "features": {
                    "ema_20": float(row["ema_20"]) if pd.notna(row["ema_20"]) else None,
                    "ema_50": float(row["ema_50"]) if pd.notna(row["ema_50"]) else None,
                    "rsi_14": float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else None,
                    "volume_vs_sma": (
                        float(row["volume"] / row["volume_sma_20"])
                        if pd.notna(row["volume_sma_20"]) and row["volume_sma_20"]
                        else None
                    ),
                },
                "forward_return_5": fwd(5),
                "forward_return_10": fwd(10),
                "forward_return_20": fwd(20),
                "max_favorable_excursion": float(mfe) if mfe is not None else None,
                "max_adverse_excursion": float(mae) if mae is not None else None,
            }
        )
    return events[-40:]


def summarize_similar(events: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    subset = [e for e in events if e["direction"] == direction and e.get("forward_return_10") is not None]
    if not subset:
        return {"matches": 0, "hit_rate": None, "avg_forward_10": None}
    hits = sum(1 for e in subset if (e["forward_return_10"] or 0) > 0)
    avg = sum(e["forward_return_10"] for e in subset) / len(subset)
    return {
        "matches": len(subset),
        "hit_rate": hits / len(subset),
        "avg_forward_10": avg,
    }
