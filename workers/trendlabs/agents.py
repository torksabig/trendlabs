from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trendlabs.indicators import enrich

Side = Literal["long", "flat"]


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Side
    stop_loss: float | None
    reason: str
    confidence: float


class BaseAgent:
    id: str
    name: str

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        raise NotImplementedError


def _frame(candles: pd.DataFrame) -> pd.DataFrame:
    return candles if "ema_20" in candles.columns else enrich(candles)


class EmaTrendAgent(BaseAgent):
    id = "ema_trend"
    name = "Agent A — EMA Trend"

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        if len(df) < 210:
            return Signal(symbol, "flat", None, "Warming up EMA stack", 0.0)
        row = df.iloc[-1]
        stop = float(row["close"] - 2 * row["atr_14"]) if pd.notna(row["atr_14"]) else None
        bullish = (
            row["ema_20"] > row["ema_50"] > row["ema_200"]
            and row["close"] > row["ema_20"]
        )
        if bullish:
            return Signal(symbol, "long", stop, "EMA 20>50>200 with price above EMA20", 0.75)
        return Signal(symbol, "flat", stop, "EMA stack not bullishly aligned", 0.55)


class BreakoutAgent(BaseAgent):
    id = "breakout"
    name = "Agent B — Breakout"

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        if len(df) < 30:
            return Signal(symbol, "flat", None, "Warming up breakout window", 0.0)
        window = df.iloc[:-1]
        high_20 = float(window["high"].tail(20).max())
        low_10 = float(window["low"].tail(10).min())
        close = float(df.iloc[-1]["close"])
        atr = float(df.iloc[-1]["atr_14"]) if pd.notna(df.iloc[-1]["atr_14"]) else close * 0.02
        stop = close - 2 * atr
        if close > high_20:
            return Signal(symbol, "long", stop, f"Close {close:.2f} broke 20-bar high {high_20:.2f}", 0.8)
        if close < low_10:
            return Signal(symbol, "flat", stop, f"Close below 10-bar low {low_10:.2f}", 0.7)
        return Signal(symbol, "flat", stop, "Inside range — no breakout", 0.35)


class MomentumAgent(BaseAgent):
    id = "momentum"
    name = "Agent C — Momentum"

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        if len(df) < 60:
            return Signal(symbol, "flat", None, "Warming up momentum filters", 0.0)
        row = df.iloc[-1]
        prev = df.iloc[-2]
        mom = (row["close"] / df.iloc[-11]["close"] - 1) * 100
        vol_ok = row["volume"] > (row["volume_sma_20"] or 0)
        trend_ok = row["ema_20"] > row["ema_50"] and (row["adx_14"] or 0) >= 22
        stop = float(row["close"] - 2 * row["atr_14"]) if pd.notna(row["atr_14"]) else None
        if mom > 0 and vol_ok and trend_ok and row["close"] >= prev["close"]:
            return Signal(
                symbol,
                "long",
                stop,
                f"Momentum {mom:.2f}% with ADX {row['adx_14']:.1f} and volume confirmation",
                0.78,
            )
        return Signal(symbol, "flat", stop, f"Momentum filters not aligned (mom={mom:.2f}%)", 0.4)


class SupertrendAgent(BaseAgent):
    id = "supertrend"
    name = "Agent D — Supertrend"

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        if len(df) < 40:
            return Signal(symbol, "flat", None, "Warming up Supertrend", 0.0)
        row = df.iloc[-1]
        stop = float(row["supertrend"]) if pd.notna(row["supertrend"]) else None
        if int(row["st_direction"]) == 1:
            return Signal(symbol, "long", stop, f"Supertrend bullish @ {row['supertrend']:.2f}", 0.72)
        return Signal(symbol, "flat", stop, f"Supertrend bearish @ {row['supertrend']:.2f}", 0.72)


def build_agents() -> list[BaseAgent]:
    return [EmaTrendAgent(), BreakoutAgent(), MomentumAgent(), SupertrendAgent()]
