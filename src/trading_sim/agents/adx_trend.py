from __future__ import annotations

import pandas as pd

from trading_sim.agents.base import Signal, TradingAgent
from trading_sim.indicators import adx


class AdxTrendAgent(TradingAgent):
    id = "adx_trend"
    label = "ADX Trend"

    def __init__(self, period: int = 14, threshold: float = 25.0) -> None:
        self.period = period
        self.threshold = threshold

    def _signal_at(self, frame: pd.DataFrame, i: int) -> Signal:
        if i + 1 < self.period * 3:
            return Signal("flat", 0.0, "Insufficient history for ADX")
        a = float(frame["adx"].iloc[i])
        plus_di = float(frame["plus_di"].iloc[i])
        minus_di = float(frame["minus_di"].iloc[i])
        if pd.isna(a) or pd.isna(plus_di) or pd.isna(minus_di):
            return Signal("flat", 0.0, "ADX warming up")
        if a >= self.threshold and plus_di > minus_di:
            conf = min(1.0, (a - self.threshold) / 25.0 + 0.4)
            return Signal("long", conf, f"ADX {a:.1f} with +DI {plus_di:.1f} > -DI {minus_di:.1f}")
        if a >= self.threshold and minus_di > plus_di:
            return Signal(
                "flat",
                min(1.0, (a - self.threshold) / 25.0 + 0.4),
                f"Downtrend: ADX {a:.1f}, -DI {minus_di:.1f} > +DI {plus_di:.1f}",
            )
        return Signal("flat", 0.3, f"Weak trend: ADX {a:.1f} below {self.threshold}")

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        frame = adx(candles["high"], candles["low"], candles["close"], self.period)
        return self._signal_at(frame, len(candles) - 1)

    def generate_signal_series(self, candles: pd.DataFrame) -> list[Signal]:
        frame = adx(candles["high"], candles["low"], candles["close"], self.period)
        return [self._signal_at(frame, i) for i in range(len(candles))]
