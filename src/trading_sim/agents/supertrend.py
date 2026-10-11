from __future__ import annotations

import pandas as pd

from trading_sim.agents.base import Signal, TradingAgent
from trading_sim.indicators import supertrend


class SupertrendAgent(TradingAgent):
    id = "supertrend"
    label = "Supertrend"

    def __init__(self, period: int = 10, multiplier: float = 3.0) -> None:
        self.period = period
        self.multiplier = multiplier

    def _signal_at(self, frame: pd.DataFrame, close: pd.Series, i: int) -> Signal:
        if i + 1 < self.period * 3:
            return Signal("flat", 0.0, "Insufficient history for Supertrend")
        direction = int(frame["direction"].iloc[i])
        level = float(frame["supertrend"].iloc[i])
        price = float(close.iloc[i])
        if pd.isna(level):
            return Signal("flat", 0.0, "Supertrend warming up")
        distance = abs(price - level) / max(price, 1e-9)
        conf = min(1.0, distance * 20 + 0.4)
        if direction == 1:
            return Signal("long", conf, f"Supertrend uptrend; support {level:.2f}")
        return Signal("flat", conf, f"Supertrend downtrend; resistance {level:.2f}")

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        frame = supertrend(
            candles["high"],
            candles["low"],
            candles["close"],
            period=self.period,
            multiplier=self.multiplier,
        )
        return self._signal_at(frame, candles["close"], len(candles) - 1)

    def generate_signal_series(self, candles: pd.DataFrame) -> list[Signal]:
        frame = supertrend(
            candles["high"],
            candles["low"],
            candles["close"],
            period=self.period,
            multiplier=self.multiplier,
        )
        return [self._signal_at(frame, candles["close"], i) for i in range(len(candles))]
