from __future__ import annotations

import pandas as pd

from trading_sim.agents.base import Signal, TradingAgent
from trading_sim.indicators import ema


class EmaCrossoverAgent(TradingAgent):
    id = "ema_crossover"
    label = "EMA Crossover"

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        self.fast = fast
        self.slow = slow

    def _signal_at(self, close: pd.Series, fast_line: pd.Series, slow_line: pd.Series, i: int) -> Signal:
        if i < self.slow + 1:
            return Signal("flat", 0.0, "Insufficient history for EMA crossover")
        prev_diff = float(fast_line.iloc[i - 1] - slow_line.iloc[i - 1])
        curr_diff = float(fast_line.iloc[i] - slow_line.iloc[i])
        price = float(close.iloc[i])
        conf = min(1.0, abs(curr_diff) / max(price * 0.01, 1e-9))
        if curr_diff > 0:
            crossed = prev_diff <= 0
            reason = (
                f"Bullish EMA({self.fast}/{self.slow}) cross"
                if crossed
                else f"Fast EMA above slow EMA ({curr_diff:.2f})"
            )
            return Signal("long", conf, reason)
        reason = (
            f"Bearish EMA({self.fast}/{self.slow}) cross"
            if prev_diff >= 0
            else f"Fast EMA below slow EMA ({curr_diff:.2f})"
        )
        return Signal("flat", conf, reason)

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        close = candles["close"]
        return self._signal_at(close, ema(close, self.fast), ema(close, self.slow), len(candles) - 1)

    def generate_signal_series(self, candles: pd.DataFrame) -> list[Signal]:
        close = candles["close"]
        fast_line = ema(close, self.fast)
        slow_line = ema(close, self.slow)
        return [self._signal_at(close, fast_line, slow_line, i) for i in range(len(candles))]
