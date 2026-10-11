from __future__ import annotations

import pandas as pd

from trading_sim.agents.base import Signal, TradingAgent
from trading_sim.indicators import roc, sma


class MomentumAgent(TradingAgent):
    id = "momentum"
    label = "Momentum"

    def __init__(self, roc_period: int = 10, sma_period: int = 50) -> None:
        self.roc_period = roc_period
        self.sma_period = sma_period

    def _signal_at(self, close: pd.Series, momentum: pd.Series, trend: pd.Series, i: int) -> Signal:
        need = max(self.roc_period, self.sma_period) + 1
        if i + 1 < need:
            return Signal("flat", 0.0, "Insufficient history for momentum")
        m = float(momentum.iloc[i])
        price = float(close.iloc[i])
        ma = float(trend.iloc[i])
        if pd.isna(m) or pd.isna(ma):
            return Signal("flat", 0.0, "Momentum warming up")
        if m > 0 and price > ma:
            conf = min(1.0, abs(m) / 5.0 + 0.35)
            return Signal(
                "long",
                conf,
                f"ROC({self.roc_period}) {m:.2f}% and price above SMA({self.sma_period})",
            )
        if m < 0 and price < ma:
            return Signal("flat", min(1.0, abs(m) / 5.0 + 0.35), f"Negative momentum {m:.2f}% below SMA")
        return Signal("flat", 0.25, f"Mixed momentum: ROC {m:.2f}%, price vs SMA {price - ma:.2f}")

    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        close = candles["close"]
        return self._signal_at(close, roc(close, self.roc_period), sma(close, self.sma_period), len(candles) - 1)

    def generate_signal_series(self, candles: pd.DataFrame) -> list[Signal]:
        close = candles["close"]
        momentum = roc(close, self.roc_period)
        trend = sma(close, self.sma_period)
        return [self._signal_at(close, momentum, trend, i) for i in range(len(candles))]
