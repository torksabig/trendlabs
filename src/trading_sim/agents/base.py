from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Side = Literal["long", "flat"]


@dataclass(frozen=True)
class Signal:
    side: Side
    confidence: float
    reason: str


class TradingAgent(ABC):
    id: str
    label: str

    @abstractmethod
    def generate_signal(self, candles: pd.DataFrame) -> Signal:
        """Return the desired position side given OHLCV history ending at the latest bar."""

    def generate_signal_series(self, candles: pd.DataFrame) -> list[Signal]:
        """Default: per-bar calls. Override for vectorized backtests."""
        return [self.generate_signal(candles.iloc[: i + 1]) for i in range(len(candles))]
