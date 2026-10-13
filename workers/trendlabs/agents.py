from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trendlabs.indicators import ema, enrich
from trendlabs.strategy_config import StrategyConfig, default_strategy_configs

Side = Literal["long", "flat"]


@dataclass(frozen=True)
class Signal:
    symbol: str
    side: Side
    stop_loss: float | None
    reason: str
    # Heuristic score in [0, 1] — not calibrated historical accuracy.
    signal_strength: float

    @property
    def confidence(self) -> float:
        """Deprecated alias — prefer signal_strength (not statistical confidence)."""
        return self.signal_strength


class BaseAgent:
    id: str
    name: str

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        raise NotImplementedError


def _frame(candles: pd.DataFrame) -> pd.DataFrame:
    return candles if "ema_20" in candles.columns else enrich(candles)


def _series_ema(df: pd.DataFrame, span: int) -> pd.Series:
    col = f"ema_{span}"
    if col in df.columns:
        return df[col]
    return ema(df["close"], span)


class EmaTrendAgent(BaseAgent):
    id = "ema_trend"
    name = "Agent A — EMA Trend"

    def __init__(
        self,
        fast_ema: int = 20,
        medium_ema: int = 50,
        slow_ema: int = 200,
        atr_stop_mult: float = 2.0,
        adx_min: float | None = None,
    ) -> None:
        self.fast_ema = fast_ema
        self.medium_ema = medium_ema
        self.slow_ema = slow_ema
        self.atr_stop_mult = atr_stop_mult
        self.adx_min = adx_min

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        if len(df) < self.slow_ema + 10:
            return Signal(symbol, "flat", None, "Warming up EMA stack", 0.0)
        row = df.iloc[-1]
        fast = float(_series_ema(df, self.fast_ema).iloc[-1])
        med = float(_series_ema(df, self.medium_ema).iloc[-1])
        slow = float(_series_ema(df, self.slow_ema).iloc[-1])
        close = float(row["close"])
        stop = (
            float(close - self.atr_stop_mult * row["atr_14"])
            if pd.notna(row["atr_14"])
            else None
        )
        adx_ok = True
        if self.adx_min is not None:
            adx_ok = float(row["adx_14"] or 0) >= self.adx_min
        bullish = fast > med > slow and close > fast and adx_ok
        if bullish:
            reason = f"EMA {self.fast_ema}>{self.medium_ema}>{self.slow_ema} with price above fast"
            if self.adx_min is not None:
                reason += f" and ADX>={self.adx_min:g}"
            return Signal(symbol, "long", stop, reason, 0.75)
        return Signal(symbol, "flat", stop, "EMA stack not bullishly aligned", 0.55)


class BreakoutAgent(BaseAgent):
    id = "breakout"
    name = "Agent B — Breakout"

    def __init__(
        self,
        entry_lookback: int = 20,
        exit_lookback: int = 10,
        atr_stop_mult: float = 2.0,
    ) -> None:
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
        self.atr_stop_mult = atr_stop_mult

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        need = max(self.entry_lookback, self.exit_lookback) + 5
        if len(df) < need:
            return Signal(symbol, "flat", None, "Warming up breakout window", 0.0)
        window = df.iloc[:-1]
        high_n = float(window["high"].tail(self.entry_lookback).max())
        low_n = float(window["low"].tail(self.exit_lookback).min())
        close = float(df.iloc[-1]["close"])
        atr = float(df.iloc[-1]["atr_14"]) if pd.notna(df.iloc[-1]["atr_14"]) else close * 0.02
        stop = close - self.atr_stop_mult * atr
        if close > high_n:
            return Signal(
                symbol,
                "long",
                stop,
                f"Close {close:.2f} broke {self.entry_lookback}-bar high {high_n:.2f}",
                0.8,
            )
        if close < low_n:
            return Signal(
                symbol,
                "flat",
                stop,
                f"Close below {self.exit_lookback}-bar low {low_n:.2f}",
                0.7,
            )
        return Signal(symbol, "flat", stop, "Inside range — no breakout", 0.35)


class MomentumAgent(BaseAgent):
    id = "momentum"
    name = "Agent C — Momentum"

    def __init__(
        self,
        momentum_bars: int = 10,
        adx_min: float = 22.0,
        atr_stop_mult: float = 2.0,
    ) -> None:
        self.momentum_bars = momentum_bars
        self.adx_min = adx_min
        self.atr_stop_mult = atr_stop_mult

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        if len(df) < max(60, self.momentum_bars + 5):
            return Signal(symbol, "flat", None, "Warming up momentum filters", 0.0)
        row = df.iloc[-1]
        prev = df.iloc[-2]
        mom = (row["close"] / df.iloc[-(self.momentum_bars + 1)]["close"] - 1) * 100
        vol_ok = row["volume"] > (row["volume_sma_20"] or 0)
        trend_ok = row["ema_20"] > row["ema_50"] and (row["adx_14"] or 0) >= self.adx_min
        stop = (
            float(row["close"] - self.atr_stop_mult * row["atr_14"])
            if pd.notna(row["atr_14"])
            else None
        )
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

    def __init__(
        self,
        atr_stop_mult: float = 2.0,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> None:
        self.atr_stop_mult = atr_stop_mult
        self.period = period
        self.multiplier = multiplier

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        df = _frame(candles)
        if len(df) < max(40, self.period + 5):
            return Signal(symbol, "flat", None, "Warming up Supertrend", 0.0)
        row = df.iloc[-1]
        # Default enrich() uses period=10, multiplier=3; custom params reserved for experiments.
        stop = float(row["supertrend"]) if pd.notna(row["supertrend"]) else None
        if int(row["st_direction"]) == 1:
            return Signal(symbol, "long", stop, f"Supertrend bullish @ {row['supertrend']:.2f}", 0.72)
        return Signal(symbol, "flat", stop, f"Supertrend bearish @ {row['supertrend']:.2f}", 0.72)


def build_agents_from_configs(configs: list[StrategyConfig] | None = None) -> list[BaseAgent]:
    configs = configs or default_strategy_configs()
    agents: list[BaseAgent] = []
    for cfg in configs:
        p = cfg.params
        atr_stop = float(cfg.risk.atr_stop_mult)
        if cfg.strategy_id == "ema_trend":
            agents.append(
                EmaTrendAgent(
                    fast_ema=int(p.get("fast_ema", 20)),
                    medium_ema=int(p.get("medium_ema", 50)),
                    slow_ema=int(p.get("slow_ema", 200)),
                    atr_stop_mult=atr_stop,
                    adx_min=float(p["adx_min"]) if p.get("adx_min") is not None else None,
                )
            )
        elif cfg.strategy_id == "breakout":
            agents.append(
                BreakoutAgent(
                    entry_lookback=int(p.get("entry_lookback", 20)),
                    exit_lookback=int(p.get("exit_lookback", 10)),
                    atr_stop_mult=atr_stop,
                )
            )
        elif cfg.strategy_id == "momentum":
            agents.append(
                MomentumAgent(
                    momentum_bars=int(p.get("momentum_bars", 10)),
                    adx_min=float(p.get("adx_min", 22.0)),
                    atr_stop_mult=atr_stop,
                )
            )
        elif cfg.strategy_id == "supertrend":
            agents.append(
                SupertrendAgent(
                    atr_stop_mult=atr_stop,
                    period=int(p.get("period", 10)),
                    multiplier=float(p.get("multiplier", 3.0)),
                )
            )
        else:
            raise KeyError(f"Unknown strategy_id: {cfg.strategy_id}")
        agents[-1].id = cfg.strategy_id
        agents[-1].name = cfg.name
    return agents


def build_agents() -> list[BaseAgent]:
    return build_agents_from_configs()
