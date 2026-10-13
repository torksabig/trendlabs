"""Configurable strategy parameters for backtests (paper agents keep defaults)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RiskConfig:
    starting_cash: float = 10_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    risk_per_trade: float = 0.01
    max_positions: int = 3
    atr_stop_mult: float = 2.0
    min_notional: float = 25.0


@dataclass(frozen=True)
class EmaTrendParams:
    fast_ema: int = 20
    medium_ema: int = 50
    slow_ema: int = 200
    # Optional ADX gate for experiment variants (None = disabled).
    adx_min: float | None = None


@dataclass(frozen=True)
class BreakoutParams:
    entry_lookback: int = 20
    exit_lookback: int = 10


@dataclass(frozen=True)
class MomentumParams:
    momentum_bars: int = 10
    adx_min: float = 22.0


@dataclass(frozen=True)
class SupertrendParams:
    # Supertrend period/multiplier live in indicators.enrich today; kept for schema completeness.
    period: int = 10
    multiplier: float = 3.0


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "params": dict(self.params),
            "risk": asdict(self.risk),
        }


def default_strategy_configs() -> list[StrategyConfig]:
    risk = RiskConfig()
    return [
        StrategyConfig(
            strategy_id="ema_trend",
            name="Agent A — EMA Trend",
            params=asdict(EmaTrendParams()),
            risk=risk,
        ),
        StrategyConfig(
            strategy_id="breakout",
            name="Agent B — Breakout",
            params=asdict(BreakoutParams()),
            risk=risk,
        ),
        StrategyConfig(
            strategy_id="momentum",
            name="Agent C — Momentum",
            params=asdict(MomentumParams()),
            risk=risk,
        ),
        StrategyConfig(
            strategy_id="supertrend",
            name="Agent D — Supertrend",
            params=asdict(SupertrendParams()),
            risk=risk,
        ),
    ]


def config_for(strategy_id: str) -> StrategyConfig:
    for cfg in default_strategy_configs():
        if cfg.strategy_id == strategy_id:
            return cfg
    raise KeyError(f"Unknown strategy_id: {strategy_id}")


def risk_from_dict(data: dict[str, Any] | None) -> RiskConfig:
    base = asdict(RiskConfig())
    if data:
        base.update({k: data[k] for k in base if k in data})
    return RiskConfig(**base)


def config_from_dict(data: dict[str, Any]) -> StrategyConfig:
    return StrategyConfig(
        strategy_id=str(data["strategy_id"]),
        name=str(data.get("name") or data["strategy_id"]),
        params=dict(data.get("params") or {}),
        risk=risk_from_dict(data.get("risk")),
    )


def override_config(
    base: StrategyConfig,
    *,
    params: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    name: str | None = None,
) -> StrategyConfig:
    merged_params = dict(base.params)
    if params:
        merged_params.update(params)
    risk_dict = asdict(base.risk)
    if risk:
        risk_dict.update(risk)
    return StrategyConfig(
        strategy_id=base.strategy_id,
        name=name or base.name,
        params=merged_params,
        risk=RiskConfig(**risk_dict),
    )
