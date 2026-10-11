from trading_sim.agents.adx_trend import AdxTrendAgent
from trading_sim.agents.base import Signal, TradingAgent
from trading_sim.agents.ema_crossover import EmaCrossoverAgent
from trading_sim.agents.momentum import MomentumAgent
from trading_sim.agents.supertrend import SupertrendAgent

AGENT_DESCRIPTIONS = {
    "ema_crossover": "Long when fast EMA crosses above slow EMA; flat on cross below.",
    "adx_trend": "Enter long when ADX > 25 and +DI > -DI; exit when trend weakens or flips.",
    "momentum": "Long when rate-of-change is positive and price is above its SMA; exit on fade.",
    "supertrend": "Follow Supertrend direction: long in uptrend, flat in downtrend.",
}


def build_agents() -> dict[str, TradingAgent]:
    return {
        "ema_crossover": EmaCrossoverAgent(),
        "adx_trend": AdxTrendAgent(),
        "momentum": MomentumAgent(),
        "supertrend": SupertrendAgent(),
    }


__all__ = [
    "AGENT_DESCRIPTIONS",
    "AdxTrendAgent",
    "EmaCrossoverAgent",
    "MomentumAgent",
    "Signal",
    "SupertrendAgent",
    "TradingAgent",
    "build_agents",
]
