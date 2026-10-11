#!/usr/bin/env python3
"""Fetch public Binance candles, run the four-agent backtest, write a research note."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_sim.config import DEFAULT_INTERVAL, DEFAULT_LOOKBACK, DEFAULT_SYMBOL  # noqa: E402
from trading_sim.engine import SimulationEngine  # noqa: E402
from trading_sim.research_agent import ResearchAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Trend Arena with a paper backtest")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    args = parser.parse_args()

    engine = SimulationEngine()
    result = engine.backtest(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.lookback,
    )
    report = ResearchAgent(engine.db).run(prefer_llm=False)

    print(f"Symbol: {result.symbol} {result.interval}")
    print(f"Bars processed: {result.bars_processed}")
    print(f"Trades: {result.trades}")
    print("Equity:")
    for agent_id, eq in sorted(result.equity.items(), key=lambda x: -x[1]):
        print(f"  {agent_id:16s} ${eq:,.2f}  |  {result.signals.get(agent_id, '')}")
    print(f"\nResearch ({report.source}): {report.summary}")
    for p in report.proposals:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
