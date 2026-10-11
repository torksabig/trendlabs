"""Collect market data, run paper agents, then produce a shared research report."""

from __future__ import annotations

import argparse
import sys

from trendlabs.collect import main as collect_main
from trendlabs.config import DEFAULT_TIMEFRAME
from trendlabs.research import main as research_main
from trendlabs.simulate import main as simulate_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed TrendLab end-to-end")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--research-symbol", default="AAPL")
    parser.add_argument("--no-ollama", action="store_true")
    args = parser.parse_args()

    collect_args = ["--timeframe", args.timeframe]
    if args.symbols:
        collect_args += ["--symbols", *args.symbols]

    old = sys.argv
    try:
        sys.argv = ["trendlabs.collect", *collect_args]
        code = collect_main()
        if code != 0:
            return code

        sim_args = ["--timeframe", args.timeframe]
        if args.symbols:
            sim_args += ["--symbols", *args.symbols]
        sys.argv = ["trendlabs.simulate", *sim_args]
        code = simulate_main()
        if code != 0:
            return code

        research_args = [
            "--symbol",
            args.research_symbol,
            "--timeframe",
            args.timeframe,
        ]
        if args.no_ollama:
            research_args.append("--no-ollama")
        sys.argv = ["trendlabs.research", *research_args]
        return research_main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
