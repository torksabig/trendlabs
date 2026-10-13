"""Collect market data, run paper agents, then produce a shared research report."""

from __future__ import annotations

import argparse
import sys

from trendlabs.collect import main as collect_main
from trendlabs.config import DEFAULT_TIMEFRAME
from trendlabs.predict import main as predict_main
from trendlabs.research import main as research_main
from trendlabs.simulate import main as simulate_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed TrendLab end-to-end")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--research-symbol", default="AAPL")
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
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
        if args.provider:
            research_args += ["--provider", args.provider]
        if args.model:
            research_args += ["--model", args.model]
        sys.argv = ["trendlabs.research", *research_args]
        code = research_main()
        if code != 0:
            return code

        sys.argv = ["trendlabs.predict", "--timeframe", args.timeframe]
        return predict_main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
