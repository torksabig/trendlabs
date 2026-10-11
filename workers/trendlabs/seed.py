"""Collect market data for all instruments and run the four-agent simulation."""

from __future__ import annotations

import argparse

from trendlabs.collect import main as collect_main
from trendlabs.config import DEFAULT_TIMEFRAME
from trendlabs.simulate import main as simulate_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed TrendLab end-to-end")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--symbols", nargs="*")
    args, unknown = parser.parse_known_args()

    collect_args = ["--timeframe", args.timeframe]
    if args.symbols:
        collect_args += ["--symbols", *args.symbols]
    # Temporarily patch argv for subcommands
    import sys

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
        return simulate_main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
