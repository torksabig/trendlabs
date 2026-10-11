from __future__ import annotations

import runpy
import sys

USAGE = """TrendLab workers

Usage:
  python -m trendlabs.collect [--symbols AAPL MSFT] [--timeframe 1h]
  python -m trendlabs.simulate [--timeframe 1h]
  python -m trendlabs.seed [--timeframe 1h]
"""


def main() -> int:
    if len(sys.argv) < 2:
        print(USAGE)
        return 1
    cmd = sys.argv[1]
    sys.argv = [f"trendlabs.{cmd}", *sys.argv[2:]]
    runpy.run_module(f"trendlabs.{cmd}", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
