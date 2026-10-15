from __future__ import annotations

import runpy
import sys

USAGE = """TrendLab workers

Usage:
  python -m trendlabs.extract [--symbols AAPL MSFT] [--timeframes 1d 1h]
  python -m trendlabs.repeating_trends [--symbols AAPL] [--timeframe 1d] [--window 20]
  python -m trendlabs.features [--symbols AAPL] [--timeframe 1d]
  python -m trendlabs.train_ml [--symbols AAPL] [--timeframe 1d]
  python -m trendlabs.ml_predict [--symbols AAPL] [--timeframe 1d]
  python -m trendlabs.ml_eval [--timeframe 1d] [--backfill]
  python -m trendlabs.collect [--symbols AAPL MSFT] [--timeframe 1h] [--full-history]
  python -m trendlabs.simulate [--timeframe 1h]
  python -m trendlabs.backtest [--timeframe 1h]
  python -m trendlabs.walk_forward [--timeframe 1h]
  python -m trendlabs.experiments [--experiment-id ema_adx20]
  python -m trendlabs.prediction_eval
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
