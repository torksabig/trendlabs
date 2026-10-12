from __future__ import annotations

import argparse
import json

from trendlabs.config import DEFAULT_TIMEFRAME, SYMBOLS
from trendlabs.db import Store
from trendlabs.simulator import PaperSimulator
from trendlabs.snapshot import write_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TrendLab paper-trading simulation")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    args = parser.parse_args()

    store = Store()
    available = store.list_symbols_with_data(args.timeframe)
    symbols = [s for s in args.symbols if s in available]
    if not symbols:
        raise SystemExit(f"No candle data for timeframe {args.timeframe}. Run collect first.")

    result = PaperSimulator(store).run(symbols=symbols, timeframe=args.timeframe)
    write_snapshot(store, timeframe=args.timeframe)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
