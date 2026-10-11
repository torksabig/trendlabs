from __future__ import annotations

import argparse
from datetime import datetime, timezone

from trendlabs.config import DEFAULT_TIMEFRAME, LOOKBACK_BARS, SYMBOLS
from trendlabs.db import Store
from trendlabs.market_data import MarketDataClient
from trendlabs.snapshot import write_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect TrendLab market candles")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--lookback", type=int, default=LOOKBACK_BARS)
    args = parser.parse_args()

    store = Store()
    client = MarketDataClient()
    total = 0
    errors: list[str] = []

    for symbol in args.symbols:
        try:
            df = client.fetch(symbol, args.timeframe, args.lookback)
            n = store.upsert_candles(df)
            total += n
            print(f"{symbol}: upserted {n} bars from {df.iloc[0]['source']}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: {exc}")
            print(f"{symbol}: ERROR {exc}")

    store.set_meta("last_collection", datetime.now(timezone.utc).isoformat())
    store.set_meta("data_source", client.preferred_source)
    store.set_meta("collection_timeframe", args.timeframe)
    write_snapshot(store, timeframe=args.timeframe)

    print(f"Done. bars={total} source={client.preferred_source} errors={len(errors)}")
    return 1 if errors and total == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
