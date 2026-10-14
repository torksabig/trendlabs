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
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Fetch max available history for the timeframe (Yahoo caps: 1h≈730d, 1d=all).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Optional list of timeframes to collect (overrides --timeframe).",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip snapshot rewrite (useful for bulk historical loads).",
    )
    args = parser.parse_args()

    store = Store()
    client = MarketDataClient()
    total = 0
    errors: list[str] = []
    timeframes = args.timeframes or [args.timeframe]

    for timeframe in timeframes:
        for symbol in args.symbols:
            try:
                df = client.fetch(
                    symbol,
                    timeframe,
                    args.lookback,
                    full_history=args.full_history,
                )
                n = store.upsert_candles(df)
                total += n
                src = df.iloc[0]["source"] if len(df) else "?"
                print(
                    f"{symbol} {timeframe}: upserted {n} bars "
                    f"({df.iloc[0]['ts']} → {df.iloc[-1]['ts']}) from {src}"
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol} {timeframe}: {exc}")
                print(f"{symbol} {timeframe}: ERROR {exc}")

    store.set_meta("last_collection", datetime.now(timezone.utc).isoformat())
    store.set_meta("data_source", client.preferred_source)
    store.set_meta("collection_timeframe", ",".join(timeframes))
    if not args.no_snapshot:
        # Snapshot still uses default/primary timeframe for the Arena.
        write_snapshot(store, timeframe=args.timeframe if args.timeframe in timeframes else timeframes[0])

    print(f"Done. bars={total} source={client.preferred_source} errors={len(errors)}")
    return 1 if errors and total == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
