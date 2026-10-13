"""Historical market-data extraction for all configured TrendLab symbols.

Usage:
  python -m trendlabs.extract
  python -m trendlabs.extract --symbols AAPL MSFT --timeframes 1d 1h
  python -m trendlabs.extract --timeframes 1d --full-history

Does not invent prices. Upserts are idempotent. Safe to re-run after offline periods.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from trendlabs.config import EXTRACT_TIMEFRAMES, SYMBOLS, TIMEFRAMES
from trendlabs.db import Store
from trendlabs.market_data import MarketDataClient, MarketDataError
from trendlabs.snapshot import write_snapshot


def extract_universe(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    full_history: bool = True,
    include_incomplete: bool = False,
    write_snap: bool = True,
    store: Store | None = None,
    client: MarketDataClient | None = None,
) -> dict[str, Any]:
    symbols = [s.upper() for s in (symbols or SYMBOLS)]
    timeframes = list(timeframes or EXTRACT_TIMEFRAMES)
    for tf in timeframes:
        if tf not in TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {tf}")

    store = store or Store()
    client = client or MarketDataClient()
    started = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    total_bars = 0
    errors: list[str] = []

    run_id = store.start_ingestion_run(
        {
            "job": "extract",
            "symbols": symbols,
            "timeframes": timeframes,
            "full_history": full_history,
            "preferred_source": client.preferred_source,
            "started_at": started,
        }
    )

    for timeframe in timeframes:
        for symbol in symbols:
            entry: dict[str, Any] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "ok",
                "bars_upserted": 0,
            }
            try:
                df = client.fetch(
                    symbol,
                    timeframe,
                    full_history=full_history,
                    include_incomplete=include_incomplete,
                )
                n = store.upsert_candles(df)
                total_bars += n
                quality = client.last_quality.to_dict() if client.last_quality else {}
                entry.update(
                    {
                        "bars_upserted": n,
                        "source": str(df.iloc[0]["source"]),
                        "first_ts": quality.get("first_ts"),
                        "last_ts": quality.get("last_ts"),
                        "quality": quality,
                    }
                )
                print(
                    f"{symbol} {timeframe}: upserted {n} "
                    f"({quality.get('first_ts')} → {quality.get('last_ts')}) "
                    f"src={entry['source']}"
                )
            except (MarketDataError, Exception) as exc:  # noqa: BLE001
                msg = f"{symbol} {timeframe}: {exc}"
                errors.append(msg)
                entry["status"] = "error"
                entry["error"] = str(exc)
                print(f"ERROR {msg}")
            results.append(entry)

    finished = datetime.now(timezone.utc).isoformat()
    summary = {
        "run_id": run_id,
        "started_at": started,
        "finished_at": finished,
        "preferred_source": client.preferred_source,
        "symbols": symbols,
        "timeframes": timeframes,
        "full_history": full_history,
        "total_bars_upserted": total_bars,
        "error_count": len(errors),
        "errors": errors,
        "results": results,
        "candle_stats": store.candle_stats(symbols=symbols, timeframes=timeframes),
    }

    store.finish_ingestion_run(
        run_id,
        status="ok" if total_bars > 0 else "error",
        details=summary,
    )
    store.set_meta("last_extraction", finished)
    store.set_meta("last_extraction_summary", json.dumps({
        "run_id": run_id,
        "total_bars_upserted": total_bars,
        "error_count": len(errors),
        "preferred_source": client.preferred_source,
        "finished_at": finished,
    }))
    store.set_meta("data_source", client.preferred_source)

    # Refresh snapshot using the first successfully extracted timeframe, else 1d/1h.
    if write_snap:
        snap_tf = next(
            (r["timeframe"] for r in results if r["status"] == "ok"),
            "1d" if "1d" in timeframes else timeframes[0],
        )
        try:
            write_snapshot(store, timeframe=snap_tf)
        except Exception as exc:  # noqa: BLE001
            summary["snapshot_error"] = str(exc)
            print(f"snapshot warning: {exc}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract historical OHLCV for configured TrendLab stocks"
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Subset of configured symbols (default: all INSTRUMENTS)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help=f"Timeframes to extract (default: {EXTRACT_TIMEFRAMES})",
    )
    parser.add_argument(
        "--full-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pull max practical history per timeframe (default: true)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Keep the latest unfinished candle (default: drop it)",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip writing dashboard snapshot JSON",
    )
    args = parser.parse_args()

    if args.symbols:
        unknown = sorted(set(s.upper() for s in args.symbols) - set(SYMBOLS))
        if unknown:
            raise SystemExit(
                f"Unknown symbols (not in TrendLab universe): {', '.join(unknown)}"
            )

    summary = extract_universe(
        symbols=args.symbols,
        timeframes=args.timeframes,
        full_history=args.full_history,
        include_incomplete=args.include_incomplete,
        write_snap=not args.no_snapshot,
    )

    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "total_bars_upserted": summary["total_bars_upserted"],
                "error_count": summary["error_count"],
                "preferred_source": summary["preferred_source"],
                "candle_stats": summary["candle_stats"],
            },
            indent=2,
        )
    )
    return 0 if summary["total_bars_upserted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
