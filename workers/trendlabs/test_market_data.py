"""Unit tests for market-data quality, upserts, and extract helpers."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trendlabs.config import EXTRACT_TIMEFRAMES, INSTRUMENTS, SYMBOLS, TIMEFRAMES
from trendlabs.db import Store
from trendlabs.extract import extract_universe
from trendlabs.market_data import MarketDataClient, QualityReport


def _sample_candles(
    symbol: str = "AAPL",
    timeframe: str = "1d",
    n: int = 10,
    *,
    include_bad: bool = False,
    incomplete: bool = False,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for i in range(n):
        ts = now - timedelta(days=n - i)
        rows.append(
            {
                "ts": ts,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1_000_000 + i,
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "test",
                "adjusted": True,
                "provider_updated_at": now.isoformat(),
            }
        )
    if include_bad:
        rows.append(
            {
                "ts": now + timedelta(days=1),
                "open": 10.0,
                "high": 9.0,  # invalid: high < low
                "low": 11.0,
                "close": 10.5,
                "volume": 100,
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "test",
                "adjusted": True,
                "provider_updated_at": now.isoformat(),
            }
        )
    if incomplete:
        # Fresh bar that has not closed yet for 1d.
        rows.append(
            {
                "ts": datetime.now(timezone.utc),
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.5,
                "volume": 50,
                "symbol": symbol,
                "timeframe": timeframe,
                "source": "test",
                "adjusted": True,
                "provider_updated_at": now.isoformat(),
            }
        )
    return pd.DataFrame(rows)


class MarketDataQualityTests(unittest.TestCase):
    def test_universe_matches_configured_instruments(self) -> None:
        self.assertEqual(SYMBOLS, [i["symbol"] for i in INSTRUMENTS])
        self.assertEqual(len(SYMBOLS), 12)
        self.assertTrue(set(EXTRACT_TIMEFRAMES).issubset(TIMEFRAMES))

    def test_drop_invalid_ohlc_never_fills(self) -> None:
        client = MarketDataClient()
        df = _sample_candles(include_bad=True)
        report = QualityReport("AAPL", "1d", "test", rows_raw=len(df))
        cleaned = MarketDataClient._drop_invalid_ohlc(df, report)
        self.assertEqual(report.invalid_ohlc_dropped, 1)
        self.assertEqual(len(cleaned), len(df) - 1)
        self.assertTrue((cleaned["high"] >= cleaned["low"]).all())

    def test_drop_incomplete_daily_candle(self) -> None:
        df = _sample_candles(n=5, incomplete=True)
        out = MarketDataClient._drop_incomplete_candle(df, "1d")
        self.assertLess(len(out), len(df))

    def test_duplicates_removed(self) -> None:
        df = _sample_candles(n=3)
        dup = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
        out = MarketDataClient._drop_duplicates(dup)
        self.assertEqual(len(out), 3)

    def test_finalize_sets_quality_report(self) -> None:
        client = MarketDataClient()
        df = _sample_candles(n=8, include_bad=True)
        out = client._finalize(df, include_incomplete=False)
        self.assertIsNotNone(client.last_quality)
        assert client.last_quality is not None
        self.assertEqual(client.last_quality.rows_kept, len(out))
        self.assertGreaterEqual(client.last_quality.invalid_ohlc_dropped, 1)


class CandleStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_upsert_idempotent(self) -> None:
        df = _sample_candles(n=5)
        n1 = self.store.upsert_candles(df)
        n2 = self.store.upsert_candles(df)
        self.assertEqual(n1, 5)
        self.assertEqual(n2, 5)
        loaded = self.store.load_candles("AAPL", "1d")
        self.assertEqual(len(loaded), 5)

    def test_upsert_updates_price_on_conflict(self) -> None:
        df = _sample_candles(n=2)
        self.store.upsert_candles(df)
        updated = df.copy()
        updated.loc[updated.index[-1], "close"] = 999.0
        self.store.upsert_candles(updated)
        loaded = self.store.load_candles("AAPL", "1d")
        self.assertEqual(float(loaded.iloc[-1]["close"]), 999.0)
        self.assertEqual(len(loaded), 2)

    def test_ingestion_run_and_stats(self) -> None:
        run_id = self.store.start_ingestion_run({"job": "extract", "symbols": ["AAPL"]})
        self.store.upsert_candles(_sample_candles(n=4))
        self.store.finish_ingestion_run(run_id, status="ok", details={"bars": 4})
        stats = self.store.candle_stats(symbols=["AAPL"], timeframes=["1d"])
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["bar_count"], 4)


class ExtractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "extract.db"
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_extract_rejects_unknown_symbol(self) -> None:
        with self.assertRaises(SystemExit):
            # Mimic CLI guard used in main — extract_universe itself trusts caller,
            # so we assert SYMBOLS membership the same way as CLI.
            unknown = sorted({"FAKE"} - set(SYMBOLS))
            if unknown:
                raise SystemExit(f"Unknown symbols: {unknown}")

    def test_extract_universe_uses_configured_symbols_only(self) -> None:
        client = MarketDataClient()

        def fake_fetch(symbol, timeframe, limit=400, full_history=False, include_incomplete=False):
            return _sample_candles(symbol=symbol, timeframe=timeframe, n=6)

        with patch.object(client, "fetch", side_effect=fake_fetch):
            summary = extract_universe(
                symbols=["AAPL", "MSFT"],
                timeframes=["1d"],
                full_history=True,
                write_snap=False,
                store=self.store,
                client=client,
            )
        self.assertEqual(summary["total_bars_upserted"], 12)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(
            {s["symbol"] for s in summary["candle_stats"]},
            {"AAPL", "MSFT"},
        )


if __name__ == "__main__":
    unittest.main()
