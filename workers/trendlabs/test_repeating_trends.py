"""Tests for repeating-trend motif discovery."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trendlabs.db import Store
from trendlabs.repeating_trends import find_repeating_trends, scan_universe


def _synthetic_series(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Build OHLCV with an intentional repeating up-ramp motif."""
    rng = np.random.default_rng(seed)
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    price = 100.0
    rows = []
    # Inject the same 20-bar rising shape every 45 bars.
    motif = np.linspace(0, 0.08, 20)
    for i in range(n):
        ts = now + timedelta(days=i)
        phase = i % 45
        if phase < 20:
            ret = float(motif[phase] - (motif[phase - 1] if phase else 0.0))
        else:
            ret = float(rng.normal(0, 0.004))
        price *= 1.0 + ret
        rows.append(
            {
                "ts": ts,
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 1_000_000,
                "symbol": "AAPL",
                "timeframe": "1d",
                "source": "test",
                "adjusted": True,
            }
        )
    return pd.DataFrame(rows)


class RepeatingTrendTests(unittest.TestCase):
    def test_finds_injected_repeating_motif(self) -> None:
        df = _synthetic_series()
        found = find_repeating_trends(
            df,
            symbol="AAPL",
            timeframe="1d",
            window=20,
            horizon=5,
            min_corr=0.85,
            min_occurrences=4,
            recent_anchors=30,
        )
        self.assertGreaterEqual(len(found), 1)
        top = found[0]
        self.assertGreaterEqual(top.occurrences, 4)
        self.assertIn(top.direction, {"up", "down", "mixed"})

    def test_scan_universe_uses_store(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "t.db")
        df = _synthetic_series(n=300)
        store.upsert_candles(df)
        payload = scan_universe(
            symbols=["AAPL"],
            timeframe="1d",
            window=20,
            horizon=5,
            min_corr=0.85,
            min_occurrences=3,
            store=store,
        )
        self.assertGreaterEqual(payload["pattern_count"], 1)
        self.assertIn("AAPL", payload["by_symbol"])


if __name__ == "__main__":
    unittest.main()
