"""Tests for walk-forward fold generation and OOS evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trendlabs.db import Store
from trendlabs.strategy_config import RiskConfig
from trendlabs.walk_forward import (
    WalkForwardEngine,
    generate_bar_folds,
    write_walk_forward_result,
)


def _synth(symbol: str, n: int = 500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    rets = rng.normal(0.00015, 0.009, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, size=n))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timeframe": "1h",
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1e6, 3e6, size=n),
            "source": "test",
            "adjusted": True,
        }
    )


class WalkForwardTests(unittest.TestCase):
    def test_generate_bar_folds_non_overlapping_tests(self) -> None:
        ts = [f"t{i:04d}" for i in range(400)]
        folds = generate_bar_folds(
            ts, warm_up=100, calibrate_bars=60, test_bars=50, step_bars=50
        )
        self.assertGreaterEqual(len(folds), 2)
        for i in range(1, len(folds)):
            self.assertGreater(folds[i].test_start_ts, folds[i - 1].test_end_ts)
        for fold in folds:
            self.assertLess(fold.calibrate_end_ts, fold.test_start_ts)

    def test_walk_forward_reproducible_and_separates_paper(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "wf.db")
        store.upsert_candles(_synth("AAPL", 520, seed=1))
        store.upsert_candles(_synth("SPY", 520, seed=2))

        before_trades = store.get_trades(limit=5)
        engine = WalkForwardEngine(store, risk=RiskConfig(starting_cash=10_000))
        a = engine.run(
            symbols=["AAPL", "SPY"],
            timeframe="1h",
            warm_up=100,
            calibrate_bars=60,
            test_bars=50,
            step_bars=50,
        )
        b = engine.run(
            symbols=["AAPL", "SPY"],
            timeframe="1h",
            warm_up=100,
            calibrate_bars=60,
            test_bars=50,
            step_bars=50,
        )
        self.assertEqual(a["run_id"], b["run_id"])
        self.assertTrue(a["out_of_sample_only"])
        self.assertTrue(a["separates_paper_trading"])
        self.assertGreaterEqual(a["fold_count"], 2)
        self.assertEqual(len(a["agents"]), 4)
        self.assertEqual(store.get_trades(limit=5), before_trades)

        path = write_walk_forward_result(a, directory=Path(tmp.name) / "wf")
        self.assertTrue(path.exists())
        self.assertTrue((Path(tmp.name) / "wf" / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
