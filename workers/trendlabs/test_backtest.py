"""Integration tests for deterministic backtest engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trendlabs.backtest import BacktestEngine, write_backtest_result
from trendlabs.db import Store
from trendlabs.strategy_config import RiskConfig


def _synth_candles(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    # Mild upward drift with noise
    rets = rng.normal(0.0002, 0.008, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, size=n))
    volume = rng.uniform(1e6, 3e6, size=n)
    return pd.DataFrame(
        {
            "symbol": "AAPL",
            "timeframe": "1h",
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "source": "test",
            "adjusted": True,
        }
    )


class BacktestEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.store = Store(self.db_path)
        aapl = _synth_candles(450, seed=1)
        spy = _synth_candles(450, seed=2)
        spy["symbol"] = "SPY"
        self.store.upsert_candles(aapl)
        self.store.upsert_candles(spy)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reproducible_run_ids_and_metrics(self) -> None:
        engine = BacktestEngine(self.store, risk=RiskConfig(starting_cash=10_000))
        a = engine.run(symbols=["AAPL", "SPY"], timeframe="1h", warm_up=220)
        b = engine.run(symbols=["AAPL", "SPY"], timeframe="1h", warm_up=220)
        self.assertEqual(a["run_id"], b["run_id"])
        self.assertEqual(a["execution_model"], "signal_on_bar_close_fill_next_open")
        self.assertTrue(a["separates_paper_trading"])
        self.assertFalse(a["look_ahead_bias"])
        self.assertEqual(len(a["agents"]), 4)
        for agent_a, agent_b in zip(a["agents"], b["agents"], strict=True):
            self.assertEqual(agent_a["metrics"]["trade_count"], agent_b["metrics"]["trade_count"])
            self.assertEqual(
                agent_a["metrics"]["total_return_pct"],
                agent_b["metrics"]["total_return_pct"],
            )

    def test_does_not_mutate_paper_trading_tables(self) -> None:
        before_trades = self.store.get_trades(limit=10)
        before_meta = self.store.get_meta("last_simulation")
        BacktestEngine(self.store).run(symbols=["AAPL", "SPY"], timeframe="1h", warm_up=220)
        self.assertEqual(self.store.get_trades(limit=10), before_trades)
        self.assertEqual(self.store.get_meta("last_simulation"), before_meta)

    def test_cash_equity_reconcile_on_curve(self) -> None:
        result = BacktestEngine(self.store).run(symbols=["AAPL", "SPY"], timeframe="1h", warm_up=220)
        for agent in result["agents"]:
            for point in agent["equity_curve"]:
                self.assertGreaterEqual(point["equity"] + 1e-6, point["cash"])

    def test_write_backtest_result(self) -> None:
        result = BacktestEngine(self.store).run(symbols=["AAPL", "SPY"], timeframe="1h", warm_up=220)
        out_dir = Path(self.tmp.name) / "bt"
        path = write_backtest_result(result, directory=out_dir)
        self.assertTrue(path.exists())
        latest = json.loads((out_dir / "latest.json").read_text())
        self.assertEqual(latest["run_id"], result["run_id"])
        self.assertIn("sharpe", latest["agents"][0]["metrics"])


if __name__ == "__main__":
    unittest.main()
