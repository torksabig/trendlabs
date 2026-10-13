"""Tests for experiment accept/reject logic and engine isolation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trendlabs.db import Store
from trendlabs.experiments import (
    AcceptanceCriteria,
    ExperimentEngine,
    ExperimentSpec,
    catalog_experiments,
    decide,
    write_experiment_result,
)
from trendlabs.strategy_config import RiskConfig, config_for, override_config


def _synth(symbol: str, n: int = 520, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    rets = rng.normal(0.0002, 0.01, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, size=n))
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


class DecideTests(unittest.TestCase):
    def test_accept_on_sharpe_lift(self) -> None:
        decision, reasons = decide(
            {"mean_fold_sharpe": 0.5, "max_drawdown_pct": -10.0, "trade_count": 40},
            {"mean_fold_sharpe": 0.8, "max_drawdown_pct": -9.0, "trade_count": 30},
            criteria=AcceptanceCriteria(min_trades=10, min_sharpe_lift=0.15),
            fold_comparisons=[
                {"variant_ge_control": True},
                {"variant_ge_control": True},
            ],
        )
        self.assertEqual(decision, "accept")
        self.assertTrue(reasons)

    def test_reject_on_weak_lift(self) -> None:
        decision, _ = decide(
            {"mean_fold_sharpe": 0.5, "max_drawdown_pct": -10.0, "trade_count": 40},
            {"mean_fold_sharpe": 0.55, "max_drawdown_pct": -9.0, "trade_count": 30},
            criteria=AcceptanceCriteria(min_trades=10, min_sharpe_lift=0.15),
        )
        self.assertEqual(decision, "reject")

    def test_insufficient_trades(self) -> None:
        decision, _ = decide(
            {"mean_fold_sharpe": 0.1, "max_drawdown_pct": -5.0, "trade_count": 40},
            {"mean_fold_sharpe": 1.0, "max_drawdown_pct": -4.0, "trade_count": 3},
            criteria=AcceptanceCriteria(min_trades=12),
        )
        self.assertEqual(decision, "insufficient")


class ExperimentEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "exp.db")
        self.store.upsert_candles(_synth("AAPL", 520, seed=1))
        self.store.upsert_candles(_synth("SPY", 520, seed=2))

    def test_catalog_nonempty(self) -> None:
        self.assertGreaterEqual(len(catalog_experiments()), 3)

    def test_evaluate_does_not_touch_paper_trades(self) -> None:
        before = self.store.get_trades(limit=5)
        ema = config_for("ema_trend")
        spec = ExperimentSpec(
            experiment_id="test_adx",
            hypothesis="ADX gate should change trade count.",
            control=ema,
            variant=override_config(ema, params={"adx_min": 20.0}, name="EMA+ADX"),
        )
        engine = ExperimentEngine(
            self.store,
            criteria=AcceptanceCriteria(min_trades=1, min_sharpe_lift=0.0, min_fold_win_rate=0.0),
        )
        result = engine.evaluate(
            spec,
            symbols=["AAPL", "SPY"],
            timeframe="1h",
            warm_up=100,
            calibrate_bars=60,
            test_bars=50,
            step_bars=50,
        )
        self.assertIn(result["decision"], {"accept", "reject", "insufficient"})
        self.assertTrue(result["separates_paper_trading"])
        self.assertFalse(result["mutates_production_strategy"])
        self.assertEqual(self.store.get_trades(limit=5), before)
        path = write_experiment_result(result, directory=Path(self.tmp.name) / "exps")
        self.assertTrue(path.exists())
        stored = self.store.get_experiment_results(limit=1)
        self.assertEqual(stored[0]["run_id"], result["run_id"])


if __name__ == "__main__":
    unittest.main()
