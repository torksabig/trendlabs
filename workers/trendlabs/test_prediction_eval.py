"""Tests for prediction outcome scoring and calibration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trendlabs.db import Store
from trendlabs.prediction_eval import (
    calibration_table,
    direction_from_move,
    realized_move_pct,
    run_eval,
    score_prediction,
)


def _candles(symbol: str = "AAPL", n: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timeframe": "1h",
            "ts": ts,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1e6,
            "source": "test",
            "adjusted": True,
        }
    )


class PredictionEvalUnitTests(unittest.TestCase):
    def test_realized_move_and_direction(self) -> None:
        df = _candles()
        move = realized_move_pct(
            df,
            predicted_at="2024-06-01T00:00:00+00:00",
            last_close=100.0,
            hours=1.0,
            timeframe="1h",
        )
        self.assertIsNotNone(move)
        self.assertGreater(move, 0)
        self.assertEqual(direction_from_move(0.2), "up")
        self.assertEqual(direction_from_move(-0.2), "down")
        self.assertEqual(direction_from_move(0.01), "sideways")

    def test_calibration_buckets(self) -> None:
        outcomes = [
            {"confidence": 0.55, "hit_1h": 1, "actual_1h_pct": 0.2},
            {"confidence": 0.58, "hit_1h": 0, "actual_1h_pct": -0.1},
            {"confidence": 0.72, "hit_1h": 1, "actual_1h_pct": 0.3},
            {"confidence": 0.91, "hit_1h": 0, "actual_1h_pct": 0.05},
            {"confidence": 0.93, "hit_1h": 0, "actual_1h_pct": -0.02},
        ]
        table = calibration_table(outcomes, horizon="1h")
        labels = {r["confidence_bucket"]: r for r in table}
        self.assertEqual(labels["50–60%"]["count"], 2)
        self.assertAlmostEqual(labels["50–60%"]["accuracy"], 0.5)
        self.assertEqual(labels["90%+"]["count"], 2)
        self.assertEqual(labels["90%+"]["accuracy"], 0.0)


class PredictionEvalIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "pred.db")
        self.store.upsert_candles(_candles("AAPL", 100))

    def test_score_and_persist(self) -> None:
        pred_id = self.store.insert_prediction(
            {
                "symbol": "AAPL",
                "timeframe": "1h",
                "horizon": "1h",
                "direction": "up",
                "confidence": 0.67,
                "last_close": 100.0,
                "drivers": {"expected_move_pct": 0.4},
                "summary": "test",
                "model": "test",
                "source": "test",
                "created_at": "2024-06-01T00:00:00+00:00",
                "next_update_at": "2024-06-01T01:15:00+00:00",
            }
        )
        pred = self.store.get_predictions(limit=1)[0]
        self.assertEqual(pred["id"], pred_id)
        row = score_prediction(self.store, pred)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertIsNotNone(row["actual_1h_pct"])
        self.store.upsert_prediction_outcome(row)
        outcomes = self.store.get_prediction_outcomes(limit=10)
        self.assertEqual(len(outcomes), 1)

        payload = run_eval(self.store, limit=50)
        # already scored → scored_this_run 0 unless rescore
        self.assertEqual(payload["summary"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
