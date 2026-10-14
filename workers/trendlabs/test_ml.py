"""Tests for ML feature matrix and XGBoost training path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trendlabs.db import Store
from trendlabs.features import FEATURE_COLUMNS, build_feature_frame, dataset_from_candles, make_labels
from trendlabs.ml_train import train_symbol, write_train_report


def _synth(symbol: str = "QQQ", n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    rets = rng.normal(0.0004, 0.012, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, size=n))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timeframe": "1d",
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, size=n),
            "source": "test",
            "adjusted": True,
        }
    )


class FeatureTests(unittest.TestCase):
    def test_dataset_shapes_and_no_future_leak_in_features(self) -> None:
        df = _synth(n=250)
        X, y, meta = dataset_from_candles(df, horizon=1, band=0.002, min_history=60)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(meta))
        self.assertFalse(X.isna().any().any())
        self.assertTrue(set(y.unique()).issubset({"up", "down", "sideways"}))
        labels = make_labels(build_feature_frame(df), horizon=1, band=0.002)
        self.assertTrue(pd.isna(labels.iloc[-1]))


class TrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "ml.db")
        self.store.upsert_candles(_synth("QQQ", n=420, seed=3))

    def test_train_symbol_writes_model(self) -> None:
        import trendlabs.ml_train as mt

        mt.MODEL_DIR = Path(self.tmp.name) / "models"
        mt.REPORT_DIR = Path(self.tmp.name)
        result = train_symbol(self.store, "QQQ", "1d", horizon=1, band=0.002)
        self.assertGreater(result["test_rows"], 10)
        self.assertTrue(Path(result["model_path"]).exists())
        self.assertIn("accuracy", result)
        path = write_train_report([result], directory=Path(self.tmp.name))
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
