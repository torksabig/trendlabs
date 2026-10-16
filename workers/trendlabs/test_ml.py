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


class PhaseBTests(unittest.TestCase):
    """Phase B: feature-version freeze, no-lookahead, embargo, feature snapshots."""

    def test_feature_frame_no_lookahead_perturbation(self) -> None:
        from trendlabs.features import FEATURE_VERSION

        df = _synth(n=400, seed=5)
        base = build_feature_frame(df)
        self.assertIsInstance(FEATURE_VERSION, str)
        self.assertTrue(FEATURE_VERSION.startswith("fv1-"))

        pert = df.copy()
        last = pert.index[-1]
        pert.loc[last, "close"] *= 2.0
        pert.loc[last, "high"] *= 2.0
        pert.loc[last, "low"] *= 2.0
        pert.loc[last, "volume"] *= 5.0
        after = build_feature_frame(pert)

        cols = FEATURE_COLUMNS
        np.testing.assert_allclose(
            base[cols].iloc[:-1].to_numpy(dtype=float),
            after[cols].iloc[:-1].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg="features of rows < last changed when the last candle changed → lookahead",
        )
        # The last row itself must react (sanity: test is not vacuous)
        self.assertFalse(np.allclose(base[cols].iloc[-1].to_numpy(dtype=float), after[cols].iloc[-1].to_numpy(dtype=float), equal_nan=True))

    def test_time_split_embargo_gap(self) -> None:
        import trendlabs.ml_train as mt

        n = 200
        train_idx, test_idx = mt._time_split(n, test_frac=0.2, embargo=5)
        # test_n = min(max(min_test=50, int(n*0.2)), n//3) = 50 for n=200
        test_n = min(max(50, int(n * 0.2)), n // 3)
        self.assertEqual(train_idx[-1], n - test_n - 5 - 1)  # train ends embargo rows before test
        self.assertEqual(test_idx[0], n - test_n)  # test window size preserved
        self.assertEqual(len(test_idx), test_n)
        gap = set(range(train_idx[-1] + 1, test_idx[0]))
        self.assertEqual(len(gap), 5)
        self.assertFalse(gap & set(train_idx.tolist()))
        self.assertFalse(gap & set(test_idx.tolist()))

    def test_market_features_roundtrip_and_prediction_lineage(self) -> None:
        from trendlabs.features import FEATURE_VERSION

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "b.db")
        store.init_schema()

        df = _synth("AAPL", n=300, seed=9)
        feat = build_feature_frame(df).dropna(subset=FEATURE_COLUMNS).iloc[60:]
        n = store.upsert_market_features(
            feat, symbol="AAPL", timeframe="1d",
            feature_version=FEATURE_VERSION, columns=FEATURE_COLUMNS,
        )
        self.assertEqual(n, len(feat))

        got = store.get_market_features("AAPL", "1d", feature_version=FEATURE_VERSION)
        self.assertEqual(len(got), len(feat))
        self.assertTrue((got["feature_version"] == FEATURE_VERSION).all())
        # spot-check a few values survived the JSON roundtrip
        self.assertEqual(len(got), len(feat))
        for col in ("ret_1", "rsi_norm", "close_location"):
            np.testing.assert_allclose(
                got[col].to_numpy(dtype=float),
                feat[col].to_numpy(dtype=float),
                rtol=1e-9,
                equal_nan=True,
            )
        # other-version reads stay empty
        self.assertTrue(store.get_market_features("AAPL", "1d", feature_version="fv0-deadbeef").empty)

        # predictions carry feature_version
        row = feat.iloc[-1]
        pid = store.insert_prediction(
            {
                "symbol": "AAPL",
                "timeframe": "1d",
                "horizon": "1d",
                "direction": "up",
                "confidence": 0.5,
                "last_close": float(row["close"]),
                "summary": "t",
                "model": "AAPL_1d_h1",
                "source": "xgboost",
                "next_update_at": "2026-01-01T00:00:00+00:00",
                "feature_version": FEATURE_VERSION,
            }
        )
        self.assertGreater(pid, 0)
        with store.connect() as conn:
            v = conn.execute("SELECT feature_version FROM predictions WHERE id=?", (pid,)).fetchone()
        self.assertEqual(v[0], FEATURE_VERSION)


if __name__ == "__main__":
    unittest.main()
