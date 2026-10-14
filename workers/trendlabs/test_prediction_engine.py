"""Unit tests for Prediction Engine V1 (leakage, targets, purge, determinism, scaling)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from trendlabs.db import Store
from trendlabs.prediction import HORIZON_DAYS, PURGE_GAP_DAYS
from trendlabs.prediction.baselines import (
    fit_ridge_predict,
    run_baselines_walk_forward,
    score_random,
)
from trendlabs.prediction.dataset import build_research_dataset, write_dataset
from trendlabs.prediction.features import FEATURE_COLUMNS, symbol_features
from trendlabs.prediction.targets import actionable_return, attach_targets, sigma_at_t, daily_log_returns
from trendlabs.prediction.train import fit_xgb_predict
from trendlabs.prediction.walk_forward import generate_purged_folds, purge_gap_ok


def _synth_ohlcv(symbol: str, n: int = 800, seed: int = 7, start: str = "2010-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="B", tz="UTC")
    rets = rng.normal(0.0003, 0.015, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.001, size=n - 1))])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timeframe": "1d",
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "source": "test",
            "adjusted": 1,
        }
    )


class TargetTests(unittest.TestCase):
    def test_target_uses_open_t1_and_close_t5(self) -> None:
        df = _synth_ohlcv("TEST", n=40, seed=1)
        R = actionable_return(df["open"], df["close"], horizon=5)
        t = 10
        expected = np.log(df["close"].iloc[t + 5] / df["open"].iloc[t + 1])
        self.assertTrue(np.isclose(R.iloc[t], expected))

    def test_sigma_uses_only_past_returns(self) -> None:
        df = _synth_ohlcv("TEST", n=50, seed=2)
        log_ret = daily_log_returns(df["close"])
        sig = sigma_at_t(log_ret, window=20)
        t = 25
        window = log_ret.iloc[t - 19 : t + 1]
        self.assertEqual(len(window), 20)
        self.assertTrue(np.isclose(sig.iloc[t], window.std(ddof=1)))
        # Future return must not enter sigma
        self.assertFalse(np.isclose(sig.iloc[t], log_ret.iloc[t - 19 : t + 2].std(ddof=1)))


class FeatureLeakageTests(unittest.TestCase):
    def test_no_future_bars_in_features(self) -> None:
        df = _synth_ohlcv("AAPL", n=400, seed=3)
        feat = symbol_features(df)
        t = 300
        # Corrupt future closes/volumes; features at t must stay identical
        poisoned = df.copy()
        poisoned.loc[t + 1 :, "close"] = poisoned.loc[t + 1 :, "close"] * 10
        poisoned.loc[t + 1 :, "volume"] = poisoned.loc[t + 1 :, "volume"] * 10
        poisoned.loc[t + 1 :, "high"] = poisoned.loc[t + 1 :, "high"] * 10
        poisoned.loc[t + 1 :, "low"] = poisoned.loc[t + 1 :, "low"] * 0.1
        feat_p = symbol_features(poisoned)
        for col in ["MOM20", "MOM60", "MOM12_1", "VOL20", "VOL120", "VOLRatio", "ATRNorm", "DollarVolume", "VolumeZ"]:
            self.assertTrue(
                np.isclose(feat[col].iloc[t], feat_p[col].iloc[t], equal_nan=True),
                msg=f"Future leak in {col}",
            )


class PurgeTests(unittest.TestCase):
    def test_purge_gap_at_least_horizon(self) -> None:
        dates = list(pd.date_range("2015-01-01", periods=800, freq="B", tz="UTC"))
        folds = generate_purged_folds(dates, min_train_days=252, test_days=63, purge_gap=PURGE_GAP_DAYS)
        self.assertGreater(len(folds), 0)
        for f in folds:
            self.assertGreaterEqual(f.purge_gap_days, PURGE_GAP_DAYS)
            self.assertGreaterEqual(f.purge_gap_days, HORIZON_DAYS)
            self.assertTrue(purge_gap_ok(f.train_end_date, f.test_start_date, dates, min_gap=PURGE_GAP_DAYS))


class DeterminismTests(unittest.TestCase):
    def test_random_baseline_deterministic(self) -> None:
        df = _synth_ohlcv("AAPL", n=100, seed=4)
        a = score_random(df, seed=42)
        b = score_random(df, seed=42)
        c = score_random(df, seed=99)
        self.assertTrue(np.allclose(a, b))
        self.assertFalse(np.allclose(a, c))


class ScalingTrainOnlyTests(unittest.TestCase):
    def test_ridge_scaler_fit_on_train_only(self) -> None:
        """Scaler/model must be fit on train only; polluting train changes OOS preds."""
        full = symbol_features(attach_targets(_synth_ohlcv("AAPL", n=700, seed=5)))
        full["RS_SPY_20"] = full["MOM20"]
        full["SPY_MOM20"] = 0.01
        full["SPY_VOL20"] = 0.01
        full["XS_Dispersion"] = 0.02
        full = full.dropna(subset=FEATURE_COLUMNS + ["z"]).reset_index(drop=True)
        split = int(len(full) * 0.75)
        train = full.iloc[:split].copy()
        test = full.iloc[split:].copy()
        self.assertGreater(len(test), 20)

        X_train = train[["MOM20"]].astype(float)
        scaler = StandardScaler().fit(X_train)
        test_shifted = test.copy()
        test_shifted["MOM20"] = test_shifted["MOM20"] + 5.0
        X_all = pd.concat([X_train, test_shifted[["MOM20"]].astype(float)], ignore_index=True)
        scaler_leaky = StandardScaler().fit(X_all)
        self.assertFalse(np.allclose(scaler.mean_, scaler_leaky.mean_, atol=1e-6))

        pred_clean = fit_ridge_predict(train, test, alpha=1.0)
        polluted = test.copy()
        polluted[FEATURE_COLUMNS] = polluted[FEATURE_COLUMNS] * 10.0
        polluted_train = pd.concat([train, polluted], ignore_index=True)
        pred_polluted = fit_ridge_predict(polluted_train, test, alpha=1.0)
        self.assertFalse(np.allclose(pred_clean.dropna(), pred_polluted.dropna()))
        pred1 = fit_ridge_predict(train, test, alpha=1.0)
        pred2 = fit_ridge_predict(train, test, alpha=1.0)
        self.assertTrue(np.allclose(pred1.dropna(), pred2.dropna()))

    def test_xgb_prediction_stable_given_train(self) -> None:
        full = symbol_features(attach_targets(_synth_ohlcv("MSFT", n=700, seed=8)))
        full["RS_SPY_20"] = full["MOM20"]
        full["SPY_MOM20"] = 0.0
        full["SPY_VOL20"] = 0.01
        full["XS_Dispersion"] = 0.01
        full = full.dropna(subset=FEATURE_COLUMNS + ["z"]).reset_index(drop=True)
        split = int(len(full) * 0.75)
        train = full.iloc[:split].copy()
        test = full.iloc[split:].copy()
        a = fit_xgb_predict(train, test)
        b = fit_xgb_predict(train, test)
        self.assertTrue(np.allclose(a.dropna(), b.dropna(), rtol=1e-5, atol=1e-5))


class IntegrationDatasetTests(unittest.TestCase):
    def test_build_and_baselines_on_synth_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            store = Store(db)
            symbols = ["AAPL", "MSFT", "NVDA", "AMD", "SPY"]
            for i, sym in enumerate(symbols):
                store.upsert_candles(_synth_ohlcv(sym, n=900, seed=10 + i))
            panel = build_research_dataset(store, symbols=symbols)
            self.assertGreater(len(panel), 100)
            self.assertTrue(set(FEATURE_COLUMNS).issubset(panel.columns))
            out = Path(tmp) / "pe"
            manifest = write_dataset(panel, out_dir=out, symbols_requested=symbols)
            self.assertIn("SURVIVORSHIP", manifest["disclaimer"])
            result = run_baselines_walk_forward(
                panel, min_train_days=252, test_days=42, purge_gap=5
            )
            self.assertIn("ridge", result["baselines"])
            self.assertGreater(result["protocol"]["n_folds"], 0)
            for gap in result["fold_purge_gaps"]:
                self.assertGreaterEqual(gap, 5)


if __name__ == "__main__":
    unittest.main()
