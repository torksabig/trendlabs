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
from trendlabs.prediction.features import FEATURE_COLUMNS, attach_market_context, symbol_features
from trendlabs.prediction.targets import actionable_return, attach_targets, sigma_at_t, daily_log_returns
from trendlabs.prediction.economic import cost_from_turnover, one_way_turnover, simulate_paper_portfolio
from trendlabs.prediction.metrics_ic import assign_quintiles
from trendlabs.prediction.train import fit_xgb_predict
from trendlabs.prediction.walk_forward import generate_purged_folds, purge_gap_ok
from xgboost import XGBRegressor

# Per-symbol features that must be leakage-safe (excludes cross-sectional broadcast cols).
_SYMBOL_FEATURE_LEAK_COLS = [
    "MOM5",
    "MOM10",
    "MOM20",
    "MOM60",
    "MOM120",
    "MOM12_1",
    "MOMACCEL_10_20",
    "MOMACCEL_20_60",
    "TREND_SLOPE20",
    "TREND_R2_20",
    "TREND_SLOPE60",
    "TREND_R2_60",
    "VOL20",
    "VOL120",
    "VOLRatio",
    "ATRNorm",
    "DollarVolume",
    "VolumeZ",
    "OVERNIGHT_RET",
    "INTRADAY_RET",
    "OVERNIGHT_RET5",
    "OVERNIGHT_RET20",
    "INTRADAY_RET5",
    "INTRADAY_RET20",
]


def _fill_market_context_stub(df: pd.DataFrame) -> pd.DataFrame:
    """Fill cross-sectional / SPY-broadcast columns for single-symbol unit tests."""
    out = df.copy()
    out["RS_SPY_5"] = out.get("MOM5", 0.0)
    out["RS_SPY_20"] = out.get("MOM20", 0.0)
    out["RS_SPY_60"] = out.get("MOM60", 0.0)
    out["SPY_MOM20"] = 0.0
    out["SPY_VOL20"] = 0.01
    out["XS_Dispersion"] = 0.01
    out["BREADTH20"] = 0.5
    out["BREADTH50"] = 0.5
    out["BREADTH200"] = 0.5
    return out


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
        # Corrupt future OHLCV; features at t must stay identical
        poisoned = df.copy()
        poisoned.loc[t + 1 :, "close"] = poisoned.loc[t + 1 :, "close"] * 10
        poisoned.loc[t + 1 :, "open"] = poisoned.loc[t + 1 :, "open"] * 10
        poisoned.loc[t + 1 :, "volume"] = poisoned.loc[t + 1 :, "volume"] * 10
        poisoned.loc[t + 1 :, "high"] = poisoned.loc[t + 1 :, "high"] * 10
        poisoned.loc[t + 1 :, "low"] = poisoned.loc[t + 1 :, "low"] * 0.1
        feat_p = symbol_features(poisoned)
        for col in _SYMBOL_FEATURE_LEAK_COLS:
            self.assertTrue(
                np.isclose(feat[col].iloc[t], feat_p[col].iloc[t], equal_nan=True),
                msg=f"Future leak in {col}",
            )

    def test_breadth_and_rs_no_future_bars(self) -> None:
        """Cross-sectional RS/breadth at date t must ignore future bars."""
        frames = []
        for i, sym in enumerate(["AAPL", "MSFT", "SPY"]):
            frames.append(_synth_ohlcv(sym, n=400, seed=20 + i))
        panel = pd.concat(frames, ignore_index=True)
        parts = []
        for _, g in panel.groupby("symbol", sort=False):
            parts.append(symbol_features(g.sort_values("ts").reset_index(drop=True)))
        feat = attach_market_context(pd.concat(parts, ignore_index=True))

        poisoned_parts = []
        for _, g in panel.groupby("symbol", sort=False):
            g = g.sort_values("ts").reset_index(drop=True)
            t_local = 300
            g = g.copy()
            g.loc[t_local + 1 :, "close"] *= 10
            g.loc[t_local + 1 :, "open"] *= 10
            g.loc[t_local + 1 :, "volume"] *= 10
            poisoned_parts.append(symbol_features(g))
        feat_p = attach_market_context(pd.concat(poisoned_parts, ignore_index=True))

        ts_t = feat.loc[feat["symbol"] == "AAPL", "ts"].iloc[300]
        row = feat[(feat["symbol"] == "AAPL") & (feat["ts"] == ts_t)].iloc[0]
        row_p = feat_p[(feat_p["symbol"] == "AAPL") & (feat_p["ts"] == ts_t)].iloc[0]
        for col in [
            "RS_SPY_5",
            "RS_SPY_20",
            "RS_SPY_60",
            "BREADTH20",
            "BREADTH50",
            "BREADTH200",
            "SPY_MOM20",
            "SPY_VOL20",
            "XS_Dispersion",
        ]:
            self.assertTrue(
                np.isclose(row[col], row_p[col], equal_nan=True),
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
        full = _fill_market_context_stub(symbol_features(attach_targets(_synth_ohlcv("AAPL", n=700, seed=5))))
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
        full = _fill_market_context_stub(symbol_features(attach_targets(_synth_ohlcv("MSFT", n=700, seed=8))))
        full = full.dropna(subset=FEATURE_COLUMNS + ["z"]).reset_index(drop=True)
        split = int(len(full) * 0.75)
        train = full.iloc[:split].copy()
        test = full.iloc[split:].copy()
        a = fit_xgb_predict(train, test)
        b = fit_xgb_predict(train, test)
        self.assertTrue(np.allclose(a.dropna(), b.dropna(), rtol=1e-5, atol=1e-5))

    def test_xgb_does_not_use_standard_scaler(self) -> None:
        """XGBoost must fit/predict on raw features; Ridge still scales."""
        import trendlabs.prediction.train as train_mod

        full = _fill_market_context_stub(symbol_features(attach_targets(_synth_ohlcv("NVDA", n=700, seed=9))))
        full = full.dropna(subset=FEATURE_COLUMNS + ["z"]).reset_index(drop=True)
        split = int(len(full) * 0.75)
        train = full.iloc[:split].copy()
        test = full.iloc[split:].copy()

        # Guard: StandardScaler must not be imported or constructed in the XGB train module
        self.assertFalse(hasattr(train_mod, "StandardScaler"))
        import inspect

        mod_src = inspect.getsource(train_mod)
        self.assertNotIn("from sklearn.preprocessing import StandardScaler", mod_src)
        self.assertNotIn("StandardScaler()", mod_src)

        pe_pred = fit_xgb_predict(train, test)
        # Manual unscaled XGB with same hyperparams defaults used in train module
        from trendlabs.prediction.train import XGB_PARAMS

        model = XGBRegressor(**XGB_PARAMS)
        X_tr = train[FEATURE_COLUMNS].astype(float)
        y_tr = train["z"].astype(float)
        ok_tr = X_tr.notna().all(axis=1) & y_tr.notna()
        model.fit(X_tr.loc[ok_tr], y_tr.loc[ok_tr])
        X_te = test[FEATURE_COLUMNS].astype(float)
        ok_te = X_te.notna().all(axis=1)
        manual = pd.Series(np.nan, index=test.index, dtype=float)
        manual.loc[ok_te] = model.predict(X_te.loc[ok_te])
        self.assertTrue(np.allclose(pe_pred.dropna(), manual.dropna(), rtol=1e-5, atol=1e-5))

        # Ridge path still uses StandardScaler (contrast)
        ridge_src = inspect.getsource(fit_ridge_predict)
        self.assertIn("StandardScaler()", ridge_src)


class QuintileDeterminismTests(unittest.TestCase):
    def test_quintiles_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        scores = pd.Series(rng.normal(size=50))
        a = assign_quintiles(scores)
        b = assign_quintiles(scores)
        self.assertTrue(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))
        self.assertEqual(set(a.dropna().unique()), {1.0, 2.0, 3.0, 4.0, 5.0})


class EconomicCostTests(unittest.TestCase):
    def test_turnover_and_cost(self) -> None:
        prev = {"A": 0.5, "B": 0.5}
        nxt = {"A": 0.0, "B": 0.5, "C": 0.5}
        # Δ: A -0.5, B 0, C +0.5 → sum abs = 1.0 → one-way = 0.5
        self.assertAlmostEqual(one_way_turnover(prev, nxt), 0.5)
        self.assertAlmostEqual(cost_from_turnover(0.5, 10.0), 0.5 * 10 / 10_000)
        self.assertAlmostEqual(cost_from_turnover(1.0, 20.0), 20 / 10_000)

    def test_no_same_bar_entry_in_target(self) -> None:
        df = _synth_ohlcv("TEST", n=40, seed=11)
        R = actionable_return(df["open"], df["close"], horizon=5)
        t = 10
        # Same-bar (wrong) would use close[t]; correct uses open[t+1]
        wrong = np.log(df["close"].iloc[t + 5] / df["close"].iloc[t])
        right = np.log(df["close"].iloc[t + 5] / df["open"].iloc[t + 1])
        self.assertTrue(np.isclose(R.iloc[t], right))
        self.assertFalse(np.isclose(R.iloc[t], wrong) and not np.isclose(right, wrong))

    def test_simulate_applies_costs(self) -> None:
        # Two signal dates, full turnover each time
        holdings = pd.DataFrame(
            [
                {"ts": pd.Timestamp("2020-01-02", tz="UTC"), "long_symbols": ("A", "B"), "short_symbols": (), "gross_R": 0.01, "n_long": 2, "n_short": 0},
                {"ts": pd.Timestamp("2020-01-09", tz="UTC"), "long_symbols": ("C", "D"), "short_symbols": (), "gross_R": 0.01, "n_long": 2, "n_short": 0},
            ]
        )
        zero = simulate_paper_portfolio(holdings, round_trip_bps=0.0, horizon_days=5, rebalance_every=1)
        taxed = simulate_paper_portfolio(holdings, round_trip_bps=20.0, horizon_days=5, rebalance_every=1)
        self.assertGreater(zero["total_return"], taxed["total_return"])
        self.assertTrue(taxed["mean_turnover"] is not None and taxed["mean_turnover"] > 0)


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
