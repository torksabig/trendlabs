"""Unit tests for PE volatility V1 (RV5 persistence vs HAR log-vol)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trendlabs.prediction.volatility import (
    build_vol_panel,
    fit_har_predict,
    future_rv,
    past_rv,
    persistence_predict,
    qlike,
    run_volatility_walk_forward,
)


def _synth_panel(symbols: list[str], n: int = 400, seed: int = 11) -> pd.DataFrame:
    frames = []
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0002, 0.012 + 0.004 * (i % 3), size=n)
        # Inject mild vol clustering so HAR has something to fit
        shock = rng.choice([0.0, 1.0], size=n, p=[0.92, 0.08])
        rets = rets * (1.0 + 2.5 * shock)
        close = 100 * np.cumprod(1 + rets)
        frames.append(pd.DataFrame({"symbol": sym, "ts": ts, "close": close}))
    return pd.concat(frames, ignore_index=True)


class FutureRVTests(unittest.TestCase):
    def test_future_rv5_matches_manual_sum(self) -> None:
        df = _synth_panel(["AAA"], n=80, seed=1)
        g = df.sort_values("ts").reset_index(drop=True)
        log_ret = np.log(g["close"] / g["close"].shift(1))
        frv = future_rv(log_ret, horizon=5)
        t = 30
        expected = np.sqrt(float(np.sum(log_ret.iloc[t + 1 : t + 6] ** 2)))
        self.assertTrue(np.isclose(frv.iloc[t], expected))

    def test_past_rv5_uses_only_past(self) -> None:
        df = _synth_panel(["AAA"], n=80, seed=2)
        g = df.sort_values("ts").reset_index(drop=True)
        log_ret = np.log(g["close"] / g["close"].shift(1))
        prv = past_rv(log_ret, window=5)
        t = 40
        expected = np.sqrt(float(np.sum(log_ret.iloc[t - 4 : t + 1] ** 2)))
        self.assertTrue(np.isclose(prv.iloc[t], expected))
        # Corruption of future returns must not change past RV at t
        poisoned = log_ret.copy()
        poisoned.iloc[t + 1 :] = poisoned.iloc[t + 1 :] * 10
        prv_p = past_rv(poisoned, window=5)
        self.assertTrue(np.isclose(prv.iloc[t], prv_p.iloc[t]))

    def test_har_lags_no_future_leak(self) -> None:
        panel = _synth_panel(["AAA"], n=120, seed=3)
        vol = build_vol_panel(panel)
        t = 60
        poisoned = panel.copy()
        poisoned.loc[t + 1 :, "close"] = poisoned.loc[t + 1 :, "close"] * 1.5
        vol_p = build_vol_panel(poisoned)
        for col in ("rv_d", "rv_w", "rv_m", "rv5_past", "log_rv_d", "log_rv_w", "log_rv_m"):
            self.assertTrue(
                np.isclose(vol[col].iloc[t], vol_p[col].iloc[t], equal_nan=True),
                msg=f"Future leak in {col}",
            )


class MetricTests(unittest.TestCase):
    def test_qlike_zero_when_perfect(self) -> None:
        y = np.array([0.01, 0.02, 0.015])
        self.assertTrue(np.isclose(qlike(y, y), 0.0, atol=1e-12))

    def test_qlike_positive_when_wrong(self) -> None:
        y = np.array([0.01, 0.02, 0.015])
        pred = y * 2.0
        self.assertGreater(qlike(y, pred), 0.0)


class ModelTests(unittest.TestCase):
    def test_persistence_equals_rv5_past(self) -> None:
        panel = _synth_panel(["AAA", "BBB"], n=100, seed=4)
        vol = build_vol_panel(panel).dropna(subset=["rv5_past", "future_rv_5"])
        pred = persistence_predict(vol)
        self.assertTrue(np.allclose(pred.to_numpy(), vol["rv5_past"].to_numpy()))

    def test_har_fit_train_only_changes_with_train(self) -> None:
        panel = _synth_panel(["AAA", "BBB", "CCC"], n=300, seed=5)
        vol = build_vol_panel(panel).dropna(
            subset=["future_rv_5", "log_future_rv_5", "log_rv_d", "log_rv_w", "log_rv_m"]
        )
        dates = sorted(vol["ts"].drop_duplicates().tolist())
        split = int(len(dates) * 0.7)
        train = vol.loc[vol["ts"].isin(dates[:split])]
        test = vol.loc[vol["ts"].isin(dates[split:])]
        pred_a = fit_har_predict(train, test)
        # Fit on a shorter train → different coefficients / predictions
        train_short = vol.loc[vol["ts"].isin(dates[: max(40, split // 2)])]
        pred_b = fit_har_predict(train_short, test)
        self.assertTrue(pred_a.notna().any())
        self.assertTrue(pred_b.notna().any())
        self.assertFalse(np.allclose(pred_a.dropna(), pred_b.reindex(pred_a.dropna().index).dropna()))


class WalkForwardTests(unittest.TestCase):
    def test_run_produces_both_models(self) -> None:
        panel = _synth_panel(["AAA", "BBB", "CCC", "SPY"], n=500, seed=6)
        result = run_volatility_walk_forward(
            panel, min_train_days=120, test_days=40, step_days=40, purge_gap=5
        )
        self.assertIn("models", result)
        self.assertIn("rv5_persistence", result["models"])
        self.assertIn("har_log_vol", result["models"])
        for name, m in result["models"].items():
            self.assertIsNotNone(m.get("MAE"), msg=name)
            self.assertIsNotNone(m.get("RMSE"), msg=name)
            self.assertIsNotNone(m.get("QLIKE"), msg=name)
            self.assertIsNotNone(m.get("RankIC"), msg=name)
        self.assertIn("comparison", result)
        self.assertIn("recommendation", result["comparison"])
        self.assertIn("EXPLORATORY", result["disclaimer"])


if __name__ == "__main__":
    unittest.main()
