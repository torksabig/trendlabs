"""Unit tests for PE intraday V1 (targets, leakage, deterministic baselines)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trendlabs.prediction.intraday import (
    HORIZON_BARS,
    attach_intraday_targets_and_features,
    evaluate_horizon_predictions,
    forward_log_return,
    past_log_return,
    range_rv_feature,
    run_intraday_walk_forward,
)


def _synth_5m(
    symbols: list[str],
    n: int = 200,
    seed: int = 9,
    start: str = "2024-01-02 14:30:00+00:00",
) -> pd.DataFrame:
    """Tiny synthetic 5m OHLCV panel for unit tests (not used as claimed results)."""
    frames = []
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0, 0.0015 + 0.0003 * (i % 3), size=n)
        close = 100.0 * np.cumprod(1.0 + rets)
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.001, size=n))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.001, size=n))
        volume = rng.integers(10_000, 50_000, size=n).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "ts": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TargetAlignmentTests(unittest.TestCase):
    def test_forward_return_bar_counts(self) -> None:
        df = _synth_5m(["QQQ"], n=80, seed=1)
        g = df.sort_values("ts").reset_index(drop=True)
        t = 20
        for name, h in HORIZON_BARS.items():
            R = forward_log_return(g["close"], h)
            expected = np.log(g["close"].iloc[t + h] / g["close"].iloc[t])
            self.assertTrue(np.isclose(R.iloc[t], expected), msg=name)
            # Tail must be NaN
            self.assertTrue(np.isnan(R.iloc[-1]))

    def test_attach_targets_match_horizons(self) -> None:
        df = _synth_5m(["AAPL"], n=100, seed=2)
        out = attach_intraday_targets_and_features(df)
        t = 40
        for name, h in HORIZON_BARS.items():
            expected = np.log(df["close"].iloc[t + h] / df["close"].iloc[t])
            self.assertTrue(np.isclose(out[f"R_{name}"].iloc[t], expected))


class LeakageTests(unittest.TestCase):
    def test_features_ignore_future_prices(self) -> None:
        df = _synth_5m(["MSFT"], n=120, seed=3)
        feat = attach_intraday_targets_and_features(df)
        t = 60
        poisoned = df.copy()
        poisoned.loc[t + 1 :, "close"] = poisoned.loc[t + 1 :, "close"] * 2.0
        poisoned.loc[t + 1 :, "high"] = poisoned.loc[t + 1 :, "high"] * 2.0
        poisoned.loc[t + 1 :, "low"] = poisoned.loc[t + 1 :, "low"] * 0.5
        poisoned.loc[t + 1 :, "open"] = poisoned.loc[t + 1 :, "open"] * 2.0
        feat_p = attach_intraday_targets_and_features(poisoned)
        for col in (
            "persistence_15m",
            "persistence_1h",
            "persistence_4h",
            "short_mom",
            "range_rv",
            "range_rv_signed",
        ):
            self.assertTrue(
                np.isclose(feat[col].iloc[t], feat_p[col].iloc[t], equal_nan=True),
                msg=f"Future leak in {col}",
            )

    def test_past_log_return_uses_only_past(self) -> None:
        df = _synth_5m(["NVDA"], n=50, seed=4)
        g = df.sort_values("ts").reset_index(drop=True)
        t = 20
        h = 12
        got = past_log_return(g["close"], h).iloc[t]
        expected = np.log(g["close"].iloc[t] / g["close"].iloc[t - h])
        self.assertTrue(np.isclose(got, expected))

    def test_range_rv_window_alignment(self) -> None:
        df = _synth_5m(["SPY"], n=40, seed=5)
        g = df.sort_values("ts").reset_index(drop=True)
        window = 12
        rv = range_rv_feature(g["high"], g["low"], g["close"], window)
        t = 20
        rng = (g["high"] - g["low"]) / g["close"]
        expected = float(rng.iloc[t - window + 1 : t + 1].mean())
        self.assertTrue(np.isclose(rv.iloc[t], expected))


class BaselineDeterminismTests(unittest.TestCase):
    def test_persistence_equals_past_horizon_return(self) -> None:
        df = _synth_5m(["QQQ", "SPY"], n=100, seed=6)
        out = attach_intraday_targets_and_features(df)
        for name, h in HORIZON_BARS.items():
            for _, g in out.groupby("symbol"):
                g = g.reset_index(drop=True)
                t = 50
                expected = past_log_return(g["close"], h).iloc[t]
                self.assertTrue(np.isclose(g[f"persistence_{name}"].iloc[t], expected))

    def test_evaluate_perfect_persistence_on_iid_noise(self) -> None:
        """On a constructed series where target == lagged return, persistence IC is high."""
        n = 80
        ts = pd.date_range("2024-06-01", periods=n, freq="5min", tz="UTC")
        # Constant step so R_h(t) == persistence at t+h — use identity case:
        # pred = target exactly
        oos = pd.DataFrame(
            {
                "symbol": ["AAA"] * n + ["BBB"] * n,
                "ts": list(ts) + list(ts),
                "pred": np.concatenate([np.linspace(-0.01, 0.01, n), np.linspace(0.02, -0.02, n)]),
                "tgt": np.concatenate([np.linspace(-0.01, 0.01, n), np.linspace(0.02, -0.02, n)]),
            }
        )
        metrics = evaluate_horizon_predictions(oos, pred_col="pred", target_col="tgt")
        self.assertEqual(metrics["n"], 2 * n)
        self.assertTrue(np.isclose(metrics["SpearmanIC"], 1.0))
        self.assertTrue(np.isclose(metrics["hit_rate"], 1.0))
        self.assertTrue(np.isclose(metrics["MAE"], 0.0))

    def test_walk_forward_runs_on_long_synth(self) -> None:
        # Enough bars for min_train + purge(48) + test
        df = _synth_5m(["QQQ", "SPY", "AAPL"], n=2200, seed=7)
        result = run_intraday_walk_forward(
            df, min_train_bars=800, test_bars=200, step_bars=200
        )
        self.assertEqual(result["experiment"], "pe_intraday_v1")
        for h in HORIZON_BARS:
            block = result["horizons"][h]
            self.assertIn("models", block, msg=h)
            self.assertIn("persistence", block["models"])
            self.assertGreater(block["models"]["persistence"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
