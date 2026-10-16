"""Tests for ML feature matrix and XGBoost training path."""

from __future__ import annotations

import json
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
        import trendlabs.features as F

        df = _synth(n=250)
        X, y, meta = dataset_from_candles(df, horizon=1, band=0.002, min_history=60)
        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(meta))
        # Core features complete; context columns may be NaN (no SPY attached)
        core = [c for c in FEATURE_COLUMNS if c not in F._CONTEXT_COLUMNS]
        self.assertFalse(X[core].isna().any().any())
        self.assertTrue(set(y.unique()).issubset({"up", "down", "sideways"}))
        labels = make_labels(build_feature_frame(df), horizon=1, band=0.002)
        self.assertTrue(pd.isna(labels.iloc[-1]))


class TrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "ml.db")
        self.store.upsert_candles(_synth("QQQ", n=420, seed=3))
        self.store.upsert_candles(_synth("SPY", n=420, seed=3))  # fv2 context panel

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
        self.assertTrue(FEATURE_VERSION.startswith("fv"))  # exact version pinned in ModelQualityTests

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
        import trendlabs.features as F

        core = [c for c in FEATURE_COLUMNS if c not in F._CONTEXT_COLUMNS]
        feat = build_feature_frame(df).dropna(subset=core).iloc[60:]
        n = store.upsert_market_features(
            feat, symbol="AAPL", timeframe="1d",
            feature_version=FEATURE_VERSION, columns=FEATURE_COLUMNS,
        )
        self.assertEqual(n, len(feat))

        got = store.get_market_features("AAPL", "1d", feature_version=FEATURE_VERSION)
        self.assertEqual(len(got), len(feat))
        self.assertTrue((got["feature_version"] == FEATURE_VERSION).all())
        # spot-check a few values survived the JSON roundtrip
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


class PhaseDTests(unittest.TestCase):
    """Phase D: calibration, train-only thresholds, paper-trade lifecycle, registry."""

    def test_isotonic_calibration_reduces_brier(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        rng = np.random.default_rng(21)
        n = 6000
        p_true = rng.uniform(0.05, 0.95, size=n)
        y = (rng.uniform(size=n) < p_true).astype(float)
        raw = np.clip(p_true + 0.25, 0, 1)  # systematically overconfident raw probs
        tr, te = np.arange(3000), np.arange(3000, n)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(raw[tr], y[tr])
        cal = iso.predict(raw[te])
        brier_raw = float(np.mean((raw[te] - y[te]) ** 2))
        brier_cal = float(np.mean((cal - y[te]) ** 2))
        self.assertLess(brier_cal, brier_raw)
        # monotone non-decreasing calibration curve
        order = np.argsort(raw[te])
        diffs = np.diff(cal[order])
        self.assertTrue((diffs >= -1e-9).all())

    def test_policy_thresholds_train_only_and_tie_to_lowest(self) -> None:
        from trendlabs.ml_train import _choose_policy_thresholds

        n = 1000
        up_cal = np.where(np.arange(n) % 2 == 0, 0.70, 0.10)  # every even row bullish
        down_cal = np.full(n, 0.05)
        y_up = np.arange(n) % 2 == 0  # bullish calls always correct on train
        y_down = np.zeros(n, dtype=bool)
        pol = _choose_policy_thresholds(up_cal, down_cal, y_up, y_down, min_calls=30)
        # all thetas give precision 1.0 → tie broken toward LOWEST theta
        self.assertEqual(pol["theta_hi"], 0.40)
        self.assertEqual(pol["chosen_on"], "train")

        # no theta produces enough calls → default 0.50
        pol2 = _choose_policy_thresholds(np.zeros(n), np.zeros(n), y_up, y_down, min_calls=30)
        self.assertEqual(pol2["theta_hi"], 0.50)

    def test_paper_trade_lifecycle_and_pnl_math(self) -> None:
        from trendlabs.ml_paper import evaluate_paper_trades

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "paper.db")
        store.init_schema()

        as_of = "2026-01-02T04:00:00+00:00"  # signal bar is t0; entry is the NEXT bar
        tid = store.insert_paper_trade(
            {
                "prediction_id": 1,
                "symbol": "UNH",
                "timeframe": "1d",
                "horizon_bars": 1,
                "policy": "bullish",
                "as_of_ts": as_of,
                "model": "UNH_1d_h1",
                "feature_version": "fv1-test",
            }
        )
        self.assertGreater(tid, 0)
        self.assertEqual(len(store.get_open_paper_trades()), 1)

        # candles: t0 = as_of signal bar; t1 = entry bar (open 100 → close 110); t2 spare
        candles = pd.DataFrame(
            {
                "symbol": "UNH",
                "timeframe": "1d",
                "ts": pd.to_datetime(["2026-01-02T04:00:00Z", "2026-01-05T04:00:00Z", "2026-01-06T04:00:00Z"]),
                "open": [90.0, 100.0, 101.0],
                "high": [92.0, 111.0, 102.0],
                "low": [89.0, 99.0, 100.0],
                "close": [91.0, 110.0, 101.5],
                "volume": [1e6] * 3,
                "source": "test",
                "adjusted": 1,
            }
        )
        store.upsert_candles(candles)
        payload = evaluate_paper_trades(store)
        self.assertEqual(payload["closed_this_run"], 1)
        trades = store.get_paper_trades()
        closed = [t for t in trades if t["status"] == "closed"][0]
        self.assertAlmostEqual(closed["pnl_pct"], 10.0, places=6)  # sign*(110/100-1)*100
        self.assertAlmostEqual(closed["entry_open"], 100.0, places=6)
        self.assertEqual(closed["entry_ts"], closed["exit_ts"])  # h=1 → exit at entry-bar close
        self.assertEqual(payload["open_count"], 0)
        self.assertAlmostEqual(payload["summary"]["win_rate"], 1.0)

        # horizon incomplete (h=3 needs 2 more bars after entry; only 1 exists) → stays open
        tid2 = store.insert_paper_trade(
            {
                "prediction_id": 2,
                "symbol": "UNH",
                "timeframe": "1d",
                "horizon_bars": 3,
                "policy": "bearish",
                "as_of_ts": as_of,
                "model": "UNH_1d_h3",
                "feature_version": "fv1-test",
            }
        )
        payload2 = evaluate_paper_trades(store)
        open_rows = store.get_open_paper_trades()
        self.assertEqual(len(open_rows), 1)
        self.assertEqual(open_rows[0]["id"], tid2)
        self.assertEqual(payload2["closed_this_run"], 0)

    def test_approval_registry_roundtrip(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        reg = {"UNH_1d_h1": {"approved": True, "gate": "approved"}, "AAPL_1d_h1": {"approved": False, "gate": "watch"}}
        (d / "approved_models.json").write_text(json.dumps(reg))
        loaded = json.loads((d / "approved_models.json").read_text())
        approved = {k.split("_")[0] for k, v in loaded.items() if v.get("approved")}
        self.assertEqual(approved, {"UNH"})


class ModelQualityTests(unittest.TestCase):
    """fv2 candidates: market-context features + vol-scaled labels."""

    def _spy(self) -> pd.DataFrame:
        return _synth("SPY", n=400, seed=31)

    def test_attach_market_context_shapes_and_pit(self) -> None:
        from trendlabs.features import attach_market_context

        df = build_feature_frame(_synth("AAPL", n=400, seed=5))
        out = attach_market_context(df, spy_candles=self._spy())
        for col in ("SPY_RET1", "SPY_RET5", "SPY_RET20", "SPY_VOL20", "RS_SPY_5", "RS_SPY_20"):
            self.assertIn(col, out.columns)
            self.assertFalse(out[col].isna().all(), col)
        # context must not shift the panel
        self.assertEqual(len(out), len(df))
        # SPY panel itself: context = own features
        spy_df = build_feature_frame(self._spy())
        spy_out = attach_market_context(spy_df, spy_candles=None)
        self.assertFalse(spy_out["SPY_RET1"].isna().all())

    def test_vol_scaled_labels_balance(self) -> None:
        from trendlabs.features import make_labels, make_labels_vol_scaled

        df = build_feature_frame(_synth("MSFT", n=600, seed=13))
        fixed = make_labels(df, horizon=1, band=0.002).dropna()
        vol = make_labels_vol_scaled(df, horizon=1).dropna()

        def _majority_share(s: pd.Series) -> float:
            return float(s.value_counts(normalize=True).iloc[0])

        def _min_share(s: pd.Series) -> float:
            return float(s.value_counts(normalize=True).iloc[-1])

        # Motivation: with band = 0.5×trailing σ, all three classes stay populated
        # and the majority-class ceiling DROPS vs a fixed tiny band (where labels
        # ≈ raw sign, majority ≈ up share, 'sideways' nearly empty).
        self.assertLess(_majority_share(vol), _majority_share(fixed) - 0.02)
        self.assertGreater(_min_share(vol), _min_share(fixed))
        self.assertTrue(set(vol.unique()).issubset({"up", "down", "sideways"}))

    def test_fv2_dataset_with_context(self) -> None:
        from trendlabs.features import FEATURE_VERSION, dataset_from_candles

        self.assertTrue(FEATURE_VERSION.startswith("fv2-"))
        X, y, meta = dataset_from_candles(
            _synth("AAPL", n=420, seed=7),
            horizon=1,
            spy_candles=self._spy(),
            label_scheme="vol_scaled",
        )
        self.assertEqual(len(X), len(y))
        self.assertFalse(X[["ret_1", "rsi_norm", "close_location"]].isna().any().any())
        # context columns present and not all-NaN
        for col in ("SPY_RET1", "RS_SPY_5", "SPY_VOL20"):
            self.assertIn(col, X.columns)
            self.assertFalse(X[col].isna().all(), col)


    def test_forward_z_math_and_nan_edges(self) -> None:
        from trendlabs.features import make_forward_z

        df = build_feature_frame(_synth("AAPL", n=420, seed=17))
        z = make_forward_z(df, horizon=5)
        # warm-up: first 20 rows have no sigma → NaN
        self.assertTrue(z.iloc[:20].isna().all())
        # tail: last 5 rows lack forward bars → NaN
        self.assertTrue(z.iloc[-5:].isna().all())
        # manual check on one row: z = fwd/(σ·√h) with trailing-20 σ
        i = 100
        close = df["close"].astype(float).to_numpy()
        fwd = close[i + 5] / close[i] - 1.0
        window = close[i - 20 : i + 1]  # rolling(20) of pct_change covers returns i-19..i
        sigma = np.std(np.diff(window) / window[:-1], ddof=1)
        expected = fwd / (sigma * np.sqrt(5))
        self.assertAlmostEqual(float(z.iloc[i]), float(expected), places=10)

    def test_reg_dataset_alignment(self) -> None:
        import trendlabs.features as F
        from trendlabs.features import dataset_reg_from_panel

        # reg datasets require the full panel (breadth/dispersion) — build a mini one
        panel = pd.concat(
            [F.build_feature_frame(_synth(s, n=600, seed=i)) for i, s in enumerate(("AAPL", "MSFT", "XOM"))],
            ignore_index=True,
        )
        panel = F.attach_market_context(
            panel, spy_candles=_synth("SPY", n=600, seed=9)
        )
        panel = F.attach_panel_context(panel)
        X, z, y_cls, meta = dataset_reg_from_panel(panel, "AAPL", horizon=5)
        self.assertGreater(len(X), 300)
        self.assertEqual(len(X), len(z))
        self.assertEqual(len(y_cls), len(X))
        self.assertEqual(len(meta), len(X))
        self.assertFalse(X.isna().any().any())
        self.assertFalse(z.isna().any())
        # z and class labels are grossly consistent: strongly positive z ⇒ mostly up
        strongly_up = z > 0.5 / np.sqrt(5)
        if strongly_up.any():
            self.assertGreater(float((y_cls[strongly_up] == "up").mean()), 0.9)


    def test_wf_policy_structure_on_synthetic(self) -> None:
        from trendlabs.ml_wf import run_wf_policy

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "wf.db")
        store.init_schema()
        # 3 symbols × 900 business days — enough for small fold params
        for i, sym in enumerate(("AAA", "MSFT", "SPY")):
            store.upsert_candles(_synth(sym, n=900, seed=40 + i))
        payload = run_wf_policy(
            store,
            timeframe="1d",
            horizon=5,
            min_train_sessions=200,
            test_sessions=60,
            step_sessions=60,
            symbols=["AAA", "MSFT"],
        )
        self.assertEqual(payload["n_symbols"], 2)
        for r in payload["rows"]:
            self.assertNotIn("error", r)
            self.assertIn("gate", r)
            self.assertGreater(r.get("n_folds", 0), 0)
        self.assertIn("created_at", payload)


    def test_symmetrize_quota_logic(self) -> None:
        from trendlabs.ml_wf import _symmetrize

        called = pd.DataFrame(
            {
                "pred_z": [0.9, 0.7, 0.5, -0.8, -0.3],
                "z": [0.5, -0.2, 0.1, 0.4, -0.6],
                "sigma": [0.01] * 5,
            }
        )
        n, correct, pnl = _symmetrize(called, horizon=5)
        # quota = min(3 longs, 2 shorts) = 2 → book: longs [0.9, 0.7] + shorts [-0.8, -0.3]
        self.assertEqual(n, 4)
        # hits: 0.9/z0.5 ✓, 0.7/z−0.2 ✗, −0.8/z0.4 ✗, −0.3/z−0.6 ✓
        self.assertEqual(correct, 2)
        self.assertAlmostEqual(pnl, 0.318034, places=6)

        # one side empty → no sym book
        called2 = called.loc[called["pred_z"] > 0]
        n2, correct2, _ = _symmetrize(called2, horizon=5)
        self.assertEqual(n2, 0)
        self.assertIsNone(correct2)


if __name__ == "__main__":
    unittest.main()
