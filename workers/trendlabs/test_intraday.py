"""Focused tests for the intraday track: resampling, targets, purge, PIT RVOL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _session_bars(date: str, n: int, tf_min: int = 5, base_price: float = 100.0) -> pd.DataFrame:
    """RTH 5m bars for one session starting 9:30 ET."""
    start = pd.Timestamp(f"{date} 09:30", tz="America/New_York")
    ts = start + pd.to_timedelta(np.arange(n) * tf_min, unit="m")
    price = base_price + np.arange(n, dtype=float) * 0.01
    return pd.DataFrame(
        {
            "ts": ts.tz_convert("UTC"),
            "open": price,
            "high": price + 0.05,
            "low": price - 0.05,
            "close": price + 0.02,
            "volume": 1000.0,
        }
    )


def test_resample_session_anchored_15m_and_4h() -> None:
    from trendlabs.intraday.dataset import resample_bars

    bars = pd.concat([_session_bars("2025-01-02", 78), _session_bars("2025-01-03", 78)], ignore_index=True)

    p15 = resample_bars(bars, "15m")
    assert len(p15) == 2 * 26
    first = p15.iloc[0]
    assert first["slot"] == 0
    assert first["ts"] == pd.Timestamp("2025-01-02 09:30", tz="America/New_York").tz_convert("UTC")
    # 15m aggregation over three 5m bars
    assert first["open"] == pytest.approx(bars.iloc[0]["open"])
    assert first["close"] == pytest.approx(bars.iloc[2]["close"])
    assert first["high"] == pytest.approx(bars.iloc[0:3]["high"].max())
    assert first["volume"] == pytest.approx(3000.0)

    p4h = resample_bars(bars, "4h")
    assert len(p4h) == 2 * 2
    et = p4h["ts"].dt.tz_convert("America/New_York")
    clock = et.dt.strftime("%H:%M").tolist()
    assert clock == ["09:30", "13:30", "09:30", "13:30"]
    # last bar of session aggregates only remaining RTH bars (13:30-16:00)
    last = p4h.iloc[1]
    assert last["volume"] == pytest.approx(1000.0 * 30)  # 13:30..15:55 = 30 bars


def test_targets_execution_rule_and_nan_tail() -> None:
    from trendlabs.intraday.dataset import attach_targets

    df = _session_bars("2025-01-02", 30)
    df["symbol"] = "AAPL"
    panel = attach_targets(df, horizon_bars=2, vol_window=5)

    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    # R_t = ln(close_{t+2} / open_{t+1})
    expect_r = np.log(close[2:] / open_[1:-1])
    got = panel["R"].to_numpy()[: len(expect_r)]
    assert np.allclose(got, expect_r, equal_nan=True)
    assert np.isnan(panel["R"].to_numpy()[-2:]).all()  # tail lacks future bars
    assert panel["sigma"].notna().sum() == 25  # vol_window=5 → first 4 NaN


def test_purged_folds_no_label_overlap_bars() -> None:
    from trendlabs.prediction.walk_forward import generate_purged_folds

    n = 200
    ts = pd.date_range("2025-01-02 09:30", periods=n, freq="15min", tz="UTC")
    folds = generate_purged_folds(list(ts), min_train_days=50, test_days=20, purge_gap=1)
    assert folds
    for fold in folds:
        train_idx = [ts.get_loc(pd.Timestamp(d)) for d in fold.train_dates]
        test_idx = [ts.get_loc(pd.Timestamp(d)) for d in fold.test_dates]
        # train's h=1 label needs bar t+1; that bar must precede the first test bar
        assert max(train_idx) + 2 <= min(test_idx)
        assert min(test_idx) - max(train_idx) - 1 >= 1


def test_rvol_slot_uses_prior_sessions_only() -> None:
    from trendlabs.intraday.features import relative_volume_slot

    d1 = _session_bars("2025-01-02", 26, tf_min=15)
    d2 = _session_bars("2025-01-03", 26, tf_min=15)
    df = pd.concat([d1, d2], ignore_index=True)
    df["symbol"] = "AAPL"
    df["slot"] = np.tile(np.arange(26), 2)
    df["volume"] = 100.0

    base = relative_volume_slot(df)
    assert np.allclose(base.dropna(), 1.0, equal_nan=False)  # constant volume → ratio 1 after day 1
    assert base.iloc[:26].isna().all()  # first session: no prior sessions

    # Perturb today's later slot; today's earlier slots must not change
    df.loc[df.index[-1], "volume"] = 100000.0
    perturbed = relative_volume_slot(df)
    np.testing.assert_array_equal(base.iloc[26:30].to_numpy(), perturbed.iloc[26:30].to_numpy())


def _score_panel() -> pd.DataFrame:
    n = 30
    ts = pd.date_range("2025-01-02 09:30", periods=n, freq="15min", tz="UTC")
    close = np.linspace(100.0, 103.0, n)
    return pd.DataFrame(
        {
            "ts": ts,
            "symbol": "AAPL",
            "open": close + 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0,
            "VOL_1d": 0.2,
        }
    )


def test_plan_baseline_scores_values_and_pit() -> None:
    from trendlabs.intraday.baselines import score_persistence, score_range_rv_signed, score_short_mom

    df = _score_panel()
    n = len(df)
    close = df["close"].to_numpy()

    pers = score_persistence(df)
    assert np.isnan(pers.iloc[0])
    assert np.allclose(pers.iloc[1:], np.diff(np.log(close)))

    short = score_short_mom(df, tf="15m")  # 2h → 8 bars
    assert short.iloc[:8].isna().all()
    assert np.allclose(short.iloc[8:], np.log(close[8:] / close[:-8]))

    rr = score_range_rv_signed(df, tf="15m")  # N = 26 trailing bars → valid from row 25
    assert rr.iloc[:25].isna().all()
    lo = close[:26].min() - 0.2
    hi = close[:26].max() + 0.2
    expected = (close[25] - lo) / (hi - lo) - 0.5
    assert rr.iloc[25] == pytest.approx(expected * 0.2)

    # PIT: perturbing the LAST bar must not change any earlier score
    df2 = df.copy()
    df2.loc[df2.index[-1], "close"] *= 3.0
    for fn in (score_persistence, lambda d: score_short_mom(d, tf="15m"), lambda d: score_range_rv_signed(d, tf="15m")):
        base, pert = fn(df), fn(df2)
        np.testing.assert_allclose(base.iloc[:-1].to_numpy(), pert.iloc[:-1].to_numpy(), equal_nan=True)


def test_runner_emits_plan_baselines_and_is_deterministic() -> None:
    from trendlabs.intraday.baselines import run_baselines_walk_forward
    from trendlabs.intraday.features import FEATURE_COLUMNS

    rng = np.random.default_rng(7)
    n_ts = 130  # 5 sessions × 26 bars on the 15m panel
    ts = pd.date_range("2025-01-02 09:30", periods=n_ts, freq="15min", tz="UTC")
    frames = []
    for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        close = 100.0 + i + np.cumsum(rng.normal(0, 0.1, n_ts))
        noise_h = np.abs(rng.normal(0, 0.05, n_ts)) + 0.05
        df = pd.DataFrame(
            {
                "ts": ts,
                "symbol": sym,
                "open": close + rng.normal(0, 0.02, n_ts),
                "high": close + noise_h,
                "low": close - noise_h,
                "close": close,
                "volume": 1000.0 + i,
                "R": rng.normal(0, 0.01, n_ts),
                "z": rng.normal(0, 1.0, n_ts),
            }
        )
        for col in FEATURE_COLUMNS:
            df[col] = rng.normal(size=n_ts)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)

    kwargs = dict(min_train_sessions=2, test_sessions=1, step_sessions=1)
    r1 = run_baselines_walk_forward(panel, tf="15m", **kwargs)
    r2 = run_baselines_walk_forward(panel, tf="15m", **kwargs)

    expected = {
        "score_random",
        "score_hist_mean",
        "score_mom",
        "score_ridge",
        "score_persistence",
        "score_short_mom",
        "score_range_rv_signed",
    }
    assert expected <= set(r1.get("baselines", {})), r1.get("error")
    for col in expected:
        stats = r1["baselines"][col]
        assert stats["hit_rate"] is not None
        assert stats["mae_z"] is not None
        assert "MeanIC" in stats["rank_ic"]

    r1.pop("generated_at"), r2.pop("generated_at")
    assert r1 == r2  # same seed + deterministic scores → identical payload
