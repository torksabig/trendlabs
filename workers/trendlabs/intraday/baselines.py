"""Phase B intraday baselines — purged walk-forward IC on the 15m/1h/4h panels.

Mirrors the daily track's Phase B: deterministic baselines evaluated out-of-sample with
expanding train / fixed test windows in BARS and purge gap >= horizon_bars.
Gate bars are imported from the daily track (trendlabs.prediction.gate) so decisions
use the same thresholds — the daily PE FAIL and HAR-vol stay method benchmarks only.

Plan-named intraday baselines (Causal+QAE slice, Phase 1):
  - score_persistence     — last panel-bar log return
  - score_short_mom       — momentum over the last 2 wall-clock hours (floor 1 bar)
  - score_range_rv_signed — close location in trailing 1-session range, RV-weighted
All three are PIT panel scores (bars <= t) and need no train fit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.intraday import BARS_PER_DAY, DATASET_VERSION, DEFAULT_HORIZON_BARS, DISCLAIMER, INTRA_DIR
from trendlabs.prediction.gate import (
    GATE_MEAN_IC_MIN,
    GATE_MIN_FOLDS_FOR_DECISION,
    GATE_MIN_QUINTILE_DAYS,
    GATE_POSITIVE_FOLD_RATE_MIN,
    GATE_Q5_Q1_SHARPE_10BPS_MIN,
)
from trendlabs.prediction.metrics_ic import daily_spearman_ic, evaluate_scores, fold_mean_ic_stats, summarize_ic
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates

RANK_EXCLUDE_SPY = True
MIN_SCORE_ROWS_RIDGE = 50


def score_random(panel: pd.DataFrame, *, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    ordered = panel.sort_values(["ts", "symbol"]).index
    scores = pd.Series(np.nan, index=panel.index, dtype=float)
    scores.loc[ordered] = rng.normal(size=len(ordered))
    return scores


def score_historical_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    by_sym = train.groupby("symbol")["z"].mean()
    global_mu = float(train["z"].dropna().mean()) if train["z"].notna().any() else 0.0
    return test["symbol"].map(by_sym).fillna(global_mu).astype(float)


def score_momentum(panel: pd.DataFrame, *, col: str = "MOM_1h") -> pd.Series:
    return panel[col].astype(float)


def score_persistence(panel: pd.DataFrame) -> pd.Series:
    """Last panel-bar log return per symbol — PIT (uses close_t vs close_{t-1})."""
    close = panel["close"].astype(float)
    prev = close.groupby(panel["symbol"], sort=False).shift(1)
    return np.log(close / prev.replace(0.0, np.nan))


def score_short_mom(panel: pd.DataFrame, *, tf: str, hours: float = 2.0) -> pd.Series:
    """Momentum over the last `hours` wall-clock hours (floor 1 bar), PIT.

    On the 4h panel the window floors to one bar, so it coincides with
    score_persistence there.
    """
    from trendlabs.intraday.features import bars_for_hours

    w = bars_for_hours(tf, hours)
    close = panel["close"].astype(float)
    base = close.groupby(panel["symbol"], sort=False).shift(w)
    return np.log(close / base.replace(0.0, np.nan))


def score_range_rv_signed(panel: pd.DataFrame, *, tf: str) -> pd.Series:
    """Signed close-location in the trailing 1-session range, RV-weighted, PIT.

    CLV_t = (close_t - min(low, N)) / (max(high, N) - min(low, N)) - 0.5, with
    N = BARS_PER_DAY[tf] trailing bars including t; score = CLV * VOL_1d so that
    volatile names weigh more in the cross-sectional ordering.
    """
    n = BARS_PER_DAY[tf]
    g = panel.groupby("symbol", sort=False)
    hi = g["high"].transform(lambda s: s.rolling(n, min_periods=n).max())
    lo = g["low"].transform(lambda s: s.rolling(n, min_periods=n).min())
    rng = (hi - lo).replace(0.0, np.nan)
    clv = (panel["close"].astype(float) - lo) / rng - 0.5
    vol = panel["VOL_1d"].astype(float) if "VOL_1d" in panel.columns else pd.Series(np.nan, index=panel.index)
    return clv * vol


def rank_ic_series(panel: pd.DataFrame, *, score_col: str, min_names: int = 2) -> pd.Series:
    """Per-bar Pearson correlation of cross-sectional ranks (score vs z)."""

    def _ric(g: pd.DataFrame) -> float:
        sub = g[[score_col, "z"]].dropna()
        if len(sub) < min_names or sub[score_col].nunique() < 2:
            return np.nan
        return float(sub[score_col].rank().corr(sub["z"].rank()))

    return panel.groupby("ts", sort=True).apply(_ric, include_groups=False)


def hit_rate(panel: pd.DataFrame, *, score_col: str) -> float | None:
    """P(sign(score) == sign(z)) across OOS rows."""
    sub = panel[[score_col, "z"]].dropna()
    if sub.empty:
        return None
    return float((np.sign(sub[score_col]) == np.sign(sub["z"])).mean())


def mae_standardized(panel: pd.DataFrame, *, score_col: str) -> float | None:
    """MAE between cross-sectionally standardized score and target z."""
    g = panel.groupby("ts", sort=False)[score_col]
    sd = g.transform(lambda s: s.std(ddof=1))
    sz = (panel[score_col] - g.transform("mean")) / sd.replace(0.0, np.nan)
    diff = (sz - panel["z"]).abs().dropna()
    return float(diff.mean()) if len(diff) else None


def fit_ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str],
    alpha: float = 1.0,
) -> pd.Series:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X_train = train[feature_cols].astype(float)
    y_train = train["z"].astype(float)
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train, y_train = X_train.loc[mask], y_train.loc[mask]
    if len(X_train) < MIN_SCORE_ROWS_RIDGE:
        return pd.Series(np.nan, index=test.index, dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y_train)

    X_test = test[feature_cols].astype(float)
    pred = pd.Series(np.nan, index=test.index, dtype=float)
    ok = X_test.notna().all(axis=1)
    if ok.any():
        pred.loc[ok] = model.predict(scaler.transform(X_test.loc[ok]))
    return pred


def session_level_ic(panel: pd.DataFrame, *, score_col: str, min_names: int = 2) -> pd.Series:
    """Mean of per-bar cross-sectional ICs within each ET session."""
    ics = daily_spearman_ic(panel, score_col=score_col, min_names=min_names)
    if ics.empty:
        return ics
    session = ics.index.tz_convert("America/New_York").date
    return ics.groupby(pd.Series(session, index=ics.index)).mean()


def run_baselines_walk_forward(
    panel: pd.DataFrame,
    *,
    tf: str,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    min_train_sessions: int = 20,
    test_sessions: int = 5,
    step_sessions: int | None = None,
    random_seed: int = 42,
    ridge_alpha: float = 1.0,
    rank_exclude_spy: bool = RANK_EXCLUDE_SPY,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    from trendlabs.intraday.features import FEATURE_COLUMNS

    feature_cols = feature_cols or list(FEATURE_COLUMNS)
    bars_per_day = BARS_PER_DAY[tf]
    work = panel.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    # Plan baselines are pure panel scores: precompute once on the full panel so
    # fold test slices keep cross-boundary history (no recompute inside test).
    work["score_persistence"] = score_persistence(work)
    work["score_short_mom"] = score_short_mom(work, tf=tf)
    work["score_range_rv_signed"] = score_range_rv_signed(work, tf=tf)
    if rank_exclude_spy:
        rank_panel = work.loc[work["symbol"] != "SPY"].copy()
    else:
        rank_panel = work

    dates = unique_sorted_dates(rank_panel)
    folds = generate_purged_folds(
        dates,
        min_train_days=min_train_sessions * bars_per_day,
        test_days=test_sessions * bars_per_day,
        step_days=None if step_sessions is None else step_sessions * bars_per_day,
        purge_gap=horizon_bars,
    )
    if not folds:
        return {"disclaimer": DISCLAIMER, "error": "no_folds", "n_bars": len(dates), "baselines": {}}

    parts: list[pd.DataFrame] = []
    for fold in folds:
        train = mask_dates(rank_panel, fold.train_dates)
        test = mask_dates(rank_panel, fold.test_dates)
        if test.empty or train.empty:
            continue
        chunk = test[["symbol", "ts", "z", "R", "MOM_1h"]].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["score_random"] = score_random(test, seed=random_seed + fold.fold_id)
        chunk["score_hist_mean"] = score_historical_mean(train, test)
        chunk["score_mom"] = score_momentum(test)
        chunk["score_ridge"] = fit_ridge_predict(train, test, feature_cols=feature_cols, alpha=ridge_alpha)
        chunk["score_persistence"] = test["score_persistence"]
        chunk["score_short_mom"] = test["score_short_mom"]
        chunk["score_range_rv_signed"] = test["score_range_rv_signed"]
        parts.append(chunk)

    if not parts:
        return {"disclaimer": DISCLAIMER, "error": "empty_oos", "baselines": {}}

    oos = pd.concat(parts, ignore_index=True)
    baselines: dict[str, Any] = {}
    for col in (
        "score_random",
        "score_hist_mean",
        "score_mom",
        "score_ridge",
        "score_persistence",
        "score_short_mom",
        "score_range_rv_signed",
    ):
        stats = evaluate_scores(oos, score_col=col, min_names_ic=2, min_names_quintile=10)
        folds_stats = fold_mean_ic_stats(oos, score_col=col)
        sess = session_level_ic(oos, score_col=col)
        stats["session_ic"] = summarize_ic(sess)
        stats["folds"] = folds_stats
        stats["rank_ic"] = summarize_ic(rank_ic_series(oos, score_col=col))
        stats["hit_rate"] = hit_rate(oos, score_col=col)
        stats["mae_z"] = mae_standardized(oos, score_col=col)
        baselines[col] = stats

    best_name = None
    for name, v in baselines.items():
        if v.get("MeanIC") is not None and (best_name is None or v["MeanIC"] > baselines[best_name]["MeanIC"]):
            best_name = name
    gate = {
        "mean_ic_min": GATE_MEAN_IC_MIN,
        "positive_fold_rate_min": GATE_POSITIVE_FOLD_RATE_MIN,
        "q5_q1_sharpe_10bps_min": GATE_Q5_Q1_SHARPE_10BPS_MIN,
        "min_folds_for_decision": GATE_MIN_FOLDS_FOR_DECISION,
        "min_quintile_days": GATE_MIN_QUINTILE_DAYS,
    }
    return {
        "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tf": tf,
        "horizon_bars": horizon_bars,
        "n_folds": len(folds),
        "n_oos_bars": int(len(oos)),
        "gate_bars": gate,
        "best_baseline": best_name,
        "baselines": baselines,
        "note": "Gate decision reserved for XGBoost stage; these are baseline ICs only.",
    }


def load_panel(tf: str) -> pd.DataFrame:
    path = INTRA_DIR / f"{DATASET_VERSION}_{tf}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No intraday panel at {path} — run dataset build first")
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run intraday baselines (purged walk-forward)")
    parser.add_argument("--tf", choices=list(BARS_PER_DAY), default="1h")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    panel = load_panel(args.tf)
    result = run_baselines_walk_forward(panel, tf=args.tf)
    out_path = args.out or (INTRA_DIR / f"baselines_{args.tf}.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for name, stats in result.get("baselines", {}).items():
        print(
            f"{name}: MeanIC={stats.get('MeanIC')} ICIR={stats.get('ICIR')} "
            f"folds+={stats['folds'].get('positive_fold_rate')}"
        )
    return 0 if result.get("baselines") else 1


if __name__ == "__main__":
    raise SystemExit(main())
