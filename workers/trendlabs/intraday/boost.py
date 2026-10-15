"""Gradient-boosting stage for the intraday panels — exploratory until the gate clears.

Adoption rule (docs/INTRADAY_V1.md): XGBoost/boosting per horizon is only a *product
candidate* if a pre-declared score clears the gate on the deep panel. This module still
runs (keyless or deep) with the output labeled EXPLORATORY so results never masquerade
as gate decisions.

Backend: xgboost when importable (its dylib needs libomp on macOS), else sklearn's
HistGradientBoostingRegressor — same histogram-GB family, no extra dependency.
Both are fit per purged-WF fold on `z`, evaluated with the same cluster-robust stats
as the baselines (session IC CI95, Q5−Q1 bps vs round-trip cost).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.intraday import BARS_PER_DAY, DATASET_VERSION, DEFAULT_HORIZON_BARS, DISCLAIMER, INTRA_DIR
from trendlabs.intraday.baselines import load_panel, q5q1_r_bps, session_bootstrap_ci
from trendlabs.intraday.features import FEATURE_COLUMNS
from trendlabs.prediction.gate import GATE_MEAN_IC_MIN, GATE_MIN_FOLDS_FOR_DECISION, GATE_POSITIVE_FOLD_RATE_MIN
from trendlabs.prediction.metrics_ic import assign_quintiles, daily_spearman_ic, summarize_ic
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates

MIN_TRAIN_ROWS = 300


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, *, feature_cols: list[str]) -> pd.Series:
    X_train = train[feature_cols].astype(float)
    y_train = train["z"].astype(float)
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train, y_train = X_train.loc[mask], y_train.loc[mask]
    if len(X_train) < MIN_TRAIN_ROWS:
        return pd.Series(np.nan, index=test.index, dtype=float)

    model: Any
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        pred_values = model.predict(test[feature_cols].astype(float))
    except Exception:  # noqa: BLE001 — missing dylib/dep → fall back to sklearn HistGB
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42
        )
        model.fit(X_train, y_train)
        pred_values = model.predict(test[feature_cols].astype(float))

    pred = pd.Series(np.nan, index=test.index, dtype=float)
    ok = test[feature_cols].notna().all(axis=1)
    pred.loc[ok] = pred_values
    return pred


def run_boost_walk_forward(
    panel: pd.DataFrame,
    *,
    tf: str,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    min_train_sessions: int = 20,
    test_sessions: int = 5,
    step_sessions: int | None = None,
    rank_exclude_spy: bool = True,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    feature_cols = feature_cols or list(FEATURE_COLUMNS)
    bars_per_day = BARS_PER_DAY[tf]
    work = panel.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    rank_panel = work.loc[work["symbol"] != "SPY"].copy() if rank_exclude_spy else work
    dates = unique_sorted_dates(rank_panel)
    folds = generate_purged_folds(
        dates,
        min_train_days=min_train_sessions * bars_per_day,
        test_days=test_sessions * bars_per_day,
        step_days=None if step_sessions is None else step_sessions * bars_per_day,
        purge_gap=horizon_bars,
    )
    if not folds:
        return {"error": "no_folds", "n_bars": len(dates)}

    parts: list[pd.DataFrame] = []
    for fold in folds:
        train = mask_dates(rank_panel, fold.train_dates)
        test = mask_dates(rank_panel, fold.test_dates)
        if test.empty or train.empty:
            continue
        chunk = test[["symbol", "ts", "z", "R"]].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["score_boost"] = _fit_predict(train, test, feature_cols=feature_cols)
        parts.append(chunk)
    if not parts:
        return {"error": "empty_oos"}
    oos = pd.concat(parts, ignore_index=True)

    daily = daily_spearman_ic(oos, score_col="score_boost", min_names=2)
    stats = summarize_ic(daily)
    sess = oos.assign(session=oos["ts"].dt.tz_convert("America/New_York").dt.date)
    sess_ic = sess.groupby("session").apply(
        lambda g: g[["score_boost", "z"]].corr(method="spearman").iloc[0, 1], include_groups=False
    ).dropna()
    stats["session_ic"] = summarize_ic(sess_ic)
    stats["session_ic"]["ci95"] = session_bootstrap_ci(sess_ic)["ci95"]

    spreads: list[float] = []
    for _, g in oos.groupby("ts", sort=True):
        sub = g[["score_boost", "R"]].dropna()
        if len(sub) >= 10:
            q = assign_quintiles(sub["score_boost"])
            if q.notna().sum() >= 10 and q.nunique(dropna=True) >= 5:
                spreads.append(sub.loc[q == 5, "R"].mean() - sub.loc[q == 1, "R"].mean())
    stats["economic"] = {
        "q5q1_bps": float(np.mean(spreads) * 1e4) if spreads else None,
        "cost_round_trip_bps": 20.0,
    }
    return {
        "status": "EXPLORATORY — not a gate decision; boosting is adoption-locked until a pre-declared score clears the gate at depth",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tf": tf,
        "horizon_bars": horizon_bars,
        "n_folds": len(folds),
        "n_oos_rows": int(len(oos)),
        "model": stats,
        "gate_reference": {"mean_ic_min": GATE_MEAN_IC_MIN, "positive_fold_rate_min": GATE_POSITIVE_FOLD_RATE_MIN,
                           "min_folds_for_decision": GATE_MIN_FOLDS_FOR_DECISION},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exploratory gradient-boosting walk-forward")
    parser.add_argument("--tf", choices=list(BARS_PER_DAY), default="1h")
    args = parser.parse_args()
    panel = load_panel(args.tf)
    result = run_boost_walk_forward(panel, tf=args.tf)
    out = INTRA_DIR / f"boost_{args.tf}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    m = result.get("model", {})
    print(f"Wrote {out} ({result.get('n_folds')} folds)")
    print(f"boost: MeanIC={m.get('MeanIC')} session_ic={m.get('session_ic', {}).get('MeanIC')} "
          f"CI95={m.get('session_ic', {}).get('ci95')} Q5Q1={m.get('economic', {}).get('q5q1_bps')}bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
