"""Evaluate the pre-declared hypotheses (H1/H2/H3) on a panel — auto verdicts.

Hypotheses and accept criteria are registered in docs/INTRADAY_V1.md and must be
declared before the panel they are evaluated on. This module only measures; it never
invents new candidates.

  H1 reversal:        score = -score_short_mom (2h wall-clock momentum, sign-flipped)
  H2 open-slot ridge: ranking skill concentrated at session-open bars
  H3 momentum-as-edge (positive short momentum) — expected reject
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.intraday import BARS_PER_DAY, DEFAULT_HORIZON_BARS, INTRA_DIR
from trendlabs.intraday.baselines import (
    fit_ridge_predict,
    load_panel,
    q5q1_r_bps,
    score_short_mom,
    session_bootstrap_ci,
)
from trendlabs.intraday.features import FEATURE_COLUMNS
from trendlabs.prediction.gate import GATE_MEAN_IC_MIN
from trendlabs.prediction.metrics_ic import summarize_ic
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates

SESSION_IC_MIN = GATE_MEAN_IC_MIN
COST_ROUND_TRIP_BPS = 20.0  # 2 × (FEE_BPS + SLIPPAGE_BPS)


def _sess_ic(oos: pd.DataFrame, score_col: str, min_names: int = 2) -> pd.Series:
    def _ric(g: pd.DataFrame) -> float:
        sub = g[[score_col, "z"]].dropna()
        if len(sub) < min_names or sub[score_col].nunique() < 2:
            return np.nan
        return float(sub[score_col].rank().corr(sub["z"].rank()))

    ics = oos.groupby("ts", sort=True).apply(_ric, include_groups=False).dropna()
    if ics.empty:
        return ics
    sess = ics.index.tz_convert("America/New_York").date
    return ics.groupby(pd.Series(sess, index=ics.index)).mean()


def _ic_stats(oos: pd.DataFrame, score_col: str) -> dict[str, Any]:
    sess = _sess_ic(oos, score_col)
    stats = summarize_ic(sess)
    stats["ci95"] = session_bootstrap_ci(sess)["ci95"]
    stats["q5q1_bps"] = q5q1_r_bps(oos, score_col=score_col)
    return stats


def _passes_h1(stats: dict[str, Any], *, n_folds: int, min_folds: int) -> tuple[bool, list[str]]:
    checks: list[str] = []
    ok = True
    mean_ic = stats.get("MeanIC")
    ci = stats.get("ci95")
    bps = stats.get("q5q1_bps")
    if mean_ic is None or mean_ic < SESSION_IC_MIN:
        ok = False
        checks.append(f"session MeanIC {mean_ic} < {SESSION_IC_MIN}")
    if not ci or ci[0] <= 0.0:
        ok = False
        checks.append(f"CI95 {ci} does not exclude 0")
    if bps is None or bps < COST_ROUND_TRIP_BPS:
        ok = False
        checks.append(f"Q5-Q1 {bps}bps < {COST_ROUND_TRIP_BPS}bps round trip")
    if n_folds < min_folds:
        ok = False
        checks.append(f"insufficient depth: {n_folds} folds < {min_folds} required")
    return ok, checks


def evaluate_hypotheses(
    panel: pd.DataFrame,
    *,
    tf: str,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    min_train_sessions: int = 20,
    test_sessions: int = 5,
    step_sessions: int | None = None,
    h1_min_folds: int = 100,
) -> dict[str, Any]:
    from trendlabs.intraday.features import bars_for_hours

    bars_per_day = BARS_PER_DAY[tf]
    work = panel.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    rank_panel = work.loc[work["symbol"] != "SPY"].copy()
    dates = unique_sorted_dates(rank_panel)
    folds = generate_purged_folds(
        dates,
        min_train_days=min_train_sessions * bars_per_day,
        test_days=test_sessions * bars_per_day,
        step_days=None if step_sessions is None else step_sessions * bars_per_day,
        purge_gap=horizon_bars,
    )
    if not folds:
        return {"error": "no_folds"}

    parts: list[pd.DataFrame] = []
    for fold in folds:
        train, test = mask_dates(rank_panel, fold.train_dates), mask_dates(rank_panel, fold.test_dates)
        if test.empty or train.empty:
            continue
        chunk = test[["symbol", "ts", "z", "R"]].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["score_neg_short_mom"] = -score_short_mom(test, tf=tf)
        chunk["score_short_mom"] = score_short_mom(test, tf=tf)
        chunk["score_ridge_open"] = fit_ridge_predict(train, test, feature_cols=list(FEATURE_COLUMNS), alpha=1.0)
        parts.append(chunk)
    if not parts:
        return {"error": "empty_oos"}
    oos = pd.concat(parts, ignore_index=True)

    frac = oos["ts"].dt.tz_convert("America/New_York").dt.hour * 60 + oos["ts"].dt.tz_convert("America/New_York").dt.minute
    open_mask = (frac - 9 * 60 - 30) < bars_for_hours(tf, 1.0)  # first wall-clock hour
    h1_stats = _ic_stats(oos, "score_neg_short_mom")
    h1_ok, h1_why = _passes_h1(h1_stats, n_folds=len(folds), min_folds=h1_min_folds)

    h2_open = _ic_stats(oos.loc[open_mask], "score_ridge_open")
    h2_rest = _ic_stats(oos.loc[~open_mask], "score_ridge_open")
    h2_ok = False
    h2_why: list[str] = []
    if h2_open.get("MeanIC") is not None and h2_open.get("ci95"):
        if h2_open["MeanIC"] < SESSION_IC_MIN:
            h2_why.append(f"open-slot session MeanIC {h2_open['MeanIC']} < {SESSION_IC_MIN}")
        elif h2_open["ci95"][0] <= 0.0:
            h2_why.append(f"open-slot CI95 {h2_open['ci95']} does not exclude 0")
        elif h2_rest.get("MeanIC") is None or h2_open["MeanIC"] < 2 * h2_rest["MeanIC"]:
            h2_why.append(
                f"open-slot IC {h2_open['MeanIC']} not ≥ 2× rest-of-session {h2_rest.get('MeanIC')}"
            )
        else:
            h2_ok = True
    else:
        h2_why.append("open-slot IC not measurable")

    h3_stats = _ic_stats(oos, "score_short_mom")
    ci = h3_stats.get("ci95")
    h3_rejected = bool(ci and ci[1] < SESSION_IC_MIN)
    h3_why = [] if h3_rejected else [f"CI95 {ci} does not rule out {SESSION_IC_MIN}"]

    return {
        "panel_version": str(panel["dataset_version"].iloc[0]) if "dataset_version" in panel.columns else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tf": tf,
        "horizon_bars": horizon_bars,
        "n_folds": len(folds),
        "n_oos_rows": int(len(oos)),
        "cost_round_trip_bps": COST_ROUND_TRIP_BPS,
        "H1_reversal": {"verdict": "ACCEPTED" if h1_ok else "REJECTED", "stats": h1_stats, "reasons": h1_why},
        "H2_open_slot": {
            "verdict": "ACCEPTED" if h2_ok else "REJECTED",
            "open_slot": h2_open,
            "rest_of_session": h2_rest,
            "reasons": h2_why,
        },
        "H3_momentum_edge": {"verdict": "REJECTED" if h3_rejected else "NOT REJECTED", "stats": h3_stats, "reasons": h3_why},
        "xgboost_unlocked": bool(h1_ok or h2_ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pre-declared hypotheses H1-H3")
    parser.add_argument("--tf", choices=list(BARS_PER_DAY), default="15m")
    args = parser.parse_args()
    panel = load_panel(args.tf)
    result = evaluate_hypotheses(panel, tf=args.tf)
    out = INTRA_DIR / f"hypothesis_eval_{args.tf}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for h in ("H1_reversal", "H2_open_slot", "H3_momentum_edge"):
        block = result.get(h, {})
        print(f"{h}: {block.get('verdict')}" + (f" ({'; '.join(block.get('reasons', []))})" if block.get("reasons") else ""))
    print(f"xgboost_unlocked: {result.get('xgboost_unlocked')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
