"""Phase B — deterministic baselines on the PE research dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from trendlabs.config import ROOT
from trendlabs.prediction import DATASET_VERSION, DISCLAIMER, PURGE_GAP_DAYS
from trendlabs.prediction.dataset import PE_DIR, load_exported_dataset
from trendlabs.prediction.features import FEATURE_COLUMNS
from trendlabs.prediction.metrics_ic import evaluate_scores
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates


def score_random(panel: pd.DataFrame, *, seed: int = 42) -> pd.Series:
    """Deterministic random ranking scores."""
    rng = np.random.default_rng(seed)
    # Stable per (symbol, ts) via hash of seed + keys would also work; use positional
    # with a fixed shuffle of a seeded array aligned to a stable sort.
    ordered = panel.sort_values(["ts", "symbol"]).index
    scores = pd.Series(np.nan, index=panel.index, dtype=float)
    scores.loc[ordered] = rng.normal(size=len(ordered))
    return scores


def score_historical_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Per-symbol train-set mean of z (cross-sectional ranks); global mean fallback."""
    by_sym = train.groupby("symbol")["z"].mean()
    global_mu = float(train["z"].dropna().mean()) if train["z"].notna().any() else 0.0
    return test["symbol"].map(by_sym).fillna(global_mu).astype(float)


def score_mom12_1(panel: pd.DataFrame) -> pd.Series:
    return panel["MOM12_1"].astype(float)


def fit_ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLUMNS,
    alpha: float = 1.0,
) -> pd.Series:
    """Ridge on standardized features; scaler + model fit on train only."""
    X_train = train[feature_cols].astype(float)
    y_train = train["z"].astype(float)
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask]
    if len(X_train) < 30:
        return pd.Series(np.nan, index=test.index, dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(Xs, y_train)

    X_test = test[feature_cols].astype(float)
    # Leave NaN rows as NaN predictions
    pred = pd.Series(np.nan, index=test.index, dtype=float)
    ok = X_test.notna().all(axis=1)
    if ok.any():
        pred.loc[ok] = model.predict(scaler.transform(X_test.loc[ok]))
    return pred


def run_baselines_walk_forward(
    panel: pd.DataFrame,
    *,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int | None = None,
    purge_gap: int = PURGE_GAP_DAYS,
    random_seed: int = 42,
    ridge_alpha: float = 1.0,
    rank_exclude_spy: bool = True,
) -> dict[str, Any]:
    """Purged WF evaluation for all Phase B baselines."""
    work = panel.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    if rank_exclude_spy:
        # Keep SPY out of ranking IC (still used in features via broadcast columns)
        rank_panel = work.loc[work["symbol"] != "SPY"].copy()
    else:
        rank_panel = work

    dates = unique_sorted_dates(rank_panel)
    folds = generate_purged_folds(
        dates,
        min_train_days=min_train_days,
        test_days=test_days,
        step_days=step_days,
        purge_gap=purge_gap,
    )
    if not folds:
        return {
            "disclaimer": DISCLAIMER,
            "error": "no_folds",
            "n_dates": len(dates),
            "baselines": {},
        }

    # Accumulate OOS predictions
    oos_parts: list[pd.DataFrame] = []
    for fold in folds:
        train = mask_dates(rank_panel, fold.train_dates)
        test = mask_dates(rank_panel, fold.test_dates)
        if test.empty or train.empty:
            continue
        chunk = test[["symbol", "ts", "z", "R", "MOM12_1"]].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["score_random"] = score_random(test, seed=random_seed + fold.fold_id)
        chunk["score_hist_mean"] = score_historical_mean(train, test)
        chunk["score_mom12_1"] = score_mom12_1(test)
        chunk["score_ridge"] = fit_ridge_predict(train, test, alpha=ridge_alpha)
        # Reindex scores to chunk index (score_* already on test.index)
        oos_parts.append(chunk)

    if not oos_parts:
        return {"disclaimer": DISCLAIMER, "error": "empty_oos", "baselines": {}}

    oos = pd.concat(oos_parts, ignore_index=True)
    baselines = {
        "random": evaluate_scores(oos, score_col="score_random"),
        "historical_mean": evaluate_scores(oos, score_col="score_hist_mean"),
        "mom12_1": evaluate_scores(oos, score_col="score_mom12_1"),
        "ridge": evaluate_scores(oos, score_col="score_ridge"),
    }

    return {
        "disclaimer": DISCLAIMER,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "min_train_days": min_train_days,
            "test_days": test_days,
            "step_days": step_days or test_days,
            "purge_gap": purge_gap,
            "n_folds": len(folds),
            "n_oos_rows": int(len(oos)),
            "rank_exclude_spy": rank_exclude_spy,
            "chronological": True,
            "random_splits": False,
        },
        "baselines": baselines,
        "fold_purge_gaps": [f.purge_gap_days for f in folds],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PE V1 baselines (purged walk-forward)")
    parser.add_argument("--data-dir", type=Path, default=PE_DIR)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--purge-gap", type=int, default=PURGE_GAP_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    panel = load_exported_dataset(args.data_dir)
    result = run_baselines_walk_forward(
        panel,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        purge_gap=args.purge_gap,
        random_seed=args.seed,
    )
    out_path = Path(args.data_dir) / "baselines_results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "disclaimer": DISCLAIMER, "baselines": result.get("baselines")}, indent=2))
    return 0 if "baselines" in result and result["baselines"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
