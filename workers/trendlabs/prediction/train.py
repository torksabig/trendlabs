"""Phase C (light) — XGBoost regressor for z_hat with purged walk-forward."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from trendlabs.prediction import DATASET_VERSION, DISCLAIMER, PURGE_GAP_DAYS
from trendlabs.prediction.baselines import run_baselines_walk_forward
from trendlabs.prediction.dataset import PE_DIR, load_exported_dataset
from trendlabs.prediction.features import FEATURE_COLUMNS
from trendlabs.prediction.metrics_ic import evaluate_scores, fold_mean_ic_stats
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates

# Simple regularized defaults — no hyperparameter hunt.
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 42,
}


def fit_xgb_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLUMNS,
    params: dict[str, Any] | None = None,
) -> pd.Series:
    """Fit XGB on raw train features only; predict z_hat on test (no StandardScaler)."""
    params = {**XGB_PARAMS, **(params or {})}
    X_train = train[feature_cols].astype(float)
    y_train = train["z"].astype(float)
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask]
    pred = pd.Series(np.nan, index=test.index, dtype=float)
    if len(X_train) < 50:
        return pred

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    X_test = test[feature_cols].astype(float)
    ok = X_test.notna().all(axis=1)
    if ok.any():
        pred.loc[ok] = model.predict(X_test.loc[ok])
    return pred


def run_xgb_walk_forward(
    panel: pd.DataFrame,
    *,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int | None = None,
    purge_gap: int = PURGE_GAP_DAYS,
    rank_exclude_spy: bool = True,
    return_oos: bool = False,
) -> dict[str, Any]:
    work = panel.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    rank_panel = work.loc[work["symbol"] != "SPY"].copy() if rank_exclude_spy else work

    dates = unique_sorted_dates(rank_panel)
    folds = generate_purged_folds(
        dates,
        min_train_days=min_train_days,
        test_days=test_days,
        step_days=step_days,
        purge_gap=purge_gap,
    )
    if not folds:
        out: dict[str, Any] = {"disclaimer": DISCLAIMER, "error": "no_folds", "xgboost": {}}
        if return_oos:
            out["oos"] = pd.DataFrame()
        return out

    oos_parts: list[pd.DataFrame] = []
    for fold in folds:
        train = mask_dates(rank_panel, fold.train_dates)
        test = mask_dates(rank_panel, fold.test_dates)
        if test.empty or train.empty:
            continue
        keep_cols = [c for c in ["symbol", "ts", "z", "R", "open", "close", "MOM12_1"] if c in test.columns]
        chunk = test[keep_cols].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["z_hat"] = fit_xgb_predict(train, test)
        oos_parts.append(chunk)

    if not oos_parts:
        out = {"disclaimer": DISCLAIMER, "error": "empty_oos", "xgboost": {}}
        if return_oos:
            out["oos"] = pd.DataFrame()
        return out

    oos = pd.concat(oos_parts, ignore_index=True)
    xgb_metrics = evaluate_scores(oos, score_col="z_hat")
    xgb_metrics["fold_ic"] = fold_mean_ic_stats(oos, score_col="z_hat")
    result: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "XGBRegressor",
        "hyperparameters": XGB_PARAMS,
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
            "scaling": "none (XGBoost on raw features; Ridge baselines still StandardScaler train-only)",
        },
        "xgboost": xgb_metrics,
        "fold_purge_gaps": [f.purge_gap_days for f in folds],
    }
    if return_oos:
        result["oos"] = oos
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PE V1 XGBoost regressor (purged WF)")
    parser.add_argument("--data-dir", type=Path, default=PE_DIR)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--purge-gap", type=int, default=PURGE_GAP_DAYS)
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="Also recompute baselines and write combined comparison JSON",
    )
    args = parser.parse_args()

    panel = load_exported_dataset(args.data_dir)
    xgb_result = run_xgb_walk_forward(
        panel,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        purge_gap=args.purge_gap,
    )
    out_path = Path(args.data_dir) / "xgb_results.json"
    out_path.write_text(json.dumps(xgb_result, indent=2), encoding="utf-8")

    if args.with_baselines:
        base = run_baselines_walk_forward(
            panel,
            min_train_days=args.min_train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            purge_gap=args.purge_gap,
        )
        combined = {
            "disclaimer": DISCLAIMER,
            "dataset_version": DATASET_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baselines": base.get("baselines", {}),
            "xgboost": xgb_result.get("xgboost", {}),
            "protocol": xgb_result.get("protocol"),
        }
        cmp_path = Path(args.data_dir) / "ic_comparison.json"
        cmp_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(json.dumps({"wrote": [str(out_path), str(cmp_path)], "comparison": combined}, indent=2))
    else:
        print(json.dumps({"wrote": str(out_path), "xgboost": xgb_result.get("xgboost")}, indent=2))

    return 0 if xgb_result.get("xgboost") else 1


if __name__ == "__main__":
    raise SystemExit(main())
