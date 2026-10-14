"""PE volatility V1 — RV5 persistence vs HAR log-volatility (research gate).

Independent of the return-ranking path. Compares:
  1) 5-day realized-vol persistence: predict future_rv_5 with past RV5
  2) HAR on log realized variance / vol (daily + weekly + monthly lags)

Metrics (OOS, purged walk-forward): MAE, RMSE, QLIKE, cross-sectional vol Rank IC.

Disclaimer: EXPLORATORY — SURVIVORSHIP BIAS NOT FULLY CONTROLLED
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from trendlabs.prediction import DATASET_VERSION, DISCLAIMER, HORIZON_DAYS, PURGE_GAP_DAYS
from trendlabs.prediction.dataset import PE_DIR, load_exported_dataset
from trendlabs.prediction.metrics_ic import daily_spearman_ic, summarize_ic
from trendlabs.prediction.targets import daily_log_returns
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates

EPS = 1e-12
HAR_FEATURE_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]


def realized_var_window(log_ret: pd.Series, window: int) -> pd.Series:
    """Sum of squared log returns over the past `window` bars ending at t (inclusive)."""
    sq = log_ret.astype(float) ** 2
    return sq.rolling(window=window, min_periods=window).sum()


def future_realized_var(log_ret: pd.Series, horizon: int = HORIZON_DAYS) -> pd.Series:
    """Sum of squared log returns over t+1 .. t+horizon (excludes r_t)."""
    sq = log_ret.astype(float) ** 2
    total = pd.Series(0.0, index=sq.index, dtype=float)
    for j in range(1, horizon + 1):
        total = total + sq.shift(-j)
    # Rows that would pull past the series end are NaN after any shift(-horizon)
    total = total.where(sq.shift(-horizon).notna())
    return total


def future_rv(log_ret: pd.Series, horizon: int = HORIZON_DAYS) -> pd.Series:
    """sqrt of future realized variance over horizon."""
    return np.sqrt(future_realized_var(log_ret, horizon=horizon))


def past_rv(log_ret: pd.Series, window: int) -> pd.Series:
    """sqrt of past realized variance over window ending at t."""
    return np.sqrt(realized_var_window(log_ret, window))


def build_vol_panel(panel: pd.DataFrame, *, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """Per-symbol RV features + future_rv_5 target from close prices.

    Columns added:
      log_ret, rv_d, rv_w, rv_m, rv5_past, future_rv_5,
      log_rv_d, log_rv_w, log_rv_m, log_future_rv_5
    where rv_d/w/m are sqrt of mean daily variance over 1/5/22 days (HAR lags).
    """
    parts: list[pd.DataFrame] = []
    for _, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("ts").reset_index(drop=True).copy()
        g["log_ret"] = daily_log_returns(g["close"])
        sq = g["log_ret"] ** 2
        var_d = sq
        var_w = sq.rolling(5, min_periods=5).mean()
        var_m = sq.rolling(22, min_periods=22).mean()
        g["rv_d"] = np.sqrt(var_d.clip(lower=0.0))
        g["rv_w"] = np.sqrt(var_w.clip(lower=0.0))
        g["rv_m"] = np.sqrt(var_m.clip(lower=0.0))
        g["rv5_past"] = past_rv(g["log_ret"], window=horizon)
        g["future_rv_5"] = future_rv(g["log_ret"], horizon=horizon)
        g["log_rv_d"] = np.log(g["rv_d"].astype(float) + EPS)
        g["log_rv_w"] = np.log(g["rv_w"].astype(float) + EPS)
        g["log_rv_m"] = np.log(g["rv_m"].astype(float) + EPS)
        g["log_future_rv_5"] = np.log(g["future_rv_5"].astype(float) + EPS)
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def qlike(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Patton QLIKE on variance scale: σ²/σ̂² − log(σ²/σ̂²) − 1 (lower is better)."""
    r2 = np.asarray(realized, dtype=float) ** 2
    p2 = np.asarray(predicted, dtype=float) ** 2
    mask = np.isfinite(r2) & np.isfinite(p2) & (p2 > 0) & (r2 >= 0)
    if not mask.any():
        return float("nan")
    ratio = r2[mask] / p2[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def regression_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return {"MAE": None, "RMSE": None, "QLIKE": None, "n": 0}
    err = yp[mask] - yt[mask]
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "QLIKE": qlike(yt[mask], yp[mask]),
        "n": int(mask.sum()),
    }


def fit_har_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """OLS HAR on log(future_rv_5) ~ log_rv_d + log_rv_w + log_rv_m; train-only fit."""
    cols = HAR_FEATURE_COLS
    X_train = train[cols].astype(float)
    y_train = train["log_future_rv_5"].astype(float)
    mask = X_train.notna().all(axis=1) & y_train.notna() & np.isfinite(y_train)
    pred = pd.Series(np.nan, index=test.index, dtype=float)
    if mask.sum() < 30:
        return pred
    model = LinearRegression()
    model.fit(X_train.loc[mask].to_numpy(), y_train.loc[mask].to_numpy())
    X_test = test[cols].astype(float)
    ok = X_test.notna().all(axis=1)
    if ok.any():
        log_hat = model.predict(X_test.loc[ok].to_numpy())
        pred.loc[ok] = np.exp(log_hat)  # back to RV scale (minus EPS negligible)
    return pred


def persistence_predict(test: pd.DataFrame) -> pd.Series:
    """5-day RV persistence: past RV5 predicts future RV5."""
    return test["rv5_past"].astype(float)


def evaluate_vol_predictions(
    oos: pd.DataFrame,
    *,
    pred_col: str,
    target_col: str = "future_rv_5",
    rank_exclude_spy: bool = True,
) -> dict[str, Any]:
    """Point metrics + cross-sectional Rank IC of predicted vs realized vol."""
    sub = oos[[pred_col, target_col, "ts", "symbol"]].copy()
    sub = sub.dropna(subset=[pred_col, target_col])
    err = regression_errors(sub[target_col].to_numpy(), sub[pred_col].to_numpy())

    rank_panel = sub.loc[sub["symbol"] != "SPY"].copy() if rank_exclude_spy else sub
    daily_ic = daily_spearman_ic(
        rank_panel, score_col=pred_col, target_col=target_col, min_names=2
    )
    ic_summary = summarize_ic(daily_ic)
    return {
        **err,
        "RankIC": ic_summary.get("MeanIC"),
        "RankICIR": ic_summary.get("ICIR"),
        "RankIC_PositiveRate": ic_summary.get("PositiveICRate"),
        "RankIC_n_days": ic_summary.get("n_days"),
    }


def run_volatility_walk_forward(
    panel: pd.DataFrame,
    *,
    min_train_days: int = 252,
    test_days: int = 63,
    step_days: int | None = None,
    purge_gap: int = PURGE_GAP_DAYS,
    rank_exclude_spy: bool = True,
) -> dict[str, Any]:
    """Purged WF: RV5 persistence vs HAR log-vol."""
    vol = build_vol_panel(panel)
    if vol.empty:
        return {"disclaimer": DISCLAIMER, "error": "empty_panel", "models": {}}

    # Require target + HAR lags + persistence feature
    need = ["future_rv_5", "rv5_past", *HAR_FEATURE_COLS, "log_future_rv_5"]
    work = vol.dropna(subset=need).copy()
    if work.empty:
        return {"disclaimer": DISCLAIMER, "error": "no_valid_rows", "models": {}}

    dates = unique_sorted_dates(work)
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
            "models": {},
        }

    oos_parts: list[pd.DataFrame] = []
    for fold in folds:
        train = mask_dates(work, fold.train_dates)
        test = mask_dates(work, fold.test_dates)
        if train.empty or test.empty:
            continue
        chunk = test[["symbol", "ts", "future_rv_5", "rv5_past"]].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["pred_persistence"] = persistence_predict(test)
        chunk["pred_har"] = fit_har_predict(train, test)
        oos_parts.append(chunk)

    if not oos_parts:
        return {"disclaimer": DISCLAIMER, "error": "empty_oos", "models": {}}

    oos = pd.concat(oos_parts, ignore_index=True)
    models = {
        "rv5_persistence": evaluate_vol_predictions(
            oos, pred_col="pred_persistence", rank_exclude_spy=rank_exclude_spy
        ),
        "har_log_vol": evaluate_vol_predictions(
            oos, pred_col="pred_har", rank_exclude_spy=rank_exclude_spy
        ),
    }

    # Gate: does HAR beat persistence on QLIKE (primary) and MAE / Rank IC?
    pers = models["rv5_persistence"]
    har = models["har_log_vol"]

    def _beats(metric: str, *, lower_better: bool) -> bool | None:
        a, b = pers.get(metric), har.get(metric)
        if a is None or b is None:
            return None
        return (float(b) < float(a)) if lower_better else (float(b) > float(a))

    comparison = {
        "har_beats_persistence_QLIKE": _beats("QLIKE", lower_better=True),
        "har_beats_persistence_MAE": _beats("MAE", lower_better=True),
        "har_beats_persistence_RMSE": _beats("RMSE", lower_better=True),
        "har_beats_persistence_RankIC": _beats("RankIC", lower_better=False),
        "recommendation": None,
    }
    qlike_win = comparison["har_beats_persistence_QLIKE"]
    mae_win = comparison["har_beats_persistence_MAE"]
    ric_win = comparison["har_beats_persistence_RankIC"]
    if qlike_win is True and (mae_win or ric_win):
        comparison["recommendation"] = (
            "HAR beats RV5 persistence on QLIKE and at least one of MAE/RankIC — "
            "proceed to OHLC estimators / XGBoost volatility."
        )
    elif qlike_win is True:
        comparison["recommendation"] = (
            "HAR improves QLIKE vs persistence; RankIC/MAE mixed — "
            "OHLC / XGB optional, keep HAR as benchmark."
        )
    else:
        comparison["recommendation"] = (
            "HAR does not clearly beat RV5 persistence — "
            "do not prioritize OHLC estimators / XGBoost volatility yet."
        )

    return {
        "disclaimer": DISCLAIMER,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "pe_volatility_v1",
        "target": "future_rv_5 = sqrt(sum_{j=1..5} r_{t+j}^2)",
        "models_described": {
            "rv5_persistence": "hat = past RV5 = sqrt(sum_{j=0..4} r_{t-j}^2)",
            "har_log_vol": (
                "log(future_rv_5) ~ log_rv_d + log_rv_w + log_rv_m "
                "(OLS, train-only; predict exp)"
            ),
        },
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
        "models": models,
        "comparison": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PE volatility V1: RV5 persistence vs HAR log-volatility"
    )
    parser.add_argument("--data-dir", type=Path, default=PE_DIR)
    parser.add_argument("--min-train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=None)
    parser.add_argument("--purge-gap", type=int, default=PURGE_GAP_DAYS)
    args = parser.parse_args()

    panel = load_exported_dataset(args.data_dir)
    # Need close for returns; keep symbol/ts/close only from PE export
    if "close" not in panel.columns:
        print("ERROR: dataset missing close column")
        return 1
    result = run_volatility_walk_forward(
        panel[["symbol", "ts", "close"]].copy(),
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        purge_gap=args.purge_gap,
    )
    out_path = Path(args.data_dir) / "volatility_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        "wrote": str(out_path),
        "disclaimer": DISCLAIMER,
        "models": result.get("models"),
        "comparison": result.get("comparison"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.get("models") else 1


if __name__ == "__main__":
    raise SystemExit(main())
