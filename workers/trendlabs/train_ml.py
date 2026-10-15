"""Per-symbol XGBoost training for binary forward-close direction.

Label: will close be higher after the next HORIZON_BARS completed candles?
Cloud workspace already has trained 1d models under models/*/1d/.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.config import ROOT, SYMBOLS
from trendlabs.db import Store, normalize_ts, utc_now
from trendlabs.features import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    compute_feature_frame,
    generate_features,
)

HORIZON_BARS = 4
MODELS_DIR = Path(ROOT) / "models"
MIN_ROWS = 400

XGB_PARAMS = {
    "n_estimators": 60,
    "max_depth": 3,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 2,
    "random_state": 42,
    "tree_method": "hist",
}


def build_labeled_dataset(
    candles: pd.DataFrame,
    *,
    horizon: int = HORIZON_BARS,
) -> pd.DataFrame:
    if candles.empty:
        return pd.DataFrame()
    raw = candles.sort_values("ts").reset_index(drop=True).copy()
    feats = compute_feature_frame(raw)
    close = raw["close"].astype(float)
    future_close = close.shift(-horizon)
    label = (future_close > close).astype("float")
    label[future_close.isna()] = np.nan
    out = feats.copy()
    out["close"] = close
    out["future_close"] = future_close
    out["label"] = label
    out["trend_up"] = (out["ema_20"] > out["ema_50"]).astype(float)
    out = out.dropna(subset=[*FEATURE_COLUMNS, "label"]).reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    return out


def chronological_split(
    n: int,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo: int = HORIZON_BARS,
) -> dict[str, slice]:
    if n < 50:
        raise ValueError(f"Need more rows for split, got {n}")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_slice = slice(0, max(1, train_end - embargo))
    val_start = min(n - 1, train_end + embargo)
    val_slice = slice(val_start, max(val_start + 1, val_end - embargo))
    test_start = min(n - 1, val_end + embargo)
    test_slice = slice(test_start, n)
    return {"train": train_slice, "val": val_slice, "test": test_slice}


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
    )
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _fit_xgb(X_train: np.ndarray, y_train: np.ndarray):
    from xgboost import XGBClassifier
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_symbol(
    symbol: str,
    timeframe: str = "1d",
    *,
    store: Store | None = None,
    feature_version: str = FEATURE_VERSION,
    horizon: int = HORIZON_BARS,
    persist_features: bool = True,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    store = store or Store()
    symbol = symbol.upper()
    candles = store.load_candles(symbol, timeframe)
    if len(candles) < MIN_ROWS:
        return {"symbol": symbol, "status": "skipped", "reason": f"insufficient_candles:{len(candles)}"}
    if persist_features:
        generate_features(symbol, timeframe, store=store, candles=candles, feature_version=feature_version, persist=True)
    labeled = build_labeled_dataset(candles, horizon=horizon)
    if len(labeled) < MIN_ROWS:
        return {"symbol": symbol, "status": "skipped", "reason": f"insufficient_labeled_rows:{len(labeled)}"}
    X = labeled[FEATURE_COLUMNS].astype(float).to_numpy()
    y = labeled["label"].astype(int).to_numpy()
    splits = chronological_split(len(labeled), embargo=horizon)
    X_train, y_train = X[splits["train"]], y[splits["train"]]
    X_test, y_test = X[splits["test"]], y[splits["test"]]
    model = _fit_xgb(X_train, y_train)
    prob_test = model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, prob_test)
    trained_at = datetime.now(timezone.utc)
    stamp = trained_at.strftime("%Y%m%dT%H%M%SZ")
    model_version = f"{symbol}_{timeframe}_xgb_{feature_version}_{stamp}"
    artifact_dir = models_dir / symbol / timeframe / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    try:
        model.save_model(str(artifact_dir / "model.json"))
    except Exception:
        pass
    artifact_rel = str(artifact_dir.resolve().relative_to(Path(ROOT).resolve()))
    metrics_payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_version": feature_version,
        "model_version": model_version,
        "horizon_bars": horizon,
        "test": test_metrics,
        "trained_at": trained_at.isoformat(),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (artifact_dir / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (artifact_dir / "meta.json").write_text(json.dumps({
        "model_version": model_version, "symbol": symbol, "timeframe": timeframe,
        "feature_version": feature_version, "horizon_bars": horizon, "artifact_path": artifact_rel,
    }, indent=2))
    store.upsert_model_version({
        "model_version": model_version, "symbol": symbol, "timeframe": timeframe,
        "feature_version": feature_version, "algorithm": "xgboost", "status": "candidate",
        "artifact_path": artifact_rel, "metrics": metrics_payload,
        "trained_at": trained_at.isoformat(),
        "dataset_start": normalize_ts(labeled["ts"].iloc[0]),
        "dataset_end": normalize_ts(labeled["ts"].iloc[-1]),
        "notes": "MVP chronological split",
    })
    print(f"{symbol} {timeframe}: model={model_version} test_acc={test_metrics['accuracy']:.3f}")
    return {"symbol": symbol, "status": "trained", "model_version": model_version, "artifact_path": artifact_rel, "test": test_metrics}


def train_universe(* , store: Store | None = None, symbols: list[str] | None = None, timeframe: str = "1d",
                   feature_version: str = FEATURE_VERSION, horizon: int = HORIZON_BARS) -> dict[str, Any]:
    store = store or Store()
    results = [train_symbol(s, timeframe, store=store, feature_version=feature_version, horizon=horizon)
               for s in [x.upper() for x in (symbols or SYMBOLS)]]
    trained = [r for r in results if r.get("status") == "trained"]
    return {"timeframe": timeframe, "n_trained": len(trained), "n_skipped": len(results) - len(trained), "results": results, "trained_at": utc_now()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train per-symbol XGBoost models")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument("--horizon", type=int, default=HORIZON_BARS)
    args = parser.parse_args()
    payload = train_universe(symbols=args.symbols, timeframe=args.timeframe,
                             feature_version=args.feature_version, horizon=args.horizon)
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload["n_trained"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
