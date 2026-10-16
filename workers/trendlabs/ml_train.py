"""Per-symbol XGBoost training on extracted candle features (research only)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from trendlabs.config import ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.features import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    dataset_from_candles,
    default_band,
    feature_importance_frame,
)


MODEL_DIR = Path(ROOT) / "data" / "ml" / "models"
REPORT_DIR = Path(ROOT) / "data" / "ml"


def _time_split(
    n: int, *, test_frac: float = 0.2, min_test: int = 50, embargo: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split with an embargo gap between train and test.

    Train labels at row i look ahead to i+horizon; embargo ≥ horizon drops the
    rows whose label windows touch the test feature window.
    """
    test_n = max(min_test, int(n * test_frac))
    test_n = min(test_n, n // 3) if n >= 90 else max(1, n // 5)
    embargo = max(0, int(embargo))
    train_end = n - test_n - embargo
    if train_end < 30:
        raise ValueError(f"Not enough rows to train/test split (n={n}, embargo={embargo})")
    idx = np.arange(n)
    return idx[:train_end], idx[train_end + embargo :]


def train_symbol(
    store: Store,
    symbol: str,
    timeframe: str,
    *,
    horizon: int = 1,
    band: float | None = None,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    candles = store.load_candles(symbol, timeframe)
    if len(candles) < 120:
        raise RuntimeError(f"{symbol} {timeframe}: need ≥120 bars, have {len(candles)}")

    band = default_band(timeframe) if band is None else band
    X, y, meta = dataset_from_candles(candles, horizon=horizon, band=band)
    if len(X) < 80:
        raise RuntimeError(f"{symbol} {timeframe}: feature rows {len(X)} too small")

    # Embargo ≥ horizon: train labels may not share their forward window with test features.
    train_idx, test_idx = _time_split(len(X), test_frac=test_frac, embargo=horizon)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train.astype(str))
    known = set(encoder.classes_)
    test_mask = y_test.astype(str).isin(known).to_numpy()
    X_test = X_test.loc[test_mask].reset_index(drop=True)
    y_test = y_test.loc[test_mask].reset_index(drop=True)
    meta_test = meta_test.loc[test_mask].reset_index(drop=True)
    y_test_enc = encoder.transform(y_test.astype(str))

    n_classes = len(encoder.classes_)
    if n_classes < 2:
        raise RuntimeError(f"{symbol} {timeframe}: need ≥2 classes in training labels")
    common = dict(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=2,
        tree_method="hist",
    )
    if n_classes == 2:
        model = XGBClassifier(**common, objective="binary:logistic", eval_metric="logloss")
    else:
        model = XGBClassifier(
            **common,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
        )

    model.fit(X_train, y_train_enc)
    proba = model.predict_proba(X_test)
    pred_enc = model.predict(X_test)
    pred_labels = encoder.inverse_transform(pred_enc)

    acc = float(accuracy_score(y_test_enc, pred_enc))
    f1 = float(f1_score(y_test_enc, pred_enc, average="macro", zero_division=0))
    report = classification_report(
        y_test.astype(str),
        pred_labels.astype(str),
        zero_division=0,
        output_dict=True,
    )

    conf = proba.max(axis=1)
    directional = [
        (p, t)
        for p, t in zip(pred_labels, y_test.astype(str), strict=True)
        if p in {"up", "down"}
    ]
    dir_acc = (
        sum(1 for p, t in directional if p == t) / len(directional) if directional else None
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol.upper()}_{timeframe}_h{horizon}"
    model_path = MODEL_DIR / f"{stem}.joblib"
    bundle = {
        "model": model,
        "encoder": encoder,
        "feature_columns": FEATURE_COLUMNS,
        "feature_version": FEATURE_VERSION,
        "embargo": int(horizon),
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "horizon": horizon,
        "band": band,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classes": list(encoder.classes_),
    }
    joblib.dump(bundle, model_path)

    holdout_rows = []
    for i in range(len(pred_labels)):
        holdout_rows.append(
            {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "ts": str(meta_test.iloc[i]["ts"]),
                "last_close": float(meta_test.iloc[i]["close"]),
                "direction": str(pred_labels[i]),
                "actual": str(y_test.iloc[i]),
                "confidence": float(conf[i]),
                "horizon": horizon,
                "model": stem,
            }
        )

    result = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "horizon": horizon,
        "band": band,
        "model_path": str(model_path),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(X_test)),
        "class_balance_train": y_train.value_counts().to_dict(),
        "class_balance_test": y_test.value_counts().to_dict(),
        "accuracy": acc,
        "macro_f1": f1,
        "directional_accuracy": dir_acc,
        "mean_confidence": float(np.mean(conf)) if len(conf) else None,
        "classification_report": report,
        "feature_importance": feature_importance_frame(model, FEATURE_COLUMNS)[:15],
        "test_start_ts": str(meta_test.iloc[0]["ts"]) if len(meta_test) else None,
        "test_end_ts": str(meta_test.iloc[-1]["ts"]) if len(meta_test) else None,
        "holdout_predictions": holdout_rows,
    }
    return result


def write_train_report(results: list[dict[str, Any]], directory: Path = REPORT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Compact holdouts in latest (keep full per-symbol files)
    compact_agents = []
    all_holdouts: list[dict[str, Any]] = []
    for r in results:
        holdouts = r.get("holdout_predictions") or []
        all_holdouts.extend(holdouts)
        per = directory / "models" / f"{r['symbol']}_{r['timeframe']}_h{r['horizon']}_report.json"
        per.parent.mkdir(parents=True, exist_ok=True)
        per.write_text(json.dumps(r, indent=2, default=str))
        summary = {k: v for k, v in r.items() if k != "holdout_predictions"}
        compact_agents.append(summary)

    payload = {
        "kind": "ml_train_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": compact_agents,
        "holdout_prediction_count": len(all_holdouts),
    }
    path = directory / "latest_train.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    (directory / "latest_holdout_predictions.json").write_text(
        json.dumps(all_holdouts, indent=2)
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train per-symbol XGBoost models")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--band", type=float, default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    store = Store()
    available = store.list_symbols_with_data(args.timeframe)
    symbols = args.symbols or ([s for s in SYMBOLS if s in available] or available)
    symbols = [s for s in symbols if s in available]
    if not symbols:
        raise SystemExit(f"No symbols with {args.timeframe} data. Collect first.")

    results = []
    errors = []
    for symbol in symbols:
        try:
            print(f"Training {symbol} {args.timeframe}…")
            result = train_symbol(
                store,
                symbol,
                args.timeframe,
                horizon=args.horizon,
                band=args.band,
                test_frac=args.test_frac,
            )
            results.append(result)
            print(
                f"  accuracy={result['accuracy']:.3f} macro_f1={result['macro_f1']:.3f} "
                f"dir_acc={result['directional_accuracy']} "
                f"train={result['train_rows']} test={result['test_rows']}"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"symbol": symbol, "error": str(exc)})
            print(f"  ERROR {exc}")

    path = write_train_report(results)
    summary = {
        "path": str(path),
        "trained": [
            {
                "symbol": r["symbol"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "directional_accuracy": r["directional_accuracy"],
                "test_rows": r["test_rows"],
                "model_path": r["model_path"],
            }
            for r in results
        ],
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
