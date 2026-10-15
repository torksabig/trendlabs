"""Immutable ML predictions from trained XGBoost artifacts.

Does not execute trades. Predictions are append-only for a given
(symbol, timeframe, model_version, predicted_at, horizon_bars) key.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.config import ROOT, SYMBOLS
from trendlabs.db import Store, normalize_ts, utc_now
from trendlabs.features import FEATURE_COLUMNS, FEATURE_VERSION, generate_features
from trendlabs.train_ml import HORIZON_BARS


def load_model_artifact(artifact_path: str | Path) -> Any:
    path = Path(artifact_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(ROOT) / path)
    pkl = None
    for base in candidates:
        probe = base / "model.pkl" if base.name != "model.pkl" else base
        if probe.exists():
            pkl = probe
            break
        if (base / "model.pkl").exists():
            pkl = base / "model.pkl"
            break
    if pkl is None or not pkl.exists():
        raise FileNotFoundError(f"Missing model artifact under: {artifact_path}")
    with pkl.open("rb") as f:
        return pickle.load(f)


def latest_feature_row(
    store: Store,
    symbol: str,
    timeframe: str,
    *,
    feature_version: str = FEATURE_VERSION,
) -> dict[str, Any] | None:
    feats = generate_features(
        symbol,
        timeframe,
        store=store,
        feature_version=feature_version,
        persist=True,
    )
    if feats.empty:
        return None
    row = feats.iloc[-1]
    return {
        "ts": normalize_ts(row["ts"]),
        "features": {c: float(row[c]) for c in FEATURE_COLUMNS},
    }


def predict_symbol_ml(
    store: Store,
    symbol: str,
    timeframe: str = "1d",
    *,
    model_version: str | None = None,
    feature_version: str = FEATURE_VERSION,
    horizon_bars: int = HORIZON_BARS,
) -> dict[str, Any] | None:
    symbol = symbol.upper()
    meta = (
        store.get_model_version(model_version)
        if model_version
        else store.latest_model_version(symbol, timeframe, status="candidate")
    )
    if not meta:
        # Fall back to any status for the pair.
        versions = store.list_model_versions(symbol=symbol, timeframe=timeframe)
        meta = versions[0] if versions else None
    if not meta:
        return None
    if meta.get("feature_version") and meta["feature_version"] != feature_version:
        feature_version = meta["feature_version"]

    feat_row = latest_feature_row(
        store, symbol, timeframe, feature_version=feature_version
    )
    if not feat_row:
        return None

    candles = store.load_candles(symbol, timeframe)
    if candles.empty:
        return None
    # Reference close must match the feature bar timestamp.
    ts = pd.to_datetime(candles["ts"], utc=True)
    target = pd.Timestamp(feat_row["ts"])
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    match = candles.loc[ts == target]
    if match.empty:
        match = candles.iloc[[-1]]
    reference_close = float(match.iloc[0]["close"])

    model = load_model_artifact(meta["artifact_path"])
    x = pd.DataFrame([feat_row["features"]])[FEATURE_COLUMNS].astype(float)
    prob_up = float(model.predict_proba(x.to_numpy())[0, 1])
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "model_version": meta["model_version"],
        "feature_version": feature_version,
        "predicted_at": feat_row["ts"],
        "horizon_bars": horizon_bars,
        "reference_close": reference_close,
        "prob_up": prob_up,
        "prob_not_up": 1.0 - prob_up,
        "features": feat_row["features"],
        "created_at": utc_now(),
    }
    pred_id = store.insert_ml_prediction(payload)
    payload["id"] = pred_id
    return payload


def run_ml_predictions(
    store: Store,
    *,
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    feature_version: str = FEATURE_VERSION,
    horizon_bars: int = HORIZON_BARS,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for symbol in [s.upper() for s in (symbols or SYMBOLS)]:
        pred = predict_symbol_ml(
            store,
            symbol,
            timeframe,
            feature_version=feature_version,
            horizon_bars=horizon_bars,
        )
        if not pred:
            print(f"{symbol}: skipped (no model or features)")
            continue
        results.append(pred)
        print(
            f"{symbol}: prob_up={pred['prob_up']:.3f} "
            f"close={pred['reference_close']:.2f} "
            f"model={pred['model_version']} @ {pred['predicted_at']}"
        )
    store.set_meta("last_ml_prediction", utc_now())
    store.set_meta("ml_prediction_count", str(len(results)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Write immutable ML predictions")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument("--horizon", type=int, default=HORIZON_BARS)
    args = parser.parse_args()

    store = Store()
    results = run_ml_predictions(
        store,
        symbols=args.symbols,
        timeframe=args.timeframe,
        feature_version=args.feature_version,
        horizon_bars=args.horizon,
    )
    print(
        json.dumps(
            {
                "count": len(results),
                "timeframe": args.timeframe,
                "predictions": [
                    {
                        "id": r["id"],
                        "symbol": r["symbol"],
                        "prob_up": r["prob_up"],
                        "model_version": r["model_version"],
                        "predicted_at": r["predicted_at"],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
