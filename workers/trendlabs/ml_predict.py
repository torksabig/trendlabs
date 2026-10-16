"""Generate ML forecasts from trained XGBoost models and score outcomes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from trendlabs.config import ROOT
from trendlabs.db import Store
from trendlabs.features import FEATURE_COLUMNS, build_feature_frame
from trendlabs.prediction_eval import (
    calibration_table,
    hit,
    summary_stats,
    write_eval_result,
)


MODEL_DIR = Path(ROOT) / "data" / "ml" / "models"
ML_DIR = Path(ROOT) / "data" / "ml"


def load_bundle(symbol: str, timeframe: str, horizon: int = 1) -> dict[str, Any]:
    path = MODEL_DIR / f"{symbol.upper()}_{timeframe}_h{horizon}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}. Run ml_train first.")
    return joblib.load(path)


def predict_latest(store: Store, symbol: str, timeframe: str, horizon: int = 1) -> dict[str, Any]:
    bundle = load_bundle(symbol, timeframe, horizon)
    candles = store.load_candles(symbol, timeframe)
    feat = build_feature_frame(candles)
    row = feat.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    X = row[FEATURE_COLUMNS].to_frame().T.apply(pd.to_numeric, errors="coerce").astype(float)
    model = bundle["model"]
    encoder = bundle["encoder"]
    proba = model.predict_proba(X)[0]
    pred_enc = int(model.predict(X)[0])
    direction = str(encoder.inverse_transform([pred_enc])[0])
    conf = float(np.max(proba))
    class_probs = {
        str(cls): float(proba[i]) for i, cls in enumerate(encoder.classes_)
    }
    atr_pct = float(row["atr_pct"]) if pd.notna(row.get("atr_pct")) else None
    if atr_pct is None:
        expected = 0.0
    elif direction == "up":
        expected = 0.5 * atr_pct * 100
    elif direction == "down":
        expected = -0.5 * atr_pct * 100
    else:
        expected = 0.0

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "horizon": f"{horizon}bar",
        "direction": direction,
        "confidence": round(conf, 4),
        "expected_move_pct": round(expected, 4),
        "last_close": float(row["close"]),
        "as_of": str(row["ts"]),
        "class_probs": class_probs,
        "model": f"{symbol.upper()}_{timeframe}_h{horizon}",
        "source": "xgboost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "executes_trades": False,
        "drivers": {
            "rsi_14": float(row["rsi_14"]) if pd.notna(row.get("rsi_14")) else None,
            "adx_14": float(row["adx_14"]) if pd.notna(row.get("adx_14")) else None,
            "market_regime": str(row.get("market_regime")),
            "ret_1": float(row["ret_1"]) if pd.notna(row.get("ret_1")) else None,
        },
    }


def score_holdout_file(
    store: Store,
    holdout_path: Path | None = None,
) -> dict[str, Any]:
    """Score offline holdout predictions using labels saved at train time.

    Prefer the training `actual` label (same band/horizon) so we do not re-enrich
    the full history for every row. Optionally attach realized % move via a
    per-symbol cached close series.
    """
    path = holdout_path or (ML_DIR / "latest_holdout_predictions.json")
    rows = json.loads(path.read_text())

    close_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _close_frame(symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)
        if key not in close_cache:
            candles = store.load_candles(symbol, timeframe)
            if candles.empty:
                close_cache[key] = pd.DataFrame()
            else:
                frame = candles[["ts", "close"]].copy()
                frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
                frame = frame.sort_values("ts").reset_index(drop=True)
                close_cache[key] = frame
        return close_cache[key]

    outcomes = []
    scored_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        actual_dir = row.get("actual")
        if actual_dir is None:
            continue
        move = None
        frame = _close_frame(row["symbol"], row["timeframe"])
        if not frame.empty:
            origin = pd.Timestamp(row["ts"])
            if origin.tzinfo is None:
                origin = origin.tz_localize("UTC")
            else:
                origin = origin.tz_convert("UTC")
            prior = frame.index[frame["ts"] <= origin]
            if len(prior):
                sig = int(prior[-1])
                horizon = int(row.get("horizon") or 1)
                target_i = sig + horizon
                if target_i < len(frame):
                    base = float(frame.iloc[sig]["close"])
                    close = float(frame.iloc[target_i]["close"])
                    if base > 0:
                        move = (close / base - 1.0) * 100.0
        hit_val = hit(row["direction"], str(actual_dir))
        outcomes.append(
            {
                "prediction_id": None,
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "predicted_at": row["ts"],
                "direction": row["direction"],
                "confidence": float(row["confidence"]),
                "expected_move_pct": None,
                "last_close": float(row["last_close"]),
                "actual_1h_pct": move if row["timeframe"] == "1h" else None,
                "actual_4h_pct": None,
                "actual_24h_pct": move if row["timeframe"] == "1d" else None,
                "hit_1h": hit_val if row["timeframe"] == "1h" else None,
                "hit_4h": None,
                "hit_24h": hit_val if row["timeframe"] == "1d" else None,
                "actual_direction": str(actual_dir),
                "label_from_train": str(actual_dir),
                "model": row.get("model"),
                "scored_at": scored_at,
            }
        )

    # Normalize for summary_stats which expects hit_1h primarily — map daily onto hit_1h for report
    normalized = []
    for o in outcomes:
        n = dict(o)
        if n.get("hit_1h") is None and n.get("hit_24h") is not None:
            n["hit_1h"] = n["hit_24h"]
            n["actual_1h_pct"] = n.get("actual_24h_pct")
        normalized.append(n)

    payload = {
        "kind": "ml_holdout_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n": len(normalized),
        "summary": summary_stats(normalized),
        "calibration_1h": calibration_table(normalized, horizon="1h"),
        "by_symbol": {},
        "outcomes_sample": normalized[:50],
    }
    by_sym: dict[str, list] = {}
    for o in normalized:
        by_sym.setdefault(o["symbol"], []).append(o)
    payload["by_symbol"] = {
        sym: summary_stats(items) for sym, items in sorted(by_sym.items())
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="XGBoost ML predictions / holdout scoring")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--score-holdout",
        action="store_true",
        help="Score training holdout predictions instead of live latest bars",
    )
    parser.add_argument("--persist", action="store_true", help="Insert latest preds into predictions table")
    args = parser.parse_args()

    store = Store()
    ML_DIR.mkdir(parents=True, exist_ok=True)

    if args.score_holdout:
        payload = score_holdout_file(store)
        path = ML_DIR / "latest_ml_eval.json"
        path.write_text(json.dumps(payload, indent=2))
        # Also mirror into prediction_eval style path for consistency
        write_eval_result(
            {
                "kind": "prediction_evaluation",
                "created_at": payload["created_at"],
                "backfilled_predictions": 0,
                "scored_this_run": payload["n"],
                "skipped_insufficient_future": 0,
                "summary": payload["summary"],
                "calibration_1h": payload["calibration_1h"],
                "outcomes_sample": payload["outcomes_sample"],
                "source": "ml_holdout",
            },
            directory=Path(ROOT) / "data" / "prediction_eval",
        )
        print(
            json.dumps(
                {
                    "path": str(path),
                    "n": payload["n"],
                    "summary": payload["summary"],
                    "by_symbol": payload["by_symbol"],
                    "calibration_1h": payload["calibration_1h"],
                },
                indent=2,
            )
        )
        return 0

    # Latest-bar inferences for available models
    models = sorted(MODEL_DIR.glob(f"*_{args.timeframe}_h{args.horizon}.joblib"))
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        models = [m for m in models if m.name.split("_")[0] in wanted]
    if not models:
        raise SystemExit("No trained models found. Run: npm run ml-train")

    preds = []
    for path in models:
        symbol = path.name.split("_")[0]
        try:
            pred = predict_latest(store, symbol, args.timeframe, args.horizon)
            preds.append(pred)
            if args.persist:
                # Map horizon label for DB schema
                db_pred = dict(pred)
                db_pred["horizon"] = args.timeframe  # keep short; details in drivers
                db_pred["next_update_at"] = datetime.now(timezone.utc).isoformat()
                db_pred["summary"] = (
                    f"{symbol} {args.timeframe}: XGBoost {pred['direction']} "
                    f"conf={pred['confidence']:.0%} as_of={pred['as_of']}"
                )
                # Lineage: freeze the exact feature snapshot used by this prediction.
                from trendlabs.features import FEATURE_VERSION, build_feature_frame

                db_pred["feature_version"] = FEATURE_VERSION
                candles = store.load_candles(symbol, args.timeframe)
                feat = build_feature_frame(candles)
                as_of_ts = pd.Timestamp(pred["as_of"])
                snap = feat.loc[pd.to_datetime(feat["ts"], utc=True) == as_of_ts].dropna(subset=FEATURE_COLUMNS)
                if not snap.empty:
                    store.upsert_market_features(
                        snap, symbol=symbol, timeframe=args.timeframe,
                        feature_version=FEATURE_VERSION, columns=FEATURE_COLUMNS,
                    )
                store.insert_prediction(db_pred)
            print(
                f"{symbol}: {pred['direction']} conf={pred['confidence']:.3f} "
                f"close={pred['last_close']:.2f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{symbol}: ERROR {exc}")

    out = ML_DIR / "latest_predictions.json"
    out.write_text(
        json.dumps(
            {
                "kind": "ml_predictions",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "timeframe": args.timeframe,
                "predictions": preds,
            },
            indent=2,
        )
    )
    print(json.dumps({"path": str(out), "count": len(preds)}, indent=2))
    return 0 if preds else 1


if __name__ == "__main__":
    raise SystemExit(main())
