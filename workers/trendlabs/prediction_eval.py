"""Score stored forecasts against realized forward returns + confidence calibration.

Does not execute trades. Predictions remain research artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.config import DEFAULT_TIMEFRAME, ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.predict import predict_symbol


EVAL_DIR = Path(ROOT) / "data" / "prediction_eval"

# Confidence buckets for calibration table
CONF_BUCKETS = [
    (0.50, 0.60, "50–60%"),
    (0.60, 0.70, "60–70%"),
    (0.70, 0.80, "70–80%"),
    (0.80, 0.90, "80–90%"),
    (0.90, 1.01, "90%+"),
]

HORIZONS_HOURS = {"1h": 1.0, "4h": 4.0, "24h": 24.0}


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _signal_index(candles: pd.DataFrame, predicted_at: str, last_close: float) -> int | None:
    """Locate the candle the forecast was conditioned on."""
    df = candles.reset_index(drop=True)
    ts = pd.to_datetime(df["ts"], utc=True)
    origin = _parse_ts(predicted_at)
    # Prefer last bar at/before prediction time
    prior = df.index[ts <= origin]
    if len(prior):
        return int(prior[-1])
    # Fallback: closest close to stored last_close
    diffs = (df["close"].astype(float) - float(last_close)).abs()
    if diffs.empty:
        return None
    return int(diffs.idxmin())


def realized_move_pct(
    candles: pd.DataFrame,
    *,
    predicted_at: str,
    last_close: float,
    hours: float,
    timeframe: str = "1h",
) -> float | None:
    """Percent change from signal bar close to a forward bar.

    For 1h bars, horizons map to bar offsets (1 / 4 / 24). Wall-clock lookup is
    used only as a fallback when bar offsets are unavailable.
    """
    if candles.empty or last_close <= 0:
        return None
    df = candles.reset_index(drop=True)
    sig = _signal_index(df, predicted_at, last_close)
    if sig is None:
        return None

    bar_hours = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}.get(timeframe, 1.0)
    offset = max(1, int(round(hours / bar_hours)))
    target_i = sig + offset
    if target_i < len(df):
        close = float(df.iloc[target_i]["close"])
        base = float(df.iloc[sig]["close"])
        if base <= 0:
            return None
        return (close / base - 1.0) * 100.0

    # Wall-clock fallback when history after the signal bar is sparse
    ts = pd.to_datetime(df["ts"], utc=True)
    origin = ts.iloc[sig]
    target = origin + pd.Timedelta(hours=hours)
    future = df[ts >= target]
    if future.empty:
        return None
    close = float(future.iloc[0]["close"])
    base = float(df.iloc[sig]["close"])
    return (close / base - 1.0) * 100.0


def direction_from_move(move_pct: float | None, *, band: float = 0.05) -> str | None:
    if move_pct is None:
        return None
    if move_pct > band:
        return "up"
    if move_pct < -band:
        return "down"
    return "sideways"


def hit(predicted: str, actual_dir: str | None) -> int | None:
    if actual_dir is None:
        return None
    return 1 if predicted == actual_dir else 0


def score_prediction(store: Store, pred: dict[str, Any]) -> dict[str, Any] | None:
    candles = store.load_candles(pred["symbol"], pred["timeframe"])
    if candles.empty:
        return None

    expected = pred.get("expected_move_pct")
    if expected is None:
        expected = (pred.get("drivers") or {}).get("expected_move_pct")

    actuals: dict[str, float | None] = {}
    hits: dict[str, int | None] = {}
    for key, hours in HORIZONS_HOURS.items():
        move = realized_move_pct(
            candles,
            predicted_at=pred["created_at"],
            last_close=float(pred["last_close"]),
            hours=hours,
            timeframe=str(pred.get("timeframe") or "1h"),
        )
        actuals[key] = move
        hits[key] = hit(pred["direction"], direction_from_move(move))

    # Need at least the primary horizon to persist
    if actuals["1h"] is None:
        return None

    return {
        "prediction_id": int(pred["id"]),
        "symbol": pred["symbol"],
        "timeframe": pred["timeframe"],
        "predicted_at": pred["created_at"],
        "direction": pred["direction"],
        "confidence": float(pred["confidence"]),
        "expected_move_pct": float(expected) if expected is not None else None,
        "last_close": float(pred["last_close"]),
        "actual_1h_pct": actuals["1h"],
        "actual_4h_pct": actuals["4h"],
        "actual_24h_pct": actuals["24h"],
        "hit_1h": hits["1h"],
        "hit_4h": hits["4h"],
        "hit_24h": hits["24h"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def calibration_table(outcomes: list[dict[str, Any]], *, horizon: str = "1h") -> list[dict[str, Any]]:
    hit_key = f"hit_{horizon}"
    rows = []
    for lo, hi, label in CONF_BUCKETS:
        bucket = [
            o
            for o in outcomes
            if o.get(hit_key) is not None and lo <= float(o["confidence"]) < hi
        ]
        if not bucket:
            rows.append(
                {
                    "confidence_bucket": label,
                    "count": 0,
                    "accuracy": None,
                    "avg_confidence": None,
                    "avg_abs_move_pct": None,
                }
            )
            continue
        acc = sum(int(o[hit_key]) for o in bucket) / len(bucket)
        moves = [
            abs(float(o[f"actual_{horizon}_pct"]))
            for o in bucket
            if o.get(f"actual_{horizon}_pct") is not None
        ]
        rows.append(
            {
                "confidence_bucket": label,
                "count": len(bucket),
                "accuracy": acc,
                "avg_confidence": sum(float(o["confidence"]) for o in bucket) / len(bucket),
                "avg_abs_move_pct": (sum(moves) / len(moves)) if moves else None,
                "overconfident": acc < (lo + hi) / 2 - 0.05 if len(bucket) >= 5 else None,
            }
        )
    return rows


def summary_stats(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    if not outcomes:
        return {
            "n": 0,
            "accuracy_1h": None,
            "accuracy_4h": None,
            "accuracy_24h": None,
            "direction_counts": {},
        }

    def _acc(key: str) -> float | None:
        vals = [int(o[key]) for o in outcomes if o.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    dirs: dict[str, int] = {}
    for o in outcomes:
        d = o["direction"]
        dirs[d] = dirs.get(d, 0) + 1

    return {
        "n": len(outcomes),
        "accuracy_1h": _acc("hit_1h"),
        "accuracy_4h": _acc("hit_4h"),
        "accuracy_24h": _acc("hit_24h"),
        "direction_counts": dirs,
        "mean_confidence": sum(float(o["confidence"]) for o in outcomes) / len(outcomes),
    }


def backfill_historical_predictions(
    store: Store,
    *,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    step: int = 6,
    min_history: int = 80,
    leave_forward: int = 25,
) -> int:
    """Insert historical rule-based forecasts so outcomes can be scored offline."""
    inserted = 0
    for symbol in symbols or SYMBOLS:
        candles = store.load_candles(symbol, timeframe)
        if len(candles) < min_history + leave_forward:
            continue
        end = len(candles) - leave_forward
        for i in range(min_history, end, max(1, step)):
            as_of = str(pd.Timestamp(candles.iloc[i]["ts"]).isoformat())
            pred = predict_symbol(
                store,
                symbol,
                timeframe,
                as_of_ts=as_of,
                candles=candles,
            )
            if not pred:
                continue
            store.insert_prediction(pred)
            inserted += 1
    return inserted


def run_eval(
    store: Store,
    *,
    limit: int = 500,
    rescore_all: bool = False,
    backfill: bool = False,
    timeframe: str = "1h",
) -> dict[str, Any]:
    backfilled = 0
    if backfill:
        backfilled = backfill_historical_predictions(store, timeframe=timeframe)

    if rescore_all:
        preds = store.get_predictions(limit=limit)
        # get_predictions is newest-first; reverse for stable scoring
        preds = list(reversed(preds))
    else:
        preds = store.get_predictions_for_eval(limit=limit)

    scored = 0
    skipped = 0
    for pred in preds:
        row = score_prediction(store, pred)
        if row is None:
            skipped += 1
            continue
        store.upsert_prediction_outcome(row)
        scored += 1

    outcomes = store.get_prediction_outcomes(limit=max(limit, 500))
    payload = {
        "kind": "prediction_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backfilled_predictions": backfilled,
        "scored_this_run": scored,
        "skipped_insufficient_future": skipped,
        "summary": summary_stats(outcomes),
        "calibration_1h": calibration_table(outcomes, horizon="1h"),
        "calibration_4h": calibration_table(outcomes, horizon="4h"),
        "outcomes_sample": outcomes[:40],
    }
    return payload


def write_eval_result(payload: dict[str, Any], directory: Path = EVAL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "latest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score TrendLab predictions vs realized moves")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--rescore-all",
        action="store_true",
        help="Re-score recent predictions even if already in outcomes table",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Generate historical rule-based forecasts from candle history, then score",
    )
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    args = parser.parse_args()

    store = Store()
    payload = run_eval(
        store,
        limit=args.limit,
        rescore_all=args.rescore_all,
        backfill=args.backfill,
        timeframe=args.timeframe,
    )
    path = write_eval_result(payload)
    print(
        json.dumps(
            {
                "path": str(path),
                "backfilled_predictions": payload["backfilled_predictions"],
                "scored_this_run": payload["scored_this_run"],
                "skipped": payload["skipped_insufficient_future"],
                "summary": payload["summary"],
                "calibration_1h": payload["calibration_1h"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
