"""Score immutable ML predictions once future candles are available.

Outcomes are write-once (never overwrite an existing outcome row).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.config import DEFAULT_TIMEFRAME, ROOT, SYMBOLS
from trendlabs.db import Store, utc_now
from trendlabs.train_ml import HORIZON_BARS

EVAL_DIR = Path(ROOT) / "data" / "ml_prediction_eval"


def resolve_actual_close(
    candles: pd.DataFrame,
    *,
    predicted_at: str,
    horizon_bars: int,
) -> tuple[float | None, str | None]:
    if candles.empty:
        return None, None
    df = candles.sort_values("ts").reset_index(drop=True)
    ts = pd.to_datetime(df["ts"], utc=True)
    origin = pd.Timestamp(predicted_at)
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    else:
        origin = origin.tz_convert("UTC")
    prior = df.index[ts <= origin]
    if len(prior) == 0:
        return None, None
    sig_i = int(prior[-1])
    target_i = sig_i + int(horizon_bars)
    if target_i >= len(df):
        return None, None
    row = df.iloc[target_i]
    return float(row["close"]), str(pd.Timestamp(row["ts"]).isoformat())


def score_ml_prediction(store: Store, pred: dict[str, Any]) -> dict[str, Any] | None:
    candles = store.load_candles(pred["symbol"], pred["timeframe"])
    actual_close, actual_ts = resolve_actual_close(
        candles,
        predicted_at=pred["predicted_at"],
        horizon_bars=int(pred["horizon_bars"]),
    )
    if actual_close is None:
        return None
    ref = float(pred["reference_close"])
    if ref <= 0:
        return None
    realized = (actual_close / ref) - 1.0
    predicted_up = float(pred["prob_up"]) >= 0.5
    actual_up = actual_close > ref
    return {
        "prediction_id": pred["id"],
        "actual_close": actual_close,
        "actual_ts": actual_ts,
        "realized_return": realized,
        "direction_correct": predicted_up == actual_up,
        "evaluated_at": utc_now(),
        "symbol": pred["symbol"],
        "prob_up": float(pred["prob_up"]),
        "predicted_up": predicted_up,
        "actual_up": actual_up,
    }


def summary_from_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    if not outcomes:
        return {"n": 0, "accuracy": None, "mean_prob_up": None, "mean_realized_return": None}
    correct = [int(o["direction_correct"]) for o in outcomes if o.get("direction_correct") is not None]
    return {
        "n": len(outcomes),
        "accuracy": (sum(correct) / len(correct)) if correct else None,
        "mean_prob_up": sum(float(o["prob_up"]) for o in outcomes) / len(outcomes),
        "mean_realized_return": sum(float(o["realized_return"]) for o in outcomes if o.get("realized_return") is not None)
        / max(1, sum(1 for o in outcomes if o.get("realized_return") is not None)),
    }


def by_symbol_from_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-symbol rollup so empty re-runs still keep readable metrics."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        buckets.setdefault(symbol, []).append(row)
    return {
        symbol: summary_from_outcomes(rows)
        for symbol, rows in sorted(buckets.items())
    }


def backfill_historical_ml_predictions(
    store: Store,
    *,
    symbols: list[str] | None = None,
    timeframe: str = "1d",
    step: int = 20,
    min_history: int = 250,
    leave_forward: int = HORIZON_BARS + 1,
    limit_per_symbol: int = 40,
) -> int:
    """Scoreable historical predictions from the latest candidate model per symbol."""
    from trendlabs.features import FEATURE_COLUMNS, FEATURE_VERSION, compute_feature_frame
    from trendlabs.ml_predict import load_model_artifact

    inserted = 0
    for symbol in [s.upper() for s in (symbols or SYMBOLS)]:
        meta = store.latest_model_version(symbol, timeframe, status="candidate")
        if not meta:
            continue
        candles = store.load_candles(symbol, timeframe)
        if len(candles) < min_history + leave_forward:
            continue
        feats = compute_feature_frame(candles).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        if feats.empty:
            continue
        # Align closes by timestamp
        c = candles.sort_values("ts").reset_index(drop=True)
        c_ts = pd.to_datetime(c["ts"], utc=True)
        model = load_model_artifact(meta["artifact_path"])
        end = len(feats) - leave_forward
        count = 0
        for i in range(min_history, end, max(1, step)):
            if count >= limit_per_symbol:
                break
            row = feats.iloc[i]
            ts = pd.Timestamp(row["ts"])
            match = c.loc[c_ts == ts]
            if match.empty:
                continue
            ref = float(match.iloc[0]["close"])
            x = row[FEATURE_COLUMNS].astype(float).to_numpy().reshape(1, -1)
            prob_up = float(model.predict_proba(x)[0, 1])
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_version": meta["model_version"],
                "feature_version": meta.get("feature_version") or FEATURE_VERSION,
                "predicted_at": str(ts.isoformat()),
                "horizon_bars": HORIZON_BARS,
                "reference_close": ref,
                "prob_up": prob_up,
                "prob_not_up": 1.0 - prob_up,
                "features": {col: float(row[col]) for col in FEATURE_COLUMNS},
                "created_at": utc_now(),
            }
            store.insert_ml_prediction(payload)
            inserted += 1
            count += 1
    return inserted


def run_ml_eval(
    store: Store,
    *,
    limit: int = 500,
    timeframe: str = "1d",
    backfill: bool = False,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    backfilled = 0
    if backfill:
        backfilled = backfill_historical_ml_predictions(
            store, symbols=symbols, timeframe=timeframe
        )

    preds = store.get_ml_predictions(
        timeframe=timeframe,
        limit=limit,
        unscored_only=True,
    )
    if symbols:
        allow = {s.upper() for s in symbols}
        preds = [p for p in preds if p["symbol"] in allow]

    scored = 0
    skipped = 0
    scored_rows: list[dict[str, Any]] = []
    for pred in preds:
        row = score_ml_prediction(store, pred)
        if row is None:
            skipped += 1
            continue
        store.insert_ml_prediction_outcome(row)
        scored += 1
        scored_rows.append(row)

    outcomes = store.get_ml_prediction_outcomes(limit=max(limit, 2000))
    if timeframe:
        outcomes = [o for o in outcomes if o.get("timeframe") == timeframe]
    if symbols:
        allow = {s.upper() for s in symbols}
        outcomes = [o for o in outcomes if str(o.get("symbol") or "").upper() in allow]
    payload = {
        "kind": "ml_prediction_evaluation",
        "created_at": utc_now(),
        "timeframe": timeframe,
        "horizon_bars": HORIZON_BARS,
        "backfilled_predictions": backfilled,
        "scored_this_run": scored,
        "skipped_insufficient_future": skipped,
        "summary": summary_from_outcomes(outcomes),
        "by_symbol": by_symbol_from_outcomes(outcomes),
        "scored_sample": scored_rows[:20],
    }
    return payload


def write_eval_result(payload: dict[str, Any], directory: Path = EVAL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "latest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ML prediction outcomes")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Insert historical ML predictions from the latest candidate model, then score",
    )
    args = parser.parse_args()

    store = Store()
    payload = run_ml_eval(
        store,
        limit=args.limit,
        timeframe=args.timeframe,
        backfill=args.backfill,
        symbols=args.symbols,
    )
    path = write_eval_result(payload)
    print(json.dumps({"path": str(path), **{k: payload[k] for k in (
        "backfilled_predictions",
        "scored_this_run",
        "skipped_insufficient_future",
        "summary",
        "by_symbol",
    )}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
