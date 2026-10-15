"""Phase A scorecard — daily ML models vs baselines on their held-out split.

Reproduces each model's exact holdout (same data, features, split, filter logic as
ml_train) and compares against simple baselines:

  majority    — most frequent actual class in the test window
  persistence — previous bar's actual label (trend-following proxy)
  always_up   — constant 'up'

Gate (Phase A): a symbol's model is APPROVED for the daily pipeline only if its
held-out accuracy beats the majority baseline; otherwise it stays 'watch'.
Output: data/ml/scorecard_{timeframe}.json + scorecard_{timeframe}.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from trendlabs.config import ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.features import dataset_from_candles, default_band
from trendlabs.ml_train import MODEL_DIR, _time_split

REPORT_DIR = Path(ROOT) / "data" / "ml"


def _brier_multiclass(proba: np.ndarray, y_enc: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y_enc]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def score_symbol(bundle: dict[str, Any], candles: pd.DataFrame, *, test_frac: float) -> dict[str, Any]:
    horizon = bundle["horizon"]
    band = bundle["band"]
    X, y, meta = dataset_from_candles(candles, horizon=horizon, band=band)
    train_idx, test_idx = _time_split(len(X), test_frac=test_frac)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test_raw = y.iloc[test_idx].reset_index(drop=True)

    encoder: LabelEncoder = bundle["encoder"]
    known = set(encoder.classes_)
    mask = y_test_raw.astype(str).isin(known).to_numpy()
    X_test, y_test_raw = X_test.loc[mask].reset_index(drop=True), y_test_raw.loc[mask].reset_index(drop=True)
    y_enc = encoder.transform(y_test_raw.astype(str))

    model = bundle["model"]
    proba = model.predict_proba(X_test)
    pred_enc = np.argmax(proba, axis=1)
    pred_labels = encoder.inverse_transform(pred_enc)

    accuracy = float(accuracy_score(y_enc, pred_enc))
    macro_f1 = float(f1_score(y_enc, pred_enc, average="macro", zero_division=0))
    brier = _brier_multiclass(proba, y_enc)

    # directional accuracy among up/down rows where the model also predicted up/down
    labels = np.array(encoder.classes_)
    dir_mask = np.isin(y_enc, [i for i, c in enumerate(labels) if c in ("up", "down")]) & np.isin(
        pred_enc, [i for i, c in enumerate(labels) if c in ("up", "down")]
    )
    directional = float((pred_enc[dir_mask] == y_enc[dir_mask]).mean()) if dir_mask.any() else None

    # Baselines on the identical test window
    classes, counts = np.unique(y_enc, return_counts=True)
    majority_class = classes[counts.argmax()]
    majority_acc = float(counts.max() / len(y_enc))
    # persistence: previous bar's actual label (labels are per-bar and consecutive here)
    prev = np.concatenate([[pred_enc[0]], y_enc[:-1]])
    persistence_acc = float((prev == y_enc).mean())
    up_idx = int(np.where(labels == "up")[0][0]) if "up" in labels else 0
    always_up_acc = float((y_enc == up_idx).mean())

    beats_majority = accuracy > majority_acc
    return {
        "symbol": bundle["symbol"],
        "timeframe": bundle["timeframe"],
        "horizon": horizon,
        "band": band,
        "test_rows": int(len(y_enc)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "brier": brier,
        "directional_accuracy": directional,
        "baselines": {
            "majority": {"class": str(labels[majority_class]), "accuracy": majority_acc},
            "persistence": {"accuracy": persistence_acc},
            "always_up": {"accuracy": always_up_acc},
        },
        "beats_majority": bool(beats_majority),
        "gate": "approved" if beats_majority else "watch",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase A scorecard: ML models vs baselines")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    store = Store()
    models = sorted(MODEL_DIR.glob(f"*_{args.timeframe}_h{args.horizon}.joblib"))
    if not models:
        raise SystemExit(f"No models for {args.timeframe} h{args.horizon}. Train first.")

    rows: list[dict[str, Any]] = []
    for path in models:
        bundle = joblib.load(path)
        candles = store.load_candles(bundle["symbol"], bundle["timeframe"])
        rows.append(score_symbol(bundle, candles, test_frac=args.test_frac))
        r = rows[-1]
        print(
            f"{r['symbol']}: acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} brier={r['brier']:.3f} "
            f"| majority={r['baselines']['majority']['accuracy']:.3f} persistence={r['baselines']['persistence']['accuracy']:.3f} "
            f"→ {r['gate']}"
        )

    payload = {
        "kind": "ml_scorecard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": args.timeframe,
        "horizon": args.horizon,
        "gate_rule": "held-out accuracy > majority baseline accuracy",
        "n_symbols": len(rows),
        "n_approved": int(sum(r["gate"] == "approved" for r in rows)),
        "rows": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"scorecard_{args.timeframe}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# ML Scorecard — {args.timeframe} h{args.horizon}",
        "",
        f"Generated {payload['created_at']} · gate: held-out accuracy > majority baseline · "
        f"{payload['n_approved']}/{payload['n_symbols']} approved",
        "",
        "| Symbol | Acc | MacroF1 | Brier | DirAcc | Majority | Persistence | AlwaysUp | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        b = r["baselines"]
        lines.append(
            f"| {r['symbol']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['brier']:.3f} | "
            f"{(r['directional_accuracy'] if r['directional_accuracy'] is not None else float('nan')):.3f} | "
            f"{b['majority']['accuracy']:.3f} | {b['persistence']['accuracy']:.3f} | {b['always_up']['accuracy']:.3f} | {r['gate']} |"
        )
    (REPORT_DIR / f"scorecard_{args.timeframe}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"scorecard: {payload['n_approved']}/{payload['n_symbols']} approved → data/ml/scorecard_{args.timeframe}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
