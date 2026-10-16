"""Pre-declared h=5 regression candidate — vol-normalized target, research-only.

Registered in docs/ML_PLAN.md BEFORE evaluation. Trains XGBRegressor on
z = fwd_ret(5)/(σ20·√5) per symbol from the fv2 context panel, evaluates on the
embargoed holdout, applies the registered gate:
  (a) holdout accuracy(pred_class vs majority) — pred class via |pred_z| vs 0.5/√h
      (consistent with the vol-scaled band 0.5σ_daily on the 5-day return)
  (b) holdout Spearman IC(pred_z, realized_z) ≥ 0.05
Bundles are suffixed `_reg` (research-only; the h=1 serving glob never matches them).
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
from trendlabs.features import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    build_panel_features,
    dataset_reg_from_panel,
    make_labels_vol_scaled,
)
from trendlabs.ml_train import MODEL_DIR, REPORT_DIR, _time_split

REG_DIR = Path(ROOT) / "data" / "ml"
MIN_IC = 0.05  # registered accept bar (b)


def _brier_regression(pred_z: np.ndarray, realized_z: np.ndarray) -> float:
    """Mean squared error in z units (the regression analogue of Brier)."""
    return float(np.mean((pred_z - realized_z) ** 2))


def train_reg_symbol(
    store: Store,
    symbol: str,
    timeframe: str,
    *,
    horizon: int = 5,
    vol_window: int = 20,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    from xgboost import XGBRegressor

    panel = build_panel_features(store, timeframe)
    X, z, y_cls, meta = dataset_reg_from_panel(panel, symbol, horizon=horizon, vol_window=vol_window)
    if len(X) < 200:
        raise RuntimeError(f"{symbol}: reg feature rows {len(X)} too small")

    train_idx, test_idx = _time_split(len(X), test_frac=test_frac, embargo=horizon)
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    z_train, z_test = z.iloc[train_idx].reset_index(drop=True), z.iloc[test_idx].reset_index(drop=True)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=2,
        tree_method="hist",
    )
    model.fit(X_train, z_train)
    pred_z = model.predict(X_test)

    # Class mapping consistent with the vol-scaled band (0.5σ_daily) on the h-day return
    z_cut = 0.5 / np.sqrt(float(horizon))
    pred_cls = np.where(pred_z > z_cut, "up", np.where(pred_z < -z_cut, "down", "sideways"))

    # Realized classes: vol-scaled labels on the SAME test rows (aligned by the builder)
    realized = y_cls.astype(str).to_numpy()[test_idx]

    classes, counts = np.unique(realized, return_counts=True)
    majority_acc = float(counts.max() / len(realized))
    accuracy = float(accuracy_score(realized, pred_cls))
    macro_f1 = float(f1_score(realized, pred_cls, average="macro", zero_division=0))
    ic = float(pd.Series(pred_z).corr(pd.Series(z_test.to_numpy()), method="spearman"))

    stem = f"{symbol.upper()}_{timeframe}_h{horizon}_reg"
    bundle = {
        "model": model,
        "encoder": None,
        "kind": "regression",
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_version": FEATURE_VERSION,
        "label_scheme": "reg_z",
        "embargo": int(horizon),
        "horizon": horizon,
        "band": 0.5,
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "z_cut": float(z_cut),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{stem}.joblib"
    joblib.dump(bundle, model_path)

    beats_majority = accuracy > majority_acc
    gate = "approved" if (beats_majority and ic >= MIN_IC) else "watch"
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "horizon": horizon,
        "model": stem,
        "model_path": str(model_path),
        "feature_version": FEATURE_VERSION,
        "embargo": int(horizon),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "majority": majority_acc,
        "spearman_ic": ic,
        "mse_z": _brier_regression(pred_z, z_test.to_numpy()),
        "z_cut": float(z_cut),
        "beats_majority": bool(beats_majority),
        "ic_clears": bool(ic >= MIN_IC),
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="h=5 regression candidate (pre-declared)")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    store = Store()
    symbols = [s for s in SYMBOLS if s in store.list_symbols_with_data(args.timeframe)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for sym in symbols:
        try:
            r = train_reg_symbol(store, sym, args.timeframe, horizon=args.horizon)
            rows.append(r)
            print(
                f"{sym}: acc={r['accuracy']:.3f} majority={r['majority']:.3f} "
                f"IC={r['spearman_ic']:+.3f} → {r['gate']}"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"symbol": sym, "error": str(exc)})
            print(f"{sym}: ERROR {exc}")

    approved = sum(r["gate"] == "approved" for r in rows)
    payload = {
        "kind": "reg_candidate_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registered": "docs/ML_PLAN.md — h=5 regression candidate (before evaluation)",
        "accept_rule": f"accuracy > majority AND spearman_ic >= {MIN_IC}",
        "timeframe": args.timeframe,
        "horizon": args.horizon,
        "feature_version": FEATURE_VERSION,
        "n_symbols": len(rows),
        "n_approved": approved,
        "rows": rows,
        "errors": errors,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REG_DIR / f"reg_scorecard_{args.timeframe}_h{args.horizon}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Approval registry: reg models get their own stems; h=1 entries untouched.
    registry_path = REG_DIR / "approved_models.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    for r in rows:
        registry[r["model"]] = {
            "approved": r["gate"] == "approved",
            "gate": r["gate"],
            "accuracy": r["accuracy"],
            "majority": r["majority"],
            "spearman_ic": r["spearman_ic"],
            "feature_version": r["feature_version"],
            "evaluated_at": payload["created_at"],
        }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"reg gate: {approved}/{len(rows)} approved → {REG_DIR / f'reg_scorecard_{args.timeframe}_h{args.horizon}.json'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
