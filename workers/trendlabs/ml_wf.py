"""C2 — rank-policy walk-forward evaluation (registered in docs/ML_PLAN.md).

Per fold (expanding train / fixed test / purge ≥ horizon, fresh OOS):
  1. fit XGBRegressor on z = fwd(h)/(σ20·√h) over the fv2 context panel;
  2. τ = |pred_z| quantile(call_quantile) on TRAIN predictions only;
  3. on TEST rows call sign(pred_z) where |pred_z| ≥ τ; pnl = sign × fwd − 20bps.

Registered gate (per symbol): mean fold hit-rate > 0.5 with bootstrap CI excluding 0.5,
AND total net pnl > 0. Research-only; nothing here auto-approves serving.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.config import ROOT, SYMBOLS
from trendlabs.db import Store
from trendlabs.features import build_panel_features, dataset_reg_from_panel
from trendlabs.ml_reg import REG_DIR
from trendlabs.prediction.walk_forward import generate_purged_folds, mask_dates, unique_sorted_dates

COST_PCT = 0.20  # round trip, percent (2 × (FEE 5 + SLIP 5) bps)


def policy_fold_rows(
    panel: pd.DataFrame,
    symbol: str,
    *,
    horizon: int,
    min_train_sessions: int,
    test_sessions: int,
    step_sessions: int,
    call_quantile: float,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], int]:
    """Per-fold policy stats (C2) + beta-control extras (C3): always-long hit on the
    called rows and on all test rows, and a symmetrized (equal long/short quota) book.

    Every metric is PIT: model + τ from train only; realized outcomes score only.
    Returns (fold_rows, n_folds_total)."""
    from xgboost import XGBRegressor

    X, z, _y_cls, meta = dataset_reg_from_panel(panel, symbol, horizon=horizon)
    if len(X) < min_train_sessions:
        return [{"error": "insufficient_rows", "n_rows": int(len(X))}], 0

    work = X.copy()
    work["ts"] = pd.to_datetime(meta["ts"], utc=True).to_numpy()
    work["z"] = z.to_numpy()
    work["sigma"] = meta["sigma"].to_numpy()
    dates = unique_sorted_dates(work)

    folds = generate_purged_folds(
        dates,
        min_train_days=min_train_sessions,
        test_days=test_sessions,
        step_days=step_sessions,
        purge_gap=horizon,
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = mask_dates(work, fold.train_dates)
        test = mask_dates(work, fold.test_dates)
        if len(train) < 300 or test.empty:
            continue
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=2,
            tree_method="hist",
        )
        model.fit(train[list(X.columns)], train["z"])
        pred_tr = model.predict(train[list(X.columns)])
        tau = float(np.quantile(np.abs(pred_tr), call_quantile))
        pred_te = model.predict(test[list(X.columns)])
        mask_called = np.abs(pred_te) >= tau
        called = test.loc[mask_called].copy()
        called["pred_z"] = pred_te[mask_called]

        row: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "n_test": int(len(test)),
            "n_calls": int(len(called)),
        }
        if called.empty:
            row.update({"n_bull": 0, "n_bear": 0, "hit_rate": None, "pnl_sum_pct": 0.0,
                        "long_called_correct": None, "long_test_correct": None,
                        "sym_calls": 0, "sym_correct": None, "sym_pnl_sum_pct": 0.0})
            fold_rows.append(row)
            continue

        direction = np.sign(called["pred_z"])
        realized_sign = np.sign(called["z"])
        correct = int((direction == realized_sign).sum())
        per_trade = direction * called["z"] * called["sigma"] * np.sqrt(float(horizon)) * 100.0 - COST_PCT

        # Always-long on the SAME called rows and on all test rows
        long_called_correct = int((called["z"] > 0).sum())
        long_test_correct = int((test["z"] > 0).sum())

        # Symmetrized book: equal long/short quotas among eligible rows (|pred_z| ≥ τ),
        # strongest |pred_z| on each side.
        sym_calls, sym_correct, sym_pnl = _symmetrize(called, horizon=horizon)

        row.update(
            {
                "n_bull": int((direction > 0).sum()),
                "n_bear": int((direction < 0).sum()),
                "hit_rate": float(correct / len(called)),
                "policy_correct": correct,
                "pnl_sum_pct": float(per_trade.sum()),
                "mean_abs_pred_z": float(np.abs(called["pred_z"]).mean()),
                "long_called_correct": long_called_correct,
                "long_test_correct": long_test_correct,
                "sym_calls": sym_calls,
                "sym_correct": sym_correct,
                "sym_pnl_sum_pct": sym_pnl,
            }
        )
        fold_rows.append(row)
    return fold_rows, len(folds)


def _symmetrize(called: pd.DataFrame, *, horizon: int) -> tuple[int, int | None, float]:
    """Equal long/short quotas among eligible calls; strongest |pred_z| per side.

    Returns (n_sym_calls, sym_correct, sym_pnl_sum_pct). n=0 when one side is empty.
    """
    longs = called.loc[called["pred_z"] > 0].sort_values("pred_z", ascending=False)
    shorts = called.loc[called["pred_z"] < 0].sort_values("pred_z", ascending=True)
    quota = min(len(longs), len(shorts))
    if quota == 0:
        return 0, None, 0.0
    book = pd.concat([longs.iloc[:quota], shorts.iloc[:quota]])
    direction = np.sign(book["pred_z"])
    correct = int((direction == np.sign(book["z"])).sum())
    pnl = float(
        (direction * book["z"] * book["sigma"] * np.sqrt(float(horizon)) * 100.0 - COST_PCT).sum()
    )
    return 2 * quota, correct, pnl


def _fold_policy_stats(
    store: Store,
    symbol: str,
    timeframe: str,
    panel: pd.DataFrame,
    *,
    horizon: int,
    min_train_sessions: int,
    test_sessions: int,
    step_sessions: int,
    call_quantile: float,
    seed: int = 42,
) -> dict[str, Any]:
    fold_rows, n_folds_total = policy_fold_rows(
        panel,
        symbol,
        horizon=horizon,
        min_train_sessions=min_train_sessions,
        test_sessions=test_sessions,
        step_sessions=step_sessions,
        call_quantile=call_quantile,
        seed=seed,
    )
    if fold_rows and "error" in fold_rows[0]:
        return fold_rows[0]

    hits = [f["hit_rate"] for f in fold_rows if f["hit_rate"] is not None]
    n_calls = sum(f["n_calls"] for f in fold_rows)
    n_bull = sum(f["n_bull"] for f in fold_rows)
    n_bear = sum(f["n_bear"] for f in fold_rows)
    total_pnl = float(sum(f["pnl_sum_pct"] for f in fold_rows))
    if not hits:
        return {"error": "no_scoring_folds", "n_folds": n_folds_total, "n_calls": n_calls}

    from trendlabs.intraday.baselines import session_bootstrap_ci

    ci = session_bootstrap_ci(pd.Series(hits), draws=2000)["ci95"]
    mean_hit = float(np.mean(hits))
    gate_hit = mean_hit > 0.5 and ci is not None and ci[0] > 0.5
    gate_pnl = total_pnl > 0
    return {
        "n_folds": n_folds_total,
        "n_scoring_folds": len(hits),
        "n_calls": n_calls,
        "n_bull": n_bull,
        "n_bear": n_bear,
        "bull_share": round(n_bull / n_calls, 4) if n_calls else None,
        "fold_hit_rates": [round(h, 4) for h in hits],
        "mean_hit_rate": round(mean_hit, 4),
        "ci95": [round(x, 4) for x in ci] if ci else None,
        "fold_pnl_sum_pct": [round(f["pnl_sum_pct"], 4) for f in fold_rows],
        "total_pnl_pct": round(total_pnl, 4),
        "gate_hit_rate": bool(gate_hit),
        "gate_pnl": bool(gate_pnl),
        "gate": "approved" if (gate_hit and gate_pnl) else "watch",
        "params": {
            "min_train_sessions": min_train_sessions,
            "test_sessions": test_sessions,
            "step_sessions": step_sessions,
            "call_quantile": call_quantile,
            "cost_pct": COST_PCT,
        },
    }


def run_wf_policy(
    store: Store,
    *,
    timeframe: str = "1d",
    horizon: int = 5,
    min_train_sessions: int = 750,
    test_sessions: int = 126,
    step_sessions: int = 126,
    call_quantile: float = 0.80,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    from trendlabs.features import FEATURE_VERSION

    panel = build_panel_features(store, timeframe)
    if panel.empty:
        return {"error": "no_panel"}
    syms = symbols or [s for s in SYMBOLS if s in store.list_symbols_with_data(timeframe)]
    rows: list[dict[str, Any]] = []
    for sym in syms:
        stats = _fold_policy_stats(
            store,
            sym,
            timeframe,
            panel,
            horizon=horizon,
            min_train_sessions=min_train_sessions,
            test_sessions=test_sessions,
            step_sessions=step_sessions,
            call_quantile=call_quantile,
        )
        rows.append({"symbol": sym.upper(), **stats})
        if "error" in stats:
            print(f"{sym}: ERROR {stats['error']}")
        else:
            print(
                f"{sym}: folds={stats['n_scoring_folds']} calls={stats['n_calls']} "
                f"hit={stats['mean_hit_rate']} CI={stats['ci95']} pnl={stats['total_pnl_pct']}% → {stats['gate']}"
            )
    approved = sum(r.get("gate") == "approved" for r in rows)
    return {
        "kind": "wf_policy_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registered": "docs/ML_PLAN.md — C2 rank-policy walk-forward (before evaluation)",
        "feature_version": FEATURE_VERSION,
        "timeframe": timeframe,
        "horizon": horizon,
        "n_symbols": len(rows),
        "n_approved": approved,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="C2 rank-policy walk-forward (registered)")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-train-sessions", type=int, default=750)
    parser.add_argument("--test-sessions", type=int, default=126)
    parser.add_argument("--step-sessions", type=int, default=126)
    parser.add_argument("--call-quantile", type=float, default=0.80)
    args = parser.parse_args()

    payload = run_wf_policy(
        Store(),
        timeframe=args.timeframe,
        horizon=args.horizon,
        min_train_sessions=args.min_train_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.step_sessions,
        call_quantile=args.call_quantile,
    )
    out = REG_DIR / f"wf_policy_{args.timeframe}_h{args.horizon}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} — approved {payload.get('n_approved')}/{payload.get('n_symbols')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
