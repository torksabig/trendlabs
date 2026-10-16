"""C3 — beta-controlled evaluation of the rank policy (registered in docs/ML_PLAN.md).

Per symbol, over the same purged walk-forward folds as C2:
  policy hit-rate vs
    (i)  always-long hit-rate on the SAME called rows (does direction selection add
         value over simply being long those days?)
    (ii) symmetrized-book hit-rate (equal long/short quotas among eligible calls)
Gate (registered): policy beats BOTH point comparisons AND total policy P&L > 0.
Fold-level bootstrap CIs on the deltas are reported as diagnostics. Research-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.config import SYMBOLS
from trendlabs.db import Store
from trendlabs.features import FEATURE_VERSION, build_panel_features
from trendlabs.ml_reg import REG_DIR
from trendlabs.ml_wf import policy_fold_rows


def _bootstrap_ci_delta(deltas: list[float], *, draws: int = 2000, seed: int = 0) -> list[float] | None:
    if len(deltas) < 5:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate_beta_control(
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
    panel = build_panel_features(store, timeframe)
    if panel.empty:
        return {"error": "no_panel"}
    syms = symbols or [s for s in SYMBOLS if s in store.list_symbols_with_data(timeframe)]
    rows: list[dict[str, Any]] = []
    for sym in syms:
        fold_rows, n_folds = policy_fold_rows(
            panel,
            sym,
            horizon=horizon,
            min_train_sessions=min_train_sessions,
            test_sessions=test_sessions,
            step_sessions=step_sessions,
            call_quantile=call_quantile,
        )
        if fold_rows and "error" in fold_rows[0]:
            rows.append({"symbol": sym.upper(), **fold_rows[0]})
            print(f"{sym}: ERROR {fold_rows[0]['error']}")
            continue
        scored = [f for f in fold_rows if f["hit_rate"] is not None]
        if not scored:
            rows.append({"symbol": sym.upper(), "error": "no_scoring_folds", "n_folds": n_folds})
            print(f"{sym}: ERROR no_scoring_folds")
            continue

        calls = sum(f["n_calls"] for f in scored)
        policy_correct = sum(f["policy_correct"] for f in scored)
        long_called_correct = sum(f["long_called_correct"] for f in scored)
        n_test = sum(f["n_test"] for f in scored)
        long_test_correct = sum(f["long_test_correct"] for f in scored)
        sym_calls = sum(f["sym_calls"] for f in scored)
        sym_correct = sum(f["sym_correct"] for f in scored if f["sym_correct"] is not None)

        policy_hit = policy_correct / calls
        long_called_hit = long_called_correct / calls
        long_test_hit = long_test_correct / n_test
        sym_hit = (sym_correct / sym_calls) if sym_calls else None

        policy_pnl = sum(f["pnl_sum_pct"] for f in scored)
        sym_pnl = sum(f["sym_pnl_sum_pct"] for f in scored)

        deltas = [f["hit_rate"] - f["long_called_correct"] / f["n_calls"] for f in scored]
        delta_ci = _bootstrap_ci_delta(deltas)

        beats_long = policy_hit > long_called_hit
        beats_sym = sym_hit is not None and policy_hit > sym_hit
        gate = "approved" if (beats_long and beats_sym and policy_pnl > 0) else "watch"
        rows.append(
            {
                "symbol": sym.upper(),
                "n_folds": n_folds,
                "n_scoring_folds": len(scored),
                "n_calls": calls,
                "policy_hit": round(policy_hit, 4),
                "long_called_hit": round(long_called_hit, 4),
                "long_test_hit": round(long_test_hit, 4),
                "sym_hit": round(sym_hit, 4) if sym_hit is not None else None,
                "delta_policy_minus_long": round(policy_hit - long_called_hit, 4),
                "delta_ci95": [round(x, 4) for x in delta_ci] if delta_ci else None,
                "policy_pnl_pct": round(policy_pnl, 2),
                "sym_pnl_pct": round(sym_pnl, 2),
                "beats_long": bool(beats_long),
                "beats_sym": bool(beats_sym),
                "pnl_positive": bool(policy_pnl > 0),
                "gate": gate,
            }
        )
        print(
            f"{sym}: policy={policy_hit:.3f} long(called)={long_called_hit:.3f} "
            f"long(test)={long_test_hit:.3f} sym={sym_hit if sym_hit is None else round(sym_hit, 3)} "
            f"ΔCI={delta_ci and [round(x, 3) for x in delta_ci]} pnl={policy_pnl:.0f}% → {gate}"
        )
    approved = sum(r.get("gate") == "approved" for r in rows)
    return {
        "kind": "beta_control_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registered": "docs/ML_PLAN.md — C3 beta-controlled (before evaluation)",
        "gate_rule": "policy_hit > long_called_hit AND policy_hit > sym_hit AND policy_pnl > 0",
        "feature_version": FEATURE_VERSION,
        "timeframe": timeframe,
        "horizon": horizon,
        "n_symbols": len(rows),
        "n_approved": approved,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 beta-controlled policy evaluation (registered)")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-train-sessions", type=int, default=750)
    parser.add_argument("--test-sessions", type=int, default=126)
    parser.add_argument("--step-sessions", type=int, default=126)
    parser.add_argument("--call-quantile", type=float, default=0.80)
    args = parser.parse_args()

    payload = evaluate_beta_control(
        Store(),
        timeframe=args.timeframe,
        horizon=args.horizon,
        min_train_sessions=args.min_train_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.step_sessions,
        call_quantile=args.call_quantile,
    )
    out = REG_DIR / f"beta_control_{args.timeframe}_h{args.horizon}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} — approved {payload.get('n_approved')}/{payload.get('n_symbols')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
