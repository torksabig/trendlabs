"""Phase D paper-trade evaluation — realized P&L of policy trades, separate from accuracy.

Trades are opened by `ml-predict --persist` (approved symbols, calibrated policy,
non-watch signals) with the execution rule: enter at the OPEN of the first bar after
the as-of bar, exit at the CLOSE of the entry bar shifted by (horizon_bars - 1).
This evaluator closes matured open trades against the candle store and reports the
paper P&L summary. It never touches the accuracy scorecard.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trendlabs.config import ROOT
from trendlabs.db import Store

ML_DIR = Path(ROOT) / "data" / "ml"


def evaluate_paper_trades(store: Store) -> dict[str, Any]:
    closed_now = 0
    for trade in store.get_open_paper_trades():
        candles = store.load_candles(trade["symbol"], trade["timeframe"])
        if candles.empty:
            continue
        as_of = pd.Timestamp(trade["as_of_ts"])
        if candles["ts"].dt.tz is None:
            candles["ts"] = pd.to_datetime(candles["ts"], utc=True)
        fut = candles.loc[candles["ts"] > as_of]
        if fut.empty:
            continue  # entry bar not yet available — stays open
        entry = fut.iloc[0]
        entry_idx = candles.index[candles["ts"] == entry["ts"]][0]
        exit_idx = int(entry_idx) + int(trade["horizon_bars"]) - 1
        if exit_idx >= len(candles):
            continue  # horizon incomplete — stays open
        exit_bar = candles.iloc[exit_idx]
        entry_open = float(entry["open"])
        exit_close = float(exit_bar["close"])
        sign = 1.0 if trade["policy"] == "bullish" else -1.0
        pnl_pct = sign * (exit_close / entry_open - 1.0) * 100.0
        store.close_paper_trade(
            trade["id"],
            entry_ts=pd.Timestamp(entry["ts"]).isoformat(),
            entry_open=entry_open,
            exit_ts=pd.Timestamp(exit_bar["ts"]).isoformat(),
            exit_close=exit_close,
            pnl_pct=pnl_pct,
        )
        closed_now += 1

    trades = store.get_paper_trades(limit=5000)
    closed = [t for t in trades if t["status"] == "closed"]
    open_trades = [t for t in trades if t["status"] == "open"]

    def _agg(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        if not rows:
            return {"policy": label, "n": 0}
        pnls = [float(t["pnl_pct"]) for t in rows]
        wins = [p for p in pnls if p > 0]
        return {
            "policy": label,
            "n": len(rows),
            "win_rate": round(len(wins) / len(rows), 4),
            "mean_pnl_pct": round(sum(pnls) / len(rows), 4),
            "total_pnl_pct": round(sum(pnls), 4),
            "best": round(max(pnls), 4),
            "worst": round(min(pnls), 4),
        }

    by_policy: dict[str, Any] = {}
    for pol in ("bullish", "bearish"):
        rows = [t for t in closed if t["policy"] == pol]
        if rows:
            by_policy[pol] = _agg(rows, pol)

    payload = {
        "kind": "paper_pnl",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Paper P&L only — separate from model accuracy. Execution: enter open(t+1), exit close(t+h).",
        "closed_this_run": closed_now,
        "summary": _agg(closed, "all"),
        "open_count": len(open_trades),
        "by_policy": by_policy,
        "recent": [
            {k: t[k] for k in ("symbol", "timeframe", "policy", "as_of_ts", "entry_ts", "exit_ts", "pnl_pct", "status")}
            for t in trades[:25]
        ],
    }
    ML_DIR.mkdir(parents=True, exist_ok=True)
    (ML_DIR / "paper_pnl.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate open paper trades against candle history")
    args = parser.parse_args()
    payload = evaluate_paper_trades(Store())
    print(json.dumps({k: payload[k] for k in ("closed_this_run", "summary", "open_count", "by_policy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
