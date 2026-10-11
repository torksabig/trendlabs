"""Export a Vercel-friendly JSON snapshot consumed by the Next.js dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trendlabs.config import AGENTS, INSTRUMENTS, SNAPSHOT_PATH, STARTING_CASH
from trendlabs.db import Store
from trendlabs.indicators import enrich
from trendlabs.trends import summarize_similar


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def build_snapshot(store: Store, timeframe: str = "1h") -> dict[str, Any]:
    markets = []
    for inst in INSTRUMENTS:
        symbol = inst["symbol"]
        candles = store.load_candles(symbol, timeframe)
        if candles.empty:
            continue
        enriched = enrich(candles)
        last = enriched.iloc[-1]
        markets.append(
            {
                "symbol": symbol,
                "name": inst["name"],
                "sector": inst["sector"],
                "timeframe": timeframe,
                "last_close": float(last["close"]),
                "trend_label": last["trend_label"],
                "ema_20": _json_safe(last["ema_20"]),
                "ema_50": _json_safe(last["ema_50"]),
                "ema_200": _json_safe(last["ema_200"]),
                "rsi_14": _json_safe(last["rsi_14"]),
                "adx_14": _json_safe(last["adx_14"]),
                "atr_14": _json_safe(last["atr_14"]),
                "source": str(candles.iloc[-1]["source"]),
                "as_of": pd.Timestamp(last["ts"]).isoformat(),
                "candles": [
                    {
                        "time": int(pd.Timestamp(r.ts).timestamp()),
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(r.volume),
                    }
                    for r in enriched.tail(180).itertuples(index=False)
                ],
            }
        )

    equity = store.get_equity()
    trades = store.get_trades(120)
    positions = store.get_positions()
    agents_db = {a["id"]: a for a in store.get_agents()}
    marks = {m["symbol"]: m["last_close"] for m in markets}

    leaderboard = []
    for agent in AGENTS:
        agent_id = agent["id"]
        cash = float(agents_db.get(agent_id, {}).get("cash", STARTING_CASH))
        agent_positions = [p for p in positions if p["agent_id"] == agent_id]
        equity_val = cash + sum(
            float(p["qty"]) * marks.get(p["symbol"], float(p["avg_entry"])) for p in agent_positions
        )
        agent_trades = [t for t in trades if t["agent_id"] == agent_id]
        sells = [t for t in agent_trades if t["side"] == "SELL"]
        # Approximate win rate from round-trips is limited; use equity path drawdown instead.
        eq_series = [e for e in equity if e["agent_id"] == agent_id]
        peak = STARTING_CASH
        max_dd = 0.0
        for point in eq_series:
            peak = max(peak, float(point["equity"]))
            dd = float(point["equity"]) / peak - 1
            max_dd = min(max_dd, dd)
        leaderboard.append(
            {
                "id": agent_id,
                "name": agent["name"],
                "strategy": agent["strategy"],
                "description": agent["description"],
                "cash": cash,
                "equity": equity_val,
                "return_pct": (equity_val / STARTING_CASH - 1) * 100,
                "max_drawdown_pct": max_dd * 100,
                "trades": len(agent_trades),
                "open_positions": len(agent_positions),
                "positions": agent_positions,
            }
        )
    leaderboard.sort(key=lambda x: x["equity"], reverse=True)

    trend_events = store.get_trend_events(200)
    research = {
        "bullish": summarize_similar(trend_events, "bullish"),
        "bearish": summarize_similar(trend_events, "bearish"),
        "recent_events": trend_events[:40],
    }

    equity_by_agent: dict[str, list] = {}
    for point in equity:
        equity_by_agent.setdefault(point["agent_id"], []).append(
            {
                "ts": point["ts"],
                "equity": float(point["equity"]),
            }
        )

    total_equity = sum(a["equity"] for a in leaderboard) if leaderboard else STARTING_CASH * 4

    return _json_safe(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timeframe": timeframe,
            "data_source": store.get_meta("data_source", "unknown"),
            "last_collection": store.get_meta("last_collection"),
            "last_simulation": store.get_meta("last_simulation"),
            "starting_cash": STARTING_CASH,
            "totals": {
                "agents": len(leaderboard),
                "instruments": len(markets),
                "total_equity": total_equity,
                "avg_return_pct": (
                    sum(a["return_pct"] for a in leaderboard) / len(leaderboard) if leaderboard else 0
                ),
            },
            "instruments": INSTRUMENTS,
            "markets": markets,
            "leaderboard": leaderboard,
            "equity_curves": equity_by_agent,
            "trades": trades,
            "research": research,
            "status": {
                "budget": "$0/month target",
                "deployment": "Vercel Hobby + GitHub Actions + Supabase Free",
                "real_money_trading": False,
            },
        }
    )


def write_snapshot(store: Store, timeframe: str = "1h", path: Path = SNAPSHOT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_snapshot(store, timeframe=timeframe)
    path.write_text(json.dumps(payload, indent=2))
    # Also copy into Next.js public path for static hosting fallback
    public_copy = Path(__file__).resolve().parents[2] / "public" / "data" / "snapshot.json"
    public_copy.parent.mkdir(parents=True, exist_ok=True)
    public_copy.write_text(json.dumps(payload))
    return path
