"""Shared research agent: multi-provider LLM with deterministic fallback.

This agent never executes trades. It only interprets verified Python analytics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from trendlabs.config import AGENTS
from trendlabs.db import Store
from trendlabs.indicators import enrich
from trendlabs.llm_providers import (
    LLMError,
    LLMProvider,
    get_registry,
    resolve_provider,
)
from trendlabs.schemas import MarketResearchSnapshot, ResearchReport
from trendlabs.trends import summarize_similar


SYSTEM_PROMPT = (
    "You are TrendLab's shared quantitative research agent (not a trading agent). "
    "Use only the supplied JSON facts. Do not invent statistics or candle values. "
    "Explain uncertainty and risks. Propose experiments Python can backtest. "
    "Never execute trades, place orders, or claim live brokerage access. "
    "Return JSON matching the required schema with executes_trades set to false."
)


def build_market_snapshot(
    store: Store,
    symbol: str = "AAPL",
    timeframe: str = "1h",
) -> MarketResearchSnapshot:
    candles = store.load_candles(symbol, timeframe)
    if candles.empty:
        raise ValueError(f"No candles for {symbol} {timeframe}. Run collect first.")

    enriched = enrich(candles)
    last = enriched.iloc[-1]
    events = [
        e
        for e in store.get_trend_events(500)
        if e["symbol"] == symbol.upper() and e["timeframe"] == timeframe
    ]
    direction = str(last["trend_label"])
    if direction not in {"bullish", "bearish"}:
        direction = "bullish"
    hist = summarize_similar(events, direction)

    agents = {a["id"]: a for a in store.get_agents()}
    positions = store.get_positions()
    trades = store.get_trades(80)
    marks = {symbol.upper(): float(last["close"])}

    leaderboard = []
    for meta in AGENTS:
        aid = meta["id"]
        cash = float(agents.get(aid, {}).get("cash", 10000))
        pos = [p for p in positions if p["agent_id"] == aid]
        equity = cash + sum(
            float(p["qty"]) * marks.get(p["symbol"], float(p["avg_entry"])) for p in pos
        )
        leaderboard.append(
            {
                "agent_id": aid,
                "name": meta["name"],
                "equity": round(equity, 2),
                "open_positions": len(pos),
                "recent_trades": len([t for t in trades if t["agent_id"] == aid]),
            }
        )
    leaderboard.sort(key=lambda x: x["equity"], reverse=True)

    # Approximate "failed" sells: price below a simple prior buy is not tracked;
    # instead surface recent SELL reasons for the LLM to interpret.
    failed = [
        {
            "agent_id": t["agent_id"],
            "symbol": t["symbol"],
            "side": t["side"],
            "price": t["price"],
            "reason": t["reason"],
            "ts": t["ts"],
        }
        for t in trades
        if t["side"] == "SELL" and t["symbol"] == symbol.upper()
    ][:8]

    return MarketResearchSnapshot(
        symbol=symbol.upper(),
        timeframe=timeframe,
        as_of=str(last["ts"]),
        last_close=float(last["close"]),
        ema20=_num(last.get("ema_20")),
        ema50=_num(last.get("ema_50")),
        ema200=_num(last.get("ema_200")),
        adx14=_num(last.get("adx_14")),
        rsi14=_num(last.get("rsi_14")),
        atr14=_num(last.get("atr_14")),
        trend_label=str(last["trend_label"]),
        historical_matches=int(hist["matches"]),
        historical_positive_outcomes=int(
            round((hist["hit_rate"] or 0) * hist["matches"])
        ),
        historical_hit_rate=hist["hit_rate"],
        avg_forward_return_10=hist["avg_forward_10"],
        agent_leaderboard=leaderboard,
        recent_failed_trades=failed,
        data_source=store.get_meta("data_source", "unknown") or "unknown",
    )


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        import math

        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def ollama_available() -> bool:
    """Backward-compatible helper used by older call sites."""
    from trendlabs.llm_providers import OllamaProvider

    return OllamaProvider().is_available()


def generate_with_provider(
    snapshot: MarketResearchSnapshot,
    provider: LLMProvider,
    model: str | None = None,
) -> ResearchReport:
    schema = ResearchReport.model_json_schema()
    user = (
        "Produce a structured research report for this verified market snapshot. "
        "JSON schema:\n"
        f"{json.dumps(schema)}\n\n"
        f"Snapshot:\n{snapshot.model_dump_json(indent=2)}"
    )
    data = provider.chat_json(SYSTEM_PROMPT, user, model=model)
    data["symbol"] = snapshot.symbol
    data["timeframe"] = snapshot.timeframe
    data["executes_trades"] = False
    report = ResearchReport.model_validate(data)
    return _reconcile_with_facts(report, snapshot)


def generate_with_ollama(
    snapshot: MarketResearchSnapshot,
    model: str | None = None,
    host: str | None = None,  # noqa: ARG001 — kept for API compatibility
) -> ResearchReport:
    from trendlabs.llm_providers import OllamaProvider

    provider = OllamaProvider()
    return generate_with_provider(snapshot, provider, model=model)


def generate_fallback_report(
    snapshot: MarketResearchSnapshot,
    reason: str = "No LLM provider configured or available",
) -> ResearchReport:
    """Deterministic research report when cloud/Ollama LLMs are unavailable."""
    hit = snapshot.historical_hit_rate
    hit_txt = f"{hit * 100:.0f}%" if hit is not None else "n/a"
    avg = snapshot.avg_forward_return_10
    avg_txt = f"{avg:+.2f}%" if avg is not None else "n/a"
    leader = snapshot.agent_leaderboard[0]["name"] if snapshot.agent_leaderboard else "n/a"

    regime: Any = snapshot.trend_label if snapshot.trend_label in {"bullish", "bearish", "sideways"} else "uncertain"
    report = ResearchReport(
        symbol=snapshot.symbol,
        timeframe=snapshot.timeframe,
        regime=regime,
        summary=(
            f"{snapshot.symbol} on {snapshot.timeframe} is classified {snapshot.trend_label} "
            f"with close {snapshot.last_close:.2f}. "
            f"EMA20/50/200 = {snapshot.ema20}, {snapshot.ema50}, {snapshot.ema200}; "
            f"ADX {snapshot.adx14}, RSI {snapshot.rsi14}. "
            f"Historical {snapshot.trend_label} matches: {snapshot.historical_matches} "
            f"(hit rate {hit_txt}, avg 10-bar forward {avg_txt}). "
            f"Current paper-trading leader: {leader}. "
            f"This fallback report was generated without an LLM ({reason})."
        ),
        key_risks=[
            "Indicator readings can flip quickly around session opens and news.",
            "Historical hit rates are descriptive, not predictive guarantees.",
            "Paper fills ignore borrow, shorting, and some liquidity constraints.",
        ],
        historical_context=(
            f"Among {snapshot.historical_matches} similar completed setups, "
            f"{snapshot.historical_positive_outcomes} had positive 10-bar forward returns "
            f"(hit rate {hit_txt})."
        ),
        agent_notes=[
            {
                "agent_id": row["agent_id"],
                "observation": f"Equity marked at {row['equity']} with {row['open_positions']} open positions.",
                "suggested_focus": "Keep risk budget and stop discipline unchanged; test only one parameter tweak at a time.",
            }
            for row in snapshot.agent_leaderboard[:4]
        ],
        proposals=[
            {
                "title": "ADX confirmation gate",
                "hypothesis": "Requiring ADX >= 25 before EMA/momentum entries may cut sideways whipsaws.",
                "test_plan": "Replay the current 1h history with an ADX>=25 filter and compare drawdown vs baseline.",
                "risk_notes": "May miss early trend acceleration; measure opportunity cost separately.",
            },
            {
                "title": "Volatility-scaled stops",
                "hypothesis": "ATR-based stops (2.0x vs 1.5x) can improve survival in choppy regimes.",
                "test_plan": "Grid-search ATR multipliers on out-of-sample final 30% of bars.",
                "risk_notes": "Wider stops increase per-trade loss size; keep 1% equity risk fixed.",
            },
        ],
        confidence=0.45 if snapshot.historical_matches < 20 else 0.6,
        disclaimers=[
            "Research only — no trade execution authority.",
            "All statistics originated from TrendLab Python calculations.",
        ],
        executes_trades=False,
    )
    return ResearchReport.model_validate(report)


def _reconcile_with_facts(report: ResearchReport, snapshot: MarketResearchSnapshot) -> ResearchReport:
    """Overwrite identity fields and clamp confidence; keep LLM prose otherwise."""
    data = report.model_dump()
    data["symbol"] = snapshot.symbol
    data["timeframe"] = snapshot.timeframe
    data["executes_trades"] = False
    if "Research only" not in " ".join(data.get("disclaimers", [])):
        data.setdefault("disclaimers", []).append("Research only — no trade execution authority.")
    return ResearchReport.model_validate(data)


def run_research(
    store: Store,
    symbol: str = "AAPL",
    timeframe: str = "1h",
    prefer_ollama: bool = True,
    provider: str | None = None,
    model: str | None = None,
    registry: dict[str, LLMProvider] | None = None,
) -> dict[str, Any]:
    """Generate a research report via the selected (or auto) LLM provider.

    prefer_ollama=False is treated as forcing fallback when no explicit cloud
    provider is requested (legacy --no-ollama / CI path).
    """
    snapshot = build_market_snapshot(store, symbol=symbol, timeframe=timeframe)
    reg = registry or get_registry()

    # --no-ollama (prefer_ollama=False) only disables the local daemon in the
    # auto chain; free-tier cloud providers still run when keys are present.
    allow_ollama = prefer_ollama
    try:
        selected, reason = resolve_provider(
            provider, registry=reg, allow_ollama=allow_ollama
        )
    except LLMError as exc:
        report = generate_fallback_report(snapshot, reason=str(exc))
        return _persist(
            store,
            snapshot,
            report,
            source="fallback_unknown_provider",
            model_name=f"rule_based_fallback:{exc.__class__.__name__}",
        )

    if selected.id == "fallback":
        report = generate_fallback_report(snapshot, reason=reason)
        return _persist(
            store,
            snapshot,
            report,
            source="fallback",
            model_name="rule_based_fallback",
        )

    try:
        report = generate_with_provider(snapshot, selected, model=model)
        used_model = model or selected.default_model()
        return _persist(
            store,
            snapshot,
            report,
            source=selected.id,
            model_name=used_model,
        )
    except Exception as exc:  # noqa: BLE001
        report = generate_fallback_report(
            snapshot, reason=f"{selected.id} error: {exc}"
        )
        return _persist(
            store,
            snapshot,
            report,
            source=f"fallback_after_{selected.id}_error",
            model_name=f"rule_based_fallback:{exc.__class__.__name__}",
        )


def _persist(
    store: Store,
    snapshot: MarketResearchSnapshot,
    report: ResearchReport,
    *,
    source: str,
    model_name: str,
) -> dict[str, Any]:
    payload = {
        "symbol": snapshot.symbol,
        "timeframe": snapshot.timeframe,
        "model": model_name,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot.model_dump(),
        "report": report.model_dump(),
    }
    store.insert_agent_report(payload)

    # Optional Supabase sync (Mac / CI with secrets)
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"):
        try:
            from trendlabs.supabase_sync import upload_agent_report

            upload_agent_report(payload)
            payload["supabase"] = "uploaded"
        except Exception as exc:  # noqa: BLE001
            payload["supabase"] = f"error:{exc}"

    store.set_meta("last_research", payload["created_at"])
    store.set_meta("research_model", model_name)
    store.set_meta("research_provider", source)
    return payload
