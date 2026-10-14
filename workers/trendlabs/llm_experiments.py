"""LLM-proposed experiment loop (OpenRouter gpt-6-luna → Python OOS evaluate).

Flow:
  1. Gather compact verified stats from walk-forward / backtest / experiment JSON
     and light DB regime summaries (Python-owned numbers only).
  2. Ask the research LLM (default: OpenRouter openai/gpt-6-luna) for 1–3
     structured ExperimentProposal objects.
  3. Map proposals → ExperimentSpec via strategy_config.override_config.
  4. Run ExperimentEngine.evaluate; persist under data/experiments/ + SQLite.

Safeguards:
  - Research LLM never executes trades.
  - AcceptanceCriteria (min trades / Sharpe lift / fold robustness) decide accept.
  - Accepted variants are NEVER auto-promoted into live paper agents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trendlabs.config import (
    DEFAULT_TIMEFRAME,
    OPENROUTER_MODEL,
    ROOT,
    SYMBOLS,
)

# Experiment loop default — pin to gpt-6-luna even if OPENROUTER_MODEL falls back to free.
DEFAULT_EXPERIMENT_MODEL = "openai/gpt-6-luna"
from trendlabs.db import Store
from trendlabs.experiments import (
    EXPERIMENT_DIR,
    AcceptanceCriteria,
    ExperimentEngine,
    ExperimentSpec,
    write_experiment_result,
    _auto_windows,
)
from trendlabs.indicators import enrich
from trendlabs.llm_providers import (
    LLMError,
    LLMProvider,
    get_registry,
    resolve_provider,
)
from trendlabs.schemas import ExperimentProposal, ExperimentProposalBatch
from trendlabs.strategy_config import config_for, override_config


WALK_FORWARD_LATEST = Path(ROOT) / "data" / "walk_forward" / "latest.json"
BACKTEST_LATEST = Path(ROOT) / "data" / "backtests" / "latest.json"
EXPERIMENT_LATEST = EXPERIMENT_DIR / "latest.json"

SYSTEM_PROMPT = (
    "You are TrendLab's quantitative experiment designer (research only). "
    "Use only the supplied verified JSON stats — do not invent metrics. "
    "Propose 1–3 structured control-vs-variant experiments Python can evaluate "
    "with walk-forward out-of-sample folds. "
    "Never execute trades, place orders, or promote variants to paper agents. "
    "Return JSON matching the required schema with executes_trades=false and "
    "promotes_to_paper_agents=false. "
    "Prefer small, testable parameter deltas (one primary change per experiment). "
    "Allowed strategy_id values: ema_trend, breakout, momentum, supertrend. "
    "Allowed params: ema_trend={fast_ema,medium_ema,slow_ema,adx_min}; "
    "breakout={entry_lookback,exit_lookback}; "
    "momentum={momentum_bars,adx_min}; "
    "supertrend={period,multiplier}. "
    "risk_overrides may include atr_stop_mult, risk_per_trade, max_positions, "
    "fee_bps, slippage_bps, starting_cash, min_notional."
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _compact_agent_metrics(agent: dict[str, Any]) -> dict[str, Any]:
    m = agent.get("metrics") or {}
    keys = (
        "total_return_pct",
        "sharpe",
        "mean_fold_sharpe",
        "max_drawdown_pct",
        "trade_count",
        "win_rate",
        "alpha_pct",
        "sortino",
        "profit_factor",
    )
    metrics = {k: m.get(k) for k in keys if k in m}
    regime_fit = agent.get("regime_fit") or {}
    regime_perf = agent.get("regime_performance") or {}
    # Keep regime PnL compact
    regime_summary = {
        regime: {
            "trade_count": stats.get("trade_count"),
            "win_rate": stats.get("win_rate"),
            "total_pnl": round(float(stats["total_pnl"]), 2)
            if stats.get("total_pnl") is not None
            else None,
        }
        for regime, stats in list(regime_perf.items())[:8]
        if isinstance(stats, dict)
    }
    return {
        "agent_id": agent.get("agent_id") or agent.get("strategy_id"),
        "name": agent.get("name"),
        "metrics": metrics,
        "regime_fit": regime_fit,
        "regime_performance": regime_summary,
    }


def _compact_walk_forward(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    agents = [_compact_agent_metrics(a) for a in (payload.get("agents") or [])[:6]]
    return {
        "run_id": payload.get("run_id"),
        "timeframe": payload.get("timeframe"),
        "fold_count": payload.get("fold_count"),
        "warm_up": payload.get("warm_up"),
        "calibrate_bars": payload.get("calibrate_bars"),
        "test_bars": payload.get("test_bars"),
        "symbols": payload.get("symbols"),
        "agents": agents,
    }


def _compact_backtest(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    agents = [_compact_agent_metrics(a) for a in (payload.get("agents") or [])[:6]]
    return {
        "run_id": payload.get("run_id"),
        "timeframe": payload.get("timeframe"),
        "bars_tested": payload.get("bars_tested"),
        "market_regime_distribution": payload.get("market_regime_distribution"),
        "agents": agents,
    }


def _compact_prior_experiments(payload: dict[str, Any] | None, limit: int = 8) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows = []
    for row in (payload.get("experiments") or [])[:limit]:
        rows.append(
            {
                "experiment_id": row.get("experiment_id"),
                "decision": row.get("decision"),
                "hypothesis": (row.get("hypothesis") or "")[:200],
                "comparison": row.get("comparison"),
            }
        )
    return rows


def _live_regime_snapshot(store: Store, symbols: list[str], timeframe: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sym in symbols[:6]:
        candles = store.load_candles(sym, timeframe)
        if candles.empty:
            continue
        enriched = enrich(candles)
        last = enriched.iloc[-1]
        rows.append(
            {
                "symbol": sym,
                "as_of": str(last["ts"]),
                "close": float(last["close"]),
                "market_regime": str(last.get("market_regime") or last.get("trend_label")),
                "adx_14": _finite(last.get("adx_14")),
                "rsi_14": _finite(last.get("rsi_14")),
                "atr_14": _finite(last.get("atr_14")),
            }
        )
    return rows


def _finite(value: Any) -> float | None:
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


def gather_verified_stats(
    store: Store,
    *,
    symbols: list[str],
    timeframe: str = DEFAULT_TIMEFRAME,
) -> dict[str, Any]:
    """Compact Python-owned facts for the LLM (no invented numbers)."""
    return {
        "note": (
            "All numeric fields were computed by TrendLab Python workers. "
            "Do not invent statistics. Do not execute or promote trades."
        ),
        "timeframe": timeframe,
        "symbols": sorted(symbols),
        "walk_forward": _compact_walk_forward(_load_json(WALK_FORWARD_LATEST)),
        "backtest": _compact_backtest(_load_json(BACKTEST_LATEST)),
        "prior_experiments": _compact_prior_experiments(_load_json(EXPERIMENT_LATEST)),
        "live_regimes": _live_regime_snapshot(store, symbols, timeframe),
        "default_strategy_params": {
            sid: config_for(sid).to_dict()
            for sid in ("ema_trend", "breakout", "momentum", "supertrend")
        },
        "acceptance_hints": {
            "min_trades": 12,
            "min_sharpe_lift": 0.15,
            "max_drawdown_worsen_pp": 3.0,
            "min_fold_win_rate": 0.5,
            "auto_promote_to_paper": False,
        },
    }


def propose_experiments(
    stats: dict[str, Any],
    *,
    provider: LLMProvider,
    model: str | None = None,
    max_experiments: int = 3,
) -> ExperimentProposalBatch:
    """Call the LLM and validate structured experiment proposals with pydantic."""
    max_experiments = max(1, min(3, int(max_experiments)))
    schema = ExperimentProposalBatch.model_json_schema()
    user = (
        f"Propose between 1 and {max_experiments} structured experiments. "
        "JSON schema:\n"
        f"{json.dumps(schema)}\n\n"
        "Verified stats (Python-owned):\n"
        f"{json.dumps(stats, indent=2, default=str)}"
    )
    data = provider.chat_json(SYSTEM_PROMPT, user, model=model)
    data["executes_trades"] = False
    data["promotes_to_paper_agents"] = False
    # Clamp if model returns too many
    if isinstance(data.get("experiments"), list):
        data["experiments"] = data["experiments"][:max_experiments]
    return ExperimentProposalBatch.model_validate(data)


def _slug_from_proposal(proposal: ExperimentProposal) -> str:
    if proposal.experiment_id:
        base = proposal.experiment_id
        if not base.startswith("llm_"):
            base = f"llm_{base}"
        return base[:80]
    raw = (
        f"{proposal.strategy_id}|{json.dumps(proposal.control_params, sort_keys=True)}|"
        f"{json.dumps(proposal.variant_params, sort_keys=True)}|"
        f"{json.dumps(proposal.risk_overrides, sort_keys=True)}|{proposal.hypothesis[:80]}"
    )
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    parts = []
    for key, value in list(proposal.variant_params.items())[:2]:
        parts.append(f"{key}{value}".replace(".", "p"))
    for key, value in list(proposal.risk_overrides.items())[:1]:
        parts.append(f"{key}{value}".replace(".", "p"))
    mid = "_".join(re.sub(r"[^a-z0-9]+", "", p.lower()) for p in parts) or "variant"
    return f"llm_{proposal.strategy_id}_{mid}_{digest}"[:80]


def proposal_to_spec(proposal: ExperimentProposal) -> ExperimentSpec:
    """Map a validated LLM proposal into an ExperimentSpec (no trade execution)."""
    base = config_for(proposal.strategy_id)
    control = override_config(
        base,
        params=proposal.control_params or None,
    )
    variant_name = proposal.variant_name or f"{control.name} (LLM variant)"
    variant = override_config(
        control,
        params=proposal.variant_params or None,
        risk=proposal.risk_overrides or None,
        name=variant_name,
    )
    return ExperimentSpec(
        experiment_id=_slug_from_proposal(proposal),
        hypothesis=proposal.hypothesis,
        control=control,
        variant=variant,
        source="llm",
        notes=(
            "LLM-proposed; evaluated by ExperimentEngine. "
            "Not auto-promoted to paper agents."
        ),
    )


def run_llm_experiments(
    store: Store,
    *,
    symbols: list[str],
    timeframe: str = DEFAULT_TIMEFRAME,
    provider_id: str | None = "openrouter",
    model: str | None = None,
    max_experiments: int = 3,
    criteria: AcceptanceCriteria | None = None,
    registry: dict[str, LLMProvider] | None = None,
    warm_up: int | None = None,
    calibrate_bars: int | None = None,
    test_bars: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Propose via LLM, evaluate each experiment OOS, persist results.

    Never mutates paper-trading books or promotes accepted variants.
    """
    stats = gather_verified_stats(store, symbols=symbols, timeframe=timeframe)
    reg = registry or get_registry()
    selected, reason = resolve_provider(provider_id, registry=reg, allow_ollama=True)
    if selected.id == "fallback":
        raise LLMError(
            f"No LLM provider available for experiment proposals ({reason}). "
            "Configure OPENROUTER_API_KEY or another research provider."
        )

    if model:
        used_model = model
    elif selected.id == "openrouter":
        # Prefer env override, else the gpt-6-luna experiment default.
        used_model = OPENROUTER_MODEL or DEFAULT_EXPERIMENT_MODEL
        if used_model == "openrouter/free":
            used_model = DEFAULT_EXPERIMENT_MODEL
    else:
        used_model = selected.default_model()
    batch = propose_experiments(
        stats,
        provider=selected,
        model=used_model,
        max_experiments=max_experiments,
    )

    specs = [proposal_to_spec(p) for p in batch.experiments]
    if dry_run:
        return {
            "kind": "llm_experiment_dry_run",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": selected.id,
            "model": used_model,
            "selection_reason": reason,
            "rationale": batch.rationale,
            "proposals": [s.to_dict() for s in specs],
            "executes_trades": False,
            "promotes_to_paper_agents": False,
            "auto_promote": False,
        }

    wu, cal, test = _auto_windows(store, symbols, timeframe)
    if warm_up is not None:
        wu = warm_up
    if calibrate_bars is not None:
        cal = calibrate_bars
    if test_bars is not None:
        test = test_bars

    engine = ExperimentEngine(store, criteria=criteria or AcceptanceCriteria())
    results: list[dict[str, Any]] = []
    for spec in specs:
        payload = engine.evaluate(
            spec,
            symbols=symbols,
            timeframe=timeframe,
            warm_up=wu,
            calibrate_bars=cal,
            test_bars=test,
        )
        # Explicit safeguards on every persisted result
        payload["source"] = "llm"
        payload["provider"] = selected.id
        payload["model"] = used_model
        payload["auto_promote"] = False
        payload["promotes_to_paper_agents"] = False
        payload["llm_executes_trades"] = False
        path = write_experiment_result(payload, directory=EXPERIMENT_DIR)
        results.append(
            {
                "experiment_id": spec.experiment_id,
                "run_id": payload["run_id"],
                "decision": payload["decision"],
                "reasons": payload["reasons"],
                "comparison": payload["comparison"],
                "path": str(path),
                "hypothesis": spec.hypothesis,
            }
        )

    return {
        "kind": "llm_experiment_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": selected.id,
        "model": used_model,
        "selection_reason": reason,
        "rationale": batch.rationale,
        "count": len(results),
        "results": results,
        "executes_trades": False,
        "promotes_to_paper_agents": False,
        "auto_promote": False,
        "walk_forward_windows": {
            "warm_up": wu,
            "calibrate_bars": cal,
            "test_bars": test,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrendLab LLM experiment loop (propose → OOS evaluate; no paper promotion)"
    )
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--provider",
        default="openrouter",
        help="LLM provider id (default: openrouter / gpt-6-luna)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model override (default OpenRouter: {DEFAULT_EXPERIMENT_MODEL})",
    )
    parser.add_argument("--max-experiments", type=int, default=3)
    parser.add_argument("--min-trades", type=int, default=12)
    parser.add_argument("--min-sharpe-lift", type=float, default=0.15)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Propose and validate only; do not run ExperimentEngine",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print compact verified stats and exit (no LLM call)",
    )
    args = parser.parse_args()

    store = Store()
    available = store.list_symbols_with_data(args.timeframe)
    symbols = [s for s in args.symbols if s in available]
    if not symbols:
        raise SystemExit(f"No candle data for {args.timeframe}. Run collect first.")

    if args.stats_only:
        print(json.dumps(gather_verified_stats(store, symbols=symbols, timeframe=args.timeframe), indent=2))
        return 0

    criteria = AcceptanceCriteria(
        min_trades=args.min_trades,
        min_sharpe_lift=args.min_sharpe_lift,
    )
    try:
        outcome = run_llm_experiments(
            store,
            symbols=symbols,
            timeframe=args.timeframe,
            provider_id=args.provider,
            model=args.model,
            max_experiments=args.max_experiments,
            criteria=criteria,
            dry_run=args.dry_run,
        )
    except LLMError as exc:
        print(json.dumps({"error": str(exc), "executes_trades": False}, indent=2))
        return 1

    print(json.dumps(outcome, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
