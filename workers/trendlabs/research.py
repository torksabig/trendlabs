"""CLI: run the shared research agent (multi-provider LLM or fallback)."""

from __future__ import annotations

import argparse
import json

from trendlabs.config import DEFAULT_TIMEFRAME
from trendlabs.db import Store
from trendlabs.llm_providers import PROVIDER_IDS
from trendlabs.research_llm import run_research
from trendlabs.snapshot import write_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="TrendLab shared AI research agent")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--provider",
        default=None,
        choices=[*PROVIDER_IDS, "auto"],
        help="LLM provider id (default: TRENDLAB_LLM_PROVIDER / auto)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the selected provider",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Disable local Ollama in the auto chain (CI / cloud VMs)",
    )
    args = parser.parse_args()

    store = Store()
    payload = run_research(
        store,
        symbol=args.symbol,
        timeframe=args.timeframe,
        prefer_ollama=not args.no_ollama,
        provider=None if args.provider in (None, "auto") else args.provider,
        model=args.model,
    )
    write_snapshot(store, timeframe=args.timeframe)
    print(json.dumps(
        {
            "symbol": payload["symbol"],
            "timeframe": payload["timeframe"],
            "model": payload["model"],
            "source": payload["source"],
            "supabase": payload.get("supabase"),
            "regime": payload["report"]["regime"],
            "summary": payload["report"]["summary"],
            "proposals": [p["title"] for p in payload["report"]["proposals"]],
            "executes_trades": payload["report"]["executes_trades"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
