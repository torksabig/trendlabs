"""Daily research agent: rule-based by default; optional LLM if API key is set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from trading_sim.config import AGENT_LABELS, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, STARTING_CASH
from trading_sim.db import Database


@dataclass
class ResearchReport:
    source: str
    summary: str
    proposals: list[str]


class ResearchAgent:
    def __init__(self, db: Database) -> None:
        self.db = db

    def _performance_table(self) -> pd.DataFrame:
        agents = self.db.get_agents()
        equity = self.db.get_equity()
        trades = self.db.get_trades()
        rows = []
        for _, agent in agents.iterrows():
            agent_id = agent["id"]
            eq = equity[equity["agent_id"] == agent_id] if not equity.empty else pd.DataFrame()
            tr = trades[trades["agent_id"] == agent_id] if not trades.empty else pd.DataFrame()
            final_eq = float(eq["equity"].iloc[-1]) if not eq.empty else float(agent["cash"])
            ret = (final_eq / STARTING_CASH - 1) * 100
            peak = eq["equity"].cummax() if not eq.empty else pd.Series([final_eq])
            dd = ((eq["equity"] / peak) - 1).min() * 100 if not eq.empty else 0.0
            rows.append(
                {
                    "agent_id": agent_id,
                    "label": AGENT_LABELS.get(agent_id, agent_id),
                    "equity": final_eq,
                    "return_pct": ret,
                    "max_drawdown_pct": float(dd) if pd.notna(dd) else 0.0,
                    "trades": int(len(tr)),
                    "position_qty": float(agent["position_qty"]),
                }
            )
        return pd.DataFrame(rows).sort_values("return_pct", ascending=False)

    def build_rule_based_report(self) -> ResearchReport:
        perf = self._performance_table()
        if perf.empty:
            return ResearchReport(
                source="rule_based",
                summary="No simulation data yet. Run a backtest first.",
                proposals=["Fetch Binance candles and run the four-agent backtest."],
            )

        leader = perf.iloc[0]
        laggard = perf.iloc[-1]
        summary = (
            f"Leader: {leader['label']} at ${leader['equity']:.2f} "
            f"({leader['return_pct']:+.2f}%, max DD {leader['max_drawdown_pct']:.2f}%, "
            f"{int(leader['trades'])} trades). "
            f"Laggard: {laggard['label']} at {laggard['return_pct']:+.2f}%."
        )

        proposals = [
            f"Keep {leader['label']} as the baseline trend template; its edge led the cohort.",
            f"Stress-test {laggard['label']} with tighter exits or a higher confirmation threshold.",
            "Try a slower interval (4h) to cut noise for momentum and EMA agents.",
            "Add a volatility filter (ATR percentile) before opening new long exposure.",
            "Run a walk-forward window: train on first 70% of bars, validate on the last 30%.",
        ]
        if leader["max_drawdown_pct"] < -15:
            proposals.insert(
                1,
                "Leader drawdown exceeded 15% — add a hard stop or reduce position size.",
            )
        return ResearchReport(source="rule_based", summary=summary, proposals=proposals)

    def build_llm_report(self) -> ResearchReport | None:
        if not OPENAI_API_KEY:
            return None

        perf = self._performance_table()
        payload_stats = perf.to_dict(orient="records")
        prompt = {
            "role": "user",
            "content": (
                "You are a quant research assistant reviewing a paper-trading experiment "
                "with four deterministic trend agents (EMA crossover, ADX, momentum, Supertrend). "
                "Given this performance JSON, write: (1) a 2-3 sentence summary, "
                "(2) exactly 4 concrete next experiments. Reply as JSON with keys "
                "summary (string) and proposals (array of strings).\n\n"
                f"{json.dumps(payload_stats, indent=2)}"
            ),
        }
        try:
            resp = requests.post(
                f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return only valid JSON with keys summary and proposals.",
                        },
                        prompt,
                    ],
                },
                timeout=45,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data: dict[str, Any] = json.loads(content)
            return ResearchReport(
                source=f"llm:{OPENAI_MODEL}",
                summary=str(data.get("summary", "")).strip(),
                proposals=[str(p) for p in data.get("proposals", [])][:6],
            )
        except Exception as exc:  # noqa: BLE001 — fall back gracefully
            fallback = self.build_rule_based_report()
            fallback.summary = f"LLM call failed ({exc}); used rule-based report. " + fallback.summary
            fallback.source = "rule_based_fallback"
            return fallback

    def run(self, prefer_llm: bool = True) -> ResearchReport:
        report = self.build_llm_report() if prefer_llm else None
        if report is None:
            report = self.build_rule_based_report()
        self.db.save_research_note(report.source, report.summary, report.proposals)
        return report
