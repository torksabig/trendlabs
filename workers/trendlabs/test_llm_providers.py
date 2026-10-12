"""Unit tests for multi-provider LLM switching (no network / no real keys)."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from trendlabs.llm_providers import (
    LLMProvider,
    configured_provider_flags,
    resolve_provider,
)
from trendlabs.research_llm import run_research
from trendlabs.schemas import MarketResearchSnapshot, ResearchReport


class FakeProvider(LLMProvider):
    def __init__(self, provider_id: str, *, configured: bool = True, available: bool = True) -> None:
        self.id = provider_id
        self.label = provider_id
        self._configured = configured
        self._available = available
        self.calls: list[tuple[str, str, str | None]] = []

    def is_configured(self) -> bool:
        return self._configured

    def is_available(self) -> bool:
        return self._available

    def default_model(self) -> str:
        return f"{self.id}-model"

    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        self.calls.append((system[:20], user[:20], model))
        return {
            "symbol": "AAPL",
            "timeframe": "1h",
            "regime": "bullish",
            "summary": "Unit-test summary from " + self.id + " with enough characters.",
            "key_risks": ["Risk one for unit test coverage."],
            "historical_context": "Historical context for unit test.",
            "agent_notes": [],
            "proposals": [
                {
                    "title": "Test proposal",
                    "hypothesis": "Hypothesis text that is long enough.",
                    "test_plan": "Test plan text that is long enough.",
                    "risk_notes": "Risk notes here.",
                }
            ],
            "confidence": 0.5,
            "disclaimers": ["Research only — no trade execution authority."],
            "executes_trades": False,
        }


def _registry(**overrides: LLMProvider) -> dict[str, LLMProvider]:
    base = {
        "google": FakeProvider("google", configured=False),
        "groq": FakeProvider("groq", configured=False),
        "openrouter": FakeProvider("openrouter", configured=False),
        "cloudflare": FakeProvider("cloudflare", configured=False),
        "cerebras": FakeProvider("cerebras", configured=False),
        "mistral": FakeProvider("mistral", configured=False),
        "ollama": FakeProvider("ollama", configured=True, available=False),
        "fallback": FakeProvider("fallback", configured=True),
    }
    base.update(overrides)
    return base


class ResolveProviderTests(unittest.TestCase):
    def test_explicit_provider_id_selects_that_client(self) -> None:
        groq = FakeProvider("groq", configured=True)
        google = FakeProvider("google", configured=True)
        reg = _registry(groq=groq, google=google)

        selected, reason = resolve_provider("groq", registry=reg)
        self.assertEqual(selected.id, "groq")
        self.assertIn("explicit:groq", reason)
        self.assertIs(selected, groq)

        selected2, _ = resolve_provider("google", registry=reg)
        self.assertIs(selected2, google)

    def test_auto_prefers_first_configured_cloud(self) -> None:
        google = FakeProvider("google", configured=True)
        groq = FakeProvider("groq", configured=True)
        reg = _registry(google=google, groq=groq)
        selected, reason = resolve_provider("auto", registry=reg, allow_ollama=False)
        self.assertEqual(selected.id, "google")
        self.assertEqual(reason, "auto:google")

    def test_auto_falls_back_when_no_keys(self) -> None:
        reg = _registry()
        selected, reason = resolve_provider("auto", registry=reg, allow_ollama=False)
        self.assertEqual(selected.id, "fallback")
        self.assertEqual(reason, "auto:fallback")

    def test_missing_key_explicit_falls_back(self) -> None:
        reg = _registry(groq=FakeProvider("groq", configured=False))
        selected, reason = resolve_provider("groq", registry=reg)
        self.assertEqual(selected.id, "fallback")
        self.assertIn("missing_key", reason)

    def test_configured_flags_are_booleans_only(self) -> None:
        reg = _registry(groq=FakeProvider("groq", configured=True))
        flags = configured_provider_flags(reg)
        self.assertTrue(flags["groq"])
        self.assertFalse(flags["google"])
        self.assertTrue(flags["fallback"])
        self.assertTrue(all(isinstance(v, bool) for v in flags.values()))


class RunResearchSwitchTests(unittest.TestCase):
    def test_switching_provider_invokes_matching_client(self) -> None:
        groq = FakeProvider("groq", configured=True)
        google = FakeProvider("google", configured=True)
        reg = _registry(groq=groq, google=google)

        snapshot = MarketResearchSnapshot(
            symbol="AAPL",
            timeframe="1h",
            as_of="2024-01-01T00:00:00Z",
            last_close=100.0,
            ema20=99.0,
            ema50=98.0,
            ema200=97.0,
            adx14=20.0,
            rsi14=55.0,
            atr14=1.2,
            trend_label="bullish",
            historical_matches=10,
            historical_positive_outcomes=6,
            historical_hit_rate=0.6,
            avg_forward_return_10=1.5,
            agent_leaderboard=[
                {
                    "agent_id": "ema_trend",
                    "name": "EMA",
                    "equity": 10000,
                    "open_positions": 0,
                    "recent_trades": 0,
                }
            ],
            recent_failed_trades=[],
            data_source="test",
        )

        store = MagicMock()
        store.insert_agent_report = MagicMock()
        store.set_meta = MagicMock()

        # Patch build_market_snapshot to avoid DB
        import trendlabs.research_llm as rl

        original = rl.build_market_snapshot
        rl.build_market_snapshot = lambda *_a, **_k: snapshot  # type: ignore[assignment]
        try:
            out_groq = run_research(
                store,
                provider="groq",
                prefer_ollama=False,
                registry=reg,
            )
            out_google = run_research(
                store,
                provider="google",
                prefer_ollama=False,
                registry=reg,
            )
        finally:
            rl.build_market_snapshot = original  # type: ignore[assignment]

        self.assertEqual(len(groq.calls), 1)
        self.assertEqual(len(google.calls), 1)
        self.assertEqual(out_groq["source"], "groq")
        self.assertEqual(out_google["source"], "google")
        self.assertFalse(out_groq["report"]["executes_trades"])
        ResearchReport.model_validate(out_groq["report"])

    def test_no_keys_uses_fallback(self) -> None:
        reg = _registry()
        snapshot = MarketResearchSnapshot(
            symbol="AAPL",
            timeframe="1h",
            as_of="2024-01-01T00:00:00Z",
            last_close=100.0,
            ema20=99.0,
            ema50=98.0,
            ema200=97.0,
            adx14=20.0,
            rsi14=55.0,
            atr14=1.2,
            trend_label="bullish",
            historical_matches=10,
            historical_positive_outcomes=6,
            historical_hit_rate=0.6,
            avg_forward_return_10=1.5,
            agent_leaderboard=[],
            recent_failed_trades=[],
            data_source="test",
        )
        store = MagicMock()
        store.insert_agent_report = MagicMock()
        store.set_meta = MagicMock()

        import trendlabs.research_llm as rl

        original = rl.build_market_snapshot
        rl.build_market_snapshot = lambda *_a, **_k: snapshot  # type: ignore[assignment]
        try:
            out = run_research(store, prefer_ollama=False, registry=reg)
        finally:
            rl.build_market_snapshot = original  # type: ignore[assignment]

        self.assertEqual(out["source"], "fallback")
        self.assertIn("rule_based", out["model"])
        self.assertFalse(out["report"]["executes_trades"])


if __name__ == "__main__":
    unittest.main()
