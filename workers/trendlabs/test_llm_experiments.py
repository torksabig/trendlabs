"""Unit tests for LLM experiment proposals (mocked LLM — no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from trendlabs.db import Store
from trendlabs.experiments import AcceptanceCriteria
from trendlabs.llm_experiments import (
    gather_verified_stats,
    proposal_to_spec,
    propose_experiments,
    run_llm_experiments,
)
from trendlabs.llm_providers import LLMProvider
from trendlabs.schemas import ExperimentProposal, ExperimentProposalBatch


def _synth(symbol: str, n: int = 520, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    rets = rng.normal(0.0002, 0.01, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, size=n))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timeframe": "1h",
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1e6, 3e6, size=n),
            "source": "test",
            "adjusted": True,
        }
    )


class FakeLLM(LLMProvider):
    def __init__(
        self,
        provider_id: str = "openrouter",
        payload: dict[str, Any] | None = None,
        *,
        configured: bool = True,
    ) -> None:
        self.id = provider_id
        self.label = f"fake-{provider_id}"
        self._configured = configured
        self._payload = payload or {
            "experiments": [
                {
                    "strategy_id": "ema_trend",
                    "hypothesis": "ADX>=20 on EMA Trend should reduce sideways whipsaws OOS.",
                    "control_params": {},
                    "variant_params": {"adx_min": 20.0},
                    "risk_overrides": {},
                    "experiment_id": "ema_adx20_llm",
                },
                {
                    "strategy_id": "ema_trend",
                    "hypothesis": "Tighter ATR stop 1.5x should improve max drawdown for EMA Trend.",
                    "control_params": {},
                    "variant_params": {},
                    "risk_overrides": {"atr_stop_mult": 1.5},
                },
            ],
            "rationale": "Unit-test proposals from verified walk-forward stats.",
            "executes_trades": False,
            "promotes_to_paper_agents": False,
        }
        self.calls: list[tuple[str, str, str | None]] = []

    def is_configured(self) -> bool:
        return self._configured

    def default_model(self) -> str:
        return "openai/gpt-6-luna"

    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        self.calls.append((system[:40], user[:40], model))
        return dict(self._payload)


def _registry(openrouter: FakeLLM) -> dict[str, LLMProvider]:
    return {
        "google": FakeLLM("google", configured=False),
        "groq": FakeLLM("groq", configured=False),
        "openrouter": openrouter,
        "cloudflare": FakeLLM("cloudflare", configured=False),
        "cerebras": FakeLLM("cerebras", configured=False),
        "mistral": FakeLLM("mistral", configured=False),
        "ollama": FakeLLM("ollama", configured=False),
        "fallback": FakeLLM("fallback", configured=True),
    }


class SchemaTests(unittest.TestCase):
    def test_valid_proposal(self) -> None:
        p = ExperimentProposal.model_validate(
            {
                "strategy_id": "momentum",
                "hypothesis": "Raising ADX floor filters weak momentum entries.",
                "control_params": {"adx_min": 22.0},
                "variant_params": {"adx_min": 28.0},
                "risk_overrides": {},
            }
        )
        self.assertEqual(p.strategy_id, "momentum")

    def test_rejects_unknown_param(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentProposal.model_validate(
                {
                    "strategy_id": "breakout",
                    "hypothesis": "Unknown param should fail validation clearly.",
                    "variant_params": {"magic_number": 99},
                }
            )

    def test_rejects_empty_delta(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentProposal.model_validate(
                {
                    "strategy_id": "ema_trend",
                    "hypothesis": "No changes should not be accepted as an experiment.",
                    "variant_params": {},
                    "risk_overrides": {},
                }
            )

    def test_batch_forces_no_trade_flags(self) -> None:
        batch = ExperimentProposalBatch.model_validate(
            {
                "experiments": [
                    {
                        "strategy_id": "breakout",
                        "hypothesis": "Longer entry lookback reduces false breakouts.",
                        "variant_params": {"entry_lookback": 30},
                    }
                ],
                "executes_trades": True,
                "promotes_to_paper_agents": True,
            }
        )
        self.assertFalse(batch.executes_trades)
        self.assertFalse(batch.promotes_to_paper_agents)


class ProposalMappingTests(unittest.TestCase):
    def test_proposal_to_spec_uses_override_config(self) -> None:
        proposal = ExperimentProposal.model_validate(
            {
                "strategy_id": "ema_trend",
                "hypothesis": "ADX gate on EMA Trend for unit mapping test coverage.",
                "variant_params": {"adx_min": 25.0},
                "risk_overrides": {"atr_stop_mult": 1.75},
                "experiment_id": "unit_adx25",
            }
        )
        spec = proposal_to_spec(proposal)
        self.assertTrue(spec.experiment_id.startswith("llm_"))
        self.assertEqual(spec.source, "llm")
        self.assertIsNone(spec.control.params.get("adx_min"))
        self.assertEqual(spec.variant.params.get("adx_min"), 25.0)
        self.assertEqual(spec.variant.risk.atr_stop_mult, 1.75)
        self.assertIn("Not auto-promoted", spec.notes)


class LlmLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "llm_exp.db")
        self.store.upsert_candles(_synth("AAPL", 520, seed=1))
        self.store.upsert_candles(_synth("SPY", 520, seed=2))
        self.fake = FakeLLM()
        self.registry = _registry(self.fake)
        self.exp_dir = Path(self.tmp.name) / "experiments"

    def test_propose_experiments_validates(self) -> None:
        stats = {"note": "test", "symbols": ["AAPL"]}
        batch = propose_experiments(stats, provider=self.fake, max_experiments=2)
        self.assertEqual(len(batch.experiments), 2)
        self.assertFalse(batch.executes_trades)
        self.assertEqual(len(self.fake.calls), 1)

    def test_gather_verified_stats_shape(self) -> None:
        stats = gather_verified_stats(self.store, symbols=["AAPL", "SPY"], timeframe="1h")
        self.assertIn("live_regimes", stats)
        self.assertIn("default_strategy_params", stats)
        self.assertFalse(stats["acceptance_hints"]["auto_promote_to_paper"])
        self.assertTrue(any(r["symbol"] == "AAPL" for r in stats["live_regimes"]))

    def test_run_llm_experiments_dry_run(self) -> None:
        outcome = run_llm_experiments(
            self.store,
            symbols=["AAPL", "SPY"],
            timeframe="1h",
            provider_id="openrouter",
            registry=self.registry,
            dry_run=True,
        )
        self.assertEqual(outcome["kind"], "llm_experiment_dry_run")
        self.assertEqual(len(outcome["proposals"]), 2)
        self.assertFalse(outcome["auto_promote"])
        self.assertFalse(outcome["promotes_to_paper_agents"])

    def test_run_llm_experiments_evaluates_without_paper_trades(self) -> None:
        before = self.store.get_trades(limit=5)
        import trendlabs.llm_experiments as mod

        original_dir = mod.EXPERIMENT_DIR
        mod.EXPERIMENT_DIR = self.exp_dir
        try:
            outcome = run_llm_experiments(
                self.store,
                symbols=["AAPL", "SPY"],
                timeframe="1h",
                provider_id="openrouter",
                registry=self.registry,
                criteria=AcceptanceCriteria(
                    min_trades=1, min_sharpe_lift=0.0, min_fold_win_rate=0.0
                ),
                warm_up=100,
                calibrate_bars=60,
                test_bars=50,
            )
        finally:
            mod.EXPERIMENT_DIR = original_dir

        self.assertEqual(outcome["kind"], "llm_experiment_run")
        self.assertGreaterEqual(outcome["count"], 1)
        for row in outcome["results"]:
            self.assertIn(row["decision"], {"accept", "reject", "insufficient"})
            self.assertTrue(Path(row["path"]).exists())
        self.assertFalse(outcome["auto_promote"])
        self.assertEqual(self.store.get_trades(limit=5), before)
        stored = self.store.get_experiment_results(limit=5)
        self.assertGreaterEqual(len(stored), 1)
        self.assertFalse(stored[0].get("auto_promote", False))
        self.assertFalse(stored[0].get("promotes_to_paper_agents", False))


if __name__ == "__main__":
    unittest.main()
