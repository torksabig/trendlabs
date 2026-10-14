"""Pydantic schemas for the shared research agent (never executes trades)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


STRATEGY_IDS = Literal["ema_trend", "breakout", "momentum", "supertrend"]

# Keys the LLM may override per strategy (Python evaluation owns the rest).
ALLOWED_STRATEGY_PARAMS: dict[str, frozenset[str]] = {
    "ema_trend": frozenset({"fast_ema", "medium_ema", "slow_ema", "adx_min"}),
    "breakout": frozenset({"entry_lookback", "exit_lookback"}),
    "momentum": frozenset({"momentum_bars", "adx_min"}),
    "supertrend": frozenset({"period", "multiplier"}),
}

ALLOWED_RISK_KEYS = frozenset(
    {
        "starting_cash",
        "fee_bps",
        "slippage_bps",
        "risk_per_trade",
        "max_positions",
        "atr_stop_mult",
        "min_notional",
    }
)


class StrategyProposal(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    hypothesis: str = Field(min_length=10, max_length=600)
    test_plan: str = Field(min_length=10, max_length=600)
    risk_notes: str = Field(min_length=5, max_length=400)


class ExperimentProposal(BaseModel):
    """Structured control-vs-variant experiment the LLM must return for Python evaluation.

    The research LLM never executes trades. Python maps this into ExperimentSpec,
    runs walk-forward OOS evaluation, and applies AcceptanceCriteria. Accepted
    variants are NOT auto-promoted into live paper agents.
    """

    strategy_id: STRATEGY_IDS
    hypothesis: str = Field(min_length=10, max_length=600)
    control_params: dict[str, Any] = Field(default_factory=dict)
    variant_params: dict[str, Any] = Field(default_factory=dict)
    risk_overrides: dict[str, Any] = Field(default_factory=dict)
    experiment_id: str | None = Field(
        default=None,
        description="Optional slug; Python generates llm_<strategy>_<hash> when omitted.",
        max_length=80,
    )
    variant_name: str | None = Field(default=None, max_length=120)

    @field_validator("experiment_id")
    @classmethod
    def slug_experiment_id(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        slug = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")
        if not slug:
            raise ValueError("experiment_id must be a non-empty slug")
        return slug[:80]

    @field_validator("control_params", "variant_params")
    @classmethod
    def non_null_dict(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        return dict(value or {})

    @field_validator("risk_overrides")
    @classmethod
    def validate_risk_keys(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(value or {})
        unknown = set(data) - ALLOWED_RISK_KEYS
        if unknown:
            raise ValueError(f"unsupported risk_overrides keys: {sorted(unknown)}")
        return data

    @model_validator(mode="after")
    def validate_param_keys_and_delta(self) -> ExperimentProposal:
        allowed = ALLOWED_STRATEGY_PARAMS[self.strategy_id]
        for label, params in (
            ("control_params", self.control_params),
            ("variant_params", self.variant_params),
        ):
            unknown = set(params) - allowed
            if unknown:
                raise ValueError(
                    f"{label} has unsupported keys for {self.strategy_id}: {sorted(unknown)}"
                )
        if not self.variant_params and not self.risk_overrides:
            raise ValueError(
                "variant_params or risk_overrides must differ from the control baseline"
            )
        return self


class ExperimentProposalBatch(BaseModel):
    """LLM response wrapper: 1–3 structured experiments, research-only."""

    experiments: list[ExperimentProposal] = Field(min_length=1, max_length=3)
    rationale: str = Field(
        default="",
        max_length=800,
        description="Brief note on why these variants were chosen from the verified stats.",
    )
    executes_trades: Literal[False] = False
    promotes_to_paper_agents: Literal[False] = False

    @field_validator("executes_trades", "promotes_to_paper_agents", mode="before")
    @classmethod
    def force_false_flags(cls, value: Any) -> bool:
        # Accept missing/true from messy model output; Python enforces false.
        return False


class AgentComparisonNote(BaseModel):
    agent_id: str
    observation: str = Field(min_length=5, max_length=400)
    suggested_focus: str = Field(min_length=5, max_length=400)


class ResearchReport(BaseModel):
    """Structured output from the local research LLM."""

    symbol: str
    timeframe: str
    regime: Literal["bullish", "bearish", "sideways", "uncertain"]
    summary: str = Field(min_length=20, max_length=1200)
    key_risks: list[str] = Field(min_length=1, max_length=6)
    historical_context: str = Field(min_length=10, max_length=800)
    agent_notes: list[AgentComparisonNote] = Field(default_factory=list, max_length=4)
    proposals: list[StrategyProposal] = Field(min_length=1, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    disclaimers: list[str] = Field(min_length=1, max_length=4)
    executes_trades: Literal[False] = False

    @field_validator("key_risks", "disclaimers")
    @classmethod
    def non_empty_strings(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v and v.strip()]
        if not cleaned:
            raise ValueError("must contain at least one non-empty string")
        return cleaned


class MarketResearchSnapshot(BaseModel):
    """Compact verified facts supplied to the LLM — Python owns the numbers."""

    symbol: str
    timeframe: str
    as_of: str
    last_close: float
    ema20: float | None
    ema50: float | None
    ema200: float | None
    adx14: float | None
    rsi14: float | None
    atr14: float | None
    trend_label: str
    historical_matches: int
    historical_positive_outcomes: int
    historical_hit_rate: float | None
    avg_forward_return_10: float | None
    agent_leaderboard: list[dict]
    recent_failed_trades: list[dict]
    data_source: str
    note: str = (
        "All numeric fields were computed by TrendLab Python workers. "
        "Do not invent additional statistics. Do not execute or recommend live trades."
    )
