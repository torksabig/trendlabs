"""Pydantic schemas for the shared Qwen research agent (never executes trades)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StrategyProposal(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    hypothesis: str = Field(min_length=10, max_length=600)
    test_plan: str = Field(min_length=10, max_length=600)
    risk_notes: str = Field(min_length=5, max_length=400)


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
