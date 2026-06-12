"""Pydantic request / response schemas for the OpenAI-compatible API.

Only the subset of OpenAI's chat-completions surface we actually serve.
Fields we don't honour are silently accepted and ignored (matches OpenAI
clients' expectations).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_cost_autopilot.classifier.complexity import ClassifierStage, ComplexityTier
from llm_cost_autopilot.storage.db import RoutingLogEntry, Stats
from llm_cost_autopilot.verifier.feedback import RoutingFailure

ChatRole = Literal["system", "user", "assistant"]


class ChatCompletionMessage(BaseModel):
    """One message in an OpenAI-compatible chat-completions request."""

    model_config = ConfigDict(extra="ignore")

    role: ChatRole
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request body for ``POST /v1/chat/completions``."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(
        default="autopilot",
        description="Ignored — autopilot picks the model based on complexity.",
    )
    messages: list[ChatCompletionMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)


class ChatCompletionUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response body."""

    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


# ── Admin / observability ───────────────────────────────────────────────────
class StatsResponse(BaseModel):
    """Snapshot returned by ``GET /v1/stats``."""

    model_config = ConfigDict(extra="forbid")

    total_requests: int
    total_actual_cost_usd: float
    total_baseline_cost_usd: float
    savings_usd: float
    savings_pct: float
    routing_accuracy_pct: float
    correction_count: int
    requests_by_tier: dict[ComplexityTier, int]

    @classmethod
    def from_stats(cls, stats: Stats) -> StatsResponse:
        return cls(**stats.model_dump())


class CorrectionsResponse(BaseModel):
    """Most-recent routing failures returned by ``GET /v1/corrections``."""

    model_config = ConfigDict(extra="forbid")

    corrections: list[RoutingFailure]
    count: int


class ExplainResponse(BaseModel):
    """Detail view returned by ``GET /v1/explain/{request_id}``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    classified_tier: ComplexityTier
    classifier_confidence: float
    classifier_stage: ClassifierStage
    model_used: str
    cost_usd: float
    baseline_cost_usd: float
    saved_usd: float
    latency_ms: float
    correction: RoutingFailure | None

    @classmethod
    def from_record(
        cls, routing: RoutingLogEntry, correction: RoutingFailure | None
    ) -> ExplainResponse:
        return cls(
            request_id=routing.request_id,
            classified_tier=routing.classified_tier,
            classifier_confidence=routing.classifier_confidence,
            classifier_stage=routing.classifier_stage,
            model_used=routing.model_used,
            cost_usd=routing.cost_usd,
            baseline_cost_usd=routing.baseline_cost_usd,
            saved_usd=routing.baseline_cost_usd - routing.cost_usd,
            latency_ms=routing.latency_ms,
            correction=correction,
        )


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    version: str
