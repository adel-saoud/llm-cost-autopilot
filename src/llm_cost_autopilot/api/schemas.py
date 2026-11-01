"""Pydantic request and response schemas for the HTTP API.

Only the subset of OpenAI's chat-completions surface that autopilot serves.
Request fields it does not honour are accepted and ignored, so stock OpenAI
clients that send extra parameters keep working.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_cost_autopilot.classifier.complexity import ClassifierStage, ComplexityTier
from llm_cost_autopilot.router.ollama import ChatRole
from llm_cost_autopilot.storage.db import LedgerStats, RoutingLogEntry
from llm_cost_autopilot.verifier.feedback import RoutingFailure


class ChatCompletionMessage(BaseModel):
    """One message in an OpenAI-compatible chat-completions request."""

    model_config = ConfigDict(extra="ignore")

    role: ChatRole
    content: str


def classification_prompt(messages: list[ChatCompletionMessage]) -> str:
    """Return the text the classifier routes on: the *last* user message.

    Earlier system prompts and conversation history affect what the model
    actually generates, but the routing decision is driven by the latest
    user turn: that's what the classifier was trained on.
    """
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    # No user turn: classify the whole conversation rather than nothing.
    return "\n".join(message.content for message in messages)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request body for ``POST /v1/chat/completions``."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(
        default="autopilot",
        description="Ignored: autopilot picks the model based on complexity.",
    )
    messages: list[ChatCompletionMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)


class ChatCompletionUsage(BaseModel):
    """Token counts in OpenAI's ``usage`` shape, taken from Ollama's eval counts."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionChoice(BaseModel):
    """One entry of OpenAI's ``choices`` list; autopilot always returns exactly one."""

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
    def from_ledger_stats(cls, stats: LedgerStats) -> StatsResponse:
        """Copy the storage aggregate into the API shape, which can evolve separately."""
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
    tenant_id: str
    matched_correction_request_id: str | None
    correction: RoutingFailure | None

    @classmethod
    def from_record(
        cls, routing: RoutingLogEntry, correction: RoutingFailure | None
    ) -> ExplainResponse:
        """Flatten a routing-log row and its optional correction, deriving ``saved_usd``."""
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
            tenant_id=routing.tenant_id,
            matched_correction_request_id=routing.matched_correction_request_id,
            correction=correction,
        )


# ── Router API (consumed by llm-gateway) ───────────────────────────────────
TenantId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        description="Who the traffic belongs to. Scopes the correction memory.",
    ),
]


class RouteRequest(BaseModel):
    """Request body for ``POST /v2/route``. Unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    tenant_id: TenantId
    messages: list[ChatCompletionMessage] = Field(
        min_length=1,
        description="OpenAI chat messages. The last user message is classified.",
    )


class RouteResponse(BaseModel):
    """Routing decision returned by ``POST /v2/route``.

    No model id on purpose: the caller owns its model registry, allowlists
    and fallbacks, and maps ``tier`` onto them.
    """

    model_config = ConfigDict(extra="forbid")

    route_id: str = Field(description="Autopilot's id for this decision, for audit logs.")
    tier: ComplexityTier
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(
        description=(
            "What decided the tier. Known values: stage_1_embedding, stage_2_zero_shot, "
            "correction_memory. Callers must accept unknown values."
        ),
    )


class HealthResponse(BaseModel):
    """Liveness payload; the version lets a deploy check confirm which build answers."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    version: str
