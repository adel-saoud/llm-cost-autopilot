"""Records the verifier produces: its outcome enum and the persisted routing failure."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.classifier.correction_memory import DEFAULT_TENANT_ID


class VerificationOutcome(StrEnum):
    """Why the verifier produced (or skipped) a result for one request."""

    AGREED = "agreed"
    """The cheap response was semantically close to the standard response."""

    DIVERGED = "diverged"
    """The two responses differed beyond the divergence threshold."""

    SKIPPED_STANDARD_TIER = "skipped_standard_tier"
    """The request was already routed to the standard model: nothing to compare."""

    SKIPPED_DISABLED = "skipped_disabled"
    """The verifier was disabled via settings."""

    SKIPPED_OVERLOADED = "skipped_overloaded"
    """Too many verifications were already waiting; this one was dropped."""

    FAILED = "failed"
    """The verifier itself errored (logged but swallowed)."""


class RoutingFailure(BaseModel):
    """One routing decision the verifier disagreed with: written to SQLite.

    The whole record is denormalised on purpose: the correction dataset is
    meant to be inspected stand-alone, not joined.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    tenant_id: str = DEFAULT_TENANT_ID
    """Whose traffic produced this correction. The correction memory only
    applies a correction to later prompts from the same tenant."""
    prompt: str
    cheap_model: str
    cheap_response: str
    cheap_tier: ComplexityTier
    classifier_confidence: float = Field(ge=0.0, le=1.0)
    standard_model: str
    standard_response: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    divergence_threshold: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
