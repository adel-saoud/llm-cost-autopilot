"""Complexity tiers, their cost order, and the immutable classification result."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ComplexityTier(StrEnum):
    """Routing tiers in increasing order of expected model cost.

    Each tier maps to exactly one model in the registry: see ``router.models``.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"


def tier_rank(tier: ComplexityTier) -> int:
    """Position of ``tier`` in increasing order of cost, starting at 0.

    Relies on ``ComplexityTier`` declaring its members cheapest first, so a new
    tier must be inserted at its cost position. The enum's own docstring is
    not the place for this note: it is published in docs/route-api.schema.json.
    """
    return list(ComplexityTier).index(tier)


def next_tier(tier: ComplexityTier) -> ComplexityTier | None:
    """Return the tier one step more capable than ``tier``, or None at the top."""
    tiers = list(ComplexityTier)
    rank = tier_rank(tier)
    return tiers[rank + 1] if rank + 1 < len(tiers) else None


class ClassifierStage(StrEnum):
    """Which part of the classifier produced the verdict.

    The value is returned as ``reason`` by ``POST /v2/route`` and as the
    ``X-Autopilot-Routing-Reason`` header, so renaming one breaks callers.
    """

    STAGE_1_EMBEDDING = "stage_1_embedding"
    """Anchor similarity was confident enough on its own."""
    STAGE_2_ZERO_SHOT = "stage_2_zero_shot"
    """Stage 1 was unsure and the NLI zero-shot model decided."""
    CORRECTION_MEMORY = "correction_memory"
    """A past correction for a similar prompt escalated the tier."""


class ClassificationResult(BaseModel):
    """Outcome of a single classification, frozen so the audit trail cannot drift."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: ComplexityTier
    confidence: float = Field(ge=0.0, le=1.0)
    stage: ClassifierStage
    latency_ms: float = Field(ge=0.0)
    scores: dict[ComplexityTier, float] = Field(
        default_factory=lambda: dict[ComplexityTier, float]({}),
        description="Per-tier raw scores from the stage that decided.",
    )
    matched_correction_request_id: str | None = Field(
        default=None,
        description="Request id of the past correction that escalated this prompt, if any.",
    )
