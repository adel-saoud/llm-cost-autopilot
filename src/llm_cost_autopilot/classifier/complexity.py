"""Complexity tiers and the immutable result type returned by the classifier."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ComplexityTier(StrEnum):
    """Routing tiers in increasing order of expected model cost.

    The tier maps 1:1 to a model in the registry — see ``router.models``.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ClassifierStage(StrEnum):
    """Which stage of the two-stage classifier produced the verdict."""

    STAGE_1_EMBEDDING = "stage_1_embedding"
    STAGE_2_ZERO_SHOT = "stage_2_zero_shot"


class ClassificationResult(BaseModel):
    """Outcome of a single classification — immutable, audit-friendly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: ComplexityTier
    confidence: float = Field(ge=0.0, le=1.0)
    stage: ClassifierStage
    latency_ms: float = Field(ge=0.0)
    scores: dict[ComplexityTier, float] = Field(
        default_factory=lambda: dict[ComplexityTier, float]({}),
        description="Per-tier raw scores from the stage that decided.",
    )
