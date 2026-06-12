"""Two-stage zero-shot classifier orchestrator.

Stage 1 — embedding-similarity scoring against anchor prompts (fast path).
Stage 2 — DeBERTa-v3-xsmall zero-shot classification (escalation path).

Backends are exposed as Protocols so production code can wire in the real
HuggingFace models while tests inject deterministic fakes — no weights are
downloaded in CI.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import structlog

from llm_cost_autopilot.classifier.complexity import (
    ClassificationResult,
    ClassifierStage,
    ComplexityTier,
)

logger = structlog.get_logger(__name__)


# ── Curated anchor prompts per tier ──────────────────────────────────────────
# These are the only "training data" the classifier needs. Edit with care —
# every change shifts the routing distribution. See decisions/001 for rationale.
ANCHORS: dict[ComplexityTier, tuple[str, ...]] = {
    ComplexityTier.SIMPLE: (
        "What is the capital of France?",
        "Translate this word to Spanish.",
        "Format this date as ISO 8601.",
        "Convert 50 USD to EUR.",
        "What day of the week is March 4th?",
    ),
    ComplexityTier.MODERATE: (
        "Summarise this article in 3 bullet points.",
        "Classify this support ticket by category.",
        "Extract the key entities from this text.",
        "Rewrite this paragraph in a more formal tone.",
        "Compare these two product descriptions.",
    ),
    ComplexityTier.COMPLEX: (
        "Analyse the tradeoffs between these two architectural approaches.",
        "Write a detailed technical proposal for this system.",
        "Debug this code and explain the root cause step by step.",
        "Draft a strategic plan with prioritised initiatives and risks.",
        "Reason through this multi-step math problem and justify each step.",
    ),
}


# ── Backend protocols ────────────────────────────────────────────────────────
@runtime_checkable
class EmbeddingBackend(Protocol):
    """Stage-1 backend — returns the best-matching tier and its cosine score."""

    def score(self, prompt: str) -> dict[ComplexityTier, float]:
        """Return per-tier max cosine similarity in [0, 1]."""
        ...


@runtime_checkable
class ZeroShotBackend(Protocol):
    """Stage-2 backend — returns NLI probabilities for each tier label."""

    def classify(self, prompt: str) -> dict[ComplexityTier, float]:
        """Return per-tier probability mass in [0, 1] summing to ~1."""
        ...


# Each tier maps to a human-readable hypothesis used by the zero-shot pipeline.
# Centralised here so the wording stays consistent with the anchors.
TIER_HYPOTHESIS_LABELS: dict[ComplexityTier, str] = {
    ComplexityTier.SIMPLE: "simple factual lookup",
    ComplexityTier.MODERATE: "moderate reasoning",
    ComplexityTier.COMPLEX: "complex multi-step analysis",
}


def argmax_tier(scores: dict[ComplexityTier, float]) -> tuple[ComplexityTier, float]:
    """Return the (tier, score) pair with the highest score.

    Ties resolve in tier-enum declaration order (SIMPLE < MODERATE < COMPLEX),
    which prefers cheaper tiers when models are genuinely indecisive.
    """
    if not scores:
        raise ValueError("scores must contain at least one tier")
    best_tier = max(scores, key=lambda t: (scores[t], -list(ComplexityTier).index(t)))
    return best_tier, scores[best_tier]


class TwoStageClassifier:
    """Routes a prompt to a complexity tier in one or two stages.

    Parameters
    ----------
    embedding_backend:
        Concrete stage-1 backend (real or fake).
    zero_shot_backend:
        Concrete stage-2 backend (real or fake).
    stage1_threshold:
        If stage-1 confidence is below this, escalate to stage 2.
    """

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        zero_shot_backend: ZeroShotBackend,
        stage1_threshold: float = 0.75,
    ) -> None:
        if not 0.0 <= stage1_threshold <= 1.0:
            raise ValueError("stage1_threshold must be in [0, 1]")
        self._embed = embedding_backend
        self._zero_shot = zero_shot_backend
        self._threshold = stage1_threshold

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify ``prompt`` into a complexity tier.

        Empty prompts short-circuit to SIMPLE — there is no meaningful work to
        route, but downstream code still expects a result.
        """
        if not prompt or not prompt.strip():
            return ClassificationResult(
                tier=ComplexityTier.SIMPLE,
                confidence=1.0,
                stage=ClassifierStage.STAGE_1_EMBEDDING,
                latency_ms=0.0,
                scores={ComplexityTier.SIMPLE: 1.0},
            )

        t0 = time.perf_counter()
        stage1_scores = self._embed.score(prompt)
        tier, confidence = argmax_tier(stage1_scores)

        if confidence >= self._threshold:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "classifier.stage1.accept",
                tier=tier.value,
                confidence=confidence,
                latency_ms=latency_ms,
            )
            return ClassificationResult(
                tier=tier,
                confidence=confidence,
                stage=ClassifierStage.STAGE_1_EMBEDDING,
                latency_ms=latency_ms,
                scores=stage1_scores,
            )

        # Escalate — stage 1 is not confident enough.
        stage2_scores = self._zero_shot.classify(prompt)
        tier, confidence = argmax_tier(stage2_scores)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "classifier.stage2.escalated",
            tier=tier.value,
            confidence=confidence,
            stage1_confidence=max(stage1_scores.values()),
            latency_ms=latency_ms,
        )
        return ClassificationResult(
            tier=tier,
            confidence=confidence,
            stage=ClassifierStage.STAGE_2_ZERO_SHOT,
            latency_ms=latency_ms,
            scores=stage2_scores,
        )
