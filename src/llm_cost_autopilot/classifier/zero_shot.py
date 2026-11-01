"""Two-stage zero-shot classifier orchestrator.

Stage 1: embedding-similarity scoring against anchor prompts (fast path).
Stage 2: DeBERTa-v3-xsmall zero-shot classification (escalation path).

Backends are exposed as Protocols so production code can wire in the real
HuggingFace models while tests inject deterministic fakes: no weights are
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
    tier_rank,
)
from llm_cost_autopilot.classifier.correction_memory import (
    DEFAULT_TENANT_ID,
    CorrectionMemory,
    Embedding,
)

logger = structlog.get_logger(__name__)


ANCHOR_PROMPTS: dict[ComplexityTier, tuple[str, ...]] = {
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
        "Analyse the tradeoffs between these two architectural approaches.",
        "Write a detailed technical proposal for this system.",
        "Debug this code and explain the root cause step by step.",
        "Draft a strategic plan with prioritised initiatives and risks.",
        "Reason through this multi-step math problem and justify each step.",
    ),
}
"""Curated example prompts per tier: the only "training data" stage 1 has.

Every edit shifts the routing distribution, so re-run the evaluation after
changing one. Rationale: docs/adr/0001-two-stage-classifier.md.
"""


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Stage-1 backend: per-tier anchor scores plus the prompt embedding."""

    def score(self, prompt: str) -> dict[ComplexityTier, float]:
        """Return per-tier max cosine similarity in [0, 1]."""
        ...

    def embed(self, prompt: str) -> Embedding:
        """Return the prompt's L2-normalised embedding (used by the correction memory)."""
        ...


@runtime_checkable
class ZeroShotBackend(Protocol):
    """Stage-2 backend: returns NLI probabilities for each tier label."""

    def classify(self, prompt: str) -> dict[ComplexityTier, float]:
        """Return per-tier probability mass in [0, 1] summing to ~1."""
        ...


TIER_HYPOTHESIS_LABELS: dict[ComplexityTier, str] = {
    ComplexityTier.SIMPLE: "simple factual lookup",
    ComplexityTier.MODERATE: "reasoning, writing or analysis",
}
"""Candidate label per tier for the stage-2 NLI pipeline.

Kept next to ``ANCHOR_PROMPTS`` so both describe the tiers in the same terms.
"""


def argmax_tier(scores: dict[ComplexityTier, float]) -> tuple[ComplexityTier, float]:
    """Return the (tier, score) pair with the highest score.

    Ties resolve in tier-enum declaration order (SIMPLE < MODERATE),
    which prefers cheaper tiers when models are genuinely indecisive.
    """
    if not scores:
        raise ValueError("scores must contain at least one tier")
    best_tier = max(scores, key=lambda tier: (scores[tier], -tier_rank(tier)))
    return best_tier, scores[best_tier]


class TwoStageClassifier:
    """Routes a prompt to a complexity tier in one or two stages.

    Parameters
    ----------
    embedding_backend:
        Stage-1 anchor scorer; also embeds prompts for the correction memory.
    zero_shot_backend:
        Stage-2 NLI classifier, called only when stage 1 is unsure.
    stage1_threshold:
        If stage-1 confidence is below this, escalate to stage 2.
    correction_memory:
        Past routing mistakes. When set, a prompt close enough to one of them
        is sent one tier up. None disables the feedback loop.
    correction_match_threshold:
        Minimum cosine similarity for a stored correction to apply.
    """

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        zero_shot_backend: ZeroShotBackend,
        stage1_threshold: float = 0.75,
        *,
        correction_memory: CorrectionMemory | None = None,
        correction_match_threshold: float = 0.80,
    ) -> None:
        if not 0.0 <= stage1_threshold <= 1.0:
            raise ValueError("stage1_threshold must be in [0, 1]")
        if not 0.0 <= correction_match_threshold <= 1.0:
            raise ValueError("correction_match_threshold must be in [0, 1]")
        self._embedding_backend = embedding_backend
        self._zero_shot_backend = zero_shot_backend
        self._stage1_threshold = stage1_threshold
        self._correction_memory = correction_memory
        self._correction_match_threshold = correction_match_threshold

    def classify(self, prompt: str, *, tenant_id: str = DEFAULT_TENANT_ID) -> ClassificationResult:
        """Classify ``prompt`` into a complexity tier.

        Empty prompts short-circuit to SIMPLE: there is no meaningful work to
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

        started_at = time.perf_counter()
        stage1_scores = self._embedding_backend.score(prompt)
        tier, confidence = argmax_tier(stage1_scores)

        if confidence >= self._stage1_threshold:
            stage = ClassifierStage.STAGE_1_EMBEDDING
            scores = stage1_scores
        else:
            stage = ClassifierStage.STAGE_2_ZERO_SHOT
            scores = self._zero_shot_backend.classify(prompt)
            tier, confidence = argmax_tier(scores)

        escalated = self._apply_corrections(prompt, tenant_id, tier, stage1_scores, started_at)
        if escalated is not None:
            return escalated

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        logger.debug(
            "classifier.decided",
            tier=tier.value,
            stage=stage.value,
            confidence=confidence,
            stage1_confidence=max(stage1_scores.values()),
            latency_ms=latency_ms,
        )
        return ClassificationResult(
            tier=tier,
            confidence=confidence,
            stage=stage,
            latency_ms=latency_ms,
            scores=scores,
        )

    def _apply_corrections(
        self,
        prompt: str,
        tenant_id: str,
        tier: ComplexityTier,
        stage1_scores: dict[ComplexityTier, float],
        started_at: float,
    ) -> ClassificationResult | None:
        """Return an escalated result when a past correction applies, else None.

        A correction applies when the prompt is at least
        ``correction_match_threshold`` similar to it, is closer to it than to
        every anchor of the tier it was misrouted to, and the anchors alone
        would have routed it below the correction's target tier. The relative
        test keeps one noisy correction from capturing prompts that clearly
        belong to the cheap tier.
        """
        if self._correction_memory is None or not self._correction_memory.has_corrections(
            tenant_id
        ):
            return None
        match = self._correction_memory.nearest(tenant_id, self._embedding_backend.embed(prompt))
        if (
            match is None
            or match.similarity < self._correction_match_threshold
            or match.similarity <= stage1_scores.get(match.misrouted_tier, 0.0)
            or tier_rank(tier) >= tier_rank(match.escalate_to)
        ):
            return None
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "classifier.correction_escalated",
            tenant_id=tenant_id,
            from_tier=tier.value,
            to_tier=match.escalate_to.value,
            similarity=match.similarity,
            matched_correction_request_id=match.request_id,
        )
        return ClassificationResult(
            tier=match.escalate_to,
            confidence=max(0.0, min(1.0, match.similarity)),
            stage=ClassifierStage.CORRECTION_MEMORY,
            latency_ms=latency_ms,
            scores=stage1_scores,
            matched_correction_request_id=match.request_id,
        )
