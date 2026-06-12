"""Background verifier — compares cheap vs. standard model output.

The verifier runs in a fire-and-forget task. It must never propagate an
exception back to the user-facing response path. All errors are logged via
structlog and absorbed.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

import structlog

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.router.models import ModelConfig
from llm_cost_autopilot.router.ollama import ChatMessage, OllamaChatResponse, OllamaClient
from llm_cost_autopilot.verifier.feedback import RoutingFailure, VerificationOutcome

logger = structlog.get_logger(__name__)


# ── Backend protocols ────────────────────────────────────────────────────────
@runtime_checkable
class SimilarityScorer(Protocol):
    """Returns a cosine-similarity-like score in [0, 1] between two strings."""

    async def score(self, a: str, b: str) -> float: ...


@runtime_checkable
class CorrectionStore(Protocol):
    """Persistence sink for divergent routing decisions."""

    async def record_failure(self, failure: RoutingFailure) -> None: ...


# ── Verifier ─────────────────────────────────────────────────────────────────
class Verifier:
    """Async quality verifier — fire-and-forget by design.

    Parameters
    ----------
    ollama:
        Shared Ollama client.
    standard_model:
        ``ModelConfig`` for the model we treat as ground truth (typically the
        MODERATE-tier entry from the registry).
    scorer:
        Cosine-similarity backend (real or fake).
    store:
        Where divergent decisions are persisted.
    divergence_threshold:
        Below this similarity, the routing decision is recorded as a failure.
    enabled:
        Master switch — false means ``verify_in_background`` records
        SKIPPED_DISABLED outcomes and never spawns work.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        standard_model: ModelConfig,
        scorer: SimilarityScorer,
        store: CorrectionStore,
        divergence_threshold: float,
        *,
        enabled: bool = True,
    ) -> None:
        if not 0.0 <= divergence_threshold <= 1.0:
            raise ValueError("divergence_threshold must be in [0, 1]")
        self._ollama = ollama
        self._standard = standard_model
        self._scorer = scorer
        self._store = store
        self._threshold = divergence_threshold
        self._enabled = enabled

    def verify_in_background(
        self,
        *,
        request_id: str,
        prompt: str,
        cheap_response: OllamaChatResponse,
        cheap_tier: ComplexityTier,
        classifier_confidence: float,
    ) -> asyncio.Task[VerificationOutcome]:
        """Schedule verification and return the task.

        Callers should NOT await the returned task on the request path — that
        is what would re-introduce blocking. The task is returned mainly so
        tests can wait on completion; the FastAPI layer fires and forgets.
        """
        coro = self._run(
            request_id=request_id,
            prompt=prompt,
            cheap_response=cheap_response,
            cheap_tier=cheap_tier,
            classifier_confidence=classifier_confidence,
        )
        return asyncio.create_task(coro, name=f"verifier:{request_id}")

    async def _run(
        self,
        *,
        request_id: str,
        prompt: str,
        cheap_response: OllamaChatResponse,
        cheap_tier: ComplexityTier,
        classifier_confidence: float,
    ) -> VerificationOutcome:
        log = logger.bind(request_id=request_id, cheap_tier=cheap_tier.value)

        if not self._enabled:
            log.debug("verifier.skipped.disabled")
            return VerificationOutcome.SKIPPED_DISABLED

        if cheap_response.model == self._standard.ollama_model:
            log.debug("verifier.skipped.standard_tier")
            return VerificationOutcome.SKIPPED_STANDARD_TIER

        try:
            standard_response = await self._ollama.chat(
                model=self._standard.ollama_model,
                messages=[ChatMessage(role="user", content=prompt)],
                max_output_tokens=self._standard.max_output_tokens,
            )
            similarity = await self._scorer.score(cheap_response.content, standard_response.content)
        except Exception as exc:
            log.warning("verifier.failed", error=str(exc), error_type=type(exc).__name__)
            return VerificationOutcome.FAILED

        if similarity >= self._threshold:
            log.debug("verifier.agreed", similarity=similarity)
            return VerificationOutcome.AGREED

        failure = RoutingFailure(
            request_id=request_id,
            prompt=prompt,
            cheap_model=cheap_response.model,
            cheap_response=cheap_response.content,
            cheap_tier=cheap_tier,
            classifier_confidence=classifier_confidence,
            standard_model=standard_response.model,
            standard_response=standard_response.content,
            similarity_score=similarity,
            divergence_threshold=self._threshold,
        )
        try:
            await self._store.record_failure(failure)
        except Exception as exc:
            log.warning(
                "verifier.persist_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return VerificationOutcome.FAILED

        log.info("verifier.diverged", similarity=similarity, threshold=self._threshold)
        return VerificationOutcome.DIVERGED
