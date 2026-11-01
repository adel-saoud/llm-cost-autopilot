"""Background verifier: compares the cheap model's answer with the standard model's.

The verifier runs in a fire-and-forget task. It must never propagate an
exception back to the user-facing response path, so every error is logged
and turned into a ``FAILED`` outcome.
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


@runtime_checkable
class SimilarityScorer(Protocol):
    """Returns a cosine-similarity-like score in [0, 1] between two strings."""

    async def score(self, first_text: str, second_text: str) -> float: ...


@runtime_checkable
class CorrectionStore(Protocol):
    """Persistence sink for divergent routing decisions."""

    async def record_failure(self, failure: RoutingFailure) -> None: ...


class Verifier:
    """Async quality verifier: fire-and-forget by design.

    Parameters
    ----------
    ollama:
        Shared Ollama client, so verifier calls reuse the request path's pool.
    standard_model:
        ``ModelConfig`` for the model we treat as ground truth (typically the
        MODERATE-tier entry from the registry).
    scorer:
        Similarity backend in [0, 1] (embedding model in production, fake in tests).
    store:
        Where divergent decisions are persisted.
    divergence_threshold:
        Below this similarity, the routing decision is recorded as a failure.
    enabled:
        Master switch: false means ``verify_in_background`` records
        SKIPPED_DISABLED outcomes and never spawns work.
    max_concurrency:
        Standard-model calls allowed at once. The verifier shares the Ollama
        daemon with user traffic, so it must not take every slot.
    max_pending:
        Verifications allowed in flight or waiting. Past this, new ones
        return SKIPPED_OVERLOADED instead of queueing without bound.
    temperature, seed:
        Sampling options for the standard-model call; evaluations pin them
        so a prompt cannot flip between agreed and diverged across runs.
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
        max_concurrency: int = 2,
        max_pending: int = 100,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= divergence_threshold <= 1.0:
            raise ValueError("divergence_threshold must be in [0, 1]")
        if max_concurrency < 1 or max_pending < 1:
            raise ValueError("max_concurrency and max_pending must be >= 1")
        self._ollama = ollama
        self._standard_model = standard_model
        self._scorer = scorer
        self._store = store
        self._divergence_threshold = divergence_threshold
        self._enabled = enabled
        self._standard_call_slots = asyncio.Semaphore(max_concurrency)
        self._max_pending = max_pending
        self._temperature = temperature
        self._seed = seed
        # The event loop keeps only weak references to tasks, so an
        # unreferenced fire-and-forget task can be garbage-collected mid-run.
        self._tasks: set[asyncio.Task[VerificationOutcome]] = set()

    @property
    def pending_count(self) -> int:
        """Verifications in flight or waiting for a slot."""
        return len(self._tasks)

    def verify_in_background(
        self,
        *,
        request_id: str,
        tenant_id: str,
        prompt: str,
        cheap_response: OllamaChatResponse,
        cheap_tier: ComplexityTier,
        classifier_confidence: float,
    ) -> asyncio.Task[VerificationOutcome]:
        """Schedule verification and return the task.

        Callers should NOT await the returned task on the request path: that
        is what would re-introduce blocking. The task is returned mainly so
        tests can wait on completion; the FastAPI layer fires and forgets.
        """
        if len(self._tasks) >= self._max_pending:
            verification = self._skip_overloaded(request_id)
        else:
            verification = self._verify(
                request_id=request_id,
                tenant_id=tenant_id,
                prompt=prompt,
                cheap_response=cheap_response,
                cheap_tier=cheap_tier,
                classifier_confidence=classifier_confidence,
            )
        task = asyncio.create_task(verification, name=f"verifier:{request_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _skip_overloaded(self, request_id: str) -> VerificationOutcome:
        """Log and return SKIPPED_OVERLOADED; a coroutine so callers get a Task either way."""
        logger.warning(
            "verifier.skipped.overloaded",
            request_id=request_id,
            max_pending=self._max_pending,
        )
        return VerificationOutcome.SKIPPED_OVERLOADED

    async def _verify(
        self,
        *,
        request_id: str,
        tenant_id: str,
        prompt: str,
        cheap_response: OllamaChatResponse,
        cheap_tier: ComplexityTier,
        classifier_confidence: float,
    ) -> VerificationOutcome:
        """Replay the prompt on the standard model, score both answers, record a divergence."""
        log = logger.bind(request_id=request_id, cheap_tier=cheap_tier.value)

        if not self._enabled:
            log.debug("verifier.skipped.disabled")
            return VerificationOutcome.SKIPPED_DISABLED

        if cheap_response.model == self._standard_model.ollama_model:
            log.debug("verifier.skipped.standard_tier")
            return VerificationOutcome.SKIPPED_STANDARD_TIER

        try:
            async with self._standard_call_slots:
                standard_response = await self._ollama.chat(
                    model=self._standard_model.ollama_model,
                    messages=[ChatMessage(role="user", content=prompt)],
                    max_output_tokens=self._standard_model.max_output_tokens,
                    temperature=self._temperature,
                    seed=self._seed,
                )
            similarity = await self._scorer.score(cheap_response.content, standard_response.content)
        except Exception as exc:
            log.warning("verifier.failed", error=str(exc), error_type=type(exc).__name__)
            return VerificationOutcome.FAILED

        if similarity >= self._divergence_threshold:
            log.debug("verifier.agreed", similarity=similarity)
            return VerificationOutcome.AGREED

        failure = RoutingFailure(
            request_id=request_id,
            tenant_id=tenant_id,
            prompt=prompt,
            cheap_model=cheap_response.model,
            cheap_response=cheap_response.content,
            cheap_tier=cheap_tier,
            classifier_confidence=classifier_confidence,
            standard_model=standard_response.model,
            standard_response=standard_response.content,
            similarity_score=similarity,
            divergence_threshold=self._divergence_threshold,
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

        log.info("verifier.diverged", similarity=similarity, threshold=self._divergence_threshold)
        return VerificationOutcome.DIVERGED
