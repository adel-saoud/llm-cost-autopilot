"""Connects the verifier's corrections to the classifier's correction memory.

``FeedbackCorrectionStore`` writes each correction to the ledger and then adds
it to the in-memory index the classifier searches, so the next similar prompt
is routed one tier up. ``load_correction_memory`` rebuilds that index from the
ledger at startup.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import structlog

from llm_cost_autopilot.classifier.correction_memory import CorrectionMemory, PromptEmbedder
from llm_cost_autopilot.verifier.async_verifier import CorrectionStore
from llm_cost_autopilot.verifier.feedback import RoutingFailure

logger = structlog.get_logger(__name__)


class FeedbackCorrectionStore:
    """``CorrectionStore`` that persists a correction, then makes it routable.

    The ledger write comes first: if it fails, the verifier reports FAILED
    and the memory stays consistent with what is on disk.
    """

    def __init__(
        self,
        ledger: CorrectionStore,
        memory: CorrectionMemory,
        embedder: PromptEmbedder,
    ) -> None:
        self._ledger = ledger
        self._memory = memory
        self._embedder = embedder

    async def record_failure(self, failure: RoutingFailure) -> None:
        """Write ``failure`` to the ledger, then add its prompt embedding to the memory."""
        await self._ledger.record_failure(failure)
        # CPU-bound encoding: a worker thread keeps the event loop serving.
        embedding = await asyncio.to_thread(self._embedder.embed, failure.prompt)
        added = self._memory.add(
            tenant_id=failure.tenant_id,
            request_id=failure.request_id,
            misrouted_tier=failure.cheap_tier,
            embedding=embedding,
        )
        logger.info(
            "correction_memory.recorded",
            request_id=failure.request_id,
            tenant_id=failure.tenant_id,
            added=added,
            tenant_correction_count=self._memory.correction_count(failure.tenant_id),
        )


def load_correction_memory(
    memory: CorrectionMemory,
    embedder: PromptEmbedder,
    failures: Iterable[RoutingFailure],
    *,
    cheap_model: str,
    standard_model: str,
) -> int:
    """Add stored corrections to ``memory`` and return how many were kept.

    Only corrections made by the currently configured model pair are loaded:
    a mistake by an older cheap model says nothing about the one serving now.
    ``failures`` must be ordered oldest first so the per-tenant cap keeps the
    most recent ones.
    """
    kept = 0
    for failure in failures:
        if failure.cheap_model != cheap_model or failure.standard_model != standard_model:
            continue
        if memory.add(
            tenant_id=failure.tenant_id,
            request_id=failure.request_id,
            misrouted_tier=failure.cheap_tier,
            embedding=embedder.embed(failure.prompt),
        ):
            kept += 1
    logger.info("correction_memory.loaded", kept=kept, correction_count=memory.correction_count())
    return kept
