"""Per-tenant nearest-neighbour memory of past routing mistakes.

The verifier records a correction when the cheap model's answer diverged from
the standard model's. This memory keeps the embeddings of those prompts so the
classifier can send close paraphrases of a past mistake one tier up, instead
of repeating it. Corrections are scoped per tenant: one tenant's traffic never
changes another tenant's routing.

Brute-force cosine search is deliberate. With a few hundred normalised
vectors per tenant, one matrix-vector product takes well under a millisecond,
so an approximate index would add a dependency without a measurable gain.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from llm_cost_autopilot.classifier.complexity import ComplexityTier, next_tier

Embedding = NDArray[np.float32]
"""A 1-D, L2-normalised sentence embedding, so cosine similarity is a dot product."""

DEFAULT_TENANT_ID = "default"
"""Tenant used when the caller does not name one (single-tenant deployments)."""


@runtime_checkable
class PromptEmbedder(Protocol):
    """Turns a prompt into an L2-normalised embedding.

    Narrower than the classifier's ``EmbeddingBackend`` so the feedback code
    depends only on ``embed`` and tests can pass a plain lookup table.
    """

    def embed(self, prompt: str) -> Embedding: ...


@dataclass(frozen=True)
class CorrectionMatch:
    """The stored correction closest to a prompt."""

    request_id: str
    """Request id of the routing decision the verifier disagreed with."""
    misrouted_tier: ComplexityTier
    """Tier the original prompt was sent to."""
    escalate_to: ComplexityTier
    """Tier a similar prompt should be sent to instead."""
    similarity: float
    """Cosine similarity between the new prompt and the stored one."""


@dataclass(frozen=True)
class _CorrectionEntry:
    """Metadata for one row of a tenant's embedding matrix (same index)."""

    request_id: str
    misrouted_tier: ComplexityTier
    escalate_to: ComplexityTier


class _TenantIndex:
    """One tenant's corrections: ``entries[i]`` describes ``matrix[i]``."""

    def __init__(self, dimension: int) -> None:
        self.entries: list[_CorrectionEntry] = []
        self.matrix: NDArray[np.float32] = np.empty((0, dimension), dtype=np.float32)


class CorrectionMemory:
    """Bounded, per-tenant store of correction embeddings.

    Parameters
    ----------
    max_per_tenant:
        Most recent corrections kept per tenant. The oldest is dropped first,
        which bounds memory, search time and the influence of stale mistakes.
    duplicate_similarity:
        A new correction this close to a stored one is skipped. Repeated
        traffic would otherwise fill the memory with copies of one prompt.
    """

    def __init__(self, *, max_per_tenant: int, duplicate_similarity: float = 0.98) -> None:
        if max_per_tenant < 1:
            raise ValueError("max_per_tenant must be >= 1")
        if not 0.0 < duplicate_similarity <= 1.0:
            raise ValueError("duplicate_similarity must be in (0, 1]")
        self._max_per_tenant = max_per_tenant
        self._duplicate_similarity = duplicate_similarity
        self._tenants: dict[str, _TenantIndex] = {}
        # Classification runs in worker threads while the verifier adds
        # corrections from the event loop, so reads and writes share a lock.
        self._lock = threading.Lock()

    def add(
        self,
        *,
        tenant_id: str,
        request_id: str,
        misrouted_tier: ComplexityTier,
        embedding: Embedding,
    ) -> bool:
        """Store one correction. Returns False when it was skipped.

        Skipped when the misrouted tier is already the most capable one
        (there is nowhere to escalate to) or when a near-identical prompt is
        already stored for this tenant.
        """
        escalate_to = next_tier(misrouted_tier)
        if escalate_to is None:
            return False
        vector = embedding.astype(np.float32, copy=False).reshape(1, -1)
        with self._lock:
            index = self._tenants.get(tenant_id)
            if index is None:
                index = _TenantIndex(vector.shape[1])
                self._tenants[tenant_id] = index
            if index.entries:
                similarities = index.matrix @ vector[0]
                if float(np.max(similarities)) >= self._duplicate_similarity:
                    return False
            index.entries.append(_CorrectionEntry(request_id, misrouted_tier, escalate_to))
            index.matrix = np.vstack([index.matrix, vector])
            if len(index.entries) > self._max_per_tenant:
                index.entries = index.entries[1:]
                index.matrix = index.matrix[1:]
        return True

    def has_corrections(self, tenant_id: str) -> bool:
        """Whether ``tenant_id`` has any stored correction (lets callers skip embedding)."""
        with self._lock:
            index = self._tenants.get(tenant_id)
            return index is not None and bool(index.entries)

    def nearest(self, tenant_id: str, embedding: Embedding) -> CorrectionMatch | None:
        """Return the stored correction closest to ``embedding``, or None if there is none."""
        with self._lock:
            index = self._tenants.get(tenant_id)
            if index is None or not index.entries:
                return None
            similarities = index.matrix @ embedding.astype(np.float32, copy=False)
            best = int(np.argmax(similarities))
            entry = index.entries[best]
            similarity = float(similarities[best])
        return CorrectionMatch(
            request_id=entry.request_id,
            misrouted_tier=entry.misrouted_tier,
            escalate_to=entry.escalate_to,
            similarity=similarity,
        )

    def correction_count(self, tenant_id: str | None = None) -> int:
        """Number of stored corrections for one tenant, or across all tenants when None."""
        with self._lock:
            if tenant_id is not None:
                index = self._tenants.get(tenant_id)
                return len(index.entries) if index is not None else 0
            return sum(len(index.entries) for index in self._tenants.values())
