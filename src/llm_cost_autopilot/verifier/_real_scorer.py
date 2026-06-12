"""Production semantic-similarity scorer backed by sentence-transformers.

Lazily imports the heavy ML stack so tests — which substitute a fake scorer
— never have to install it. Excluded from pyright/coverage; same pattern as
``classifier/_real_backends.py``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    import numpy as np
    from numpy.typing import NDArray

    NDArrayF32 = NDArray[np.float32]


class EmbeddingSimilarityScorer:
    """Cosine similarity between sentence embeddings, scaled into [0, 1].

    Cosine similarity is in [-1, 1]; we map it via ``(s + 1) / 2`` so the
    verifier's threshold semantics ("higher is more similar") line up with
    the rest of the codebase.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    async def score(self, a: str, b: str) -> float:
        # encode() is CPU-bound — offload so the event loop keeps turning.
        return await asyncio.to_thread(self._score_sync, a, b)

    def _score_sync(self, a: str, b: str) -> float:
        import numpy as np

        embs = cast(
            "NDArrayF32",
            self._model.encode([a, b], normalize_embeddings=True),
        )
        cosine = float(np.dot(embs[0], embs[1]))
        return (cosine + 1.0) / 2.0
