"""Production semantic-similarity scorer backed by sentence-transformers.

``sentence_transformers`` comes from the optional ``[ml]`` extra and is
imported inside the constructor, so tests, which substitute a fake scorer,
never install it. Excluded from pyright and coverage like
``classifier/huggingface_backends.py``.
"""

from __future__ import annotations

import asyncio
from typing import cast

import numpy as np
from numpy.typing import NDArray


class EmbeddingSimilarityScorer:
    """Cosine similarity between sentence embeddings, scaled into [0, 1].

    Cosine similarity is in [-1, 1]; ``(cosine + 1) / 2`` maps it onto the
    [0, 1] range that ``AUTOPILOT_VERIFIER_DIVERGENCE_THRESHOLD`` is expressed in.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    async def score(self, first_text: str, second_text: str) -> float:
        """Return the scaled similarity; encoding runs in a worker thread (CPU-bound)."""
        return await asyncio.to_thread(self._score_sync, first_text, second_text)

    def _score_sync(self, first_text: str, second_text: str) -> float:
        embeddings = cast(
            "NDArray[np.float32]",
            self._model.encode([first_text, second_text], normalize_embeddings=True),
        )
        cosine = float(np.dot(embeddings[0], embeddings[1]))
        return (cosine + 1.0) / 2.0
