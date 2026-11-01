"""Production classifier backends that wrap HuggingFace models.

``sentence_transformers`` and ``transformers`` come from the optional ``[ml]``
extra and are imported inside the constructors, so the test suite, which
substitutes fake backends, never installs or downloads the ML stack. The
module is excluded from pyright and coverage for the same reason.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.classifier.correction_memory import Embedding
from llm_cost_autopilot.classifier.zero_shot import ANCHOR_PROMPTS, TIER_HYPOTHESIS_LABELS


class SentenceTransformerEmbeddingBackend:
    """Stage-1 backend: cosine similarity to the anchor prompts with ``sentence-transformers``."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._anchor_embeddings: dict[ComplexityTier, NDArray[np.float32]] = {
            tier: cast(
                "NDArray[np.float32]",
                self._model.encode(list(prompts), normalize_embeddings=True),
            )
            for tier, prompts in ANCHOR_PROMPTS.items()
        }

        # score() and embed() run on the same prompt when the correction
        # memory is active; caching the encoding keeps that to one forward pass.
        self._encode_cached = lru_cache(maxsize=1024)(self._encode)

    def _encode(self, prompt: str) -> Embedding:
        return cast(
            "NDArray[np.float32]",
            self._model.encode([prompt], normalize_embeddings=True),
        )[0]

    def embed(self, prompt: str) -> Embedding:
        """Return the prompt's L2-normalised embedding, shared with ``score``'s cache."""
        return self._encode_cached(prompt)

    def score(self, prompt: str) -> dict[ComplexityTier, float]:
        """Return the prompt's best cosine similarity to each tier's anchors."""
        prompt_embedding = self._encode_cached(prompt)
        # Both sides are L2-normalised, so the dot product is the cosine similarity.
        return {
            tier: float(np.max(anchor_embeddings @ prompt_embedding))
            for tier, anchor_embeddings in self._anchor_embeddings.items()
        }


class DebertaZeroShotBackend:
    """Stage-2 backend: a ``transformers`` zero-shot-classification (NLI) pipeline."""

    HYPOTHESIS_TEMPLATE = "This request requires {}."
    """Filled with each ``TIER_HYPOTHESIS_LABELS`` value to form the NLI hypothesis."""

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self._pipeline = pipeline("zero-shot-classification", model=model_name)
        self._label_to_tier: dict[str, ComplexityTier] = {
            label: tier for tier, label in TIER_HYPOTHESIS_LABELS.items()
        }

    def classify(self, prompt: str) -> dict[ComplexityTier, float]:
        """Return the pipeline's probability for each tier label (they sum to 1)."""
        result = cast(
            "dict[str, Any]",
            self._pipeline(
                prompt,
                candidate_labels=list(self._label_to_tier.keys()),
                hypothesis_template=self.HYPOTHESIS_TEMPLATE,
            ),
        )
        labels = cast("list[str]", result["labels"])
        scores = cast("list[float]", result["scores"])
        return {
            self._label_to_tier[label]: float(score)
            for label, score in zip(labels, scores, strict=True)
        }
