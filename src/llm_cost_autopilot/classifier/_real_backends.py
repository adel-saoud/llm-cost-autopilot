"""Production backends that wrap real HuggingFace models.

Imports of ``sentence_transformers`` and ``transformers`` are performed lazily
inside the constructors so the test suite — which substitutes fake backends —
never has to install or download the heavy ML stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.classifier.zero_shot import (
    ANCHORS,
    TIER_HYPOTHESIS_LABELS,
)

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    import numpy as np
    from numpy.typing import NDArray

    NDArrayF32 = NDArray[np.float32]


class SentenceTransformerEmbeddingBackend:
    """Stage-1 backend backed by ``sentence-transformers``."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # Pre-encode anchors once — they never change at runtime.
        self._anchor_embeddings: dict[ComplexityTier, NDArrayF32] = {
            tier: cast(
                "NDArrayF32",
                self._model.encode(list(prompts), normalize_embeddings=True),
            )
            for tier, prompts in ANCHORS.items()
        }

    def score(self, prompt: str) -> dict[ComplexityTier, float]:
        import numpy as np

        prompt_emb = cast(
            "NDArrayF32",
            self._model.encode([prompt], normalize_embeddings=True),
        )
        scores: dict[ComplexityTier, float] = {}
        for tier, anchor_embs in self._anchor_embeddings.items():
            # Both sides are L2-normalised so cosine == dot product.
            sims = anchor_embs @ prompt_emb.T
            scores[tier] = float(np.max(sims))
        return scores


class DebertaZeroShotBackend:
    """Stage-2 backend backed by a HuggingFace zero-shot pipeline."""

    HYPOTHESIS_TEMPLATE = "This request requires {}."

    def __init__(
        self,
        model_name: str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    ) -> None:
        from transformers import pipeline

        self._pipeline = pipeline("zero-shot-classification", model=model_name)
        # Maintain a fixed label order so we can map results back to tiers.
        self._label_to_tier: dict[str, ComplexityTier] = {
            label: tier for tier, label in TIER_HYPOTHESIS_LABELS.items()
        }

    def classify(self, prompt: str) -> dict[ComplexityTier, float]:
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
