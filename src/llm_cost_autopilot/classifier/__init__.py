"""Two-stage zero-shot complexity classifier.

Stage 1 — embedding similarity against curated anchor prompts (fast, ~5 ms).
Stage 2 — DeBERTa-v3-xsmall zero-shot classification (only when stage 1 is
unsure, ~50 ms).

Public surface:
    ComplexityTier        — the three routing tiers
    ClassificationResult  — frozen record of one classification
    TwoStageClassifier    — the orchestrator
    EmbeddingBackend      — Protocol for stage-1 backend (mockable in tests)
    ZeroShotBackend       — Protocol for stage-2 backend (mockable in tests)
"""

from llm_cost_autopilot.classifier.complexity import (
    ClassificationResult,
    ClassifierStage,
    ComplexityTier,
)
from llm_cost_autopilot.classifier.zero_shot import (
    ANCHORS,
    EmbeddingBackend,
    TwoStageClassifier,
    ZeroShotBackend,
)

__all__ = [
    "ANCHORS",
    "ClassificationResult",
    "ClassifierStage",
    "ComplexityTier",
    "EmbeddingBackend",
    "TwoStageClassifier",
    "ZeroShotBackend",
]
