"""Unit tests for the two-stage classifier and its result types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_cost_autopilot.classifier import (
    ANCHOR_PROMPTS,
    ClassificationResult,
    ClassifierStage,
    ComplexityTier,
    TwoStageClassifier,
)
from llm_cost_autopilot.classifier.zero_shot import (
    TIER_HYPOTHESIS_LABELS,
    argmax_tier,
)
from tests.conftest import MakeClassifier


class TestComplexityTier:
    def test_tiers_are_string_enum(self) -> None:
        assert ComplexityTier.SIMPLE == "simple"
        assert ComplexityTier.MODERATE == "moderate"

    def test_anchors_cover_every_tier(self) -> None:
        assert set(ANCHOR_PROMPTS.keys()) == set(ComplexityTier)
        for tier, prompts in ANCHOR_PROMPTS.items():
            assert len(prompts) >= 3, f"{tier} needs at least 3 anchors for stability"

    def test_hypothesis_labels_cover_every_tier(self) -> None:
        assert set(TIER_HYPOTHESIS_LABELS.keys()) == set(ComplexityTier)


class TestClassificationResult:
    def test_frozen_rejects_mutation(self) -> None:
        result = ClassificationResult(
            tier=ComplexityTier.SIMPLE,
            confidence=0.9,
            stage=ClassifierStage.STAGE_1_EMBEDDING,
            latency_ms=4.2,
        )
        with pytest.raises(ValidationError, match="frozen"):
            result.confidence = 0.5  # type: ignore[misc]

    def test_confidence_must_be_in_range(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ClassificationResult(
                tier=ComplexityTier.SIMPLE,
                confidence=1.5,
                stage=ClassifierStage.STAGE_1_EMBEDDING,
                latency_ms=1.0,
            )


class TestArgmaxTier:
    def test_picks_highest_score(self) -> None:
        scores = {
            ComplexityTier.SIMPLE: 0.2,
            ComplexityTier.MODERATE: 0.8,
        }
        tier, score = argmax_tier(scores)
        assert tier is ComplexityTier.MODERATE
        assert score == 0.8

    def test_tie_prefers_cheaper_tier(self) -> None:
        scores = {
            ComplexityTier.SIMPLE: 0.5,
            ComplexityTier.MODERATE: 0.5,
        }
        tier, _ = argmax_tier(scores)
        assert tier is ComplexityTier.SIMPLE

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one tier"):
            argmax_tier({})


class TestTwoStageClassifier:
    def test_rejects_bad_threshold(self) -> None:
        from tests.conftest import FakeEmbeddingBackend, FakeZeroShotBackend

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            TwoStageClassifier(
                FakeEmbeddingBackend({}),
                FakeZeroShotBackend({}),
                stage1_threshold=1.5,
            )

    def test_stage1_accepts_confident_simple_prompt(self, make_classifier: MakeClassifier) -> None:
        clf, embed, zero_shot = make_classifier(
            stage1={
                "Capital of France?": {
                    ComplexityTier.SIMPLE: 0.92,
                    ComplexityTier.MODERATE: 0.10,
                },
            },
            threshold=0.75,
        )
        result = clf.classify("Capital of France?")

        assert result.tier is ComplexityTier.SIMPLE
        assert result.confidence == pytest.approx(0.92)
        assert result.stage is ClassifierStage.STAGE_1_EMBEDDING
        assert result.latency_ms >= 0.0
        assert embed.calls == ["Capital of France?"]
        assert zero_shot.calls == []  # stage 2 not invoked

    def test_escalates_to_stage2_when_unsure(self, make_classifier: MakeClassifier) -> None:
        prompt = "Help me think about pricing tiers."
        clf, embed, zero_shot = make_classifier(
            stage1={
                prompt: {
                    ComplexityTier.SIMPLE: 0.40,
                    ComplexityTier.MODERATE: 0.50,
                },
            },
            stage2={
                prompt: {
                    ComplexityTier.SIMPLE: 0.18,
                    ComplexityTier.MODERATE: 0.82,
                },
            },
            threshold=0.75,
        )
        result = clf.classify(prompt)

        assert result.tier is ComplexityTier.MODERATE
        assert result.confidence == pytest.approx(0.82)
        assert result.stage is ClassifierStage.STAGE_2_ZERO_SHOT
        assert embed.calls == [prompt]
        assert zero_shot.calls == [prompt]

    def test_stage1_at_exactly_threshold_does_not_escalate(
        self, make_classifier: MakeClassifier
    ) -> None:
        prompt = "borderline"
        clf, _, zero_shot = make_classifier(
            stage1={
                prompt: {
                    ComplexityTier.SIMPLE: 0.75,
                    ComplexityTier.MODERATE: 0.05,
                },
            },
            threshold=0.75,
        )
        result = clf.classify(prompt)

        assert result.stage is ClassifierStage.STAGE_1_EMBEDDING
        assert zero_shot.calls == []

    def test_empty_prompt_short_circuits_to_simple(self, make_classifier: MakeClassifier) -> None:
        clf, embed, zero_shot = make_classifier()

        result = clf.classify("   ")
        assert result.tier is ComplexityTier.SIMPLE
        assert result.confidence == 1.0
        assert embed.calls == []
        assert zero_shot.calls == []

    def test_reasoning_prompt_routes_to_moderate(self, make_classifier: MakeClassifier) -> None:
        prompt = "Architect a multi-region failover plan and weigh the tradeoffs."
        clf, _, _ = make_classifier(
            stage1={
                prompt: {
                    ComplexityTier.SIMPLE: 0.10,
                    ComplexityTier.MODERATE: 0.88,
                },
            },
        )
        result = clf.classify(prompt)
        assert result.tier is ComplexityTier.MODERATE
        assert result.stage is ClassifierStage.STAGE_1_EMBEDDING

    def test_scores_dict_carries_decision_basis(self, make_classifier: MakeClassifier) -> None:
        prompt = "summarise"
        clf, _, _ = make_classifier(
            stage1={
                prompt: {
                    ComplexityTier.SIMPLE: 0.20,
                    ComplexityTier.MODERATE: 0.91,
                },
            },
        )
        result = clf.classify(prompt)
        # Stage 1 decided, so scores should reflect stage 1's scoring.
        assert result.scores[ComplexityTier.MODERATE] == pytest.approx(0.91)
        assert set(result.scores.keys()) == set(ComplexityTier)
