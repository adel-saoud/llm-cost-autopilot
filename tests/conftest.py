"""Shared pytest fixtures.

Heavy ML models are never loaded in the test suite. Each test that needs a
classifier wires in a deterministic fake backend defined here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pytest

from llm_cost_autopilot.classifier import (
    ComplexityTier,
    EmbeddingBackend,
    TwoStageClassifier,
    ZeroShotBackend,
)


class FakeEmbeddingBackend:
    """Stage-1 fake — returns hard-coded per-prompt scores."""

    def __init__(self, scores_by_prompt: Mapping[str, Mapping[ComplexityTier, float]]) -> None:
        self._scores = {k: dict(v) for k, v in scores_by_prompt.items()}
        self.calls: list[str] = []

    def score(self, prompt: str) -> dict[ComplexityTier, float]:
        self.calls.append(prompt)
        if prompt not in self._scores:
            raise KeyError(f"FakeEmbeddingBackend has no score for prompt: {prompt!r}")
        return dict(self._scores[prompt])


class FakeZeroShotBackend:
    """Stage-2 fake — returns hard-coded per-prompt label probabilities."""

    def __init__(self, scores_by_prompt: Mapping[str, Mapping[ComplexityTier, float]]) -> None:
        self._scores = {k: dict(v) for k, v in scores_by_prompt.items()}
        self.calls: list[str] = []

    def classify(self, prompt: str) -> dict[ComplexityTier, float]:
        self.calls.append(prompt)
        if prompt not in self._scores:
            raise KeyError(f"FakeZeroShotBackend has no score for prompt: {prompt!r}")
        return dict(self._scores[prompt])


ClassifierTriple = tuple[TwoStageClassifier, FakeEmbeddingBackend, FakeZeroShotBackend]


class MakeClassifier(Protocol):
    """Typed signature for the ``make_classifier`` fixture factory."""

    def __call__(
        self,
        stage1: Mapping[str, Mapping[ComplexityTier, float]] | None = ...,
        stage2: Mapping[str, Mapping[ComplexityTier, float]] | None = ...,
        threshold: float = ...,
    ) -> ClassifierTriple: ...


@pytest.fixture
def make_classifier() -> MakeClassifier:
    """Return a factory: build a TwoStageClassifier from per-prompt score maps."""

    def _factory(
        stage1: Mapping[str, Mapping[ComplexityTier, float]] | None = None,
        stage2: Mapping[str, Mapping[ComplexityTier, float]] | None = None,
        threshold: float = 0.75,
    ) -> ClassifierTriple:
        embed = FakeEmbeddingBackend(stage1 or {})
        zero_shot = FakeZeroShotBackend(stage2 or {})
        # The Protocol check is structural; the assertion documents intent.
        assert isinstance(embed, EmbeddingBackend)
        assert isinstance(zero_shot, ZeroShotBackend)
        clf = TwoStageClassifier(embed, zero_shot, stage1_threshold=threshold)
        return clf, embed, zero_shot

    return _factory
