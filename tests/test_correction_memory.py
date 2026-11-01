"""Correction memory: the store, the classifier's escalation rule, and the
verifier-to-memory bridge. Embeddings are small hand-built unit vectors."""

from __future__ import annotations

import numpy as np
import pytest

from llm_cost_autopilot.classifier import ClassifierStage, ComplexityTier, TwoStageClassifier
from llm_cost_autopilot.classifier.complexity import next_tier, tier_rank
from llm_cost_autopilot.classifier.correction_memory import (
    CorrectionMemory,
    Embedding,
    PromptEmbedder,
)
from llm_cost_autopilot.verifier.correction_feedback import (
    FeedbackCorrectionStore,
    load_correction_memory,
)
from llm_cost_autopilot.verifier.feedback import RoutingFailure
from tests.conftest import FakeEmbeddingBackend, FakeZeroShotBackend, unit_vector

MISROUTED = unit_vector(1.0, 0.0, 0.0)
PARAPHRASE = unit_vector(0.95, 0.31, 0.0)  # cosine ~0.95 with MISROUTED
UNRELATED = unit_vector(0.0, 0.0, 1.0)

SIMPLE_ANCHOR_SCORES = {
    ComplexityTier.SIMPLE: 0.85,
    ComplexityTier.MODERATE: 0.10,
}


def _memory_with_correction(tenant_id: str = "tenant-a") -> CorrectionMemory:
    memory = CorrectionMemory(max_per_tenant=10)
    assert memory.add(
        tenant_id=tenant_id,
        request_id="req-misrouted",
        misrouted_tier=ComplexityTier.SIMPLE,
        embedding=MISROUTED,
    )
    return memory


def _classifier(
    memory: CorrectionMemory | None,
    *,
    stage1: dict[str, dict[ComplexityTier, float]],
    embeddings: dict[str, Embedding] | None = None,
    stage2: dict[str, dict[ComplexityTier, float]] | None = None,
    match_threshold: float = 0.80,
) -> tuple[TwoStageClassifier, FakeEmbeddingBackend]:
    embed = FakeEmbeddingBackend(stage1, embeddings)
    clf = TwoStageClassifier(
        embed,
        FakeZeroShotBackend(stage2 or {}),
        stage1_threshold=0.75,
        correction_memory=memory,
        correction_match_threshold=match_threshold,
    )
    return clf, embed


def _failure(
    *,
    request_id: str = "req-1",
    tenant_id: str = "tenant-a",
    prompt: str = "Who painted the Mona Lisa?",
    cheap_model: str = "qwen3.5:0.8b",
    standard_model: str = "qwen3.5:4b",
    cheap_tier: ComplexityTier = ComplexityTier.SIMPLE,
) -> RoutingFailure:
    return RoutingFailure(
        request_id=request_id,
        tenant_id=tenant_id,
        prompt=prompt,
        cheap_model=cheap_model,
        cheap_response="Sorry, I can't assist with that.",
        cheap_tier=cheap_tier,
        classifier_confidence=0.9,
        standard_model=standard_model,
        standard_response="Leonardo da Vinci.",
        similarity_score=0.4,
        divergence_threshold=0.7,
    )


# ── Tier helpers ─────────────────────────────────────────────────────────────
def test_next_tier_climbs_and_stops_at_top() -> None:
    assert next_tier(ComplexityTier.SIMPLE) is ComplexityTier.MODERATE
    assert next_tier(ComplexityTier.MODERATE) is None


def test_tier_rank_follows_cost_order() -> None:
    ranks = [tier_rank(t) for t in ComplexityTier]
    assert ranks == sorted(ranks)


# ── Store ────────────────────────────────────────────────────────────────────
class TestCorrectionMemory:
    def test_rejects_bad_parameters(self) -> None:
        with pytest.raises(ValueError, match="max_per_tenant"):
            CorrectionMemory(max_per_tenant=0)
        with pytest.raises(ValueError, match="duplicate_similarity"):
            CorrectionMemory(max_per_tenant=1, duplicate_similarity=0.0)

    def test_nearest_returns_closest_with_escalation_target(self) -> None:
        memory = _memory_with_correction()
        memory.add(
            tenant_id="tenant-a",
            request_id="req-other",
            misrouted_tier=ComplexityTier.SIMPLE,
            embedding=UNRELATED,
        )
        match = memory.nearest("tenant-a", PARAPHRASE)
        assert match is not None
        assert match.request_id == "req-misrouted"
        assert match.misrouted_tier is ComplexityTier.SIMPLE
        assert match.escalate_to is ComplexityTier.MODERATE
        assert match.similarity == pytest.approx(float(MISROUTED @ PARAPHRASE))

    def test_tenants_are_isolated(self) -> None:
        memory = _memory_with_correction("tenant-a")
        assert memory.nearest("tenant-b", MISROUTED) is None
        assert not memory.has_corrections("tenant-b")
        assert memory.has_corrections("tenant-a")
        assert memory.correction_count("tenant-b") == 0

    def test_skips_near_duplicates(self) -> None:
        memory = _memory_with_correction()
        added = memory.add(
            tenant_id="tenant-a",
            request_id="req-repeat",
            misrouted_tier=ComplexityTier.SIMPLE,
            embedding=MISROUTED,
        )
        assert not added
        assert memory.correction_count("tenant-a") == 1

    def test_skips_top_tier_because_nothing_is_above_it(self) -> None:
        memory = CorrectionMemory(max_per_tenant=10)
        added = memory.add(
            tenant_id="tenant-a",
            request_id="req-complex",
            misrouted_tier=ComplexityTier.MODERATE,
            embedding=MISROUTED,
        )
        assert not added
        assert memory.correction_count() == 0

    def test_cap_drops_oldest_first(self) -> None:
        memory = CorrectionMemory(max_per_tenant=2)
        for request_id, vector in [
            ("req-old", unit_vector(1.0, 0.0, 0.0)),
            ("req-mid", unit_vector(0.0, 1.0, 0.0)),
            ("req-new", unit_vector(0.0, 0.0, 1.0)),
        ]:
            memory.add(
                tenant_id="tenant-a",
                request_id=request_id,
                misrouted_tier=ComplexityTier.SIMPLE,
                embedding=vector,
            )
        assert memory.correction_count("tenant-a") == 2
        match = memory.nearest("tenant-a", unit_vector(1.0, 0.0, 0.0))
        assert match is not None
        assert match.request_id != "req-old"

    def test_size_sums_tenants(self) -> None:
        memory = _memory_with_correction("tenant-a")
        memory.add(
            tenant_id="tenant-b",
            request_id="req-b",
            misrouted_tier=ComplexityTier.SIMPLE,
            embedding=MISROUTED,
        )
        assert memory.correction_count() == 2


# ── Classifier escalation rule ───────────────────────────────────────────────
class TestClassifierEscalation:
    def test_paraphrase_of_a_misroute_is_escalated(self) -> None:
        clf, _ = _classifier(
            _memory_with_correction(),
            stage1={"Who made the Mona Lisa?": SIMPLE_ANCHOR_SCORES},
            embeddings={"Who made the Mona Lisa?": PARAPHRASE},
        )
        result = clf.classify("Who made the Mona Lisa?", tenant_id="tenant-a")
        assert result.tier is ComplexityTier.MODERATE
        assert result.stage is ClassifierStage.CORRECTION_MEMORY
        assert result.matched_correction_request_id == "req-misrouted"
        assert result.confidence == pytest.approx(float(MISROUTED @ PARAPHRASE))

    def test_other_tenant_is_not_escalated(self) -> None:
        clf, embed = _classifier(
            _memory_with_correction("tenant-a"),
            stage1={"Who made the Mona Lisa?": SIMPLE_ANCHOR_SCORES},
            embeddings={"Who made the Mona Lisa?": PARAPHRASE},
        )
        result = clf.classify("Who made the Mona Lisa?", tenant_id="tenant-b")
        assert result.tier is ComplexityTier.SIMPLE
        assert result.matched_correction_request_id is None
        # No stored correction for tenant-b, so the prompt is never embedded.
        assert embed.embed_calls == []

    def test_below_match_threshold_is_not_escalated(self) -> None:
        clf, _ = _classifier(
            _memory_with_correction(),
            stage1={"Who made the Mona Lisa?": SIMPLE_ANCHOR_SCORES},
            embeddings={"Who made the Mona Lisa?": PARAPHRASE},
            match_threshold=0.99,
        )
        assert clf.classify("Who made the Mona Lisa?", tenant_id="tenant-a").tier is (
            ComplexityTier.SIMPLE
        )

    def test_closer_to_a_cheap_anchor_than_to_the_correction_is_not_escalated(self) -> None:
        scores = {**SIMPLE_ANCHOR_SCORES, ComplexityTier.SIMPLE: 0.97}
        clf, _ = _classifier(
            _memory_with_correction(),
            stage1={"What is the capital of France?": scores},
            embeddings={"What is the capital of France?": PARAPHRASE},
        )
        result = clf.classify("What is the capital of France?", tenant_id="tenant-a")
        assert result.tier is ComplexityTier.SIMPLE
        assert result.stage is ClassifierStage.STAGE_1_EMBEDDING

    def test_never_downgrades_a_higher_tier(self) -> None:
        prompt = "Design a multi-region failover plan."
        clf, _ = _classifier(
            _memory_with_correction(),
            stage1={
                prompt: {
                    ComplexityTier.SIMPLE: 0.10,
                    ComplexityTier.MODERATE: 0.90,
                }
            },
            embeddings={prompt: PARAPHRASE},
        )
        result = clf.classify(prompt, tenant_id="tenant-a")
        assert result.tier is ComplexityTier.MODERATE
        assert result.stage is ClassifierStage.STAGE_1_EMBEDDING

    def test_applies_after_stage_two(self) -> None:
        prompt = "Who made that famous portrait?"
        unsure = {
            ComplexityTier.SIMPLE: 0.60,
            ComplexityTier.MODERATE: 0.20,
        }
        clf, _ = _classifier(
            _memory_with_correction(),
            stage1={prompt: unsure},
            embeddings={prompt: PARAPHRASE},
            stage2={
                prompt: {
                    ComplexityTier.SIMPLE: 0.7,
                    ComplexityTier.MODERATE: 0.1,
                }
            },
        )
        result = clf.classify(prompt, tenant_id="tenant-a")
        assert result.tier is ComplexityTier.MODERATE
        assert result.stage is ClassifierStage.CORRECTION_MEMORY

    def test_without_memory_the_prompt_is_never_embedded(self) -> None:
        clf, embed = _classifier(
            None,
            stage1={"Who made the Mona Lisa?": SIMPLE_ANCHOR_SCORES},
        )
        assert clf.classify("Who made the Mona Lisa?").tier is ComplexityTier.SIMPLE
        assert embed.embed_calls == []

    def test_rejects_bad_match_threshold(self) -> None:
        with pytest.raises(ValueError, match="correction_match_threshold"):
            _classifier(CorrectionMemory(max_per_tenant=1), stage1={}, match_threshold=1.5)


# ── Verifier-to-memory bridge ────────────────────────────────────────────────
class _Ledger:
    def __init__(self, *, raises: Exception | None = None) -> None:
        self.raises = raises
        self.failures: list[RoutingFailure] = []

    async def record_failure(self, failure: RoutingFailure) -> None:
        if self.raises is not None:
            raise self.raises
        self.failures.append(failure)


class _Embedder:
    def __init__(self, vectors: dict[str, Embedding]) -> None:
        self.vectors = vectors

    def embed(self, prompt: str) -> Embedding:
        return self.vectors[prompt]


class TestFeedbackCorrectionStore:
    def test_embedder_satisfies_protocol(self) -> None:
        assert isinstance(_Embedder({}), PromptEmbedder)

    async def test_persists_then_makes_the_correction_routable(self) -> None:
        ledger = _Ledger()
        memory = CorrectionMemory(max_per_tenant=10)
        store = FeedbackCorrectionStore(
            ledger, memory, _Embedder({"Who painted the Mona Lisa?": MISROUTED})
        )
        await store.record_failure(_failure())
        assert [f.request_id for f in ledger.failures] == ["req-1"]
        match = memory.nearest("tenant-a", PARAPHRASE)
        assert match is not None
        assert match.request_id == "req-1"

    async def test_ledger_failure_leaves_memory_untouched(self) -> None:
        memory = CorrectionMemory(max_per_tenant=10)
        store = FeedbackCorrectionStore(
            _Ledger(raises=RuntimeError("db down")),
            memory,
            _Embedder({"Who painted the Mona Lisa?": MISROUTED}),
        )
        with pytest.raises(RuntimeError, match="db down"):
            await store.record_failure(_failure())
        assert memory.correction_count() == 0


class TestLoadCorrectionMemory:
    def test_loads_only_the_current_model_pair(self) -> None:
        memory = CorrectionMemory(max_per_tenant=10)
        embedder = _Embedder(
            {
                "current": unit_vector(1.0, 0.0, 0.0),
                "old cheap model": unit_vector(0.0, 1.0, 0.0),
                "old standard model": unit_vector(0.0, 0.0, 1.0),
            }
        )
        kept = load_correction_memory(
            memory,
            embedder,
            [
                _failure(request_id="r1", prompt="current"),
                _failure(request_id="r2", prompt="old cheap model", cheap_model="qwen2:0.5b"),
                _failure(request_id="r3", prompt="old standard model", standard_model="llama3:8b"),
            ],
            cheap_model="qwen3.5:0.8b",
            standard_model="qwen3.5:4b",
        )
        assert kept == 1
        assert memory.correction_count("tenant-a") == 1

    def test_counts_only_corrections_actually_added(self) -> None:
        memory = CorrectionMemory(max_per_tenant=10)
        embedder = _Embedder({"same": np.asarray(MISROUTED)})
        kept = load_correction_memory(
            memory,
            embedder,
            [_failure(request_id="r1", prompt="same"), _failure(request_id="r2", prompt="same")],
            cheap_model="qwen3.5:0.8b",
            standard_model="qwen3.5:4b",
        )
        assert kept == 1
