"""Tests for the async quality verifier and the feedback loop.

The OllamaClient, scorer, and correction store are all fakes: no daemon,
no model downloads.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.router.models import ModelConfig
from llm_cost_autopilot.router.ollama import OllamaChatResponse, OllamaClient
from llm_cost_autopilot.verifier import (
    CorrectionStore,
    RoutingFailure,
    SimilarityScorer,
    VerificationOutcome,
    Verifier,
)

# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeScorer:
    def __init__(self, score: float, *, raises: Exception | None = None) -> None:
        self._score = score
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    async def score(self, first_text: str, second_text: str) -> float:
        self.calls.append((first_text, second_text))
        if self._raises is not None:
            raise self._raises
        return self._score


class FakeStore:
    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises
        self.failures: list[RoutingFailure] = []

    async def record_failure(self, failure: RoutingFailure) -> None:
        if self._raises is not None:
            raise self._raises
        self.failures.append(failure)


def _standard_cfg(model: str = "qwen3.5:4b") -> ModelConfig:
    return ModelConfig(
        tier=ComplexityTier.MODERATE,
        ollama_model=model,
        usd_per_1m_input_tokens=3.0,
        usd_per_1m_output_tokens=15.0,
        max_output_tokens=512,
    )


def _cheap_response(content: str = "Paris.", model: str = "qwen3.5:0.8b") -> OllamaChatResponse:
    return OllamaChatResponse(
        model=model,
        content=content,
        prompt_tokens=5,
        completion_tokens=2,
        total_duration_ns=1_000_000,
        latency_ms=12.3,
    )


def _ollama_client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    return OllamaClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x.test"),
        max_attempts=1,
    )


def _ollama_returns(
    content: str = "The capital of France is Paris.",
    model: str = "qwen3.5:4b",
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": model,
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 8,
                "total_duration": 1,
            },
        )

    return handler


# ── Protocol structural checks ──────────────────────────────────────────────
def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeScorer(0.9), SimilarityScorer)
    assert isinstance(FakeStore(), CorrectionStore)


# ── Threshold + construction ────────────────────────────────────────────────
def test_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Verifier(
            ollama=_ollama_client(_ollama_returns()),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.5),
            store=FakeStore(),
            divergence_threshold=1.5,
        )


# ── Behaviour ────────────────────────────────────────────────────────────────
class TestVerifyInBackground:
    async def test_skipped_when_disabled(self) -> None:
        store = FakeStore()
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns()),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.5),
            store=store,
            divergence_threshold=0.7,
            enabled=False,
        )
        outcome = await _run_verifier(verifier, _cheap_response())
        assert outcome is VerificationOutcome.SKIPPED_DISABLED
        assert store.failures == []

    async def test_skipped_when_already_on_standard_model(self) -> None:
        scorer = FakeScorer(0.5)
        store = FakeStore()
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns()),
            standard_model=_standard_cfg("qwen3.5:4b"),
            scorer=scorer,
            store=store,
            divergence_threshold=0.7,
        )
        outcome = await _run_verifier(verifier, _cheap_response(model="qwen3.5:4b"))
        assert outcome is VerificationOutcome.SKIPPED_STANDARD_TIER
        assert scorer.calls == []
        assert store.failures == []

    async def test_agreed_when_similarity_at_or_above_threshold(self) -> None:
        store = FakeStore()
        scorer = FakeScorer(0.85)
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns(content="Paris is the capital.")),
            standard_model=_standard_cfg(),
            scorer=scorer,
            store=store,
            divergence_threshold=0.70,
        )
        outcome = await _run_verifier(verifier, _cheap_response("Paris."))
        assert outcome is VerificationOutcome.AGREED
        assert store.failures == []
        assert scorer.calls == [("Paris.", "Paris is the capital.")]

    async def test_diverged_records_failure_with_full_context(self) -> None:
        store = FakeStore()
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns(content="The Eiffel Tower is in Paris.")),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.30),
            store=store,
            divergence_threshold=0.70,
        )
        outcome = await _run_verifier(verifier, _cheap_response("Paris."))
        assert outcome is VerificationOutcome.DIVERGED
        assert len(store.failures) == 1

        failure = store.failures[0]
        assert failure.request_id == "req-1"
        assert failure.cheap_response == "Paris."
        assert failure.standard_response == "The Eiffel Tower is in Paris."
        assert failure.cheap_tier is ComplexityTier.SIMPLE
        assert failure.similarity_score == pytest.approx(0.30)
        assert failure.divergence_threshold == pytest.approx(0.70)

    async def test_ollama_failure_returns_failed_outcome_not_raises(self) -> None:
        def broken_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        store = FakeStore()
        verifier = Verifier(
            ollama=_ollama_client(broken_handler),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.9),
            store=store,
            divergence_threshold=0.70,
        )
        outcome = await _run_verifier(verifier, _cheap_response())
        assert outcome is VerificationOutcome.FAILED
        assert store.failures == []  # nothing persisted on transport failure

    async def test_store_failure_returns_failed_outcome_not_raises(self) -> None:
        store = FakeStore(raises=RuntimeError("db down"))
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns()),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.1),  # would diverge
            store=store,
            divergence_threshold=0.70,
        )
        outcome = await _run_verifier(verifier, _cheap_response())
        assert outcome is VerificationOutcome.FAILED

    async def test_scorer_failure_returns_failed_outcome_not_raises(self) -> None:
        store = FakeStore()
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns()),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.0, raises=RuntimeError("embedding broke")),
            store=store,
            divergence_threshold=0.70,
        )
        outcome = await _run_verifier(verifier, _cheap_response())
        assert outcome is VerificationOutcome.FAILED
        assert store.failures == []

    async def test_threshold_boundary_is_inclusive(self) -> None:
        store = FakeStore()
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns()),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.70),  # exactly threshold
            store=store,
            divergence_threshold=0.70,
        )
        outcome = await _run_verifier(verifier, _cheap_response())
        assert outcome is VerificationOutcome.AGREED


async def _run_verifier(verifier: Verifier, cheap: OllamaChatResponse) -> VerificationOutcome:
    """Drive ``verify_in_background`` to completion for assertions."""
    task = verifier.verify_in_background(
        request_id="req-1",
        tenant_id="tenant-a",
        prompt="What is the capital of France?",
        cheap_response=cheap,
        cheap_tier=ComplexityTier.SIMPLE,
        classifier_confidence=0.82,
    )
    return await task


# ── Feedback type ────────────────────────────────────────────────────────────
class TestRoutingFailure:
    def test_is_frozen(self) -> None:
        failure = RoutingFailure(
            request_id="x",
            prompt="p",
            cheap_model="a",
            cheap_response="b",
            cheap_tier=ComplexityTier.SIMPLE,
            classifier_confidence=0.5,
            standard_model="c",
            standard_response="d",
            similarity_score=0.1,
            divergence_threshold=0.7,
        )
        with pytest.raises(ValidationError, match="frozen"):
            failure.prompt = "other"  # type: ignore[misc]

    def test_default_created_at_is_recent(self) -> None:
        from datetime import UTC, datetime

        failure = RoutingFailure(
            request_id="x",
            prompt="p",
            cheap_model="a",
            cheap_response="b",
            cheap_tier=ComplexityTier.SIMPLE,
            classifier_confidence=0.5,
            standard_model="c",
            standard_response="d",
            similarity_score=0.1,
            divergence_threshold=0.7,
        )
        now = datetime.now(UTC)
        assert abs((now - failure.created_at).total_seconds()) < 5


# ── Load limits and reproducibility ─────────────────────────────────────────
class TestVerifierLimits:
    def test_rejects_bad_limits(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency"):
            Verifier(
                ollama=_ollama_client(_ollama_returns()),
                standard_model=_standard_cfg(),
                scorer=FakeScorer(0.9),
                store=FakeStore(),
                divergence_threshold=0.7,
                max_concurrency=0,
            )

    async def test_skips_when_too_many_are_pending(self) -> None:
        verifier = Verifier(
            ollama=_ollama_client(_ollama_returns()),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.9),
            store=FakeStore(),
            divergence_threshold=0.7,
            max_pending=1,
        )
        first = verifier.verify_in_background(
            request_id="req-1",
            tenant_id="t",
            prompt="p",
            cheap_response=_cheap_response(),
            cheap_tier=ComplexityTier.SIMPLE,
            classifier_confidence=0.9,
        )
        second = verifier.verify_in_background(
            request_id="req-2",
            tenant_id="t",
            prompt="p",
            cheap_response=_cheap_response(),
            cheap_tier=ComplexityTier.SIMPLE,
            classifier_confidence=0.9,
        )
        assert verifier.pending_count == 2
        assert await second is VerificationOutcome.SKIPPED_OVERLOADED
        assert await first is VerificationOutcome.AGREED
        assert verifier.pending_count == 0

    async def test_pins_sampling_and_records_tenant(self) -> None:
        captured: dict[str, object] = {}
        respond = _ollama_returns()

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return respond(request)

        store = FakeStore()
        verifier = Verifier(
            ollama=_ollama_client(handler),
            standard_model=_standard_cfg(),
            scorer=FakeScorer(0.1),
            store=store,
            divergence_threshold=0.7,
            temperature=0.0,
            seed=7,
        )
        assert await _run_verifier(verifier, _cheap_response()) is VerificationOutcome.DIVERGED
        assert captured["options"] == {"num_predict": 512, "temperature": 0.0, "seed": 7}
        assert store.failures[0].tenant_id == "tenant-a"
