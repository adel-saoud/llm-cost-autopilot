"""Production wiring in ``create_app``: retention pruning, correction-memory
reload and the feedback store. The HuggingFace backends are replaced by fakes
so no model is downloaded."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llm_cost_autopilot.api.app import create_app
from llm_cost_autopilot.api.dependencies import AppState
from llm_cost_autopilot.classifier import huggingface_backends
from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.classifier.correction_memory import Embedding
from llm_cost_autopilot.router.ollama import OllamaClient, OllamaVersionError
from llm_cost_autopilot.storage.db import open_ledger
from llm_cost_autopilot.verifier import embedding_scorer
from llm_cost_autopilot.verifier.correction_feedback import FeedbackCorrectionStore
from llm_cost_autopilot.verifier.feedback import RoutingFailure
from tests.conftest import unit_vector


class _FakeEmbedding:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def score(self, prompt: str) -> dict[ComplexityTier, float]:
        return dict.fromkeys(ComplexityTier, 0.5)

    def embed(self, prompt: str) -> Embedding:
        return unit_vector(float(len(prompt)), 1.0)


class _FakeZeroShot:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def classify(self, prompt: str) -> dict[ComplexityTier, float]:
        return dict.fromkeys(ComplexityTier, 1 / 3)


class _FakeScorer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def score(self, first_text: str, second_text: str) -> float:
        return 1.0


def _failure(request_id: str, prompt: str, created_at: datetime) -> RoutingFailure:
    return RoutingFailure(
        request_id=request_id,
        tenant_id="team-1",
        prompt=prompt,
        cheap_model="qwen3.5:0.8b",
        cheap_response="?",
        cheap_tier=ComplexityTier.SIMPLE,
        classifier_confidence=0.9,
        standard_model="qwen3.5:4b",
        standard_response="!",
        similarity_score=0.3,
        divergence_threshold=0.7,
        created_at=created_at,
    )


@pytest.fixture
def fake_ml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(huggingface_backends, "SentenceTransformerEmbeddingBackend", _FakeEmbedding)
    monkeypatch.setattr(huggingface_backends, "DebertaZeroShotBackend", _FakeZeroShot)
    monkeypatch.setattr(embedding_scorer, "EmbeddingSimilarityScorer", _FakeScorer)
    monkeypatch.setattr(OllamaClient, "server_version", _fake_server_version)


async def _fake_server_version(self: OllamaClient) -> str:
    return "0.23.2"


@pytest.mark.usefixtures("fake_ml")
async def test_startup_prunes_old_rows_and_reloads_recent_corrections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "ledger.db"
    now = datetime.now(UTC)
    async with open_ledger(db_path) as ledger:
        await ledger.record_failure(_failure("old", "an old mistake", now - timedelta(days=60)))
        await ledger.record_failure(_failure("new", "a recent mistake", now))

    monkeypatch.setenv("AUTOPILOT_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("AUTOPILOT_RETENTION_DAYS", "30")
    app = create_app()

    async with app.router.lifespan_context(app):
        state = app.state.autopilot
        assert isinstance(state, AppState)
        assert isinstance(state.verifier._store, FeedbackCorrectionStore)  # pyright: ignore[reportPrivateUsage]
        remaining = await state.ledger.list_failures_oldest_first()
        memory = state.classifier._correction_memory  # pyright: ignore[reportPrivateUsage]

    assert [f.request_id for f in remaining] == ["new"]
    assert memory is not None
    assert memory.correction_count("team-1") == 1


@pytest.mark.usefixtures("fake_ml")
async def test_feedback_can_be_switched_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTOPILOT_SQLITE_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("AUTOPILOT_CORRECTION_FEEDBACK_ENABLED", "false")
    monkeypatch.setenv("AUTOPILOT_API_TOKEN", "s3cret")
    app = create_app()

    async with app.router.lifespan_context(app):
        state = app.state.autopilot
        assert isinstance(state, AppState)
        assert state.classifier._correction_memory is None  # pyright: ignore[reportPrivateUsage]
        assert not isinstance(state.verifier._store, FeedbackCorrectionStore)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.usefixtures("fake_ml")
async def test_old_ollama_stops_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def old_version(self: OllamaClient) -> str:
        return "0.3.0"

    monkeypatch.setattr(OllamaClient, "server_version", old_version)
    monkeypatch.setenv("AUTOPILOT_SQLITE_PATH", str(tmp_path / "ledger.db"))
    app = create_app()
    with pytest.raises(OllamaVersionError, match="older than"):
        async with app.router.lifespan_context(app):
            pass
