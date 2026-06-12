"""End-to-end tests for the FastAPI app.

All collaborators are real except the HTTP transport (mocked) and the
verifier scorer (fake). The classifier uses the FakeEmbeddingBackend from
the shared conftest so we don't load any HuggingFace models.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from llm_cost_autopilot.api.app import build_app
from llm_cost_autopilot.api.dependencies import AppState
from llm_cost_autopilot.classifier import TwoStageClassifier
from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.cost.tracker import CostTracker
from llm_cost_autopilot.router import OllamaClient
from llm_cost_autopilot.router.decision import build_default_registry
from llm_cost_autopilot.storage.db import open_ledger
from llm_cost_autopilot.verifier import Verifier
from tests.conftest import FakeEmbeddingBackend, FakeZeroShotBackend


class FakeScorer:
    def __init__(self, score: float = 0.95) -> None:
        self._score = score

    async def score(self, a: str, b: str) -> float:
        return self._score


def _ollama_handler(
    responses_by_model: dict[str, dict[str, Any]],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        model = body["model"]
        if model not in responses_by_model:
            return httpx.Response(404, text=f"unknown model: {model}")
        return httpx.Response(200, json=responses_by_model[model])

    return handler


def _ollama_body(
    content: str, model: str, *, prompt_tokens: int = 8, eval_tokens: int = 12
) -> dict[str, Any]:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
        "total_duration": 1_000_000,
    }


@pytest_asyncio.fixture
async def app_factory() -> AsyncIterator[Callable[..., FastAPI]]:
    """Build a fresh FastAPI app per test with controllable fakes."""

    async with open_ledger(":memory:") as ledger:

        def factory(
            *,
            stage1_scores: dict[str, dict[ComplexityTier, float]] | None = None,
            stage2_scores: dict[str, dict[ComplexityTier, float]] | None = None,
            ollama_responses: dict[str, dict[str, Any]] | None = None,
            scorer_score: float = 0.95,
            verifier_enabled: bool = True,
        ) -> FastAPI:
            settings = Settings()
            registry = build_default_registry(settings)

            classifier = TwoStageClassifier(
                embedding_backend=FakeEmbeddingBackend(stage1_scores or {}),
                zero_shot_backend=FakeZeroShotBackend(stage2_scores or {}),
                stage1_threshold=settings.stage1_threshold,
            )

            http = httpx.AsyncClient(
                transport=httpx.MockTransport(_ollama_handler(ollama_responses or {})),
                base_url="http://ollama.test",
            )
            ollama = OllamaClient(http, max_retries=1)
            tracker = CostTracker(ledger=ledger, standard_model=registry.standard)
            verifier = Verifier(
                ollama=ollama,
                standard_model=registry.for_tier(ComplexityTier.MODERATE),
                scorer=FakeScorer(scorer_score),
                store=ledger,
                divergence_threshold=settings.verifier_divergence_threshold,
                enabled=verifier_enabled,
            )
            state = AppState(
                settings=settings,
                classifier=classifier,
                registry=registry,
                ollama=ollama,
                ledger=ledger,
                tracker=tracker,
                verifier=verifier,
            )
            return build_app(state)

        yield factory


async def _drain_verifier_tasks() -> None:
    """Await any in-flight ``verifier:*`` background tasks."""
    pending = [
        t for t in asyncio.all_tasks() if t.get_name().startswith("verifier:") and not t.done()
    ]
    if pending:
        await asyncio.gather(*pending)


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# ── Health ───────────────────────────────────────────────────────────────────
class TestHealth:
    async def test_health_returns_ok(self, app_factory: Callable[..., FastAPI]) -> None:
        app = app_factory()
        async with await _client(app) as ac:
            r = await ac.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "version" in r.json()


# ── Chat completions ────────────────────────────────────────────────────────
class TestChatCompletions:
    async def test_routes_simple_prompt_to_budget_model(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        app = app_factory(
            stage1_scores={
                "Capital of France?": {
                    ComplexityTier.SIMPLE: 0.92,
                    ComplexityTier.MODERATE: 0.20,
                    ComplexityTier.COMPLEX: 0.10,
                }
            },
            ollama_responses={
                settings.budget_model: _ollama_body("Paris.", settings.budget_model),
                settings.standard_model: _ollama_body(
                    "Paris is the capital.", settings.standard_model
                ),
            },
        )

        async with await _client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Capital of France?"}]},
            )
        await _drain_verifier_tasks()

        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "Paris."
        assert body["model"] == f"ollama/{settings.budget_model}"
        assert body["usage"]["total_tokens"] == 20

        assert r.headers["X-Autopilot-Complexity-Tier"] == "simple"
        assert r.headers["X-Autopilot-Model-Used"] == f"ollama/{settings.budget_model}"
        assert float(r.headers["X-Autopilot-Confidence"]) == pytest.approx(0.92)
        assert float(r.headers["X-Autopilot-Cost-USD"]) > 0
        assert float(r.headers["X-Autopilot-Savings-USD"]) > 0
        assert r.headers["X-Autopilot-Request-Id"].startswith("chatcmpl-")

    async def test_routes_complex_prompt_to_standard_model(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        prompt = "Architect a multi-region failover plan and weigh the tradeoffs."
        app = app_factory(
            stage1_scores={
                prompt: {
                    ComplexityTier.SIMPLE: 0.10,
                    ComplexityTier.MODERATE: 0.30,
                    ComplexityTier.COMPLEX: 0.88,
                }
            },
            ollama_responses={
                settings.standard_model: _ollama_body("Detailed plan...", settings.standard_model),
            },
        )

        async with await _client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}]},
            )
        await _drain_verifier_tasks()

        assert r.status_code == 200
        assert r.headers["X-Autopilot-Complexity-Tier"] == "complex"
        assert r.headers["X-Autopilot-Model-Used"] == f"ollama/{settings.standard_model}"
        # Savings are zero (or negative) when we routed to standard.
        assert float(r.headers["X-Autopilot-Savings-USD"]) == pytest.approx(0.0)

    async def test_classifier_uses_last_user_message(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        app = app_factory(
            stage1_scores={
                "What about tomorrow?": {
                    ComplexityTier.SIMPLE: 0.85,
                    ComplexityTier.MODERATE: 0.10,
                    ComplexityTier.COMPLEX: 0.05,
                }
            },
            ollama_responses={
                settings.budget_model: _ollama_body("yes", settings.budget_model),
                settings.standard_model: _ollama_body("yes too", settings.standard_model),
            },
        )

        async with await _client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What's the weather today?"},
                        {"role": "assistant", "content": "Sunny."},
                        {"role": "user", "content": "What about tomorrow?"},
                    ]
                },
            )
        await _drain_verifier_tasks()
        assert r.status_code == 200

    async def test_ollama_502_returns_502(self, app_factory: Callable[..., FastAPI]) -> None:
        app = app_factory(
            stage1_scores={
                "p": {
                    ComplexityTier.SIMPLE: 0.92,
                    ComplexityTier.MODERATE: 0.10,
                    ComplexityTier.COMPLEX: 0.05,
                }
            },
            ollama_responses={},  # unknown model → 404 from Ollama → OllamaError
        )

        async with await _client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "p"}]},
            )
        assert r.status_code == 502


# ── Stats / corrections / explain ───────────────────────────────────────────
class TestObservability:
    async def test_stats_starts_empty(self, app_factory: Callable[..., FastAPI]) -> None:
        app = app_factory()
        async with await _client(app) as ac:
            r = await ac.get("/v1/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_requests"] == 0
        assert body["correction_count"] == 0
        assert body["routing_accuracy_pct"] == 100.0

    async def test_request_is_recorded_and_explainable(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        app = app_factory(
            stage1_scores={
                "hi": {
                    ComplexityTier.SIMPLE: 0.92,
                    ComplexityTier.MODERATE: 0.10,
                    ComplexityTier.COMPLEX: 0.05,
                }
            },
            ollama_responses={
                settings.budget_model: _ollama_body("hello", settings.budget_model),
                settings.standard_model: _ollama_body("hello there", settings.standard_model),
            },
            scorer_score=0.95,  # agreed
        )

        async with await _client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            request_id = r.headers["X-Autopilot-Request-Id"]
            await _drain_verifier_tasks()

            stats = await ac.get("/v1/stats")
            explain = await ac.get(f"/v1/explain/{request_id}")
            corrections = await ac.get("/v1/corrections")

        assert stats.json()["total_requests"] == 1
        assert stats.json()["correction_count"] == 0  # scorer agreed
        assert explain.status_code == 200
        assert explain.json()["model_used"] == settings.budget_model
        assert explain.json()["correction"] is None
        assert corrections.json()["count"] == 0

    async def test_divergent_request_appears_in_corrections(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        app = app_factory(
            stage1_scores={
                "tricky": {
                    ComplexityTier.SIMPLE: 0.92,
                    ComplexityTier.MODERATE: 0.10,
                    ComplexityTier.COMPLEX: 0.05,
                }
            },
            ollama_responses={
                settings.budget_model: _ollama_body("uhh idk", settings.budget_model),
                settings.standard_model: _ollama_body(
                    "Here's a careful answer.", settings.standard_model
                ),
            },
            scorer_score=0.20,  # diverged
        )

        async with await _client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "tricky"}]},
            )
            request_id = r.headers["X-Autopilot-Request-Id"]
            await _drain_verifier_tasks()

            stats = await ac.get("/v1/stats")
            corrections = await ac.get("/v1/corrections")
            explain = await ac.get(f"/v1/explain/{request_id}")

        assert stats.json()["correction_count"] == 1
        assert stats.json()["routing_accuracy_pct"] == pytest.approx(0.0)
        assert corrections.json()["count"] == 1
        assert corrections.json()["corrections"][0]["request_id"] == request_id
        assert explain.json()["correction"] is not None
        assert explain.json()["correction"]["similarity_score"] == pytest.approx(0.20)

    async def test_explain_returns_404_for_unknown_id(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        app = app_factory()
        async with await _client(app) as ac:
            r = await ac.get("/v1/explain/does-not-exist")
        assert r.status_code == 404
