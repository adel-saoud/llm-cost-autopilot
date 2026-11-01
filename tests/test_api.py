"""End-to-end tests for the FastAPI app.

All collaborators are real except the HTTP transport (mocked) and the
verifier scorer (fake). The classifier uses the FakeEmbeddingBackend from
the shared conftest so we don't load any HuggingFace models.
"""

from __future__ import annotations

import asyncio
import json
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
from llm_cost_autopilot.classifier.correction_memory import CorrectionMemory, Embedding
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.cost.tracker import CostTracker
from llm_cost_autopilot.router import OllamaClient
from llm_cost_autopilot.router.models import build_default_registry
from llm_cost_autopilot.storage.db import open_ledger
from llm_cost_autopilot.verifier import CorrectionStore, Verifier
from llm_cost_autopilot.verifier.correction_feedback import FeedbackCorrectionStore
from tests.conftest import FakeEmbeddingBackend, FakeZeroShotBackend, unit_vector


class FakeScorer:
    def __init__(self, score: float = 0.95) -> None:
        self._score = score

    async def score(self, first_text: str, second_text: str) -> float:
        return self._score


def _ollama_handler(
    responses_by_model: dict[str, dict[str, Any]],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
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
            embeddings: dict[str, Embedding] | None = None,
            memory: CorrectionMemory | None = None,
            settings_overrides: dict[str, Any] | None = None,
        ) -> FastAPI:
            settings = Settings(**(settings_overrides or {}))
            registry = build_default_registry(settings)

            embed = FakeEmbeddingBackend(stage1_scores or {}, embeddings)
            classifier = TwoStageClassifier(
                embedding_backend=embed,
                zero_shot_backend=FakeZeroShotBackend(stage2_scores or {}),
                stage1_threshold=settings.stage1_threshold,
                correction_memory=memory,
                correction_match_threshold=settings.correction_match_threshold,
            )

            http = httpx.AsyncClient(
                transport=httpx.MockTransport(_ollama_handler(ollama_responses or {})),
                base_url="http://ollama.test",
            )
            ollama = OllamaClient(http, max_attempts=1)
            tracker = CostTracker(ledger=ledger, standard_model=registry.standard_model)
            store: CorrectionStore = (
                ledger if memory is None else FeedbackCorrectionStore(ledger, memory, embed)
            )
            verifier = Verifier(
                ollama=ollama,
                standard_model=registry.standard_model,
                scorer=FakeScorer(scorer_score),
                store=store,
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
    async def test_routes_simple_prompt_to_simple_model(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        app = app_factory(
            stage1_scores={
                "Capital of France?": {
                    ComplexityTier.SIMPLE: 0.92,
                    ComplexityTier.MODERATE: 0.10,
                }
            },
            ollama_responses={
                settings.simple_model: _ollama_body("Paris.", settings.simple_model),
                settings.moderate_model: _ollama_body(
                    "Paris is the capital.", settings.moderate_model
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
        assert body["model"] == f"ollama/{settings.simple_model}"
        assert body["usage"]["total_tokens"] == 20

        assert r.headers["X-Autopilot-Complexity-Tier"] == "simple"
        assert r.headers["X-Autopilot-Model-Used"] == f"ollama/{settings.simple_model}"
        assert float(r.headers["X-Autopilot-Confidence"]) == pytest.approx(0.92)
        assert float(r.headers["X-Autopilot-Cost-USD"]) > 0
        assert float(r.headers["X-Autopilot-Savings-USD"]) > 0
        assert r.headers["X-Autopilot-Request-Id"].startswith("chatcmpl-")

    async def test_routes_reasoning_prompt_to_standard_model(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        prompt = "Architect a multi-region failover plan and weigh the tradeoffs."
        app = app_factory(
            stage1_scores={
                prompt: {
                    ComplexityTier.SIMPLE: 0.10,
                    ComplexityTier.MODERATE: 0.88,
                }
            },
            ollama_responses={
                settings.moderate_model: _ollama_body("Detailed plan...", settings.moderate_model),
            },
        )

        async with await _client(app) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}]},
            )
        await _drain_verifier_tasks()

        assert r.status_code == 200
        assert r.headers["X-Autopilot-Complexity-Tier"] == "moderate"
        assert r.headers["X-Autopilot-Model-Used"] == f"ollama/{settings.moderate_model}"
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
                    ComplexityTier.MODERATE: 0.05,
                }
            },
            ollama_responses={
                settings.simple_model: _ollama_body("yes", settings.simple_model),
                settings.moderate_model: _ollama_body("yes too", settings.moderate_model),
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

    async def test_ollama_error_returns_502(self, app_factory: Callable[..., FastAPI]) -> None:
        app = app_factory(
            stage1_scores={
                "p": {
                    ComplexityTier.SIMPLE: 0.92,
                    ComplexityTier.MODERATE: 0.05,
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
                    ComplexityTier.MODERATE: 0.05,
                }
            },
            ollama_responses={
                settings.simple_model: _ollama_body("hello", settings.simple_model),
                settings.moderate_model: _ollama_body("hello there", settings.moderate_model),
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
        assert explain.json()["model_used"] == settings.simple_model
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
                    ComplexityTier.MODERATE: 0.05,
                }
            },
            ollama_responses={
                settings.simple_model: _ollama_body("uhh idk", settings.simple_model),
                settings.moderate_model: _ollama_body(
                    "Here's a careful answer.", settings.moderate_model
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


# ── Auth ─────────────────────────────────────────────────────────────────────
_SIMPLE = {
    ComplexityTier.SIMPLE: 0.92,
    ComplexityTier.MODERATE: 0.05,
}


class TestApiToken:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/v1/stats"),
            ("GET", "/v1/corrections"),
            ("GET", "/v1/explain/any"),
            ("POST", "/v2/route"),
            ("POST", "/v1/chat/completions"),
        ],
    )
    async def test_rejects_missing_or_wrong_token(
        self, app_factory: Callable[..., FastAPI], method: str, path: str
    ) -> None:
        app = app_factory(settings_overrides={"api_token": "s3cret"})
        async with await _client(app) as ac:
            missing = await ac.request(method, path, json={})
            wrong = await ac.request(
                method, path, json={}, headers={"Authorization": "Bearer nope"}
            )
        assert missing.status_code == 401
        assert wrong.status_code == 401
        assert missing.headers["WWW-Authenticate"] == "Bearer"

    async def test_accepts_the_configured_token(self, app_factory: Callable[..., FastAPI]) -> None:
        app = app_factory(settings_overrides={"api_token": "s3cret"})
        async with await _client(app) as ac:
            r = await ac.get("/v1/stats", headers={"Authorization": "Bearer s3cret"})
            health = await ac.get("/health")
        assert r.status_code == 200
        assert health.status_code == 200


# ── Router API ───────────────────────────────────────────────────────────────
class TestRouteEndpoint:
    async def test_returns_tier_without_model_or_correction_ids(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        app = app_factory(stage1_scores={"Capital of France?": _SIMPLE})
        async with await _client(app) as ac:
            r = await ac.post(
                "/v2/route",
                json={
                    "tenant_id": "team-1",
                    "messages": [{"role": "user", "content": "Capital of France?"}],
                    "unknown_field": "ignored",
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"route_id", "tier", "confidence", "reason"}
        assert body["tier"] == "simple"
        assert body["reason"] == "stage_1_embedding"
        assert body["route_id"].startswith("rt-")

    async def test_requires_a_tenant(self, app_factory: Callable[..., FastAPI]) -> None:
        app = app_factory()
        async with await _client(app) as ac:
            r = await ac.post("/v2/route", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 422

    def test_published_json_schema_matches_the_models(self) -> None:
        from pathlib import Path

        from scripts.export_route_schema import route_api_schema

        path = Path(__file__).resolve().parents[1] / "docs" / "route-api.schema.json"
        published = json.loads(path.read_text(encoding="utf-8"))
        assert published == route_api_schema(), (
            "docs/route-api.schema.json is stale: "
            "run `uv run python scripts/export_route_schema.py`"
        )


# ── Feedback loop, end to end ────────────────────────────────────────────────
class TestFeedbackLoop:
    async def test_divergence_escalates_the_next_paraphrase(
        self, app_factory: Callable[..., FastAPI]
    ) -> None:
        settings = Settings()
        original = "Who painted the Mona Lisa?"
        paraphrase = "Who was the painter of the Mona Lisa?"
        app = app_factory(
            stage1_scores={original: _SIMPLE, paraphrase: _SIMPLE},
            embeddings={
                original: unit_vector(1.0, 0.0),
                paraphrase: unit_vector(0.97, 0.24),
            },
            memory=CorrectionMemory(max_per_tenant=10),
            ollama_responses={
                settings.simple_model: _ollama_body(
                    "Sorry, I can't help with that.", settings.simple_model
                ),
                settings.moderate_model: _ollama_body(
                    "Leonardo da Vinci.", settings.moderate_model
                ),
            },
            scorer_score=0.30,  # diverged
        )
        tenant = {"X-Autopilot-Tenant-Id": "team-1"}

        async with await _client(app) as ac:
            first = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": original}]},
                headers=tenant,
            )
            await _drain_verifier_tasks()
            second = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": paraphrase}]},
                headers=tenant,
            )
            other_tenant = await ac.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": paraphrase}]},
                headers={"X-Autopilot-Tenant-Id": "team-2"},
            )
            await _drain_verifier_tasks()
            explain = await ac.get(f"/v1/explain/{second.headers['X-Autopilot-Request-Id']}")

        assert first.headers["X-Autopilot-Complexity-Tier"] == "simple"
        assert second.headers["X-Autopilot-Complexity-Tier"] == "moderate"
        assert second.headers["X-Autopilot-Routing-Reason"] == "correction_memory"
        assert second.headers["X-Autopilot-Model-Used"] == f"ollama/{settings.moderate_model}"
        assert other_tenant.headers["X-Autopilot-Complexity-Tier"] == "simple"
        assert explain.json()["tenant_id"] == "team-1"
        assert explain.json()["classifier_stage"] == "correction_memory"
        assert (
            explain.json()["matched_correction_request_id"]
            == first.headers["X-Autopilot-Request-Id"]
        )
