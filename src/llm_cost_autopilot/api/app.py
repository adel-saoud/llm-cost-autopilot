"""FastAPI app factory + lifespan wiring.

Production code calls ``create_app()``. Tests call ``build_app(state)`` with
a pre-built ``AppState`` containing fakes, skipping the lifespan handler.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI

from llm_cost_autopilot.api.completions import router as completions_router
from llm_cost_autopilot.api.dependencies import AppState
from llm_cost_autopilot.api.health import router as health_router
from llm_cost_autopilot.api.stats import router as stats_router
from llm_cost_autopilot.classifier import TwoStageClassifier
from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.config import get_settings
from llm_cost_autopilot.cost.tracker import CostTracker
from llm_cost_autopilot.router import OllamaClient
from llm_cost_autopilot.router.decision import build_default_registry
from llm_cost_autopilot.storage.db import open_ledger
from llm_cost_autopilot.verifier import Verifier

logger = structlog.get_logger(__name__)


def build_app(state: AppState) -> FastAPI:
    """Build an app bound to a pre-constructed ``AppState`` — used by tests."""
    app = FastAPI(
        title="LLM Cost Autopilot",
        version="0.1.0",
        description="Zero-shot complexity routing for LLM requests.",
    )
    app.state.autopilot = state
    app.include_router(health_router)
    app.include_router(completions_router)
    app.include_router(stats_router)
    return app


def create_app() -> FastAPI:
    """Production entry point — wires up real collaborators via lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        settings = get_settings()
        logger.info("app.lifespan.startup", ollama_base_url=settings.ollama_base_url)

        # Heavy ML imports happen only inside the lifespan — keeping the
        # module top-level light means ``import llm_cost_autopilot.api`` is
        # cheap for tools, scripts, and tests.
        from llm_cost_autopilot.classifier._real_backends import (
            DebertaZeroShotBackend,
            SentenceTransformerEmbeddingBackend,
        )
        from llm_cost_autopilot.verifier._real_scorer import EmbeddingSimilarityScorer

        registry = build_default_registry(settings)
        classifier = TwoStageClassifier(
            embedding_backend=SentenceTransformerEmbeddingBackend(settings.stage1_model),
            zero_shot_backend=DebertaZeroShotBackend(settings.stage2_model),
            stage1_threshold=settings.stage1_threshold,
        )

        async with (
            httpx.AsyncClient(
                base_url=settings.ollama_base_url, timeout=httpx.Timeout(60.0)
            ) as http,
            open_ledger(settings.sqlite_path) as ledger,
        ):
            ollama = OllamaClient(http)
            tracker = CostTracker(ledger=ledger, standard_model=registry.standard)
            verifier = Verifier(
                ollama=ollama,
                standard_model=registry.for_tier(ComplexityTier.MODERATE),
                scorer=EmbeddingSimilarityScorer(settings.stage1_model),
                store=ledger,
                divergence_threshold=settings.verifier_divergence_threshold,
                enabled=settings.verifier_enabled,
            )

            app.state.autopilot = AppState(
                settings=settings,
                classifier=classifier,
                registry=registry,
                ollama=ollama,
                ledger=ledger,
                tracker=tracker,
                verifier=verifier,
            )
            try:
                yield
            finally:
                logger.info("app.lifespan.shutdown")

    app = FastAPI(
        title="LLM Cost Autopilot",
        version="0.1.0",
        description="Zero-shot complexity routing for LLM requests.",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(completions_router)
    app.include_router(stats_router)
    return app


__all__ = ["build_app", "create_app"]
