"""FastAPI app factory and lifespan wiring.

Production code calls ``create_app()``. Tests call ``build_app(state)`` with
a pre-built ``AppState`` containing fakes, skipping the lifespan handler.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
import structlog
from fastapi import FastAPI

from llm_cost_autopilot import __version__
from llm_cost_autopilot.api.completions import router as completions_router
from llm_cost_autopilot.api.dependencies import AppState
from llm_cost_autopilot.api.health import router as health_router
from llm_cost_autopilot.api.route import router as route_router
from llm_cost_autopilot.api.stats import router as stats_router
from llm_cost_autopilot.classifier import TwoStageClassifier
from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.classifier.correction_memory import CorrectionMemory
from llm_cost_autopilot.config import get_settings
from llm_cost_autopilot.cost.tracker import CostTracker
from llm_cost_autopilot.router import OllamaClient
from llm_cost_autopilot.router.models import build_default_registry
from llm_cost_autopilot.router.ollama import require_min_ollama_version
from llm_cost_autopilot.storage.db import open_ledger
from llm_cost_autopilot.verifier import CorrectionStore, Verifier
from llm_cost_autopilot.verifier.correction_feedback import (
    FeedbackCorrectionStore,
    load_correction_memory,
)

logger = structlog.get_logger(__name__)


_APP_TITLE = "LLM Cost Autopilot"
_APP_DESCRIPTION = "Zero-shot complexity routing for LLM requests."
_OLLAMA_TIMEOUT_SECONDS = 60.0
"""Per-call Ollama timeout, generous because MODERATE answers may run to 1,024 tokens locally."""


def _include_routers(app: FastAPI) -> None:
    app.include_router(health_router)
    app.include_router(completions_router)
    app.include_router(route_router)
    app.include_router(stats_router)


def build_app(state: AppState) -> FastAPI:
    """Build an app bound to a pre-constructed ``AppState``, skipping the lifespan (tests)."""
    app = FastAPI(title=_APP_TITLE, version=__version__, description=_APP_DESCRIPTION)
    app.state.autopilot = state
    _include_routers(app)
    return app


def create_app() -> FastAPI:
    """Production entry point: the lifespan loads the models and opens Ollama and SQLite.

    Retention pruning runs before the correction memory is rebuilt, so
    expired corrections never reach the classifier.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        settings = get_settings()
        logger.info("app.lifespan.startup", ollama_base_url=settings.ollama_base_url)

        # Imported here so ``import llm_cost_autopilot.api`` works without the [ml] extra.
        from llm_cost_autopilot.classifier.huggingface_backends import (
            DebertaZeroShotBackend,
            SentenceTransformerEmbeddingBackend,
        )
        from llm_cost_autopilot.verifier.embedding_scorer import EmbeddingSimilarityScorer

        if settings.api_token is None:
            logger.warning("app.lifespan.api_open", reason="AUTOPILOT_API_TOKEN is not set")

        registry = build_default_registry(settings)
        embedding_backend = SentenceTransformerEmbeddingBackend(settings.stage1_model)
        correction_memory = (
            CorrectionMemory(max_per_tenant=settings.correction_max_per_tenant)
            if settings.correction_feedback_enabled
            else None
        )
        classifier = TwoStageClassifier(
            embedding_backend=embedding_backend,
            zero_shot_backend=DebertaZeroShotBackend(settings.stage2_model),
            stage1_threshold=settings.stage1_threshold,
            correction_memory=correction_memory,
            correction_match_threshold=settings.correction_match_threshold,
        )

        async with (
            httpx.AsyncClient(
                base_url=settings.ollama_base_url, timeout=httpx.Timeout(_OLLAMA_TIMEOUT_SECONDS)
            ) as http_client,
            open_ledger(settings.sqlite_path) as ledger,
        ):
            if settings.retention_days is not None:
                cutoff = datetime.now(UTC) - timedelta(days=settings.retention_days)
                routing_rows, correction_rows = await ledger.prune_older_than(cutoff)
                logger.info(
                    "app.lifespan.pruned",
                    retention_days=settings.retention_days,
                    routing_rows=routing_rows,
                    correction_rows=correction_rows,
                )

            correction_store: CorrectionStore = ledger
            if correction_memory is not None:
                load_correction_memory(
                    correction_memory,
                    embedding_backend,
                    await ledger.list_failures_oldest_first(),
                    cheap_model=registry.for_tier(ComplexityTier.SIMPLE).ollama_model,
                    standard_model=registry.standard_model.ollama_model,
                )
                correction_store = FeedbackCorrectionStore(
                    ledger, correction_memory, embedding_backend
                )

            ollama = OllamaClient(http_client, think=settings.ollama_think)
            require_min_ollama_version(await ollama.server_version())
            tracker = CostTracker(ledger=ledger, standard_model=registry.standard_model)
            verifier = Verifier(
                ollama=ollama,
                standard_model=registry.standard_model,
                scorer=EmbeddingSimilarityScorer(settings.stage1_model),
                store=correction_store,
                divergence_threshold=settings.verifier_divergence_threshold,
                enabled=settings.verifier_enabled,
                max_concurrency=settings.verifier_max_concurrency,
                max_pending=settings.verifier_max_pending,
                temperature=settings.ollama_temperature,
                seed=settings.ollama_seed,
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
        title=_APP_TITLE,
        version=__version__,
        description=_APP_DESCRIPTION,
        lifespan=lifespan,
    )
    _include_routers(app)
    return app


__all__ = ["build_app", "create_app"]
