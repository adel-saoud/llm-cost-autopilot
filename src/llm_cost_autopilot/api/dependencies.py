"""Dependency-injection wiring for the FastAPI app.

The app state holds the long-lived collaborators (classifier, ledger, etc).
Routes pull these via ``Annotated[T, Depends(...)]`` so tests can override
any of them with ``app.dependency_overrides[...]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from llm_cost_autopilot.classifier import TwoStageClassifier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.cost.tracker import CostTracker
from llm_cost_autopilot.router import ModelRegistry, OllamaClient
from llm_cost_autopilot.storage.db import SQLiteLedger
from llm_cost_autopilot.verifier import Verifier


@dataclass
class AppState:
    """Long-lived collaborators wired up at lifespan startup."""

    settings: Settings
    classifier: TwoStageClassifier
    registry: ModelRegistry
    ollama: OllamaClient
    ledger: SQLiteLedger
    tracker: CostTracker
    verifier: Verifier


# ── Plumbing ────────────────────────────────────────────────────────────────
def get_state(request: Request) -> AppState:
    """Pull the AppState off the FastAPI app instance."""
    state = request.app.state.autopilot
    assert isinstance(state, AppState)
    return state


def get_settings(state: Annotated[AppState, Depends(get_state)]) -> Settings:
    return state.settings


def get_classifier(state: Annotated[AppState, Depends(get_state)]) -> TwoStageClassifier:
    return state.classifier


def get_registry(state: Annotated[AppState, Depends(get_state)]) -> ModelRegistry:
    return state.registry


def get_ollama(state: Annotated[AppState, Depends(get_state)]) -> OllamaClient:
    return state.ollama


def get_ledger(state: Annotated[AppState, Depends(get_state)]) -> SQLiteLedger:
    return state.ledger


def get_tracker(state: Annotated[AppState, Depends(get_state)]) -> CostTracker:
    return state.tracker


def get_verifier(state: Annotated[AppState, Depends(get_state)]) -> Verifier:
    return state.verifier


# ── Convenience aliases used by route signatures ────────────────────────────
SettingsDep = Annotated[Settings, Depends(get_settings)]
ClassifierDep = Annotated[TwoStageClassifier, Depends(get_classifier)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama)]
LedgerDep = Annotated[SQLiteLedger, Depends(get_ledger)]
TrackerDep = Annotated[CostTracker, Depends(get_tracker)]
VerifierDep = Annotated[Verifier, Depends(get_verifier)]
