"""Dependency-injection wiring for the FastAPI app.

The app state holds the long-lived collaborators (classifier, ledger, etc).
Routes pull each one through a one-line getter and an ``Annotated[T,
Depends(getter)]`` alias, so tests can override any of them with
``app.dependency_overrides[getter]``.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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


def get_state(request: Request) -> AppState:
    """Return the ``AppState`` that ``create_app``'s lifespan or ``build_app`` attached."""
    state = request.app.state.autopilot
    if not isinstance(state, AppState):  # pragma: no cover
        raise TypeError("app.state.autopilot is not an AppState; was create_app() used?")
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


_bearer_scheme = HTTPBearer(auto_error=False, description="Value of AUTOPILOT_API_TOKEN.")
"""``auto_error=False`` so an unset token can leave the API open instead of always demanding one."""


def require_api_token(
    settings: Annotated[Settings, Depends(get_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> None:
    """Reject the request unless it carries ``AUTOPILOT_API_TOKEN`` as a bearer token.

    The admin endpoints return raw prompts and model answers, so they must
    not be readable by anyone who can reach the port. With no token
    configured the check is off, which only suits a local demo.
    """
    expected_token = settings.api_token
    if expected_token is None:
        return
    supplied_token = credentials.credentials if credentials is not None else ""
    # Constant-time comparison so response timing does not leak the token.
    if not secrets.compare_digest(
        supplied_token.encode(), expected_token.get_secret_value().encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClassifierDep = Annotated[TwoStageClassifier, Depends(get_classifier)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama)]
LedgerDep = Annotated[SQLiteLedger, Depends(get_ledger)]
TrackerDep = Annotated[CostTracker, Depends(get_tracker)]
VerifierDep = Annotated[Verifier, Depends(get_verifier)]
