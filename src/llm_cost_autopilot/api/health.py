"""Liveness probe."""

from __future__ import annotations

from fastapi import APIRouter

from llm_cost_autopilot import __version__
from llm_cost_autopilot.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)
