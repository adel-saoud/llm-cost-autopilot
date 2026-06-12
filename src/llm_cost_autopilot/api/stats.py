"""Read-only admin endpoints — stats, corrections, explain."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from llm_cost_autopilot.api.dependencies import LedgerDep, RegistryDep
from llm_cost_autopilot.api.schemas import (
    CorrectionsResponse,
    ExplainResponse,
    StatsResponse,
)
from llm_cost_autopilot.classifier.complexity import ComplexityTier

router = APIRouter(prefix="/v1")


@router.get("/stats", response_model=StatsResponse)
async def get_stats(ledger: LedgerDep, registry: RegistryDep) -> StatsResponse:
    standard = registry.for_tier(ComplexityTier.MODERATE).ollama_model
    stats = await ledger.get_stats(standard_model=standard)
    return StatsResponse.from_stats(stats)


@router.get("/corrections", response_model=CorrectionsResponse)
async def get_corrections(
    ledger: LedgerDep,
    limit: int = Query(default=100, ge=1, le=1000),
) -> CorrectionsResponse:
    rows = await ledger.list_failures(limit=limit)
    return CorrectionsResponse(corrections=rows, count=len(rows))


@router.get("/explain/{request_id}", response_model=ExplainResponse)
async def get_explain(request_id: str, ledger: LedgerDep) -> ExplainResponse:
    record = await ledger.get_explain(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")
    return ExplainResponse.from_record(record.routing, record.correction)
