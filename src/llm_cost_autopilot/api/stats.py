"""Read-only admin endpoints: stats, corrections, explain.

They return raw prompts and answers, hence the bearer-token dependency on the router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from llm_cost_autopilot.api.dependencies import LedgerDep, RegistryDep, require_api_token
from llm_cost_autopilot.api.schemas import (
    CorrectionsResponse,
    ExplainResponse,
    StatsResponse,
)

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_token)])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(ledger: LedgerDep, registry: RegistryDep) -> StatsResponse:
    """Totals, savings and routing accuracy over the whole ledger."""
    stats = await ledger.get_stats(standard_ollama_model=registry.standard_model.ollama_model)
    return StatsResponse.from_ledger_stats(stats)


@router.get("/corrections", response_model=CorrectionsResponse)
async def get_corrections(
    ledger: LedgerDep,
    limit: int = Query(default=100, ge=1, le=1000),
) -> CorrectionsResponse:
    """The ``limit`` most recent routing failures, newest first."""
    failures = await ledger.list_failures(limit=limit)
    return CorrectionsResponse(corrections=failures, count=len(failures))


@router.get("/explain/{request_id}", response_model=ExplainResponse)
async def get_explain(request_id: str, ledger: LedgerDep) -> ExplainResponse:
    """One request's routing decision, cost and, if the verifier disagreed, its correction."""
    record = await ledger.get_explain(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {request_id}")
    return ExplainResponse.from_record(record.routing, record.correction)
