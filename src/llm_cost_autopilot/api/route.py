"""``POST /v2/route``: routing decision only, for an upstream gateway.

The gateway sends the conversation, gets back a tier, and serves the request
with its own providers. Autopilot makes no model call here, so the endpoint
costs one classification. Verification and correction learning happen only
on traffic autopilot serves itself through ``/v1/chat/completions``.

The router API is versioned on its own, by path. v2 dropped the ``complex``
tier; v1 is gone rather than aliased, since the tier set it promised no
longer exists.
"""

from __future__ import annotations

import asyncio
import uuid

import structlog
from fastapi import APIRouter, Depends

from llm_cost_autopilot.api.dependencies import ClassifierDep, require_api_token
from llm_cost_autopilot.api.schemas import RouteRequest, RouteResponse, classification_prompt

logger = structlog.get_logger(__name__)

ROUTE_API_VERSION = "2"
"""Router API version: the path prefix, and ``api_version`` in docs/route-api.schema.json."""

ROUTE_PATH = f"/v{ROUTE_API_VERSION}/route"

router = APIRouter(dependencies=[Depends(require_api_token)])


@router.post(ROUTE_PATH, response_model=RouteResponse)
async def route_request(body: RouteRequest, classifier: ClassifierDep) -> RouteResponse:
    """Classify the last user message for ``body.tenant_id`` and return the tier only.

    The matched correction id stays in autopilot's logs: it points at another
    request's prompt and must not cross the API boundary.
    """
    route_id = f"rt-{uuid.uuid4().hex[:24]}"
    prompt = classification_prompt(body.messages)
    classification = await asyncio.to_thread(classifier.classify, prompt, tenant_id=body.tenant_id)
    logger.info(
        "route.decided",
        route_id=route_id,
        tenant_id=body.tenant_id,
        tier=classification.tier.value,
        reason=classification.stage.value,
        matched_correction_request_id=classification.matched_correction_request_id,
    )
    return RouteResponse(
        route_id=route_id,
        tier=classification.tier,
        confidence=classification.confidence,
        reason=classification.stage.value,
    )
