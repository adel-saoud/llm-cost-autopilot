"""``POST /v1/chat/completions``: classify, serve on the routed Ollama model, verify later."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Response

from llm_cost_autopilot.api.dependencies import (
    ClassifierDep,
    OllamaDep,
    RegistryDep,
    SettingsDep,
    TrackerDep,
    VerifierDep,
    require_api_token,
)
from llm_cost_autopilot.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    classification_prompt,
)
from llm_cost_autopilot.classifier.correction_memory import DEFAULT_TENANT_ID
from llm_cost_autopilot.router import ChatMessage
from llm_cost_autopilot.router.ollama import OllamaError

logger = structlog.get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_token)])


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    settings: SettingsDep,
    classifier: ClassifierDep,
    registry: RegistryDep,
    ollama: OllamaDep,
    tracker: TrackerDep,
    verifier: VerifierDep,
    tenant_id: Annotated[
        str, Header(alias="X-Autopilot-Tenant-Id", min_length=1, max_length=128)
    ] = DEFAULT_TENANT_ID,
) -> ChatCompletionResponse:
    """Classify, serve on the routed Ollama model, record cost, then verify in the background.

    Only the last user message is classified and verified; the full history
    is still sent to the model. Ollama failures surface as 502.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    prompt = classification_prompt(body.messages)

    # CPU-bound model inference: a worker thread keeps the event loop serving.
    classification = await asyncio.to_thread(classifier.classify, prompt, tenant_id=tenant_id)
    routed_model = registry.for_tier(classification.tier)

    log = logger.bind(request_id=request_id, tier=classification.tier.value)
    log.info(
        "completions.routed",
        model=routed_model.ollama_model,
        confidence=classification.confidence,
        stage=classification.stage.value,
    )

    chat_messages = [
        ChatMessage(role=message.role, content=message.content) for message in body.messages
    ]
    try:
        ollama_response = await ollama.chat(
            model=routed_model.ollama_model,
            messages=chat_messages,
            max_output_tokens=body.max_tokens or routed_model.max_output_tokens,
            temperature=(
                body.temperature if body.temperature is not None else settings.ollama_temperature
            ),
            seed=settings.ollama_seed,
        )
    except OllamaError as exc:
        log.warning("completions.ollama_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    savings = await tracker.record(
        request_id=request_id,
        tenant_id=tenant_id,
        prompt=prompt,
        classification=classification,
        model_used=routed_model,
        response=ollama_response,
    )

    verifier.verify_in_background(
        request_id=request_id,
        tenant_id=tenant_id,
        prompt=prompt,
        cheap_response=ollama_response,
        cheap_tier=classification.tier,
        classifier_confidence=classification.confidence,
    )

    # Decision-trail headers, documented in docs/api.md.
    response.headers["X-Autopilot-Request-Id"] = request_id
    response.headers["X-Autopilot-Model-Used"] = f"ollama/{routed_model.ollama_model}"
    response.headers["X-Autopilot-Complexity-Tier"] = classification.tier.value
    response.headers["X-Autopilot-Confidence"] = f"{classification.confidence:.4f}"
    response.headers["X-Autopilot-Routing-Reason"] = classification.stage.value
    response.headers["X-Autopilot-Cost-USD"] = f"{savings.actual_cost_usd:.6f}"
    response.headers["X-Autopilot-Savings-USD"] = f"{savings.saved_usd:.6f}"

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=f"ollama/{routed_model.ollama_model}",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=ollama_response.content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=ollama_response.prompt_tokens,
            completion_tokens=ollama_response.completion_tokens,
            total_tokens=ollama_response.prompt_tokens + ollama_response.completion_tokens,
        ),
    )
