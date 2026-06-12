"""``POST /v1/chat/completions`` — the autopilot routing endpoint."""

from __future__ import annotations

import asyncio
import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Response

from llm_cost_autopilot.api.dependencies import (
    ClassifierDep,
    OllamaDep,
    RegistryDep,
    TrackerDep,
    VerifierDep,
)
from llm_cost_autopilot.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from llm_cost_autopilot.router import ChatMessage, route
from llm_cost_autopilot.router.ollama import OllamaError

logger = structlog.get_logger(__name__)

router = APIRouter()


def _extract_classification_prompt(messages: list[ChatCompletionMessage]) -> str:
    """Use the *last* user message for complexity classification.

    Earlier system prompts and conversation history affect what the model
    actually generates, but the routing decision is driven by the latest
    user turn — that's what the classifier was trained on.
    """
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    # No user turn — should be impossible given the request schema, but
    # fall back to joining everything so we still produce a classification.
    return "\n".join(m.content for m in messages)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    response: Response,
    classifier: ClassifierDep,
    registry: RegistryDep,
    ollama: OllamaDep,
    tracker: TrackerDep,
    verifier: VerifierDep,
) -> ChatCompletionResponse:
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    prompt = _extract_classification_prompt(body.messages)

    # Classification — sync but CPU-bound. Offload so we don't stall the loop.
    classification = await asyncio.to_thread(classifier.classify, prompt)
    model_cfg = route(classification.tier, registry)

    log = logger.bind(request_id=request_id, tier=classification.tier.value)
    log.info(
        "completions.routed",
        model=model_cfg.ollama_model,
        confidence=classification.confidence,
        stage=classification.stage.value,
    )

    chat_messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    try:
        ollama_response = await ollama.chat(
            model=model_cfg.ollama_model,
            messages=chat_messages,
            max_output_tokens=body.max_tokens or model_cfg.max_output_tokens,
            temperature=body.temperature,
        )
    except OllamaError as exc:
        log.warning("completions.ollama_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    savings = await tracker.record(
        request_id=request_id,
        prompt=prompt,
        classification=classification,
        model_used=model_cfg,
        response=ollama_response,
    )

    # Schedule the verifier on the user's prompt only — not the full history.
    # Fire-and-forget: any failure stays inside the task.
    verifier.verify_in_background(
        request_id=request_id,
        prompt=prompt,
        cheap_response=ollama_response,
        cheap_tier=classification.tier,
        classifier_confidence=classification.confidence,
    )

    # Decision-trail headers — match the spec in CLAUDE.md.
    response.headers["X-Autopilot-Request-Id"] = request_id
    response.headers["X-Autopilot-Model-Used"] = f"ollama/{model_cfg.ollama_model}"
    response.headers["X-Autopilot-Complexity-Tier"] = classification.tier.value
    response.headers["X-Autopilot-Confidence"] = f"{classification.confidence:.4f}"
    response.headers["X-Autopilot-Cost-USD"] = f"{savings.actual_cost_usd:.6f}"
    response.headers["X-Autopilot-Savings-USD"] = f"{savings.saved_usd:.6f}"

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=f"ollama/{model_cfg.ollama_model}",
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
