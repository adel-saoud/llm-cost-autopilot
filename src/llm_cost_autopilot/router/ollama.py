"""Async adapter for the local Ollama HTTP API.

We only use the ``POST /api/chat`` endpoint (non-streaming). The structured
response carries the token counts the cost tracker needs.

Reference: https://docs.ollama.com/api/chat
"""

from __future__ import annotations

import time
from typing import Any, Literal, cast

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = structlog.get_logger(__name__)


# ── Wire types ──────────────────────────────────────────────────────────────
Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A single chat-turn message — the wire format Ollama expects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str


class OllamaChatResponse(BaseModel):
    """Structured response from a non-streaming ``/api/chat`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    content: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_duration_ns: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class OllamaError(RuntimeError):
    """Raised when Ollama returns a non-2xx response or an unparsable body."""


# ── Client ──────────────────────────────────────────────────────────────────
class OllamaClient:
    """Thin async wrapper around the Ollama HTTP API.

    The underlying ``httpx.AsyncClient`` is injected so tests can plug in a
    transport mock (`httpx.MockTransport`).

    Parameters
    ----------
    http:
        An ``httpx.AsyncClient`` configured with the desired base URL/timeout.
    max_retries:
        Total attempts on transient transport errors (network, timeout, 5xx).
        Use 1 to disable retries entirely.
    """

    def __init__(self, http: httpx.AsyncClient, max_retries: int = 3) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._http = http
        self._max_retries = max_retries

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> OllamaChatResponse:
        """Send one chat turn and return a structured, token-counted response."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
        }
        options: dict[str, Any] = {}
        if max_output_tokens is not None:
            # Ollama's option key is `num_predict`.
            options["num_predict"] = max_output_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options

        t0 = time.perf_counter()
        response = await self._post_with_retry("/api/chat", payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        body = self._parse_chat_body(response)

        return OllamaChatResponse(
            model=cast("str", body.get("model", model)),
            content=cast("str", body["message"]["content"]),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            total_duration_ns=int(body.get("total_duration", 0)),
            latency_ms=latency_ms,
        )

    async def _post_with_retry(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with tenacity-backed retry on transient transport failures."""
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(initial=0.1, max=2.0),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                response = await self._http.post(path, json=payload)
                if response.status_code >= 500:
                    # Treat 5xx as retryable; raise so tenacity can decide.
                    raise httpx.HTTPStatusError(
                        f"Ollama returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    raise OllamaError(
                        f"Ollama returned {response.status_code}: {response.text[:200]}"
                    )
                return response
        # Unreachable — AsyncRetrying always raises or returns.
        raise OllamaError("retry loop exited without a response")  # pragma: no cover

    @staticmethod
    def _parse_chat_body(response: httpx.Response) -> dict[str, Any]:
        try:
            body = cast("dict[str, Any]", response.json())
        except ValueError as exc:
            raise OllamaError(f"Ollama returned non-JSON body: {response.text[:200]}") from exc
        if "message" not in body or "content" not in body["message"]:
            raise OllamaError(f"Ollama response missing message/content: {body}")
        return body
