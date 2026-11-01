"""Async adapter for the local Ollama HTTP API.

Requests use ``POST /api/chat``, non-streaming, because its final body carries
the token counts the cost tracker needs (``prompt_eval_count``, ``eval_count``).
``GET /api/version`` is read once at startup to reject a server that is too old.

References: https://docs.ollama.com/api/chat, https://docs.ollama.com/api/version
"""

from __future__ import annotations

import re
import time
from typing import Any, Final, Literal, cast

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


ChatRole = Literal["system", "user", "assistant"]
"""Message roles accepted by both the OpenAI-compatible API and Ollama's chat endpoint."""


class ChatMessage(BaseModel):
    """A single chat-turn message: the wire format Ollama expects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ChatRole
    content: str


class OllamaChatResponse(BaseModel):
    """Structured response from a non-streaming ``/api/chat`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    content: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_duration_ns: int = Field(ge=0, description="Ollama's server-side total_duration.")
    latency_ms: float = Field(
        ge=0.0, description="Wall-clock time of the HTTP call, retries included."
    )


MIN_OLLAMA_VERSION: Final = "0.17.5"
"""First Ollama release with the small Qwen3.5 sizes (0.8B, 4B), the default tier
models, and the fix for Qwen3.5 repeating itself
(https://github.com/ollama/ollama/releases/tag/v0.17.5)."""

_VERSION_RE: Final = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable, returns a non-2xx status, or an unparsable body.

    The single error type lets the API map every Ollama failure to 502.
    """


class OllamaVersionError(RuntimeError):
    """The Ollama server is older than ``MIN_OLLAMA_VERSION`` or reports no readable version."""


def require_min_ollama_version(found: str) -> None:
    """Raise ``OllamaVersionError`` naming the minimum when ``found`` is older or unreadable."""
    match = _VERSION_RE.match(found.strip())
    minimum = tuple(int(part) for part in MIN_OLLAMA_VERSION.split("."))
    if match is None:
        raise OllamaVersionError(
            f"Ollama reported version {found!r}, which is not readable; "
            f"autopilot needs Ollama {MIN_OLLAMA_VERSION} or newer."
        )
    if tuple(int(part) for part in match.groups()) < minimum:
        raise OllamaVersionError(
            f"Ollama {found} is older than {MIN_OLLAMA_VERSION}, the minimum autopilot "
            "supports. Upgrade Ollama."
        )


class OllamaClient:
    """Thin async wrapper around the Ollama HTTP API.

    The ``httpx.AsyncClient`` is injected so one connection pool is shared per
    process and tests can plug in ``httpx.MockTransport``.

    Parameters
    ----------
    http_client:
        An ``httpx.AsyncClient`` configured with Ollama's base URL and timeout.
    max_attempts:
        Total attempts on transient failures (network, timeout, 5xx).
        Use 1 to disable retries entirely.
    think:
        Sent as Ollama's ``think`` on every chat call. False keeps reasoning
        models such as Qwen3.5 from spending tokens on a hidden reasoning trace,
        which would inflate cost and latency and differ from what is served.
        None leaves the model's own default.
    """

    def __init__(
        self, http_client: httpx.AsyncClient, max_attempts: int = 3, think: bool | None = None
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._http_client = http_client
        self._max_attempts = max_attempts
        self._think = think

    async def server_version(self) -> str:
        """Return the version string from ``GET /api/version``, without retries."""
        try:
            response = await self._http_client.get("/api/version")
            response.raise_for_status()
            return str(response.json()["version"])
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise OllamaError(f"Cannot read the Ollama version: {exc}") from exc

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> OllamaChatResponse:
        """Send one chat turn and return a structured, token-counted response.

        ``temperature=0`` plus a fixed ``seed`` makes Ollama's output
        reproducible, which the evaluation needs to compare runs fairly.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
        }
        if self._think is not None:
            payload["think"] = self._think
        options: dict[str, Any] = {}
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if seed is not None:
            options["seed"] = seed
        if options:
            payload["options"] = options

        started_at = time.perf_counter()
        response = await self._post_chat_with_retry(payload)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        body = self._parse_chat_body(response)

        return OllamaChatResponse(
            model=cast("str", body.get("model", model)),
            content=cast("str", body["message"]["content"]),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
            total_duration_ns=int(body.get("total_duration", 0)),
            latency_ms=latency_ms,
        )

    async def _post_chat_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        """POST to ``/api/chat``, retrying transport errors and 5xx, failing fast on 4xx.

        A 4xx (unknown model, bad request) will not succeed on retry. Once
        retries run out the last error is re-raised as ``OllamaError``.
        """
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=0.1, max=2.0),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._http_client.post("/api/chat", json=payload)
                    if response.status_code >= 500:
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
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise OllamaError(f"Ollama failed after {self._max_attempts} attempts: {exc}") from exc
        raise AssertionError("unreachable: AsyncRetrying returns or raises")  # pragma: no cover

    @staticmethod
    def _parse_chat_body(response: httpx.Response) -> dict[str, Any]:
        """Decode the JSON body and check it holds ``message.content``."""
        try:
            body = cast("dict[str, Any]", response.json())
        except ValueError as exc:
            raise OllamaError(f"Ollama returned non-JSON body: {response.text[:200]}") from exc
        if "message" not in body or "content" not in body["message"]:
            raise OllamaError(f"Ollama response missing message/content: {body}")
        return body
