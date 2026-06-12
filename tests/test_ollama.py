"""Async tests for the Ollama HTTP adapter.

Uses ``httpx.MockTransport`` so no daemon is required and the behaviour is
deterministic across runs.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from llm_cost_autopilot.router.ollama import (
    ChatMessage,
    OllamaClient,
    OllamaError,
)


def _build_client(handler: httpx.MockTransport, *, max_retries: int = 3) -> OllamaClient:
    http = httpx.AsyncClient(transport=handler, base_url="http://ollama.test")
    return OllamaClient(http, max_retries=max_retries)


def _ok_response(
    content: str = "hi", prompt_tokens: int = 10, eval_tokens: int = 5
) -> dict[str, Any]:
    return {
        "model": "qwen2.5:0.5b",
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "total_duration": 123_456_789,
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
    }


class TestOllamaClient:
    async def test_chat_round_trips_payload_and_parses_response(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_response(content="hello world"))

        client = _build_client(httpx.MockTransport(handler))

        result = await client.chat(
            model="qwen2.5:0.5b",
            messages=[ChatMessage(role="user", content="Hi?")],
            max_output_tokens=64,
            temperature=0.2,
        )

        assert captured["url"] == "/api/chat"
        assert captured["body"]["model"] == "qwen2.5:0.5b"
        assert captured["body"]["stream"] is False
        assert captured["body"]["messages"] == [{"role": "user", "content": "Hi?"}]
        assert captured["body"]["options"] == {"num_predict": 64, "temperature": 0.2}
        assert result.content == "hello world"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.latency_ms >= 0.0

    async def test_omits_options_when_unset(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_response())

        client = _build_client(httpx.MockTransport(handler))
        await client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])

        assert "options" not in captured["body"]

    async def test_5xx_is_retried_then_succeeds(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 2:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, json=_ok_response(content="ok"))

        client = _build_client(httpx.MockTransport(handler), max_retries=3)
        result = await client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])
        assert result.content == "ok"
        assert len(attempts) == 2

    async def test_4xx_raises_without_retry(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(400, text="bad model")

        client = _build_client(httpx.MockTransport(handler), max_retries=3)
        with pytest.raises(OllamaError, match="400"):
            await client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])
        assert len(attempts) == 1  # no retry on 4xx

    async def test_persistent_5xx_raises_after_max_retries(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(500, text="broken")

        client = _build_client(httpx.MockTransport(handler), max_retries=2)
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])
        assert len(attempts) == 2

    async def test_malformed_body_raises(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model": "x", "done": True})  # no message

        client = _build_client(httpx.MockTransport(handler))
        with pytest.raises(OllamaError, match="missing message"):
            await client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])

    async def test_non_json_body_raises(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not-json")

        client = _build_client(httpx.MockTransport(handler))
        with pytest.raises(OllamaError, match="non-JSON"):
            await client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])

    async def test_transport_error_is_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 2:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(200, json=_ok_response())

        client = _build_client(httpx.MockTransport(handler), max_retries=3)
        result = await client.chat(model="x", messages=[ChatMessage(role="user", content="hi")])
        assert result.content == "hi"
        assert len(attempts) == 2

    def test_rejects_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            OllamaClient(httpx.AsyncClient(), max_retries=0)

    async def test_chat_message_is_frozen(self) -> None:
        msg = ChatMessage(role="user", content="hi")
        with pytest.raises(ValidationError, match="frozen"):
            msg.content = "other"  # type: ignore[misc]
