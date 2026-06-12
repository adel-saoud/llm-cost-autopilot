"""Model registry and Ollama routing.

Public surface:
    ModelConfig         — frozen per-tier configuration (model id, simulated cost).
    ModelRegistry       — immutable map of tier → ModelConfig.
    build_default_registry — wire a registry from current settings.
    route               — pick the registry entry for a complexity tier.
    OllamaClient        — async adapter for the local Ollama HTTP API.
    OllamaChatResponse  — structured response with token usage.
"""

from llm_cost_autopilot.router.decision import build_default_registry, route
from llm_cost_autopilot.router.models import ModelConfig, ModelRegistry
from llm_cost_autopilot.router.ollama import (
    ChatMessage,
    OllamaChatResponse,
    OllamaClient,
    OllamaError,
)

__all__ = [
    "ChatMessage",
    "ModelConfig",
    "ModelRegistry",
    "OllamaChatResponse",
    "OllamaClient",
    "OllamaError",
    "build_default_registry",
    "route",
]
