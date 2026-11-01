"""Model registry and Ollama adapter.

Picking a model is a registry lookup once the classifier has chosen a tier;
the routing work happens in ``classifier``.

Public surface:
    ModelConfig             frozen per-tier configuration (model id, simulated price).
    MODEL_PRICES            simulated price per Ollama model, with its source.
    ModelRegistry           immutable map of tier to ModelConfig.
    build_default_registry  build the registry from current settings.
    OllamaClient            async adapter for the local Ollama HTTP API.
    OllamaChatResponse      structured response with token usage.
"""

from llm_cost_autopilot.router.models import ModelConfig, ModelRegistry, build_default_registry
from llm_cost_autopilot.router.ollama import (
    ChatMessage,
    OllamaChatResponse,
    OllamaClient,
    OllamaError,
)
from llm_cost_autopilot.router.pricing import MODEL_PRICES, ModelPrice, price_for

__all__ = [
    "MODEL_PRICES",
    "ChatMessage",
    "ModelConfig",
    "ModelPrice",
    "ModelRegistry",
    "OllamaChatResponse",
    "OllamaClient",
    "OllamaError",
    "build_default_registry",
    "price_for",
]
