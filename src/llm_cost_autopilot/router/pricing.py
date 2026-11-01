"""Simulated per-model prices: the single table every cost figure comes from.

Ollama runs locally and is free, so each local model is priced at the list
price of the same model on a hosted API, named in ``priced_as``. Prices go
stale; ``PRICES_RETRIEVED_ON`` says when they were read.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PRICES_SOURCE = "https://api.deepinfra.com/models/list"
"""DeepInfra's public model list, which publishes prices per token for each Qwen3.5 size."""

PRICES_RETRIEVED_ON = "2026-10-24"


class ModelPrice(BaseModel):
    """Simulated USD price of one Ollama model, per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    usd_per_1m_input_tokens: float = Field(ge=0.0)
    usd_per_1m_output_tokens: float = Field(ge=0.0)
    priced_as: str = Field(min_length=1, description="DeepInfra model id whose list price is used.")


MODEL_PRICES: dict[str, ModelPrice] = {
    "qwen3.5:0.8b": ModelPrice(
        usd_per_1m_input_tokens=0.01,
        usd_per_1m_output_tokens=0.05,
        priced_as="Qwen/Qwen3.5-0.8B",
    ),
    "qwen3.5:4b": ModelPrice(
        usd_per_1m_input_tokens=0.03,
        usd_per_1m_output_tokens=0.15,
        priced_as="Qwen/Qwen3.5-4B",
    ),
}


def price_for(ollama_model: str) -> ModelPrice:
    """Return the simulated price of ``ollama_model``.

    Raises instead of defaulting to zero, so a model added to the settings
    without a price cannot report free requests and inflated savings.
    """
    try:
        return MODEL_PRICES[ollama_model]
    except KeyError:
        raise ValueError(
            f"No simulated price for Ollama model {ollama_model!r}. "
            f"Add it to MODEL_PRICES in {__name__}; priced models: {sorted(MODEL_PRICES)}"
        ) from None
