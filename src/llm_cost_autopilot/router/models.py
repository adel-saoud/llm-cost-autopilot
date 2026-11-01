"""Per-tier model configuration, the registry that maps tiers to models, and its default."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.router.pricing import price_for


class ModelConfig(BaseModel):
    """Concrete model bound to one complexity tier: immutable.

    Prices are simulated (Ollama is free) and come from ``router.pricing``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: ComplexityTier
    ollama_model: str = Field(min_length=1, description="Ollama model id, e.g. 'qwen3.5:0.8b'.")
    usd_per_1m_input_tokens: float = Field(ge=0.0, description="Simulated USD per 1M input tokens.")
    usd_per_1m_output_tokens: float = Field(
        ge=0.0, description="Simulated USD per 1M output tokens."
    )
    max_output_tokens: int = Field(
        default=512,
        ge=1,
        description=(
            "Tokens the model may produce (Ollama num_predict) when the request sets no max_tokens."
        ),
    )

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Total simulated cost in USD for one request."""
        return (
            prompt_tokens * self.usd_per_1m_input_tokens
            + completion_tokens * self.usd_per_1m_output_tokens
        ) / 1_000_000


class ModelRegistry(BaseModel):
    """Frozen, exhaustive map from every ``ComplexityTier`` to a ``ModelConfig``.

    The registry rejects construction if any tier is missing, so the "which
    model serves tier X" lookup is a total function rather than a runtime
    error waiting to happen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: dict[ComplexityTier, ModelConfig]

    def model_post_init(self, context: object, /) -> None:
        """Reject a registry that misses a tier or files a config under the wrong tier."""
        missing = set(ComplexityTier) - set(self.entries.keys())
        if missing:
            missing_names = sorted(tier.value for tier in missing)
            raise ValueError(
                f"ModelRegistry must define an entry for every tier; missing: {missing_names}"
            )
        for tier, config in self.entries.items():
            if config.tier is not tier:
                raise ValueError(
                    f"ModelRegistry entry for {tier.value} has mismatched "
                    f"ModelConfig.tier={config.tier.value}"
                )

    def for_tier(self, tier: ComplexityTier) -> ModelConfig:
        """Return the model bound to ``tier`` (never raises: the registry is total)."""
        return self.entries[tier]

    @property
    def standard_model(self) -> ModelConfig:
        """The MODERATE-tier model: the cost baseline and the verifier's reference."""
        return self.entries[ComplexityTier.MODERATE]


def build_default_registry(settings: Settings) -> ModelRegistry:
    """Construct the registry from ``Settings`` and the simulated price table.

    One model per tier. The output caps (256 and 1024 tokens) bound worst-case
    latency and simulated cost per tier; a request's own ``max_tokens``
    replaces them.
    """

    def config(tier: ComplexityTier, ollama_model: str, max_output_tokens: int) -> ModelConfig:
        price = price_for(ollama_model)
        return ModelConfig(
            tier=tier,
            ollama_model=ollama_model,
            usd_per_1m_input_tokens=price.usd_per_1m_input_tokens,
            usd_per_1m_output_tokens=price.usd_per_1m_output_tokens,
            max_output_tokens=max_output_tokens,
        )

    return ModelRegistry(
        entries={
            ComplexityTier.SIMPLE: config(ComplexityTier.SIMPLE, settings.simple_model, 256),
            ComplexityTier.MODERATE: config(ComplexityTier.MODERATE, settings.moderate_model, 1024),
        }
    )
