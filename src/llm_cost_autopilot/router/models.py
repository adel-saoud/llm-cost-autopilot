"""Per-tier model configuration and the registry that maps tiers to models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llm_cost_autopilot.classifier.complexity import ComplexityTier


class ModelConfig(BaseModel):
    """Concrete model bound to one complexity tier — immutable.

    Costs are *simulated* — Ollama runs locally and is free. The simulated
    prices let the cost tracker produce realistic savings numbers using a
    public pricing curve (cheap small model vs. mid-tier model).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: ComplexityTier
    ollama_model: str = Field(min_length=1, description="Ollama model id, e.g. 'qwen2.5:0.5b'.")
    cost_per_1k_input: float = Field(ge=0.0, description="Simulated USD per 1K input tokens.")
    cost_per_1k_output: float = Field(ge=0.0, description="Simulated USD per 1K output tokens.")
    max_output_tokens: int = Field(
        default=512,
        ge=1,
        description="Hard cap on tokens the model may produce per request.",
    )

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Total simulated cost in USD for one request."""
        return (
            prompt_tokens / 1000.0 * self.cost_per_1k_input
            + completion_tokens / 1000.0 * self.cost_per_1k_output
        )


class ModelRegistry(BaseModel):
    """Frozen, exhaustive map from every ``ComplexityTier`` to a ``ModelConfig``.

    The registry rejects construction if any tier is missing — this makes the
    "what model do I use for tier X" lookup a total function rather than a
    runtime error waiting to happen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: dict[ComplexityTier, ModelConfig]

    def model_post_init(self, _: object) -> None:
        missing = set(ComplexityTier) - set(self.entries.keys())
        if missing:
            missing_names = sorted(t.value for t in missing)
            raise ValueError(
                f"ModelRegistry must define an entry for every tier; missing: {missing_names}"
            )
        for tier, cfg in self.entries.items():
            if cfg.tier is not tier:
                raise ValueError(
                    f"ModelRegistry entry for {tier.value} has mismatched "
                    f"ModelConfig.tier={cfg.tier.value}"
                )

    def for_tier(self, tier: ComplexityTier) -> ModelConfig:
        """Return the model bound to ``tier`` (never raises — registry is total)."""
        return self.entries[tier]

    @property
    def standard(self) -> ModelConfig:
        """The fallback / verifier model — STANDARD tier by convention."""
        return self.entries[ComplexityTier.MODERATE]
