"""ModelConfig and ModelRegistry invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.router.decision import build_default_registry, route
from llm_cost_autopilot.router.models import ModelConfig, ModelRegistry


def _cfg(tier: ComplexityTier, ollama_model: str = "test:0.5b") -> ModelConfig:
    return ModelConfig(
        tier=tier,
        ollama_model=ollama_model,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
    )


class TestModelConfig:
    def test_cost_is_token_weighted_sum(self) -> None:
        cfg = ModelConfig(
            tier=ComplexityTier.SIMPLE,
            ollama_model="qwen2.5:0.5b",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0002,
        )
        # 1000 in + 500 out → 0.0001 + 0.0001 = 0.0002
        assert cfg.cost_usd(prompt_tokens=1000, completion_tokens=500) == pytest.approx(0.0002)

    def test_zero_tokens_costs_zero(self) -> None:
        cfg = _cfg(ComplexityTier.SIMPLE)
        assert cfg.cost_usd(prompt_tokens=0, completion_tokens=0) == 0.0

    def test_rejects_empty_model_id(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                tier=ComplexityTier.SIMPLE,
                ollama_model="",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
            )

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                tier=ComplexityTier.SIMPLE,
                ollama_model="x",
                cost_per_1k_input=-1.0,
                cost_per_1k_output=0.002,
            )

    def test_is_frozen(self) -> None:
        cfg = _cfg(ComplexityTier.SIMPLE)
        with pytest.raises(ValidationError, match="frozen"):
            cfg.ollama_model = "other"  # type: ignore[misc]


class TestModelRegistry:
    def test_for_tier_returns_bound_config(self) -> None:
        registry = ModelRegistry(
            entries={
                ComplexityTier.SIMPLE: _cfg(ComplexityTier.SIMPLE, "a"),
                ComplexityTier.MODERATE: _cfg(ComplexityTier.MODERATE, "b"),
                ComplexityTier.COMPLEX: _cfg(ComplexityTier.COMPLEX, "c"),
            }
        )
        assert registry.for_tier(ComplexityTier.SIMPLE).ollama_model == "a"
        assert registry.for_tier(ComplexityTier.COMPLEX).ollama_model == "c"

    def test_rejects_incomplete_registry(self) -> None:
        with pytest.raises(ValidationError, match="missing"):
            ModelRegistry(
                entries={
                    ComplexityTier.SIMPLE: _cfg(ComplexityTier.SIMPLE),
                }
            )

    def test_rejects_mismatched_tier(self) -> None:
        with pytest.raises(ValidationError, match="mismatched"):
            ModelRegistry(
                entries={
                    ComplexityTier.SIMPLE: _cfg(ComplexityTier.MODERATE),
                    ComplexityTier.MODERATE: _cfg(ComplexityTier.MODERATE),
                    ComplexityTier.COMPLEX: _cfg(ComplexityTier.COMPLEX),
                }
            )

    def test_standard_property_is_moderate(self) -> None:
        registry = build_default_registry(Settings())
        assert registry.standard.tier is ComplexityTier.MODERATE


class TestDefaultRegistry:
    def test_simple_uses_budget_model(self) -> None:
        settings = Settings()
        registry = build_default_registry(settings)
        cfg = registry.for_tier(ComplexityTier.SIMPLE)
        assert cfg.ollama_model == settings.budget_model
        assert cfg.cost_per_1k_input == pytest.approx(settings.budget_cost_per_1k_input)

    def test_moderate_uses_standard_model(self) -> None:
        settings = Settings()
        registry = build_default_registry(settings)
        cfg = registry.for_tier(ComplexityTier.MODERATE)
        assert cfg.ollama_model == settings.standard_model

    def test_complex_has_larger_token_budget_than_moderate(self) -> None:
        registry = build_default_registry(Settings())
        moderate = registry.for_tier(ComplexityTier.MODERATE)
        complex_ = registry.for_tier(ComplexityTier.COMPLEX)
        assert complex_.max_output_tokens > moderate.max_output_tokens

    def test_route_picks_registry_entry(self) -> None:
        registry = build_default_registry(Settings())
        assert route(ComplexityTier.SIMPLE, registry) is registry.for_tier(ComplexityTier.SIMPLE)
