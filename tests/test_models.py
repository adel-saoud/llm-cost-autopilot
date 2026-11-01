"""ModelConfig and ModelRegistry invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.router.models import ModelConfig, ModelRegistry, build_default_registry
from llm_cost_autopilot.router.pricing import MODEL_PRICES


def _cfg(tier: ComplexityTier, ollama_model: str = "test:0.5b") -> ModelConfig:
    return ModelConfig(
        tier=tier,
        ollama_model=ollama_model,
        usd_per_1m_input_tokens=1.0,
        usd_per_1m_output_tokens=2.0,
    )


class TestModelConfig:
    def test_cost_is_token_weighted_sum(self) -> None:
        cfg = ModelConfig(
            tier=ComplexityTier.SIMPLE,
            ollama_model="qwen3.5:0.8b",
            usd_per_1m_input_tokens=0.1,
            usd_per_1m_output_tokens=0.2,
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
                usd_per_1m_input_tokens=1.0,
                usd_per_1m_output_tokens=2.0,
            )

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                tier=ComplexityTier.SIMPLE,
                ollama_model="x",
                usd_per_1m_input_tokens=-1.0,
                usd_per_1m_output_tokens=2.0,
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
                ComplexityTier.MODERATE: _cfg(ComplexityTier.MODERATE, "c"),
            }
        )
        assert registry.for_tier(ComplexityTier.SIMPLE).ollama_model == "a"
        assert registry.for_tier(ComplexityTier.MODERATE).ollama_model == "c"

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
                }
            )

    def test_standard_model_is_the_moderate_tier(self) -> None:
        registry = build_default_registry(Settings())
        assert registry.standard_model.tier is ComplexityTier.MODERATE


class TestDefaultRegistry:
    def test_each_tier_uses_its_own_model_and_price(self) -> None:
        settings = Settings()
        registry = build_default_registry(settings)
        simple = registry.for_tier(ComplexityTier.SIMPLE)
        moderate = registry.for_tier(ComplexityTier.MODERATE)
        assert simple.ollama_model == settings.simple_model
        assert moderate.ollama_model == settings.moderate_model
        assert simple.usd_per_1m_output_tokens == pytest.approx(
            MODEL_PRICES[settings.simple_model].usd_per_1m_output_tokens
        )
        assert moderate.usd_per_1m_output_tokens > simple.usd_per_1m_output_tokens

    def test_moderate_has_larger_token_budget_than_simple(self) -> None:
        registry = build_default_registry(Settings())
        simple = registry.for_tier(ComplexityTier.SIMPLE)
        moderate = registry.for_tier(ComplexityTier.MODERATE)
        assert moderate.max_output_tokens > simple.max_output_tokens

    def test_unpriced_model_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="No simulated price"):
            build_default_registry(Settings(moderate_model="unpriced:1b"))
