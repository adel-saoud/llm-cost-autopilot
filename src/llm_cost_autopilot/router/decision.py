"""Tier → model decision logic.

This module is intentionally tiny: the interesting routing work happens in the
classifier. Once a tier is known, picking a model is a dictionary lookup.
"""

from __future__ import annotations

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.router.models import ModelConfig, ModelRegistry


def build_default_registry(settings: Settings) -> ModelRegistry:
    """Construct the default registry from ``Settings``.

    The COMPLEX tier reuses the standard Ollama model with a larger output
    budget — Ollama runs locally and we don't want to require a second model
    download. The cost differential between MODERATE and COMPLEX is purely
    the longer simulated output, which is realistic for production: a complex
    request *is* more expensive because it generates more tokens.
    """
    return ModelRegistry(
        entries={
            ComplexityTier.SIMPLE: ModelConfig(
                tier=ComplexityTier.SIMPLE,
                ollama_model=settings.budget_model,
                cost_per_1k_input=settings.budget_cost_per_1k_input,
                cost_per_1k_output=settings.budget_cost_per_1k_output,
                max_output_tokens=256,
            ),
            ComplexityTier.MODERATE: ModelConfig(
                tier=ComplexityTier.MODERATE,
                ollama_model=settings.standard_model,
                cost_per_1k_input=settings.standard_cost_per_1k_input,
                cost_per_1k_output=settings.standard_cost_per_1k_output,
                max_output_tokens=512,
            ),
            ComplexityTier.COMPLEX: ModelConfig(
                tier=ComplexityTier.COMPLEX,
                ollama_model=settings.standard_model,
                cost_per_1k_input=settings.standard_cost_per_1k_input,
                cost_per_1k_output=settings.standard_cost_per_1k_output,
                max_output_tokens=2048,
            ),
        }
    )


def route(tier: ComplexityTier, registry: ModelRegistry) -> ModelConfig:
    """Pick the ``ModelConfig`` for ``tier``.

    Wraps ``ModelRegistry.for_tier`` so callers depend on a function rather
    than the registry's method — convenient for typing and future policy
    changes (e.g. tier-rewriting fallbacks if a model is unhealthy).
    """
    return registry.for_tier(tier)
