"""Cost tracker — assembles a RoutingLogEntry and writes it to the ledger.

This is the only place that needs to know how the pieces fit together:
classification result, the model that was chosen, the Ollama response, and
the baseline standard-model cost. Everything else stays a pure function.
"""

from __future__ import annotations

import structlog

from llm_cost_autopilot.classifier.complexity import ClassificationResult
from llm_cost_autopilot.cost.calculator import Savings, compute_savings
from llm_cost_autopilot.router.models import ModelConfig
from llm_cost_autopilot.router.ollama import OllamaChatResponse
from llm_cost_autopilot.storage.db import RoutingLogEntry, SQLiteLedger

logger = structlog.get_logger(__name__)


class CostTracker:
    """Records routed requests + computes per-request savings."""

    def __init__(self, ledger: SQLiteLedger, standard_model: ModelConfig) -> None:
        self._ledger = ledger
        self._standard = standard_model

    async def record(
        self,
        *,
        request_id: str,
        prompt: str,
        classification: ClassificationResult,
        model_used: ModelConfig,
        response: OllamaChatResponse,
    ) -> Savings:
        """Persist one routing event and return the savings record."""
        actual = model_used.cost_usd(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        baseline = self._standard.cost_usd(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        savings = compute_savings(actual_cost_usd=actual, baseline_cost_usd=baseline)

        entry = RoutingLogEntry(
            request_id=request_id,
            prompt=prompt,
            classified_tier=classification.tier,
            classifier_confidence=classification.confidence,
            classifier_stage=classification.stage,
            model_used=model_used.ollama_model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=actual,
            baseline_cost_usd=baseline,
            latency_ms=response.latency_ms,
        )
        await self._ledger.record_routing(entry)
        logger.debug(
            "cost.recorded",
            request_id=request_id,
            tier=classification.tier.value,
            saved_usd=savings.saved_usd,
        )
        return savings
