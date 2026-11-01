"""Cost tracker: assembles a RoutingLogEntry and writes it to the ledger.

The one place that combines the classification, the chosen model, the Ollama
token counts and the standard-model baseline, so the savings math in
``calculator`` stays a pure function.
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
    """Records each routed request and its savings against the standard model."""

    def __init__(self, ledger: SQLiteLedger, standard_model: ModelConfig) -> None:
        self._ledger = ledger
        self._standard_model = standard_model

    async def record(
        self,
        *,
        request_id: str,
        tenant_id: str,
        prompt: str,
        classification: ClassificationResult,
        model_used: ModelConfig,
        response: OllamaChatResponse,
    ) -> Savings:
        """Persist one routing event and return the savings record.

        The baseline prices the same token counts at the standard model's
        rates, so savings measure the routing choice alone.
        """
        actual_cost_usd = model_used.cost_usd(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        baseline_cost_usd = self._standard_model.cost_usd(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        savings = compute_savings(
            actual_cost_usd=actual_cost_usd, baseline_cost_usd=baseline_cost_usd
        )

        entry = RoutingLogEntry(
            request_id=request_id,
            tenant_id=tenant_id,
            prompt=prompt,
            classified_tier=classification.tier,
            classifier_confidence=classification.confidence,
            classifier_stage=classification.stage,
            model_used=model_used.ollama_model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=actual_cost_usd,
            baseline_cost_usd=baseline_cost_usd,
            latency_ms=response.latency_ms,
            matched_correction_request_id=classification.matched_correction_request_id,
        )
        await self._ledger.record_routing(entry)
        logger.debug(
            "cost.recorded",
            request_id=request_id,
            tier=classification.tier.value,
            saved_usd=savings.saved_usd,
        )
        return savings
