"""Cost calculator + CostTracker tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from llm_cost_autopilot.classifier.complexity import (
    ClassificationResult,
    ClassifierStage,
    ComplexityTier,
)
from llm_cost_autopilot.cost.calculator import compute_savings
from llm_cost_autopilot.cost.tracker import CostTracker
from llm_cost_autopilot.router.models import ModelConfig
from llm_cost_autopilot.router.ollama import OllamaChatResponse
from llm_cost_autopilot.storage.db import SQLiteLedger, open_ledger


# ── Pure savings math ───────────────────────────────────────────────────────
class TestComputeSavings:
    def test_positive_savings(self) -> None:
        s = compute_savings(actual_cost_usd=0.0001, baseline_cost_usd=0.0010)
        assert s.saved_usd == pytest.approx(0.0009)
        assert s.saved_pct == pytest.approx(90.0)

    def test_zero_baseline_means_zero_pct(self) -> None:
        s = compute_savings(actual_cost_usd=0.0, baseline_cost_usd=0.0)
        assert s.saved_usd == 0.0
        assert s.saved_pct == 0.0

    def test_negative_savings_when_autopilot_is_more_expensive(self) -> None:
        # Edge case: classifier put a small request in the MODERATE tier somehow.
        s = compute_savings(actual_cost_usd=0.002, baseline_cost_usd=0.001)
        assert s.saved_usd == pytest.approx(-0.001)
        assert s.saved_pct == pytest.approx(-100.0)

    def test_rejects_negative_inputs(self) -> None:
        with pytest.raises(ValueError, match="actual"):
            compute_savings(actual_cost_usd=-1.0, baseline_cost_usd=0.0)
        with pytest.raises(ValueError, match="baseline"):
            compute_savings(actual_cost_usd=0.0, baseline_cost_usd=-1.0)


# ── CostTracker ─────────────────────────────────────────────────────────────
def _budget_cfg() -> ModelConfig:
    return ModelConfig(
        tier=ComplexityTier.SIMPLE,
        ollama_model="qwen3.5:0.8b",
        usd_per_1m_input_tokens=0.1,
        usd_per_1m_output_tokens=0.2,
    )


def _standard_cfg() -> ModelConfig:
    return ModelConfig(
        tier=ComplexityTier.MODERATE,
        ollama_model="qwen3.5:4b",
        usd_per_1m_input_tokens=3.0,
        usd_per_1m_output_tokens=15.0,
    )


def _classification(tier: ComplexityTier = ComplexityTier.SIMPLE) -> ClassificationResult:
    return ClassificationResult(
        tier=tier,
        confidence=0.9,
        stage=ClassifierStage.STAGE_1_EMBEDDING,
        latency_ms=4.0,
    )


def _response(prompt_tokens: int = 1000, completion_tokens: int = 1000) -> OllamaChatResponse:
    return OllamaChatResponse(
        model="qwen3.5:0.8b",
        content="hi",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_duration_ns=1,
        latency_ms=15.0,
    )


@pytest_asyncio.fixture
async def ledger() -> AsyncIterator[SQLiteLedger]:
    async with open_ledger(":memory:") as lg:
        yield lg


class TestCostTracker:
    async def test_records_entry_with_actual_and_baseline_cost(self, ledger: SQLiteLedger) -> None:
        tracker = CostTracker(ledger=ledger, standard_model=_standard_cfg())
        savings = await tracker.record(
            tenant_id="default",
            request_id="req-1",
            prompt="hi",
            classification=_classification(),
            model_used=_budget_cfg(),
            response=_response(),
        )
        # 1K in + 1K out at budget rates = 0.0001 + 0.0002 = 0.0003
        # standard baseline:                0.003  + 0.015  = 0.018
        assert savings.actual_cost_usd == pytest.approx(0.0003)
        assert savings.baseline_cost_usd == pytest.approx(0.018)
        assert savings.saved_usd == pytest.approx(0.0177)

        record = await ledger.get_explain("req-1")
        assert record is not None
        assert record.routing.cost_usd == pytest.approx(0.0003)
        assert record.routing.baseline_cost_usd == pytest.approx(0.018)
        assert record.routing.model_used == "qwen3.5:0.8b"

    async def test_records_zero_cost_for_empty_response(self, ledger: SQLiteLedger) -> None:
        tracker = CostTracker(ledger=ledger, standard_model=_standard_cfg())
        savings = await tracker.record(
            tenant_id="default",
            request_id="req-2",
            prompt="",
            classification=_classification(),
            model_used=_budget_cfg(),
            response=_response(prompt_tokens=0, completion_tokens=0),
        )
        assert savings.actual_cost_usd == 0.0
        assert savings.baseline_cost_usd == 0.0
        assert savings.saved_usd == 0.0
