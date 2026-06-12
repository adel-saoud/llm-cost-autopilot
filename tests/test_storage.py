"""SQLiteLedger — schema migration, writes, reads, aggregate stats.

All tests use ``:memory:`` so they're hermetic and parallel-safe.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from llm_cost_autopilot.classifier.complexity import ClassifierStage, ComplexityTier
from llm_cost_autopilot.storage.db import (
    CURRENT_SCHEMA_VERSION,
    RoutingLogEntry,
    SQLiteLedger,
    open_ledger,
)
from llm_cost_autopilot.verifier.feedback import RoutingFailure


@pytest_asyncio.fixture
async def ledger() -> AsyncIterator[SQLiteLedger]:
    async with open_ledger(":memory:") as lg:
        yield lg


def _entry(
    *,
    request_id: str = "req-1",
    tier: ComplexityTier = ComplexityTier.SIMPLE,
    model: str = "qwen2.5:0.5b",
    cost_usd: float = 0.0001,
    baseline_cost_usd: float = 0.0010,
) -> RoutingLogEntry:
    return RoutingLogEntry(
        request_id=request_id,
        prompt="hi",
        classified_tier=tier,
        classifier_confidence=0.82,
        classifier_stage=ClassifierStage.STAGE_1_EMBEDDING,
        model_used=model,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=cost_usd,
        baseline_cost_usd=baseline_cost_usd,
        latency_ms=12.0,
    )


def _failure(*, request_id: str = "req-1", similarity: float = 0.2) -> RoutingFailure:
    return RoutingFailure(
        request_id=request_id,
        prompt="hi",
        cheap_model="qwen2.5:0.5b",
        cheap_response="x",
        cheap_tier=ComplexityTier.SIMPLE,
        classifier_confidence=0.82,
        standard_model="llama3.2:3b",
        standard_response="y",
        similarity_score=similarity,
        divergence_threshold=0.7,
    )


# ── Schema ──────────────────────────────────────────────────────────────────
class TestSchema:
    async def test_migration_seeds_version(self, ledger: SQLiteLedger) -> None:
        # The fixture has already run migrations.
        assert await ledger.schema_version() == CURRENT_SCHEMA_VERSION

    async def test_migration_is_idempotent(self) -> None:
        async with open_ledger(":memory:") as lg:
            # The connect() helper already migrated once; call again.
            await lg.migrate()
            await lg.migrate()
            # Still reports the same version — no duplicate rows, no corruption.
            assert await lg.schema_version() == CURRENT_SCHEMA_VERSION


# ── Writes + reads ──────────────────────────────────────────────────────────
class TestRoutingLog:
    async def test_record_and_explain(self, ledger: SQLiteLedger) -> None:
        await ledger.record_routing(_entry())
        record = await ledger.get_explain("req-1")
        assert record is not None
        assert record.routing.request_id == "req-1"
        assert record.routing.classified_tier is ComplexityTier.SIMPLE
        assert record.correction is None

    async def test_explain_returns_none_for_unknown(self, ledger: SQLiteLedger) -> None:
        assert await ledger.get_explain("nope") is None

    async def test_record_is_idempotent_on_request_id(self, ledger: SQLiteLedger) -> None:
        await ledger.record_routing(_entry(cost_usd=0.0001))
        await ledger.record_routing(_entry(cost_usd=0.0099))  # update
        record = await ledger.get_explain("req-1")
        assert record is not None
        assert record.routing.cost_usd == pytest.approx(0.0099)


class TestCorrections:
    async def test_record_failure_persists(self, ledger: SQLiteLedger) -> None:
        await ledger.record_routing(_entry())
        await ledger.record_failure(_failure())
        record = await ledger.get_explain("req-1")
        assert record is not None
        assert record.correction is not None
        assert record.correction.similarity_score == pytest.approx(0.2)

    async def test_list_failures_returns_most_recent_first(self, ledger: SQLiteLedger) -> None:
        await ledger.record_failure(_failure(request_id="a", similarity=0.1))
        await ledger.record_failure(_failure(request_id="b", similarity=0.2))
        await ledger.record_failure(_failure(request_id="c", similarity=0.3))
        rows = await ledger.list_failures(limit=2)
        assert [r.request_id for r in rows] == ["c", "b"]

    async def test_list_failures_rejects_zero_limit(self, ledger: SQLiteLedger) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            await ledger.list_failures(limit=0)


# ── Aggregate stats ─────────────────────────────────────────────────────────
class TestStats:
    async def test_empty_ledger_returns_zero_stats(self, ledger: SQLiteLedger) -> None:
        stats = await ledger.get_stats(standard_model="llama3.2:3b")
        assert stats.total_requests == 0
        assert stats.total_actual_cost_usd == 0.0
        assert stats.total_baseline_cost_usd == 0.0
        assert stats.savings_usd == 0.0
        assert stats.savings_pct == 0.0
        assert stats.correction_count == 0
        assert stats.routing_accuracy_pct == 100.0  # default with nothing to judge

    async def test_savings_is_baseline_minus_actual(self, ledger: SQLiteLedger) -> None:
        await ledger.record_routing(
            _entry(request_id="a", cost_usd=0.0001, baseline_cost_usd=0.0010)
        )
        await ledger.record_routing(
            _entry(request_id="b", cost_usd=0.0002, baseline_cost_usd=0.0020)
        )
        stats = await ledger.get_stats(standard_model="llama3.2:3b")
        assert stats.total_actual_cost_usd == pytest.approx(0.0003)
        assert stats.total_baseline_cost_usd == pytest.approx(0.0030)
        assert stats.savings_usd == pytest.approx(0.0027)
        assert stats.savings_pct == pytest.approx(90.0)

    async def test_accuracy_only_counts_non_standard_requests(self, ledger: SQLiteLedger) -> None:
        # Two cheap-routed requests, one diverged → 50% accuracy.
        await ledger.record_routing(
            _entry(request_id="a", model="qwen2.5:0.5b", tier=ComplexityTier.SIMPLE)
        )
        await ledger.record_routing(
            _entry(request_id="b", model="qwen2.5:0.5b", tier=ComplexityTier.SIMPLE)
        )
        # One standard-routed request — should NOT count against accuracy.
        await ledger.record_routing(
            _entry(request_id="c", model="llama3.2:3b", tier=ComplexityTier.MODERATE)
        )
        await ledger.record_failure(_failure(request_id="a"))

        stats = await ledger.get_stats(standard_model="llama3.2:3b")
        assert stats.routing_accuracy_pct == pytest.approx(50.0)
        assert stats.correction_count == 1

    async def test_requests_by_tier_counts_every_tier(self, ledger: SQLiteLedger) -> None:
        await ledger.record_routing(_entry(request_id="a", tier=ComplexityTier.SIMPLE))
        await ledger.record_routing(_entry(request_id="b", tier=ComplexityTier.SIMPLE))
        await ledger.record_routing(_entry(request_id="c", tier=ComplexityTier.COMPLEX))
        stats = await ledger.get_stats(standard_model="llama3.2:3b")
        assert stats.requests_by_tier[ComplexityTier.SIMPLE] == 2
        assert stats.requests_by_tier[ComplexityTier.MODERATE] == 0
        assert stats.requests_by_tier[ComplexityTier.COMPLEX] == 1
