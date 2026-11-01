"""The demo ledger seed behind the README dashboard steps."""

from __future__ import annotations

from pathlib import Path

from scripts.seed_demo_ledger import TIER_WEIGHTS, main

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.storage.db import open_ledger


def test_tier_weights_cover_every_tier() -> None:
    assert set(TIER_WEIGHTS) == set(ComplexityTier)


async def test_seed_writes_failures_for_cheap_tier_only(tmp_path: Path) -> None:
    path = str(tmp_path / "demo.db")
    await main(path)
    async with open_ledger(path) as ledger:
        failures = await ledger.list_failures_oldest_first()
    assert failures
    assert {failure.cheap_tier for failure in failures} == {ComplexityTier.SIMPLE}
