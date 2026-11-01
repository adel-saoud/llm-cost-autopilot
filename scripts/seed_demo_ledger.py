"""Seed a demo SQLite ledger with synthetic routing decisions + corrections.

Used to populate the dashboard for screenshots and manual QA without having
to run the load test (which needs Ollama). Pure synthetic data: no LLM calls.

Run:
    uv run python scripts/seed_demo_ledger.py [path]
"""

from __future__ import annotations

import asyncio
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta

from llm_cost_autopilot.classifier.complexity import ClassifierStage, ComplexityTier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.router.models import build_default_registry
from llm_cost_autopilot.storage.db import RoutingLogEntry, open_ledger
from llm_cost_autopilot.verifier.feedback import RoutingFailure

TIER_WEIGHTS: dict[ComplexityTier, float] = {
    ComplexityTier.SIMPLE: 0.65,
    ComplexityTier.MODERATE: 0.35,
}
"""Traffic mix, skewed toward SIMPLE so the dashboard shows savings. Keyed by tier
so a tier added to or removed from `ComplexityTier` fails the seed test."""


async def main(path: str) -> None:
    """Write 240 synthetic routing rows over six hours, with failures that thin out."""
    settings = Settings()
    registry = build_default_registry(settings)
    rng = random.Random(42)
    start = datetime.now(UTC) - timedelta(hours=6)

    async with open_ledger(path) as ledger:
        request_count = 240
        for index in range(request_count):
            tier = rng.choices(list(TIER_WEIGHTS), weights=list(TIER_WEIGHTS.values()), k=1)[0]
            cfg = registry.for_tier(tier)
            prompt_tokens = rng.randint(50, 400)
            completion_tokens = rng.randint(40, 600)
            cost = cfg.cost_usd(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            baseline = registry.standard_model.cost_usd(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            entry = RoutingLogEntry(
                request_id=request_id,
                prompt=f"Synthetic prompt #{index}",
                classified_tier=tier,
                classifier_confidence=round(rng.uniform(0.65, 0.98), 2),
                classifier_stage=(
                    ClassifierStage.STAGE_1_EMBEDDING
                    if rng.random() > 0.25
                    else ClassifierStage.STAGE_2_ZERO_SHOT
                ),
                model_used=cfg.ollama_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                baseline_cost_usd=baseline,
                latency_ms=round(rng.uniform(80, 1200), 1),
                created_at=start + timedelta(minutes=index * 1.5),
            )
            await ledger.record_routing(entry)

            # Routing accuracy drifts up over time: early failures, fewer later.
            if cfg.ollama_model != registry.standard_model.ollama_model:
                failure_probability = max(0.05, 0.30 - index / request_count * 0.25)
                if rng.random() < failure_probability:
                    similarity = round(rng.uniform(0.30, 0.65), 2)
                    await ledger.record_failure(
                        RoutingFailure(
                            request_id=request_id,
                            prompt=entry.prompt,
                            cheap_model=cfg.ollama_model,
                            cheap_response="Synthetic cheap response, concise.",
                            cheap_tier=tier,
                            classifier_confidence=entry.classifier_confidence,
                            standard_model=registry.standard_model.ollama_model,
                            standard_response=(
                                "Synthetic standard response, longer and more "
                                "carefully reasoned, with additional caveats."
                            ),
                            similarity_score=similarity,
                            divergence_threshold=settings.verifier_divergence_threshold,
                            created_at=entry.created_at,
                        )
                    )

    print(f"Seeded ledger at {path}: {request_count} routing decisions.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "autopilot-demo.db"
    asyncio.run(main(target))
