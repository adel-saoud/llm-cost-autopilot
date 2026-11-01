"""The evaluation's scoring, traffic split and report: the code behind the
published numbers. No model runs here; rows are synthetic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from scripts.eval_feedback_loop import (
    CachingOllamaClient,
    ask,
    build_traffic,
    is_correct,
    normalize,
    prefetch,
    report,
)

from llm_cost_autopilot.classifier.complexity import ComplexityTier
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.router import build_default_registry
from llm_cost_autopilot.verifier import Verifier
from llm_cost_autopilot.verifier.feedback import RoutingFailure


def test_normalize_drops_case_punctuation_and_articles() -> None:
    assert normalize("The  Beatles!") == "beatles"


def test_is_correct_matches_whole_words_only() -> None:
    assert is_correct("It was painted by Leonardo da Vinci.", ["Leonardo da Vinci"])
    assert not is_correct("I can't help with that.", ["Leonardo da Vinci"])
    assert not is_correct("Born in 19455.", ["1945"])
    assert not is_correct("anything", ["the"])  # normalises to empty: never a match


def _intents(n: int) -> list[dict[str, Any]]:
    return [
        {
            "intent_id": f"i{k}",
            "question": f"Q{k}?",
            "paraphrases": [f"P{k}a?", f"P{k}b?", f"P{k}c?"],
            "answers": [f"a{k}"],
        }
        for k in range(n)
    ]


def test_traffic_holds_out_paraphrases_and_whole_intents() -> None:
    warmup, measure = build_traffic(_intents(20), seed=1)
    warm_prompts = {i.prompt for i in warmup}
    qa = [i for i in measure if i.kind == "qa"]

    assert len(warmup) == 20  # 10 paraphrase hold-out intents x 2
    assert not warm_prompts & {i.prompt for i in qa}
    seen_intents = {i.intent_id for i in warmup}
    assert {i.intent_id for i in qa if i.split == "paraphrase_holdout"} == seen_intents
    assert not seen_intents & {i.intent_id for i in qa if i.split == "intent_holdout"}
    # 40 QA prompts measured, mixed 50/35/15 with moderate and complex prompts.
    assert len(qa) == 40
    assert sum(i.kind == "moderate" for i in measure) == 28
    assert sum(i.kind == "multi_step" for i in measure) == 12
    # Same seed, same traffic: both runs replay identical streams.
    assert build_traffic(_intents(20), seed=1) == (warmup, measure)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": "measure",
        "kind": "qa",
        "split": "paraphrase_holdout",
        "intent_id": "i0",
        "tier": "simple",
        "stage": "stage_1_embedding",
        "cost_usd": 0.1,
        "baseline_cost_usd": 1.0,
        "verifier": "agreed",
        "correct": True,
    }
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_report_scores_served_answers_and_applies_the_claim_bar(tmp_path: Path) -> None:
    run_a = [
        # Cheap model wrong, verifier flags it: a true correction.
        _row(intent_id="i0", correct=False, verifier="diverged"),
        _row(intent_id="i1", correct=True),
        _row(intent_id="i2", split="intent_holdout", correct=True),
        _row(
            kind="moderate",
            split="",
            intent_id="",
            tier="moderate",
            correct=None,
            verifier="skipped_standard_tier",
            cost_usd=1.0,
        ),
    ]
    run_b = [
        # Escalated by the memory and now right: a useful escalation.
        _row(
            intent_id="i0",
            tier="moderate",
            stage="correction_memory",
            correct=True,
            verifier="skipped_standard_tier",
            cost_usd=1.0,
        ),
        _row(intent_id="i1", correct=True),
        _row(intent_id="i2", split="intent_holdout", correct=True),
        _row(
            kind="moderate",
            split="",
            intent_id="",
            tier="moderate",
            correct=None,
            verifier="skipped_standard_tier",
            cost_usd=1.0,
        ),
    ]
    text = report(_write(tmp_path / "a.jsonl", run_a), _write(tmp_path / "b.jsonl", run_b), 0)

    assert "| Gold accuracy, paraphrase hold-out | 50.0% | 100.0% |" in text
    assert (
        "useful (A's cheap answer wrong, B right): 1; false (A's cheap answer already right): 0"
        in text
    )
    assert "precision 100.0%" in text
    assert "| Requests decided by stage 1 (anchors) | 100.0% | 75.0% |" in text
    # The cost bar fails (B costs far more than A), so the claim is withheld.
    assert "- [ ] cost increase <= 10%" in text
    assert "Verdict: claim NOT supported." in text


class _AlwaysAgrees:
    async def score(self, first_text: str, second_text: str) -> float:
        return 1.0


class _NoStore:
    async def record_failure(self, failure: RoutingFailure) -> None:
        raise AssertionError("an agreeing verifier records nothing")


async def test_replay_after_prefetch_makes_no_model_calls(tmp_path: Path) -> None:
    """Prefetch must cover the routed call on each tier and the verifier's call."""
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body["model"])
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": "an answer"},
                "done": True,
                "prompt_eval_count": 3,
                "eval_count": 2,
            },
        )

    settings = Settings()
    registry = build_default_registry(settings)
    prompts = ["Who painted the Mona Lisa?", "Plan a migration.", "Who painted the Mona Lisa?"]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.test"
    ) as http:
        ollama = CachingOllamaClient(http, tmp_path, think=False)
        covered = await prefetch(ollama, registry, prompts, seed=7, concurrency=2)
        assert covered == len(requests) == 4  # 2 unique prompts x 2 tier models

        ollama.misses = 0
        verifier = Verifier(
            ollama=ollama,
            standard_model=registry.standard_model,
            scorer=_AlwaysAgrees(),
            store=_NoStore(),
            divergence_threshold=0.7,
            temperature=0.0,
            seed=7,
        )
        for prompt in prompts:
            for tier in ComplexityTier:
                response = await ask(ollama, registry.for_tier(tier), prompt, seed=7)
                await verifier.verify_in_background(
                    request_id="r",
                    tenant_id="t",
                    prompt=prompt,
                    cheap_response=response,
                    cheap_tier=tier,
                    classifier_confidence=0.5,
                )

    assert ollama.misses == 0
    assert len(requests) == 4
