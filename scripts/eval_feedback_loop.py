"""Measure the correction feedback loop: run A (memory off) vs run B (memory on).

Both runs replay the same seeded traffic through the real classifier, router,
verifier and local Ollama models, one request at a time so the memory state
is identical at every step except for what the loop itself learns.

Traffic:
    Intents from eval_data/nq_open_paraphrases.jsonl are split in half.
    * Paraphrase hold-out intents: the question and its first paraphrase are
      sent during warm-up; the other two paraphrases are measured. This asks
      "does a past mistake stop repeating on reworded prompts?"
    * Intent hold-out intents: never seen during warm-up; the question and
      its first paraphrase are measured. This asks "does the loop escalate
      prompts it has no reason to?"
    Prompts from the moderate and multi-step pools of scripts/load_test.py are
    mixed into the measured stream in the 50/35/15 mix of the original load
    test. Pools are prompt sources, not tiers: the router has two tiers.

Quality is the gold accuracy of the answer actually served (NQ-Open answers,
normalised containment match), not agreement with the standard model, which
escalation would raise by construction. Ollama runs at temperature 0 with a
fixed seed, and responses are cached on disk, so both runs see identical
answers for identical calls.

The run has two phases. Prefetch asks both models for every prompt, a few
calls at a time, and stores the answers in the response cache. The replay
then runs A and B one request at a time as described above and is served
entirely from that cache, so learning order is kept without paying for
sequential model calls. Concurrent inference is not bit-identical to
one-at-a-time inference, even at temperature 0, so the cache, not a fresh
rerun, is what makes the published numbers reproducible.

The claim bar below is fixed before any run.

    uv run python -m scripts.eval_feedback_loop
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import structlog

from llm_cost_autopilot.classifier import TwoStageClassifier
from llm_cost_autopilot.classifier.correction_memory import CorrectionMemory
from llm_cost_autopilot.classifier.huggingface_backends import (
    DebertaZeroShotBackend,
    SentenceTransformerEmbeddingBackend,
)
from llm_cost_autopilot.config import Settings
from llm_cost_autopilot.cost.tracker import CostTracker
from llm_cost_autopilot.router import (
    ModelConfig,
    ModelRegistry,
    OllamaClient,
    build_default_registry,
)
from llm_cost_autopilot.router.ollama import ChatMessage, OllamaChatResponse
from llm_cost_autopilot.storage.db import open_ledger
from llm_cost_autopilot.verifier import CorrectionStore, Verifier
from llm_cost_autopilot.verifier.correction_feedback import FeedbackCorrectionStore
from llm_cost_autopilot.verifier.embedding_scorer import EmbeddingSimilarityScorer
from scripts.load_test import MODERATE_PROMPTS, MULTI_STEP_PROMPTS

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "eval_data" / "nq_open_paraphrases.jsonl"
"""Frozen eval set behind the published numbers, committed with its CC BY-SA 3.0 notice.

Built once by scripts/build_feedback_eval_set.py; that script writes to the
gitignored data/eval/ so a rebuild never overwrites this file.
"""
EVAL_SET_PARAPHRASE_MODEL = "llama3.2:3b"
"""Model that wrote the paraphrases of the frozen eval set behind the published numbers."""
OLLAMA_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
"""Fails a stalled call (a laptop sleep drops the connection) in minutes, not hours.

120 s covers a 1,024-token answer from the moderate model with room for a
short queue; ``OllamaClient`` retries a timeout before the run stops.
"""
PREFETCH_CONCURRENCY = 4
"""Ollama calls in flight while prefetching; the replay itself stays sequential."""
OUT_DIR = ROOT / "data" / "eval"
REPORT = ROOT / "docs" / "eval-feedback-loop.md"
EVAL_TENANT_ID = "eval"
"""Single tenant for the whole replay, so every correction can apply to every later prompt."""

# Pre-registered claim bar: a feedback-loop gain may be claimed only if all four hold.
MIN_VERIFIER_PRECISION = 0.70
"""Share of verifier corrections where the cheap answer was really wrong."""
MIN_PARAPHRASE_GAIN_CI_LOW = 0.0
"""Lower bound of the 95% CI of (B - A) gold accuracy on paraphrase hold-out."""
MIN_INTENT_GAIN_CI_LOW = -0.02
"""Lower bound of the 95% CI of (B - A) gold accuracy on intent hold-out (no regression)."""
MAX_COST_INCREASE = 0.10
"""Run B may cost at most 10% more than run A on the measured stream."""

BOOTSTRAP_SAMPLES = 10_000
"""Resamples for the paired bootstrap CI over intents."""


def normalize(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation, articles and extra spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(response: str, answers: list[str]) -> bool:
    """True when any gold answer appears in the response after normalisation.

    Containment is lenient (a long answer that mentions the gold string
    counts), which favours the verbose standard model slightly; the same
    rule applies to both runs, so the A/B difference is unaffected.
    """
    haystack = f" {normalize(response)} "
    return any(f" {normalize(a)} " in haystack for a in answers if normalize(a))


class CachingOllamaClient(OllamaClient):
    """Ollama client that replays identical calls from disk.

    With temperature 0 and a fixed seed the cache only removes run-to-run
    noise and time: run B re-serves run A's exact answer for every call both
    runs make, so any A/B difference comes from routing alone.
    """

    def __init__(self, http_client: httpx.AsyncClient, cache_dir: Path, think: bool | None) -> None:
        super().__init__(http_client, think=think)
        self._cache_dir = cache_dir
        self.misses = 0
        """Calls that reached Ollama; zero during a replay after a full prefetch."""
        cache_dir.mkdir(parents=True, exist_ok=True)

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> OllamaChatResponse:
        """Return the cached response for these exact arguments, calling Ollama on a miss."""
        key = hashlib.sha256(
            json.dumps(
                [
                    model,
                    [m.model_dump() for m in messages],
                    max_output_tokens,
                    temperature,
                    seed,
                    self._think,
                ]
            ).encode()
        ).hexdigest()
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return OllamaChatResponse.model_validate_json(path.read_text(encoding="utf-8"))
        self.misses += 1
        response = await super().chat(
            model=model,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
        )
        path.write_text(response.model_dump_json(), encoding="utf-8")
        return response


async def ask(
    ollama: OllamaClient, model: ModelConfig, prompt: str, seed: int
) -> OllamaChatResponse:
    """One user prompt on ``model``, with the arguments the verifier also uses.

    The routed call and the verifier's call on the standard model share
    these arguments, so one cached answer serves both and a prefetch
    through this function covers every call the replay makes.
    """
    return await ollama.chat(
        model=model.ollama_model,
        messages=[ChatMessage(role="user", content=prompt)],
        max_output_tokens=model.max_output_tokens,
        temperature=0.0,
        seed=seed,
    )


async def prefetch(
    ollama: OllamaClient,
    registry: ModelRegistry,
    prompts: list[str],
    *,
    seed: int,
    concurrency: int,
) -> int:
    """Cache every answer the replay can need, ``concurrency`` calls at a time.

    Routing is unknown before the replay, so each prompt is asked on every
    tier's model. Returns the number of (prompt, model) pairs covered.
    """
    slots = asyncio.Semaphore(concurrency)
    models = list(dict.fromkeys(registry.entries.values()))
    pairs = [(model, prompt) for prompt in dict.fromkeys(prompts) for model in models]
    done = 0

    async def one(model: ModelConfig, prompt: str) -> None:
        nonlocal done
        async with slots:
            await ask(ollama, model, prompt, seed)
        done += 1
        if done % 200 == 0:
            print(f"[prefetch] {done}/{len(pairs)}", flush=True)

    await asyncio.gather(*(one(model, prompt) for model, prompt in pairs))
    return len(pairs)


@dataclass(frozen=True)
class TrafficItem:
    """One request of the replayed traffic, with what it takes to score it."""

    phase: str
    """``warmup`` (teaches the memory, not measured) or ``measure``."""
    kind: str
    """``qa`` (gold-scored NQ-Open prompt), or the load-test pool: ``moderate`` or ``complex``."""
    split: str
    """``paraphrase_holdout``, ``intent_holdout``, or empty for non-QA prompts."""
    intent_id: str
    prompt: str
    answers: tuple[str, ...]
    """Gold answers; empty for non-QA prompts."""


def build_traffic(
    intents: list[dict[str, Any]], seed: int
) -> tuple[list[TrafficItem], list[TrafficItem]]:
    """Split intents into hold-outs and return ``(warmup, measure)``, both shuffled by ``seed``."""
    rng = random.Random(seed)
    shuffled = intents[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    warmup: list[TrafficItem] = []
    measure: list[TrafficItem] = []
    for index, intent in enumerate(shuffled):
        answers = tuple(intent["answers"])
        variants = [intent["question"], *intent["paraphrases"]]
        if index < half:
            for prompt in variants[:2]:
                warmup.append(
                    TrafficItem(
                        "warmup", "qa", "paraphrase_holdout", intent["intent_id"], prompt, answers
                    )
                )
            for prompt in variants[2:4]:
                measure.append(
                    TrafficItem(
                        "measure", "qa", "paraphrase_holdout", intent["intent_id"], prompt, answers
                    )
                )
        else:
            for prompt in variants[:2]:
                measure.append(
                    TrafficItem(
                        "measure", "qa", "intent_holdout", intent["intent_id"], prompt, answers
                    )
                )
    qa_count = len(measure)
    for kind, pool, share in (
        ("moderate", MODERATE_PROMPTS, 0.35),
        ("multi_step", MULTI_STEP_PROMPTS, 0.15),
    ):
        for _ in range(round(qa_count * share / 0.50)):
            measure.append(TrafficItem("measure", kind, "", "", rng.choice(pool), ()))
    rng.shuffle(warmup)
    rng.shuffle(measure)
    return warmup, measure


async def run(
    label: str,
    *,
    feedback: bool,
    warmup: list[TrafficItem],
    measure: list[TrafficItem],
    settings: Settings,
    embedding: SentenceTransformerEmbeddingBackend,
    zero_shot: DebertaZeroShotBackend,
    scorer: EmbeddingSimilarityScorer,
    seed: int,
    out_dir: Path,
    cache_dir: Path,
) -> Path:
    """Replay warm-up then measured traffic through one pipeline; return the run's JSONL path.

    Requests run one at a time and each verification is awaited, so the
    memory state at every step depends only on the traffic before it.
    """
    registry = build_default_registry(settings)
    memory = (
        CorrectionMemory(max_per_tenant=settings.correction_max_per_tenant) if feedback else None
    )
    classifier = TwoStageClassifier(
        embedding,
        zero_shot,
        stage1_threshold=settings.stage1_threshold,
        correction_memory=memory,
        correction_match_threshold=settings.correction_match_threshold,
    )
    out = out_dir / f"run_{label}.jsonl"
    # Kept after the run so the dashboard can be captured on measured traffic.
    ledger_path = out_dir / f"ledger_{label}.db"
    for stale in out_dir.glob(f"ledger_{label}.db*"):
        stale.unlink()
    async with (
        httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=OLLAMA_TIMEOUT) as http_client,
        open_ledger(ledger_path) as ledger,
    ):
        ollama = CachingOllamaClient(http_client, cache_dir, settings.ollama_think)
        store: CorrectionStore = (
            ledger if memory is None else FeedbackCorrectionStore(ledger, memory, embedding)
        )
        verifier = Verifier(
            ollama=ollama,
            standard_model=registry.standard_model,
            scorer=scorer,
            store=store,
            divergence_threshold=settings.verifier_divergence_threshold,
            temperature=0.0,
            seed=seed,
        )
        tracker = CostTracker(ledger=ledger, standard_model=registry.standard_model)
        with out.open("w", encoding="utf-8") as out_file:
            for step, item in enumerate([*warmup, *measure]):
                classification = await asyncio.to_thread(
                    classifier.classify, item.prompt, tenant_id=EVAL_TENANT_ID
                )
                routed_model = registry.for_tier(classification.tier)
                response = await ask(ollama, routed_model, item.prompt, seed)
                request_id = f"{label}-{step}"
                savings = await tracker.record(
                    request_id=request_id,
                    tenant_id=EVAL_TENANT_ID,
                    prompt=item.prompt,
                    classification=classification,
                    model_used=routed_model,
                    response=response,
                )
                # Awaited, not fire-and-forget: the next request must see
                # whatever this one taught the memory.
                outcome = await verifier.verify_in_background(
                    request_id=request_id,
                    tenant_id=EVAL_TENANT_ID,
                    prompt=item.prompt,
                    cheap_response=response,
                    cheap_tier=classification.tier,
                    classifier_confidence=classification.confidence,
                )
                record = {
                    **asdict(item),
                    "step": step,
                    "tier": classification.tier.value,
                    "stage": classification.stage.value,
                    "classifier_latency_ms": classification.latency_ms,
                    "matched_correction_request_id": classification.matched_correction_request_id,
                    "model": routed_model.ollama_model,
                    "response": response.content,
                    "cost_usd": savings.actual_cost_usd,
                    "baseline_cost_usd": savings.baseline_cost_usd,
                    "verifier": outcome.value,
                    "correct": is_correct(response.content, list(item.answers))
                    if item.kind == "qa"
                    else None,
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if step % 100 == 0:
                    print(f"[{label}] {step}/{len(warmup) + len(measure)}", flush=True)
        print(f"[{label}] corrections in memory: {memory.correction_count() if memory else 0}")
        if ollama.misses:
            print(
                f"[{label}] WARNING: {ollama.misses} Ollama calls missed the prefetched cache; "
                "the replay's call arguments differ from prefetch()",
                flush=True,
            )
    return out


def load(path: Path) -> list[dict[str, Any]]:
    """Read one run's JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def paired_gain_ci(
    a: list[dict[str, Any]], b: list[dict[str, Any]], split: str, seed: int
) -> tuple[float, float, float, int]:
    """Mean per-intent (B - A) gold accuracy with a 95% bootstrap CI over intents."""
    by_intent: dict[str, list[float]] = {}
    for ra, rb in zip(a, b, strict=True):
        if ra["phase"] == "measure" and ra["kind"] == "qa" and ra["split"] == split:
            by_intent.setdefault(ra["intent_id"], []).append(
                float(rb["correct"]) - float(ra["correct"])
            )
    gains = np.array([np.mean(v) for v in by_intent.values()])
    rng = np.random.default_rng(seed)
    samples = rng.choice(gains, size=(BOOTSTRAP_SAMPLES, gains.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(gains.mean()), float(low), float(high), int(gains.size)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Cost, gold accuracy per split, and routing shares over one run's measured requests."""
    measured = [r for r in rows if r["phase"] == "measure"]
    qa = [r for r in measured if r["kind"] == "qa"]
    cost = sum(r["cost_usd"] for r in measured)
    baseline = sum(r["baseline_cost_usd"] for r in measured)
    verifiable = [r for r in measured if r["verifier"] in ("agreed", "diverged")]
    diverged = [r for r in verifiable if r["verifier"] == "diverged"]

    def accuracy(split: str) -> float:
        subset = [r for r in qa if r["split"] == split]
        return sum(bool(r["correct"]) for r in subset) / len(subset)

    return {
        "requests": float(len(measured)),
        "cost_usd": cost,
        "cost_reduction": 1.0 - cost / baseline,
        "accuracy_paraphrase_holdout": accuracy("paraphrase_holdout"),
        "accuracy_intent_holdout": accuracy("intent_holdout"),
        "verifier_agreement": 1.0 - len(diverged) / len(verifiable) if verifiable else 1.0,
        "verified_requests": float(len(verifiable)),
        "escalation_rate_qa": sum(r["stage"] == "correction_memory" for r in qa) / len(qa),
        "cheap_share_qa": sum(r["tier"] == "simple" for r in qa) / len(qa),
        "stage_1_share": sum(r["stage"] == "stage_1_embedding" for r in measured) / len(measured),
    }


def verifier_precision_recall(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    """Precision and recall of the verifier against gold, over every QA request in run A."""
    cheap_qa = [r for r in rows if r["kind"] == "qa" and r["verifier"] in ("agreed", "diverged")]
    flagged = [r for r in cheap_qa if r["verifier"] == "diverged"]
    wrong = [r for r in cheap_qa if not r["correct"]]
    true_flags = [r for r in flagged if not r["correct"]]
    precision = len(true_flags) / len(flagged) if flagged else 0.0
    recall = len(true_flags) / len(wrong) if wrong else 0.0
    return precision, recall, len(flagged)


def report(path_a: Path, path_b: Path, seed: int) -> str:
    """Render the Markdown report, including the claim-bar verdict, from both runs."""
    a, b = load(path_a), load(path_b)
    sa, sb = summarize(a), summarize(b)
    precision, recall, flagged = verifier_precision_recall(a)
    p_gain = paired_gain_ci(a, b, "paraphrase_holdout", seed)
    i_gain = paired_gain_ci(a, b, "intent_holdout", seed)
    cost_increase = sb["cost_usd"] / sa["cost_usd"] - 1.0

    measured = [(ra, rb) for ra, rb in zip(a, b, strict=True) if ra["phase"] == "measure"]
    escalated = [(ra, rb) for ra, rb in measured if rb["stage"] == "correction_memory"]
    false_escalations = sum(1 for ra, _ in escalated if ra["tier"] == "simple" and ra["correct"])
    useful_escalations = sum(
        1 for ra, rb in escalated if ra["tier"] == "simple" and not ra["correct"] and rb["correct"]
    )

    bar = {
        f"verifier precision >= {MIN_VERIFIER_PRECISION:.2f}": precision >= MIN_VERIFIER_PRECISION,
        "paraphrase hold-out gain CI low > 0": p_gain[1] > MIN_PARAPHRASE_GAIN_CI_LOW,
        f"intent hold-out gain CI low > {MIN_INTENT_GAIN_CI_LOW:+.2f}": i_gain[1]
        > MIN_INTENT_GAIN_CI_LOW,
        f"cost increase <= {MAX_COST_INCREASE:.0%}": cost_increase <= MAX_COST_INCREASE,
    }

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    lines = [
        "# Feedback-loop evaluation",
        "",
        "Generated by `scripts/eval_feedback_loop.py`. Do not edit by hand.",
        "",
        f"- Eval set: {len({r['intent_id'] for r in a if r['kind'] == 'qa'})} NQ-Open intents "
        f"(CC BY-SA 3.0), paraphrases written by {EVAL_SET_PARAPHRASE_MODEL}",
        f"- Measured requests per run: {int(sa['requests'])} (50/35/15 NQ-Open questions, "
        "moderate-pool and multi-step-pool prompts)",
        "- Ollama temperature 0, fixed seed, responses cached so both runs share identical answers",
        "",
        "| Metric (measured stream) | Run A: memory off | Run B: memory on |",
        "|---|---|---|",
        f"| Cost reduction vs all-standard | {pct(sa['cost_reduction'])} | "
        f"{pct(sb['cost_reduction'])} |",
        f"| Gold accuracy, paraphrase hold-out | {pct(sa['accuracy_paraphrase_holdout'])} | "
        f"{pct(sb['accuracy_paraphrase_holdout'])} |",
        f"| Gold accuracy, intent hold-out | {pct(sa['accuracy_intent_holdout'])} | "
        f"{pct(sb['accuracy_intent_holdout'])} |",
        f"| Requests decided by stage 1 (anchors) | {pct(sa['stage_1_share'])} | "
        f"{pct(sb['stage_1_share'])} |",
        f"| Share of QA prompts on the cheap model | {pct(sa['cheap_share_qa'])} | "
        f"{pct(sb['cheap_share_qa'])} |",
        f"| QA prompts escalated by a correction | {pct(sa['escalation_rate_qa'])} | "
        f"{pct(sb['escalation_rate_qa'])} |",
        f"| Verifier agreement (not a quality metric) | {pct(sa['verifier_agreement'])} | "
        f"{pct(sb['verifier_agreement'])} |",
        "",
        "| Paired difference B - A (95% bootstrap CI over intents) | Gain | CI | Intents |",
        "|---|---|---|---|",
        f"| Paraphrase hold-out | {p_gain[0] * 100:+.1f} pts | "
        f"[{p_gain[1] * 100:+.1f}, {p_gain[2] * 100:+.1f}] | {p_gain[3]} |",
        f"| Intent hold-out | {i_gain[0] * 100:+.1f} pts | "
        f"[{i_gain[1] * 100:+.1f}, {i_gain[2] * 100:+.1f}] | {i_gain[3]} |",
        "",
        f"- Cost change B vs A: {cost_increase * 100:+.1f}%",
        f"- Escalations in run B: {len(escalated)}; useful (A's cheap answer wrong, B right): "
        f"{useful_escalations}; false (A's cheap answer already right): {false_escalations}",
        f"- Verifier vs gold (run A, all QA requests): precision {pct(precision)}, "
        f"recall {pct(recall)}, {flagged} corrections",
        "",
        "## Pre-registered claim bar",
        "",
        *[f"- [{'x' if ok else ' '}] {name}" for name, ok in bar.items()],
        "",
        f"Verdict: {'claim supported' if all(bar.values()) else 'claim NOT supported'}.",
        "",
    ]
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> None:
    """Load the models once, run A then B, and write the report."""
    intents = [json.loads(line) for line in args.eval_set.read_text(encoding="utf-8").splitlines()]
    warmup, measure = build_traffic(intents, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"warm-up {len(warmup)} requests, measured {len(measure)} requests per run")

    settings = Settings()
    embedding = SentenceTransformerEmbeddingBackend(settings.stage1_model)
    zero_shot = DebertaZeroShotBackend(settings.stage2_model)
    scorer = EmbeddingSimilarityScorer(settings.stage1_model)
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=OLLAMA_TIMEOUT
    ) as http_client:
        covered = await prefetch(
            CachingOllamaClient(http_client, args.out_dir / "ollama_cache", settings.ollama_think),
            build_default_registry(settings),
            [item.prompt for item in [*warmup, *measure]],
            seed=args.seed,
            concurrency=args.prefetch_concurrency,
        )
    print(f"[prefetch] {covered} prompt and model pairs cached")
    common: dict[str, Any] = {
        "warmup": warmup,
        "measure": measure,
        "settings": settings,
        "embedding": embedding,
        "zero_shot": zero_shot,
        "scorer": scorer,
        "seed": args.seed,
        "out_dir": args.out_dir,
        "cache_dir": args.out_dir / "ollama_cache",
    }
    path_a = await run("A", feedback=False, **common)
    path_b = await run("B", feedback=True, **common)
    text = report(path_a, path_b, args.seed)
    args.report.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    """Parse the CLI; ``--report-only`` rebuilds the report without calling any model."""
    # Per-request debug lines would bury the progress output of a long run.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Where run_A/run_B go")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument(
        "--prefetch-concurrency",
        type=int,
        default=PREFETCH_CONCURRENCY,
        help="Ollama calls in flight during prefetch (the replay is always sequential)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the report from existing run_A/run_B files without calling any model",
    )
    args = parser.parse_args()
    if args.report_only:
        text = report(args.out_dir / "run_A.jsonl", args.out_dir / "run_B.jsonl", args.seed)
        args.report.write_text(text, encoding="utf-8")
        print(text)
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
