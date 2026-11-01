"""Regenerate every number in the README and record how it was produced.

Steps, each delegated to the script that owns it:
    1. Run the feedback-loop evaluation on the frozen eval set in eval_data/,
       A then B (scripts/eval_feedback_loop.py), which also rewrites
       docs/eval-feedback-loop.md.
    2. Write results/manifest.json: environment, models, dataset hash and
       every metric the README shows, each with its sample size.

``--timing`` instead measures each tier's Ollama latency on uncached calls,
one at a time, and writes results/tier_latency.json, which the manifest
includes. Run it when nothing else uses Ollama: it records which other
models were loaded, and the README labels the numbers accordingly.

Needs the ``ml`` extra and a local Ollama with the configured models. No paid
API is called. Responses are cached under the gitignored data/eval/ollama_cache,
so a second run on the same machine only recomputes the metrics. A fresh clone
regenerates that cache, so its answers can differ slightly from the published
ones (see the manifest's reproducibility note).

    uv run --extra ml python scripts/reproduce.py
    uv run --extra ml python scripts/reproduce.py --manifest-only
    uv run --extra ml python scripts/reproduce.py --timing 20
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_cost_autopilot.classifier.complexity import ComplexityTier  # noqa: E402
from llm_cost_autopilot.config import Settings  # noqa: E402
from llm_cost_autopilot.router.models import ModelConfig, build_default_registry  # noqa: E402
from llm_cost_autopilot.router.pricing import (  # noqa: E402
    MODEL_PRICES,
    PRICES_RETRIEVED_ON,
    PRICES_SOURCE,
)
from scripts.eval_feedback_loop import (  # noqa: E402
    EVAL_SET,
    EVAL_SET_PARAPHRASE_MODEL,
    OUT_DIR,
    PREFETCH_CONCURRENCY,
    CachingOllamaClient,
    ask,
    is_correct,
    load,
    paired_gain_ci,
    summarize,
    verifier_precision_recall,
)

MANIFEST = ROOT / "results" / "manifest.json"
TIER_LATENCY = ROOT / "results" / "tier_latency.json"
SEED = 42
PACKAGES = (
    "llm-cost-autopilot",
    "numpy",
    "pydantic",
    "sentence-transformers",
    "torch",
    "transformers",
)


def run_step(module: str, *args: str) -> None:
    """Run ``python -m module`` from the repo root with this interpreter; stop on failure."""
    print(f"$ python -m {module} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)  # noqa: S603


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    """HEAD's commit hash, or ``uncommitted`` when the working tree differs from it."""
    git = ["git", "-C", str(ROOT)]
    dirty = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True)  # noqa: S603
    if dirty.returncode != 0 or dirty.stdout.strip():
        return "uncommitted"
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)  # noqa: S603
    return head.stdout.strip()


def hardware() -> dict[str, Any]:
    """CPU, memory and OS of the machine that produced the numbers."""
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "cpu": cpu,
        "memory_gb": round(memory / 1024**3, 1),
        "os": platform.platform(),
    }


def ollama_models(settings: Settings) -> dict[str, Any]:
    """Server version and the digest of each configured model, read from the local Ollama."""
    with httpx.Client(base_url=settings.ollama_base_url, timeout=10.0) as client:
        version = client.get("/api/version").json()["version"]
        tags = {m["name"]: m["digest"] for m in client.get("/api/tags").json()["models"]}
    return {
        "server_version": version,
        "models": {
            role: {"name": name, "digest": tags.get(name, "not pulled")}
            for role, name in (
                ("simple_tier", settings.simple_model),
                ("moderate_tier", settings.moderate_model),
            )
        },
    }


def huggingface_revision(model_id: str) -> str:
    """Commit hash of the locally cached Hugging Face model, or ``not cached``."""
    from huggingface_hub import scan_cache_dir

    for repo in scan_cache_dir().repos:
        if repo.repo_id == model_id and repo.revisions:
            return max(repo.revisions, key=lambda r: r.last_modified).commit_hash
    return "not cached"


def metric(value: float, n: int, unit: str, description: str) -> dict[str, Any]:
    """One manifest metric; ``n`` is the number of items the value is computed over."""
    return {"value": round(value, 4), "n": n, "unit": unit, "description": description}


def tier_accuracy(run_a: list[dict[str, Any]]) -> dict[str, Any]:
    """Gold accuracy of each tier's model on every measured QA prompt, read from the cache.

    Prefetch asked both models every prompt, so this compares the tiers on
    the same 500 questions regardless of where the router sent them. The
    HTTP transport refuses every request: a cache miss raises instead of
    calling a model.
    """
    settings = Settings()
    registry = build_default_registry(settings)
    qa = [r for r in run_a if r["phase"] == "measure" and r["kind"] == "qa"]

    def refuse(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"cache miss: {request.content[:120]!r}")

    async def score() -> dict[str, Any]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(refuse), base_url="http://cache-only"
        ) as http:
            ollama = CachingOllamaClient(http, OUT_DIR / "ollama_cache", settings.ollama_think)
            out: dict[str, Any] = {}
            for tier in ComplexityTier:
                model = registry.for_tier(tier)
                right = 0
                for row in qa:
                    answer = await ask(ollama, model, row["prompt"], SEED)
                    right += is_correct(answer.content, row["answers"])
                out[f"{tier.value}_model_accuracy_qa"] = metric(
                    right / len(qa),
                    len(qa),
                    "fraction",
                    f"Gold accuracy of {model.ollama_model} on every measured QA prompt",
                )
            return out

    return asyncio.run(score())


def metrics(run_a: list[dict[str, Any]], run_b: list[dict[str, Any]]) -> dict[str, Any]:
    """Every number the README shows, from the two runs."""
    sa, sb = summarize(run_a), summarize(run_b)
    precision, recall, flagged = verifier_precision_recall(run_a)
    p_gain = paired_gain_ci(run_a, run_b, "paraphrase_holdout", SEED)
    i_gain = paired_gain_ci(run_a, run_b, "intent_holdout", SEED)
    measured_a = [r for r in run_a if r["phase"] == "measure"]
    qa_a = [r for r in measured_a if r["kind"] == "qa"]
    # Same population as verifier_precision_recall: every QA request of run A.
    qa_wrong_cheap = [
        r
        for r in run_a
        if r["kind"] == "qa" and r["verifier"] in ("agreed", "diverged") and not r["correct"]
    ]
    pairs = [(a, b) for a, b in zip(run_a, run_b, strict=True) if a["phase"] == "measure"]
    escalated = [(a, b) for a, b in pairs if b["stage"] == "correction_memory"]
    n_req, n_qa = len(measured_a), len(qa_a)
    n_para = sum(r["split"] == "paraphrase_holdout" for r in qa_a)
    n_intent = sum(r["split"] == "intent_holdout" for r in qa_a)

    def latency(stage: str) -> tuple[float, int]:
        values = [r["classifier_latency_ms"] for r in measured_a if r["stage"] == stage]
        return (statistics.median(values) if values else 0.0), len(values)

    stage1_ms, stage1_n = latency("stage_1_embedding")
    stage2_ms, stage2_n = latency("stage_2_zero_shot")
    out: dict[str, Any] = {}
    for label, s, rows in (("a", sa, run_a), ("b", sb, run_b)):
        corrections = sum(
            r["verifier"] == "diverged" for r in rows if r["phase"] in ("warmup", "measure")
        )
        out |= {
            f"run_{label}_cost_reduction": metric(
                s["cost_reduction"],
                n_req,
                "fraction",
                "Cost saved vs sending every request to the moderate-tier model",
            ),
            f"run_{label}_accuracy_paraphrase_holdout": metric(
                s["accuracy_paraphrase_holdout"],
                n_para,
                "fraction",
                "Gold accuracy on reworded versions of warm-up questions",
            ),
            f"run_{label}_accuracy_intent_holdout": metric(
                s["accuracy_intent_holdout"],
                n_intent,
                "fraction",
                "Gold accuracy on questions never seen in warm-up",
            ),
            f"run_{label}_cheap_share_qa": metric(
                s["cheap_share_qa"], n_qa, "fraction", "QA prompts routed to the simple tier"
            ),
            f"run_{label}_escalation_rate_qa": metric(
                s["escalation_rate_qa"],
                n_qa,
                "fraction",
                "QA prompts escalated by a past correction",
            ),
            f"run_{label}_stage_1_share": metric(
                s["stage_1_share"], n_req, "fraction", "Requests decided by stage 1 (anchors)"
            ),
            f"run_{label}_verifier_agreement": metric(
                s["verifier_agreement"],
                int(s["verified_requests"]),
                "fraction",
                "Cheap answers the verifier accepted, over requests served by the simple tier",
            ),
            f"run_{label}_corrections": metric(
                corrections,
                len(rows),
                "count",
                "Corrections recorded over warm-up and measured traffic",
            ),
        }
    out |= {
        "paraphrase_gain_points": metric(
            p_gain[0] * 100,
            p_gain[3],
            "points",
            "B - A gold accuracy, paraphrase hold-out, over intents",
        ),
        "paraphrase_gain_ci_low_points": metric(
            p_gain[1] * 100, p_gain[3], "points", "95% bootstrap CI low"
        ),
        "paraphrase_gain_ci_high_points": metric(
            p_gain[2] * 100, p_gain[3], "points", "95% bootstrap CI high"
        ),
        "intent_gain_points": metric(
            i_gain[0] * 100,
            i_gain[3],
            "points",
            "B - A gold accuracy, intent hold-out, over intents",
        ),
        "intent_gain_ci_low_points": metric(
            i_gain[1] * 100, i_gain[3], "points", "95% bootstrap CI low"
        ),
        "intent_gain_ci_high_points": metric(
            i_gain[2] * 100, i_gain[3], "points", "95% bootstrap CI high"
        ),
        "run_b_escalations": metric(
            len(escalated), len(pairs), "count", "Measured requests escalated in run B"
        ),
        "run_b_useful_escalations": metric(
            sum(a["tier"] == "simple" and not a["correct"] and b["correct"] for a, b in escalated),
            len(escalated),
            "count",
            "Escalations that turned a wrong cheap answer into a right one",
        ),
        "run_b_false_escalations": metric(
            sum(a["tier"] == "simple" and bool(a["correct"]) for a, _ in escalated),
            len(escalated),
            "count",
            "Escalations of a cheap answer that was already right",
        ),
        "verifier_precision": metric(
            precision,
            flagged,
            "fraction",
            "Run A corrections where the cheap answer was wrong against gold",
        ),
        "verifier_recall": metric(
            recall,
            len(qa_wrong_cheap),
            "fraction",
            "Wrong cheap QA answers the verifier flagged, run A warm-up and measured",
        ),
        **tier_accuracy(run_a),
        "stage_1_median_latency_ms": metric(
            stage1_ms, stage1_n, "ms", "Median classifier latency when stage 1 decides, run A"
        ),
        "stage_2_median_latency_ms": metric(
            stage2_ms, stage2_n, "ms", "Median classifier latency when stage 2 decides, run A"
        ),
    }
    return out


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of ``values``."""
    ranked = sorted(values)
    return ranked[max(0, round(fraction * len(ranked)) - 1)]


def timed_chat(
    client: httpx.Client, settings: Settings, model: ModelConfig, prompt: str
) -> dict[str, Any]:
    """One uncached /api/chat call with the router's options; returns Ollama's timings."""
    body: dict[str, Any] = {
        "model": model.ollama_model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": model.max_output_tokens, "temperature": 0.0, "seed": SEED},
    }
    if settings.ollama_think is not None:
        body["think"] = settings.ollama_think
    response = client.post("/api/chat", json=body)
    response.raise_for_status()
    return response.json()


def measure_tier_latency(prompt_count: int) -> dict[str, Any]:
    """Time ``prompt_count`` uncached NQ-Open questions per tier, one call at a time.

    Calls Ollama directly instead of through the response cache, with the same
    ``think``, temperature, seed and output cap as the router, so the numbers
    are what a request pays on this machine. One unmeasured call per model
    loads it first, so load time is not counted.
    """
    settings = Settings()
    registry = build_default_registry(settings)
    questions = [row["question"] for row in load(EVAL_SET)[:prompt_count]]
    with httpx.Client(base_url=settings.ollama_base_url, timeout=600.0) as client:
        configured = {settings.simple_model, settings.moderate_model}
        loaded = [m["name"] for m in client.get("/api/ps").json()["models"]]
        others = sorted(set(loaded) - configured)
        tiers: dict[str, Any] = {}
        for tier in ComplexityTier:
            model = registry.for_tier(tier)

            timed_chat(client, settings, model, questions[0])
            calls = [timed_chat(client, settings, model, q) for q in questions]
            total_ms = [c["total_duration"] / 1e6 for c in calls]
            tokens_per_s = [c["eval_count"] / (c["eval_duration"] / 1e9) for c in calls]
            tiers[tier.value] = {
                "model": model.ollama_model,
                "n": len(calls),
                "median_latency_ms": round(statistics.median(total_ms), 1),
                "p95_latency_ms": round(percentile(total_ms, 0.95), 1),
                "median_output_tokens": statistics.median(c["eval_count"] for c in calls),
                "median_output_tokens_per_s": round(statistics.median(tokens_per_s), 1),
            }
    result = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "other_ollama_models_loaded": others,
        "conditions": "quiet" if not others else "under contention",
        "tiers": tiers,
    }
    TIER_LATENCY.parent.mkdir(parents=True, exist_ok=True)
    TIER_LATENCY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TIER_LATENCY.relative_to(ROOT)} ({result['conditions']})")
    return result


def write_manifest() -> dict[str, Any]:
    """Compute the metrics from the saved runs and write results/manifest.json."""
    settings = Settings()
    run_a, run_b = load(OUT_DIR / "run_A.jsonl"), load(OUT_DIR / "run_B.jsonl")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "uv run --extra ml python scripts/reproduce.py",
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "hardware": hardware(),
        "ollama": ollama_models(settings),
        "classifier_models": {
            "stage_1": {
                "name": settings.stage1_model,
                "revision": huggingface_revision(settings.stage1_model),
            },
            "stage_2": {
                "name": settings.stage2_model,
                "revision": huggingface_revision(settings.stage2_model),
            },
        },
        "simulated_prices": {
            "source": PRICES_SOURCE,
            "retrieved_on": PRICES_RETRIEVED_ON,
            "usd_per_1m_tokens": {
                model: price.model_dump() for model, price in MODEL_PRICES.items()
            },
        },
        "seed": SEED,
        "dataset": {
            "path": str(EVAL_SET.relative_to(ROOT)),
            "sha256": sha256(EVAL_SET),
            "source": "google-research-datasets/nq_open, validation split, CC BY-SA 3.0",
            "paraphrase_model": EVAL_SET_PARAPHRASE_MODEL,
            "intents": len(load(EVAL_SET)),
        },
        "temperature": 0.0,
        "prefetch_concurrency": PREFETCH_CONCURRENCY,
        "reproducibility_note": (
            "Answers were generated with this many concurrent Ollama calls, and concurrent "
            "inference is not bit-identical to one-at-a-time inference even at temperature 0. "
            "The response cache under data/eval/ollama_cache on the machine that produced "
            "these numbers is the reproducibility source; a fresh clone regenerates it."
        ),
        "metrics": metrics(run_a, run_b),
        "metrics_latency_note": (
            "stage_1 and stage_2 latencies are measured inside the evaluation, on a "
            "machine shared with other workloads; tier_latency is the dedicated measurement"
        ),
        "tier_latency": (
            json.loads(TIER_LATENCY.read_text(encoding="utf-8"))
            if TIER_LATENCY.exists()
            else "not measured"
        ),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return manifest


def main() -> None:
    """Build the eval set if missing, run the evaluation, then write the manifest."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rebuild results/manifest.json from the saved runs without calling any model",
    )
    parser.add_argument(
        "--timing",
        type=int,
        metavar="PROMPTS",
        help="Measure per-tier Ollama latency on this many questions, then rewrite the manifest",
    )
    args = parser.parse_args()
    if args.timing:
        measure_tier_latency(args.timing)
    elif not args.manifest_only:
        if not EVAL_SET.exists():
            raise SystemExit(f"Frozen eval set missing: {EVAL_SET.relative_to(ROOT)}")
        run_step(
            "scripts.eval_feedback_loop",
            "--seed",
            str(SEED),
            "--prefetch-concurrency",
            str(PREFETCH_CONCURRENCY),
        )
    write_manifest()


if __name__ == "__main__":
    main()
