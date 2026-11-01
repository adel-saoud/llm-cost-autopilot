"""Load test: fire synthetic prompts and report the cost-savings headline.

Default mix, by prompt pool:
    50% simple   (factual lookups, conversions, single-step tasks)
    35% moderate (summarisation, classification, extraction, light reasoning)
    15% multi-step (multi-step reasoning, planning, deep analysis)

The router has two tiers, so prompts from the moderate and multi-step pools
are both expected on the moderate tier; the multi-step pool keeps the mix heavy.

Run against a live autopilot (``docker compose up -d`` first):
    uv run python scripts/load_test.py --requests 1000 --concurrency 12
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 30 prompts per pool, varied in domain (factual, math, code, business,
# creative, support, language, planning) so the classifier does not see one
# template padded with filler words. eval_feedback_loop.py reuses the
# moderate and multi-step pools.
SIMPLE_PROMPTS = [
    "What is the capital of France?",
    "What's 15 percent of 240?",
    "Convert 100 USD to EUR at a 1.07 rate.",
    "Translate 'good morning' to Spanish.",
    "Format 2024-03-15 as DD/MM/YYYY.",
    "What day of the week was July 4, 1776?",
    "List the planets of the solar system.",
    "What is the boiling point of water in Celsius?",
    "Spell 'necessary' correctly.",
    "What's the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "How many continents are there?",
    "What year did World War 2 end?",
    "Convert 32 Fahrenheit to Celsius.",
    "What's the speed of light in km/s?",
    "Capitalize this sentence: the quick brown fox.",
    "Reverse this string: hello world.",
    "What's 2 to the power of 10?",
    "How many days are there in a leap year?",
    "What's the largest ocean on Earth?",
    "Define the word 'ephemeral'.",
    "What's the opposite of 'abundant'?",
    "What's 7 multiplied by 8?",
    "Translate 'thank you' to Japanese.",
    "List 3 primary colors.",
    "What's the currency symbol for the British pound?",
    "Convert 5 feet to centimeters.",
    "What does the acronym 'ASAP' mean?",
    "Who wrote the play 'Hamlet'?",
    "What is the capital of Australia?",
]

MODERATE_PROMPTS = [
    "Summarise this paragraph in 3 bullet points: 'Our company uses event-driven "
    "architecture for the order pipeline. It scales well during peak hours but "
    "introduces complexity around debugging and observability.'",
    "Classify this support ticket by priority (low/medium/high) and category: "
    "'My account was charged twice for the same subscription this month.'",
    "Extract all named entities (people, places, organizations) from: "
    "'Tim Cook announced at the Cupertino keynote that Apple is partnering with OpenAI.'",
    "Rewrite this email in a more professional tone: "
    "'hey just wanted to know if you got my last message about the contract'",
    "Compare these two product descriptions and list the key differences: "
    "(A) 'Wireless noise-cancelling headphones, 30hr battery'. "
    "(B) 'Bluetooth headphones with ANC, all-day battery life'.",
    "Generate 5 alternative titles for this blog post: "
    "'How we cut our AWS bill by 40% in 90 days'.",
    "Convert this meeting transcript into action items: "
    "'Alex will draft the proposal by Friday. Sam is reviewing the legal terms. "
    "Jordan owns the customer call next week.'",
    "Explain monads to a non-technical audience using one analogy.",
    "Translate this paragraph to French and keep the casual tone: "
    "'Hey, just checking if you got the package I sent on Tuesday.'",
    "Categorize these three reviews by sentiment (positive/neutral/negative): "
    "(1) 'Loved it, fast shipping!'. (2) 'It works as expected'. (3) 'Broke in a week.'",
    "Identify the main thesis of this essay opening: "
    "'Software estimation is broken because we treat it as a calculation, not a forecast.'",
    "Draft a polite decline for this invitation: "
    "'We'd love for you to keynote at our conference in October.'",
    "Generate a 100-word product description from this spec sheet: "
    "'12.9-inch OLED display, M3 chip, 16GB RAM, 1TB SSD, all-day battery.'",
    "Identify any logical inconsistencies in this argument: "
    "'All birds can fly. Penguins are birds. Therefore penguins can fly.'",
    "Summarise the main pros and cons in this short article: "
    "'Remote work boosts focus but isolates new hires from team culture.'",
    "List 5 follow-up questions to ask after reading: "
    "'Our churn rate doubled in Q3 with no obvious cause.'",
    "Convert this resume line into a quantified achievement: "
    "'Improved the onboarding flow for new users.'",
    "Suggest 3 ways to shorten this paragraph without losing meaning: "
    "'In order to make sure that we deliver the highest quality product to our customers, "
    "we have decided to extend the testing phase by an additional two weeks.'",
    "Identify the call-to-action in this marketing copy: "
    "'Save 30% this weekend only. Limited stock, free returns. Shop now.'",
    "Write a 50-word abstract for this research summary: "
    "'We trained a small transformer on 100M tokens of Python and measured "
    "completion accuracy on a held-out test set.'",
    "Generate 3 FAQ entries from this support doc: "
    "'Refunds are processed within 5 business days. Items must be unopened. "
    "Original receipt is required.'",
    "Identify any potentially sensitive PII in this text: 'Customer John Doe "
    "(DOB 1985-03-12) contacted us from john.doe@example.com about order #4421.'",
    "Rewrite this code comment to be clearer: '# do the thing with the array'.",
    "Compose a tweet (under 280 chars) summarising this announcement: 'Our "
    "open-source library now supports 12 new languages, distributed inference, "
    "and a managed cloud tier.'",
    "Classify these three issues by severity (S1/S2/S3): "
    "(1) 'Login broken for all users'. (2) 'Avatar upload fails'. (3) 'Typo in footer'.",
    "Extract date, amount, and vendor from this invoice line: "
    "'Acme Corp invoiced $1,250.00 on 2024-04-12 for cloud services.'",
    "Reformat this JSON sample with proper indentation: "
    '\'{"name":"Alice","roles":["admin","editor"],"active":true}\'.',
    "Suggest improvements to this short bio: "
    "'I am a developer with skills in many things and I love coding.'",
    "Identify the target user persona in this product brief: "
    "'A budgeting app for first-time parents juggling daycare costs and irregular income.'",
    "Convert this informal status update into a project-management ticket "
    "(title + description + priority): "
    "'login is super slow today, like 5 seconds, started after the deploy'",
]

MULTI_STEP_PROMPTS = [
    "Analyse the tradeoffs between event-driven and request/response architectures "
    "for a multi-tenant SaaS billing pipeline. Recommend an approach with justification.",
    "Write a detailed technical proposal for migrating a monolithic Django application "
    "to microservices. Cover phases, risks, observability, and rollback strategy.",
    "Debug this code snippet and explain the root cause step by step: "
    "'def avg(xs): return sum(xs) / len(xs)' - it raises ZeroDivisionError sometimes "
    "in production. Walk through why and how to fix it robustly.",
    "Draft a strategic 90-day plan with prioritised initiatives and risks for launching "
    "a fintech startup serving SMBs in Europe.",
    "Reason through this multi-step probability problem and justify each step: "
    "A bag has 3 red, 5 blue, and 2 green marbles. Two marbles are drawn without "
    "replacement. What is the probability that at least one is red?",
    "Design a database schema for a multi-tenant SaaS analytics platform. "
    "Explain normalization choices, partitioning strategy, and index design.",
    "Compare 3 approaches to LLM cost optimization (prompt caching, model routing, "
    "batch processing) and recommend one for a startup with $50k/mo in OpenAI spend.",
    "Write an incident postmortem covering root cause, contributing factors, and "
    "5 specific lessons learned, given: a payment processor outage lasting 47 minutes "
    "during Black Friday peak traffic.",
    "Walk through implementing OAuth2 with PKCE from scratch in a Next.js app. "
    "Cover token storage, refresh flow, and 3 common security pitfalls.",
    "Critique this product roadmap and propose a re-prioritization given a team of 4 "
    "engineers and 6 months runway: '(1) AI assistant. (2) Mobile app. (3) Team plan. "
    "(4) SSO. (5) Public API.'",
    "Reason through a Bayesian update problem: a test for a disease has 99% sensitivity "
    "and 95% specificity. The disease prevalence is 1%. A patient tests positive. "
    "Compute the posterior probability they have the disease and show all steps.",
    "Design a feature flag rollout plan for a payments API change affecting 100M+ users. "
    "Cover blast radius, automated rollback, and observability.",
    "Plan a multi-region failover strategy for a global e-commerce platform with "
    "sub-second SLA requirements. Cover data replication, traffic shifting, and split-brain risk.",
    "Compare and contrast 3 vector database options (pgvector, Pinecone, Qdrant) for a "
    "RAG application serving 10M queries/day. Recommend one with justification.",
    "Conduct a SWOT analysis of this business proposition: 'A subscription service "
    "delivering hand-curated, region-specific specialty coffee monthly to North America.'",
    "Walk through the mathematical derivation of backpropagation for a 2-layer neural "
    "network with sigmoid activations and MSE loss. Show each gradient step.",
    "Design a rate-limiting system for an API gateway handling burst traffic patterns. "
    "Compare token bucket, leaky bucket, and sliding window; recommend one with reasoning.",
    "Design a CI/CD pipeline for a polyglot monorepo with Python and TypeScript "
    "services. Cover dependency change detection, parallel test execution, and progressive deploy.",
    "Analyse this legal contract clause for ambiguity and suggest improvements: "
    "'The vendor shall use reasonable efforts to maintain availability of the service.'",
    "Compare 4 React state-management libraries (Redux, Zustand, Jotai, Recoil) and "
    "recommend one for a real-time collaborative editor with optimistic UI.",
    "Design a system to detect fraudulent credit-card transactions in real time at "
    "100k tx/sec with under 50ms latency. Cover model serving, fallback, and feedback loops.",
    "Draft a 5-year technical roadmap for an early-stage AI startup pivoting from "
    "B2C consumer to B2B enterprise. Cover team, product, and infra evolution.",
    "Plan an A/B test with proper statistical power to detect a 2% conversion lift "
    "on a 50k DAU site. Cover sample size calculation, guardrail metrics, and stop rules.",
    "Reason through how to scale a Postgres database from 100GB to 10TB while "
    "maintaining sub-100ms p99 latency. Cover sharding, read replicas, and connection pooling.",
    "Analyse the threat model for this authentication flow and identify 3 plausible "
    "attack vectors: 'Email magic-link sign-in with a 15-minute expiry, link clicked via SMS.'",
    "Critique this engineering hiring rubric and suggest 4 specific improvements "
    "to reduce false negatives: 'Score candidates 1-5 on coding, design, "
    "communication, and culture-add.'",
    "Design a system architecture for a chat application supporting 1M concurrent "
    "users with end-to-end encryption. Cover key management, presence, and message ordering.",
    "Walk through implementing distributed transactions using the saga pattern for "
    "an order-processing system. Cover compensating actions and idempotency.",
    "Plan a migration from a $50k/yr Snowflake bill to DuckDB + S3 for a 5TB analytics "
    "workload. Include cost model, risks, and rollback path.",
    "Build a 60-day onboarding plan for a senior staff engineer joining a 100-person "
    "platform team. Cover ramp-up, first deliverables, and influence-without-authority moves.",
]


def _build_prompts(request_count: int, rng: random.Random) -> list[tuple[str, str]]:
    """Return ``(expected_tier, prompt)`` pairs in a 50/35/15 pool mix, shuffled by ``rng``."""
    counts = {
        "simple": int(request_count * 0.50),
        "moderate": int(request_count * 0.35),
        "multi_step": 0,
    }
    counts["multi_step"] = request_count - counts["simple"] - counts["moderate"]
    pools = {
        "simple": ("simple", SIMPLE_PROMPTS),
        "moderate": ("moderate", MODERATE_PROMPTS),
        "multi_step": ("moderate", MULTI_STEP_PROMPTS),
    }
    prompts: list[tuple[str, str]] = []
    for pool_name, pool_count in counts.items():
        expected_tier, pool = pools[pool_name]
        prompts.extend((expected_tier, rng.choice(pool)) for _ in range(pool_count))
    rng.shuffle(prompts)
    return prompts


async def _send(
    client: httpx.AsyncClient,
    expected_tier: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, object]:
    """Send one prompt and return its decision headers, cost and latency, or the error."""
    async with semaphore:
        started_at = time.perf_counter()
        try:
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}]},
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc), "expected": expected_tier}

        return {
            "ok": True,
            "expected": expected_tier,
            "actual_tier": response.headers.get("x-autopilot-complexity-tier", "?"),
            "model": response.headers.get("x-autopilot-model-used", "?"),
            "cost": float(response.headers.get("x-autopilot-cost-usd", "0") or 0),
            "saved": float(response.headers.get("x-autopilot-savings-usd", "0") or 0),
            "latency_ms": elapsed_ms,
        }


async def main(base_url: str, request_count: int, concurrency: int, seed: int) -> None:
    """Fire the mix at ``base_url`` and print savings, tier match and latency."""
    console = Console()
    rng = random.Random(seed)
    prompts = _build_prompts(request_count, rng)
    semaphore = asyncio.Semaphore(concurrency)

    console.print(
        Panel.fit(
            f"[bold]Load test[/bold]\n"
            f"  target       : {base_url}\n"
            f"  requests     : {request_count}\n"
            f"  concurrency  : {concurrency}\n"
            f"  pool mix     : 50% simple · 35% moderate · 15% multi-step\n"
            f"  pool sizes   : {len(SIMPLE_PROMPTS)} simple · "
            f"{len(MODERATE_PROMPTS)} moderate · {len(MULTI_STEP_PROMPTS)} multi-step",
            border_style="cyan",
        )
    )

    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0)) as client:
        started_at = time.perf_counter()
        results = await asyncio.gather(
            *[_send(client, tier, prompt, semaphore) for tier, prompt in prompts]
        )
        wall_seconds = time.perf_counter() - started_at

    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]

    if not oks:
        first_error = fails[0]["error"]
        console.print(f"[red]All {len(fails)} requests failed. First error: {first_error}[/red]")
        return

    total_cost = sum(float(r["cost"]) for r in oks)
    total_saved = sum(float(r["saved"]) for r in oks)
    baseline = total_cost + total_saved
    saved_pct = (total_saved / baseline * 100.0) if baseline > 0 else 0.0

    latencies = sorted(float(r["latency_ms"]) for r in oks)
    p50 = statistics.median(latencies)
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]

    tier_counts: dict[str, int] = {}
    for r in oks:
        tier_counts[str(r["actual_tier"])] = tier_counts.get(str(r["actual_tier"]), 0) + 1
    routing_match = sum(1 for r in oks if r["expected"] == r["actual_tier"])
    routing_match_pct = routing_match / len(oks) * 100.0

    summary = Table(show_header=False, box=None)
    summary.add_column(style="dim")
    summary.add_column(style="bold")
    summary.add_row("Wall time", f"{wall_seconds:.1f}s ({len(oks) / wall_seconds:.1f} req/s)")
    summary.add_row("Successful / failed", f"{len(oks)} / {len(fails)}")
    summary.add_row("Actual cost", f"${total_cost:,.4f}")
    summary.add_row("Baseline cost (all standard)", f"${baseline:,.4f}")
    summary.add_row("Saved", f"${total_saved:,.4f}  [green]({saved_pct:.1f}% reduction)[/green]")
    summary.add_row("Expected-tier match", f"{routing_match_pct:.1f}%")
    summary.add_row("Latency p50 / p95", f"{p50:.0f}ms / {p95:.0f}ms")

    console.print(Panel(summary, title="Results", border_style="green"))

    tiers = Table(title="Requests by actual tier", header_style="bold cyan")
    tiers.add_column("Tier")
    tiers.add_column("Count", justify="right")
    tiers.add_column("Share", justify="right")
    for tier in ("simple", "moderate"):
        count = tier_counts.get(tier, 0)
        tiers.add_row(tier, str(count), f"{count / len(oks) * 100:.1f}%")
    console.print(tiers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="autopilot base URL")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.requests, args.concurrency, args.seed))
