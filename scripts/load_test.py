"""Load test — fires synthetic prompts and reports the cost-savings headline.

Default mix:
    50% simple   (factual lookups, conversions, reformatting)
    35% moderate (summarisation, classification, extraction)
    15% complex  (multi-step reasoning, analysis, planning)

Run against a live autopilot (docker-compose up first):
    uv run python scripts/load_test.py --requests 1000 --concurrency 16
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

# ── Prompt pools — light, deterministic seed corpus ─────────────────────────
SIMPLE_PROMPTS = [
    "What is the capital of {country}?",
    "Convert {amount} USD to EUR.",
    "Format the date {date} as ISO 8601.",
    "Translate '{word}' to Spanish.",
    "What day of the week is {date}?",
]
MODERATE_PROMPTS = [
    "Summarise this paragraph in 3 bullet points: '{text}'",
    "Classify this support ticket: '{text}'",
    "Extract the named entities from: '{text}'",
    "Rewrite this in a more formal tone: '{text}'",
    "Compare these two product descriptions and call out the key differences: '{text}'",
]
COMPLEX_PROMPTS = [
    "Analyse the tradeoffs between event-driven and request/response architectures "
    "for the system described: '{text}'. Recommend an approach with justification.",
    "Write a detailed technical proposal for migrating the following legacy system to "
    "a modern stack: '{text}'.",
    "Debug the following snippet and explain the root cause step by step: '{text}'.",
    "Draft a strategic 90-day plan with prioritised initiatives and risks for: '{text}'.",
    "Reason through this multi-step math problem and justify each step: '{text}'.",
]

FILLERS = {
    "country": ["France", "Japan", "Brazil", "Canada", "Egypt"],
    "amount": ["50", "100", "240", "999"],
    "date": ["March 4th", "December 31st", "July 14"],
    "word": ["library", "weather", "morning", "river"],
    "text": [
        "Streamlit re-executes the script on each interaction, which simplifies state "
        "management but can be slow for large queries without caching.",
        "Our billing pipeline ingests events from Stripe, attributes them to tenants "
        "using a metadata field, and rolls them up daily for invoicing.",
        "The function fails intermittently in production but never in CI, even though "
        "the dependencies and Python version are pinned.",
    ],
}


def _format(template: str, rng: random.Random) -> str:
    return template.format(**{k: rng.choice(v) for k, v in FILLERS.items()})


def _build_prompts(n: int, rng: random.Random) -> list[tuple[str, str]]:
    counts = {"simple": int(n * 0.50), "moderate": int(n * 0.35), "complex": 0}
    counts["complex"] = n - counts["simple"] - counts["moderate"]
    prompts: list[tuple[str, str]] = []
    pools = {
        "simple": SIMPLE_PROMPTS,
        "moderate": MODERATE_PROMPTS,
        "complex": COMPLEX_PROMPTS,
    }
    for tier, tier_count in counts.items():
        pool = pools[tier]
        prompts.extend((tier, _format(rng.choice(pool), rng)) for _ in range(tier_count))
    rng.shuffle(prompts)
    return prompts


# ── Driver ──────────────────────────────────────────────────────────────────
async def _send(
    client: httpx.AsyncClient,
    expected_tier: str,
    prompt: str,
    sem: asyncio.Semaphore,
) -> dict[str, object]:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}]},
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            r.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc), "expected": expected_tier}

        return {
            "ok": True,
            "expected": expected_tier,
            "actual_tier": r.headers.get("x-autopilot-complexity-tier", "?"),
            "model": r.headers.get("x-autopilot-model-used", "?"),
            "cost": float(r.headers.get("x-autopilot-cost-usd", "0") or 0),
            "saved": float(r.headers.get("x-autopilot-savings-usd", "0") or 0),
            "latency_ms": elapsed_ms,
        }


async def main(base_url: str, n: int, concurrency: int, seed: int) -> None:
    console = Console()
    rng = random.Random(seed)
    prompts = _build_prompts(n, rng)
    sem = asyncio.Semaphore(concurrency)

    console.print(
        Panel.fit(
            f"[bold]Load test[/bold]\n"
            f"  target       : {base_url}\n"
            f"  requests     : {n}\n"
            f"  concurrency  : {concurrency}\n"
            f"  mix          : 50% simple · 35% moderate · 15% complex",
            border_style="cyan",
        )
    )

    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0)) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[_send(client, tier, p, sem) for tier, p in prompts])
        wall_seconds = time.perf_counter() - t0

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
    for tier in ("simple", "moderate", "complex"):
        count = tier_counts.get(tier, 0)
        tiers.add_row(tier, str(count), f"{count / len(oks) * 100:.1f}%")
    console.print(tiers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="autopilot base URL")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.requests, args.concurrency, args.seed))
