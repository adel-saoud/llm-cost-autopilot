"""Live routing demo: sends a handful of prompts to a running autopilot.

Run with:
    docker compose up -d                              # autopilot + ollama
    uv run python scripts/demo_autopilot.py
"""

from __future__ import annotations

import argparse
import asyncio

import httpx
from rich.console import Console
from rich.table import Table

DEMO_PROMPTS: list[tuple[str, str]] = [
    ("simple", "What is the capital of France?"),
    ("simple", "Convert 100 USD to EUR."),
    (
        "moderate",
        "Summarise this paragraph in three bullet points: "
        "Streamlit is a Python framework for building interactive data apps "
        "that runs entirely on the server and re-executes the script on each "
        "user interaction.",
    ),
    (
        "moderate",
        "Classify this support ticket by category: "
        "'My account was charged twice for the same subscription.'",
    ),
    (
        "moderate",
        "Analyse the tradeoffs between event-driven and request/response "
        "architectures for a multi-tenant SaaS billing pipeline, and recommend "
        "an approach with justification.",
    ),
]


async def send_one(client: httpx.AsyncClient, prompt: str) -> dict[str, str]:
    """Send one prompt and return its ``x-autopilot-*`` decision headers."""
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
    )
    response.raise_for_status()
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower().startswith("x-autopilot-")
    }


async def main(base_url: str) -> None:
    """Send every demo prompt in turn and print the routing decisions as a table."""
    console = Console()
    console.print(f"[bold]Sending {len(DEMO_PROMPTS)} prompts to {base_url}[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("Expected", style="dim", no_wrap=True)
    table.add_column("Routed tier", no_wrap=True)
    table.add_column("Conf.", justify="right")
    table.add_column("Model", no_wrap=True)
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Saved (USD)", justify="right")
    table.add_column("Prompt (excerpt)")

    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(120.0)) as client:
        for expected, prompt in DEMO_PROMPTS:
            try:
                decision_headers = await send_one(client, prompt)
            except httpx.HTTPError as exc:
                console.print(f"[red]Request failed: {exc}[/red]")
                continue
            tier = decision_headers.get("x-autopilot-complexity-tier", "?")
            tier_colours = {"simple": "green", "moderate": "yellow"}
            tier_colour = tier_colours.get(tier, "white")
            table.add_row(
                expected,
                f"[{tier_colour}]{tier}[/{tier_colour}]",
                decision_headers.get("x-autopilot-confidence", "?"),
                decision_headers.get("x-autopilot-model-used", "?"),
                decision_headers.get("x-autopilot-cost-usd", "?"),
                decision_headers.get("x-autopilot-savings-usd", "?"),
                prompt[:60] + ("…" if len(prompt) > 60 else ""),
            )

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="autopilot base URL")
    args = parser.parse_args()
    asyncio.run(main(args.url))
