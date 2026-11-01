"""Build the feedback-loop evaluation set from NQ-Open plus model-written paraphrases.

Each "intent" is one NQ-Open validation question with its gold answers and
three paraphrases written by a local Ollama model (temperature 0, fixed
seed, thinking off). The default is the moderate-tier model from the
settings. The published eval set was written by ``llama3.2:3b`` and is kept
frozen; rebuilding with another model produces a different set, so the
published numbers only reproduce from that file. Paraphrases come from a
model, not from the author, so the test set is not shaped by whoever tunes
the router.

Source: google-research-datasets/nq_open (validation split), CC BY-SA 3.0,
fetched through the Hugging Face dataset-viewer ``/rows`` API at run time.
The output is a derivative under the same share-alike licence. It is written
to the gitignored ``data/eval/`` folder; the frozen set the published numbers
use lives in ``eval_data/`` with its licence notice, and this script never
writes there. Pass ``--eval-set`` to scripts/eval_feedback_loop.py to evaluate
a rebuilt set.

    uv run python scripts/build_feedback_eval_set.py --intents 250
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

import httpx

from llm_cost_autopilot.config import Settings

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "google-research-datasets/nq_open"
DATASET_CONFIG = "nq_open"
DATASET_SPLIT = "validation"
PAGE_SIZE = 100
"""Maximum ``length`` the dataset-viewer ``/rows`` endpoint accepts per call."""

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "nq_open_paraphrases.jsonl"

PARAPHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "paraphrases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["paraphrases"],
}

PARAPHRASE_INSTRUCTION = (
    "Rewrite the question below in three different ways. Each rewrite must ask for "
    "exactly the same fact, so it has the same answer. Vary the wording and word "
    "order; do not add or remove information. Return JSON with a 'paraphrases' "
    "list of three questions.\n\nQuestion: {question}"
)


def as_question(text: str) -> str:
    """NQ questions are lowercase without punctuation; make them read like user input."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith("?") else f"{text}?"


async def fetch_rows(http: httpx.AsyncClient, count: int) -> list[dict[str, Any]]:
    """Page through the split until ``count`` rows are collected or it runs out."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < count:
        response = await http.get(
            ROWS_URL,
            params={
                "dataset": DATASET,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "offset": offset,
                "length": PAGE_SIZE,
            },
        )
        response.raise_for_status()
        page = cast("list[dict[str, Any]]", response.json()["rows"])
        if not page:
            break
        rows.extend(cast("dict[str, Any]", item["row"]) for item in page)
        offset += PAGE_SIZE
    return rows


def is_short_factual(row: dict[str, Any]) -> bool:
    """Short factual questions with short answers: the traffic a cheap model should take."""
    answers = cast("list[str]", row["answer"])
    return len(str(row["question"])) <= 100 and all(len(a.split()) <= 5 for a in answers)


async def paraphrase(
    http: httpx.AsyncClient, model: str, question: str, seed: int, think: bool | None
) -> list[str]:
    """Ask Ollama for three paraphrases as structured JSON; drop duplicates of the original."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PARAPHRASE_INSTRUCTION.format(question=question)}],
        "stream": False,
        "format": PARAPHRASE_SCHEMA,
        "options": {"temperature": 0, "seed": seed},
    }
    if think is not None:
        body["think"] = think
    response = await http.post("/api/chat", json=body)
    response.raise_for_status()
    content = cast("str", response.json()["message"]["content"])
    try:
        raw = cast("list[object]", json.loads(content)["paraphrases"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    seen = {question.lower()}
    out: list[str] = []
    for item in raw:
        text = as_question(str(item))
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:3]


async def main_async(args: argparse.Namespace) -> None:
    """Fetch, filter and paraphrase NQ-Open questions until ``--intents`` are written."""
    async with httpx.AsyncClient(timeout=30.0) as hub:
        rows = [r for r in await fetch_rows(hub, args.intents * 3) if is_short_factual(r)]

    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=args.ollama_url, timeout=120.0) as ollama:
        with args.out.open("w", encoding="utf-8") as out_file:
            for index, row in enumerate(rows):
                if written >= args.intents:
                    break
                question = as_question(str(row["question"]))
                paraphrases = await paraphrase(ollama, args.model, question, args.seed, args.think)
                if len(paraphrases) < 3:
                    continue  # model output unusable; skip rather than hand-fix
                record = {
                    "intent_id": f"nq-val-{index}",
                    "question": question,
                    "paraphrases": paraphrases,
                    "answers": row["answer"],
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % 25 == 0:
                    print(f"{written}/{args.intents} intents")
    print(f"wrote {written} intents to {args.out}")


def main() -> None:
    """Parse the CLI and run the build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", type=int, default=250)
    settings = Settings()
    parser.add_argument(
        "--model",
        default=settings.moderate_model,
        help="Ollama model that paraphrases (default: AUTOPILOT_MODERATE_MODEL)",
    )
    parser.add_argument("--ollama-url", default=settings.ollama_base_url)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.think = settings.ollama_think
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
