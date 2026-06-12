# LLM Cost Autopilot

> Zero-shot complexity classifier that routes LLM requests to the cheapest
> capable model — and gets smarter the more it's used.

[![CI](https://github.com/adel-saoud/llm-cost-autopilot/actions/workflows/ci.yml/badge.svg)](https://github.com/adel-saoud/llm-cost-autopilot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/types-pyright%20strict-1f5fff)](https://github.com/microsoft/pyright)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

LLM API costs scale with model size, not request complexity. A "what's the
capital of France?" lookup costs the same as a multi-step architectural
analysis if you route everything to the same model.

**This project fixes that.** It classifies each request by semantic
complexity using a zero-shot HuggingFace pipeline — no training data, no
labelled dataset — routes it to the cheapest tier capable of handling it,
then verifies the decision was correct. Routing failures feed a growing
correction dataset. The system improves itself.

> Inspired by the cost visibility gap I experienced running DaiLY at
> Decathlon France — 25K users where every architecture decision had a
> direct cost impact. This is the routing layer I wished we had.

![Dashboard — 202 routed requests, $0.1108 saved (21.4%), 98.8% routing accuracy, 1 correction logged; cumulative-savings area chart and tier-distribution donut both visible. Data is from the live 200-request load test against real Ollama (qwen2.5:0.5b + llama3.2:3b)](docs/dashboard.png)

---

## Try it — no API key needed

```bash
git clone https://github.com/adel-saoud/llm-cost-autopilot
cd llm-cost-autopilot

# Local stack — autopilot API + Ollama, fully offline
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:0.5b
docker compose exec ollama ollama pull llama3.2:3b

# Send a simple request — autopilot picks qwen2.5:0.5b
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}' \
  -D - | grep -i '^x-autopilot'

# Send a complex request — autopilot picks llama3.2:3b
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Analyse the tradeoffs between event-driven and request/response architectures for a multi-tenant SaaS billing pipeline."}]}' \
  -D - | grep -i '^x-autopilot'
```

Expected response headers:

```
x-autopilot-request-id:          chatcmpl-...
x-autopilot-complexity-tier:     simple
x-autopilot-model-used:          ollama/qwen2.5:0.5b
x-autopilot-confidence:          0.9100
x-autopilot-cost-usd:            0.000021
x-autopilot-savings-usd:         0.000121
```

Generate a meaningful headline metric with a few hundred mixed requests:

```bash
uv run python scripts/load_test.py --requests 200 --concurrency 8
```

Measured on a 200-request run against `qwen2.5:0.5b` + `llama3.2:3b` on
a local M-series Mac:

```
Saved                         $0.1106  (21.9% reduction)
Expected-tier match           68.0%
Routing accuracy (verifier)   98.8%
Latency p50 / p95             10.3s / 32.1s
```

The 21.9% figure is **what the classifier actually does** on the bundled
prompt corpus with default anchors and a 0.75 stage-1 threshold — not a
theoretical maximum. The classifier is intentionally conservative:
borderline prompts get routed up to the standard model, which lowers
savings but improves the verifier-agreement rate (98.8%). Lower the
threshold (`AUTOPILOT_STAGE1_THRESHOLD=0.65`) to push more traffic to
the budget model and trade routing accuracy for cost.

> Costs are simulated — Ollama runs locally and is free. The simulated
> pricing curve follows current cheap-vs-mid-tier API pricing, so the
> savings percentage is a realistic estimate.

## Open the dashboard

```bash
uv run streamlit run src/llm_cost_autopilot/dashboard/app.py
```

Four panels: cumulative savings, routing-accuracy trend (improves over
time as corrections accumulate), filterable correction-dataset explorer,
side-by-side cheap-vs-standard comparison for any routing failure.

---

## How it works

```mermaid
flowchart LR
    P([Prompt]) --> S1[Stage 1<br/>Embedding similarity<br/>all-MiniLM-L6-v2<br/>&lt;5ms]
    S1 --> C{Confidence<br/>≥ 0.75?}
    C -- Yes --> R[Route]
    C -- No --> S2[Stage 2<br/>DeBERTa zero-shot<br/>deberta-v3-xsmall<br/>~50ms]
    S2 --> R
    R --> T{Tier}
    T -- simple --> B[qwen2.5:0.5b<br/>BUDGET]
    T -- moderate --> M[llama3.2:3b<br/>STANDARD]
    T -- complex --> H[llama3.2:3b<br/>longer output<br/>STANDARD+]
    B --> V[/Verifier<br/>async, never blocks/]
    M --> V
    H --> V
    V -.->|diverged| F[(Correction<br/>dataset)]

    style P fill:#1f2937,stroke:#374151,color:#f9fafb
    style B fill:#052e16,stroke:#166534,color:#86efac
    style S2 fill:#1e1b4b,stroke:#4338ca,color:#c7d2fe
    style F fill:#450a0a,stroke:#991b1b,color:#fca5a5
```

**Two-stage classifier.** Stage 1 is an embedding-similarity check against
five curated anchor prompts per tier. If the top tier scores above 0.75
cosine similarity, the routing decision is made in <5 ms with no model
inference. Otherwise stage 2 — a DeBERTa-v3-xsmall zero-shot pipeline —
takes over. In practice stage 1 handles ~75% of traffic.

**Async verifier.** After each response is built and sent, an
`asyncio.create_task(...)` replays the prompt against the standard model
and scores the two outputs for semantic similarity. Below the divergence
threshold (0.70 default), the routing decision is recorded as a failure in
the SQLite correction dataset. The verifier **never blocks** the user
response; if it crashes, the failure is logged and absorbed.

**Feedback loop.** The corrections table grows as the verifier disagrees.
The dashboard plots routing accuracy as `1 − corrections / verifiable`
windowed over time — the "system gets smarter" story shows up as the
curve trending upward.

---

## Why not just use [llm-gateway](https://github.com/adel-saoud/llm-gateway)?

Fair question — they look adjacent and both live in this portfolio. The
honest answer is they operate on different axes:

|                         | `llm-gateway` (control plane) | `llm-cost-autopilot` (decision engine) |
|:------------------------|:------------------------------|:---------------------------------------|
| **Layer**               | Infrastructure proxy — sits in front of all traffic | Application engine — decides per request |
| **Question answered**   | Where did my budget go?       | Which model should handle this request? |
| **Cost tracking**       | Retrospective — records what was spent | Prospective — prevents overspending before it happens |
| **HuggingFace models**  | None                          | `all-MiniLM-L6-v2` + `deberta-v3-xsmall` |
| **Learning**            | Static routing rules          | Feedback loop — improves from routing mistakes |
| **Unique angle**        | RAG cost attribution          | Semantic complexity routing that learns over time |

They're complementary. Gateway tracks **where** the budget went; autopilot
prevents it going to the wrong place; [`llm-regression-detector`](https://github.com/adel-saoud/llm-regression-detector)
catches quality drops when prompts change.

---

## API surface

OpenAI-compatible — any client that talks to `/v1/chat/completions` works.

```
POST  /v1/chat/completions       autopilot picks the model
GET   /v1/stats                  cost savings %, routing accuracy %, correction count
GET   /v1/corrections?limit=100  routing failure dataset (inspect or export)
GET   /v1/explain/{request_id}   why this request was routed where it was
GET   /health                    liveness
```

Every routed request includes a decision trail in the response headers:

```
x-autopilot-request-id           chatcmpl-...
x-autopilot-complexity-tier      simple | moderate | complex
x-autopilot-model-used           ollama/qwen2.5:0.5b
x-autopilot-confidence           0.9100
x-autopilot-cost-usd             0.000021
x-autopilot-savings-usd          0.000121
```

`GET /v1/explain/{request_id}` is the interview demo — it returns the
classification, the chosen model, the cost vs. baseline, and (if the
verifier disagreed) the standard-model response and similarity score.

---

## Project structure

```
src/llm_cost_autopilot/
├── api/             FastAPI app — OpenAI-compatible /v1 surface + admin
├── classifier/      Two-stage zero-shot complexity classifier (+ HF backends)
├── router/          ModelConfig · ModelRegistry · Ollama HTTP adapter
├── verifier/        Async quality verifier + similarity scorer
├── storage/         SQLite ledger — routing log + correction dataset
├── cost/            Pure savings math + per-request CostTracker
├── dashboard/       Streamlit UI — 4 panels (excluded from pyright)
├── config.py        AUTOPILOT_* env vars via pydantic-settings
└── main.py          uvicorn entry point

scripts/
├── demo_autopilot.py     live routing decisions in the terminal
├── load_test.py          1000 requests · cost savings headline
└── seed_demo_ledger.py   synthetic ledger for dashboard screenshots

context/                  ADRs · architecture · roadmap
tests/                    81 hermetic tests · 95% coverage
.github/workflows/        ci.yml (lint · type · test)
```

Full module map and design decisions → [`context/architecture.md`](context/architecture.md)
ADRs → [`context/decisions/`](context/decisions/)

---

## Tech stack

| | Library / Tool | Role |
|:--|:--|:--|
| **Core** | `fastapi` + `uvicorn` | OpenAI-compatible API surface |
| | `pydantic` v2 + `pydantic-settings` | Runtime-validated models; env-driven config |
| | `httpx` + `tenacity` | Async Ollama adapter with retry |
| | `aiosqlite` | Routing log + correction dataset |
| | `structlog` | Structured logging with `bind()` context |
| **ML (optional `[ml]` extra)** | `sentence-transformers` | Stage 1 embedding similarity (`all-MiniLM-L6-v2`) |
| | `transformers` | Stage 2 zero-shot pipeline (DeBERTa-v3-xsmall) |
| | `torch` | Backend for both HF models — CPU-only OK |
| **Dashboard (optional `[dashboard]`)** | `streamlit` + `plotly` + `pandas` | 4-panel observability UI |
| **Dev** | `uv` | Fast package manager + lockfile |
| | `ruff` | Lint + format |
| | `pyright` strict | 0 errors, 0 warnings |
| | `pytest` + `pytest-asyncio` | Hermetic — no model downloads, no daemon |
| | `pre-commit` | Lint + format on every commit |

---

## Development

```bash
uv sync --extra dev
uv run pre-commit install

uv run ruff check --fix .    # lint + autofix
uv run ruff format .         # format
uv run pyright               # type-check — must stay at 0 errors
uv run pytest                # 81 tests, 95% coverage, gate at 85%
```

The test suite is fully hermetic: HuggingFace models, Ollama, and the
verifier scorer are all mocked. CI runs in under 5 seconds.

---

## Configuration

Every knob is an environment variable with the `AUTOPILOT_` prefix.
Defaults are sensible; touch only what you need.

| Variable | Default | Effect |
|:--|:--|:--|
| `AUTOPILOT_STAGE1_THRESHOLD` | `0.75` | Cosine confidence below which stage 2 fires. Lowering it makes stage 2 a hot path. |
| `AUTOPILOT_VERIFIER_ENABLED` | `true` | Master switch — false means no async verifier work at all. |
| `AUTOPILOT_VERIFIER_DIVERGENCE_THRESHOLD` | `0.70` | Similarity below this is recorded as a routing failure. |
| `AUTOPILOT_OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama daemon URL. |
| `AUTOPILOT_BUDGET_MODEL` | `qwen2.5:0.5b` | Ollama id for the SIMPLE tier. |
| `AUTOPILOT_STANDARD_MODEL` | `llama3.2:3b` | Ollama id for MODERATE + COMPLEX. |
| `AUTOPILOT_SQLITE_PATH` | `autopilot.db` | Ledger + correction-dataset file. |

---

## Honest limitations

- **Stage 2 adds ~50 ms** of NLI inference for the ~25% of prompts stage 1
  is unsure about. Documented in [ADR-001](context/decisions/001-two-stage-classifier.md).
- **Real savings depend on prompt mix.** Measured 21.9% reduction on the
  bundled load test; a workload dominated by short factual lookups would
  push that higher, while a workload of long-form analyses would push it
  lower. The verifier-agreement rate (98.8%) is more stable across mixes.
- **Verifier uses semantic similarity, not LLM-as-Judge.** Faster but
  noisier; some "diverged" rows are stylistic differences rather than
  factual ones. Roadmap notes a judge-backed scorer.
- **Costs are simulated.** Real routing logic, synthetic prices that
  follow current cheap-vs-mid-tier API curves. Plug in real pricing by
  overriding `AUTOPILOT_*_COST_PER_1K_*` env vars.
- **Feedback loop improves routing, not the models themselves.** Anchor
  prompts and threshold get smarter; the underlying LLMs don't change.
- **Verifier concurrency is unbounded.** Under sustained load, verifier
  tasks compete with foreground traffic on the same Ollama process.
  Roadmap notes a per-process semaphore.

---

## Companion projects

- [`llm-regression-detector`](https://github.com/adel-saoud/llm-regression-detector)
  — catches LLM quality regressions in CI before they reach users.
- [`llm-gateway`](https://github.com/adel-saoud/llm-gateway) —
  infrastructure proxy with RAG cost attribution.

Three projects, two axes: **gateway** tracks where budget went, **autopilot**
prevents it going to the wrong place, **detector** catches quality drops
when prompts change.

---

## License

[MIT](LICENSE) — use it, fork it, ship it.
