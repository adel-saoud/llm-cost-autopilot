# Project structure

```
src/llm_cost_autopilot/
├── api/             FastAPI app, OpenAI-compatible /v1 surface and admin endpoints
├── classifier/      Two-stage zero-shot classifier, correction memory, HF backends
├── router/          ModelConfig, ModelRegistry, simulated price table, Ollama HTTP adapter
├── verifier/        Async quality verifier, similarity scorer, memory feedback
├── storage/         SQLite ledger: routing log and correction dataset
├── cost/            Pure savings math and the per-request CostTracker
├── dashboard/       Streamlit UI, 4 panels (excluded from pyright and coverage)
├── config.py        AUTOPILOT_* env vars via pydantic-settings
└── main.py          ASGI entry point for uvicorn

scripts/
├── demo_autopilot.py           live routing decisions in the terminal
├── load_test.py                1,000 requests against a running API
├── build_feedback_eval_set.py  rebuilds an NQ-Open eval set into the gitignored data/eval/
├── composite_dashboard.py      stitches four panel screenshots into docs/dashboard.png
├── eval_feedback_loop.py       run A (memory off) against run B (memory on), gold-scored
├── export_route_schema.py      writes docs/route-api.schema.json
├── reproduce.py                runs the evaluation and writes results/manifest.json
└── seed_demo_ledger.py         synthetic ledger for the dashboard

docs/
├── architecture.md      components, request lifecycle, data model, failure modes
├── adr/                 architecture decision records (MADR 4), indexed in adr/README.md
├── api.md               endpoints, router API v2, decision headers
├── configuration.md     every AUTOPILOT_* variable
├── eval-feedback-loop.md  generated report of the feedback-loop evaluation
├── roadmap.md           open work
└── route-api.schema.json  published router API v2 contract

eval_data/
├── nq_open_paraphrases.jsonl  frozen eval set (CC BY-SA 3.0)
└── README.md            source, attribution and licence

results/
├── manifest.json        every README number with its n, versions, digests and seed
└── tier_latency.json    per-tier Ollama latency, timed alone (reproduce.py --timing)

tests/                   hermetic: models, Ollama and the scorer are faked
```

The module-level map with file names is in [architecture.md](architecture.md).

## Tech stack

| | Library or tool | Role |
|:--|:--|:--|
| Core | `fastapi`, `uvicorn` | OpenAI-compatible API surface |
| | `pydantic` v2, `pydantic-settings` | Validated models, env-driven config |
| | `httpx`, `tenacity` | Async Ollama adapter with retry |
| | `aiosqlite` | Routing log and correction dataset |
| | `numpy` | Correction-memory nearest-neighbour search |
| | `structlog` | Structured logging |
| ML (`[ml]` extra) | `sentence-transformers` | Stage 1 embedding similarity (`all-MiniLM-L6-v2`) |
| | `transformers`, `torch` | Stage 2 zero-shot pipeline (DeBERTa-v3-xsmall), CPU is enough |
| Dashboard (`[dashboard]` extra) | `streamlit`, `plotly`, `pandas` | 4-panel UI |
| Dev | `uv`, `uv_build` | Environment, lockfile, build |
| | `ruff`, `pyright` (strict) | Lint, format, types |
| | `pytest`, `pytest-asyncio`, `pytest-cov` | Tests, coverage gate at 90% |
| | `pre-commit`, `gitleaks`, `commitizen` | Hooks, secret scan, Conventional Commits and releases |
