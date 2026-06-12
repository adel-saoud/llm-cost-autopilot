# Architecture

## One-paragraph summary

The autopilot is a thin async layer in front of a local LLM (Ollama). Every
request first goes through a **two-stage zero-shot classifier** — a fast
embedding-similarity pass that resolves ~75% of traffic in <5 ms, escalating
to DeBERTa-v3-xsmall zero-shot classification only when stage 1 is unsure.
The classified tier picks a model from a frozen registry. After the response
returns, an **async verifier** replays the prompt against the standard model
and scores the two outputs for semantic similarity. Divergences are persisted
as routing-failure rows in SQLite — the correction dataset visible to the
dashboard and the `/v1/corrections` endpoint.

## Layered module map

```
api/             FastAPI surface — OpenAI-compatible /v1/chat/completions
  ├ app.py       lifespan + DI wiring
  ├ completions.py  routing endpoint (classify → route → call → verify)
  ├ stats.py        /v1/stats · /v1/corrections · /v1/explain/{id}
  ├ health.py       /health
  └ schemas.py      request/response pydantic models

classifier/      Two-stage zero-shot classifier
  ├ complexity.py    ComplexityTier · ClassificationResult
  ├ zero_shot.py     TwoStageClassifier (orchestrator + anchors)
  └ _real_backends.py  HF model wrappers (lazy ML imports)

router/          Tier → model mapping + Ollama adapter
  ├ models.py        ModelConfig · ModelRegistry (frozen, exhaustive)
  ├ decision.py      build_default_registry · route(tier)
  └ ollama.py        OllamaClient (async, tenacity retry)

verifier/        Async quality verifier + feedback loop
  ├ async_verifier.py  Verifier (fire-and-forget)
  ├ feedback.py        RoutingFailure · VerificationOutcome
  └ _real_scorer.py    EmbeddingSimilarityScorer (lazy ML imports)

storage/         Persistent ledger
  └ db.py        SQLiteLedger · open_ledger · RoutingLogEntry · Stats

cost/            Pure savings math + per-request tracker
  ├ calculator.py  compute_savings (pure)
  └ tracker.py     CostTracker (orchestrates ledger writes)

dashboard/       Streamlit UI (excluded from pyright + coverage)
  └ app.py       4 panels: savings · accuracy trend · corrections · comparison

config.py        AUTOPILOT_* env vars via pydantic-settings
main.py          uvicorn entry point
```

## Request lifecycle

1. **Receive** — FastAPI deserialises the OpenAI-compatible chat-completions body.
2. **Classify** — `TwoStageClassifier.classify(last_user_message)` runs in
   `asyncio.to_thread` so the CPU-bound embedding work doesn't block the loop.
3. **Route** — `route(tier, registry)` picks a `ModelConfig`.
4. **Generate** — `OllamaClient.chat(...)` posts to local Ollama; retries on
   transport errors and 5xx, fails fast on 4xx.
5. **Record** — `CostTracker.record(...)` computes actual + baseline cost and
   inserts one `routing_log` row.
6. **Verify (background)** — `Verifier.verify_in_background(...)` schedules an
   `asyncio.Task` named `verifier:{request_id}`. The task replays the prompt on
   the standard model, scores similarity, and writes a `corrections` row when
   it diverges beyond the threshold. All errors stay inside the task.
7. **Return** — response carries `X-Autopilot-*` headers describing the
   decision trail (model used, tier, confidence, cost, savings, request id).

## Data model

Two tables plus a version row.

- `routing_log` (PK `request_id`, indexed on `created_at`) — one row per
  routed request. Captures the classification, the chosen model, token
  counts, actual and baseline cost, latency.
- `corrections` (autoincrement `id`, indexed on `request_id`) — one row per
  verifier disagreement. Denormalised by design — meant to be inspected
  stand-alone.
- `schema_version` — single row recording `CURRENT_SCHEMA_VERSION`. The
  app refuses to start against a newer schema than it understands.

## Concurrency model

- Single Ollama process, single SQLite file — both serialised internally.
- `WAL` journal mode so the dashboard can read while the API writes.
- One `httpx.AsyncClient` per process, shared across requests.
- Verifier tasks are fire-and-forget; no per-process semaphore (yet). Under
  sustained load this can pile up — addressed in `roadmap.md`.

## What's deliberately out of scope

- Real paid API keys / multi-tenant rate limits.
- Active retraining of the classifier (it's zero-shot — no training set).
- Kubernetes / service mesh.
- Semantic caching (different problem, see `llm-gateway`).
