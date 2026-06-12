# Roadmap

Open work, in rough priority order. Items here are **not** committed — they
exist so future-me (or a contributor) doesn't have to re-discover them.

## Near-term

- **Bounded verifier concurrency** — under sustained load the verifier
  spawns one task per request and competes with foreground traffic on the
  same Ollama process. A `asyncio.Semaphore(N)` inside `Verifier` would cap
  the in-flight verifier load; pick `N` from `AUTOPILOT_VERIFIER_CONCURRENCY`.
- **Graceful shutdown drains background tasks** — `lifespan` should
  `gather()` any `verifier:*` tasks before yielding shutdown so the
  correction dataset isn't truncated when uvicorn cycles.
- **Persist verifier outcome on `routing_log`** — accuracy currently uses
  `1 - corrections / verifiable`. Adding a nullable `verification_outcome`
  column would let us distinguish `AGREED` from `FAILED` (currently both
  count as "not a failure"), and the dashboard could break out per-tier
  accuracy.
- **Per-tier savings panel** — Panel 1 shows aggregate savings; per-tier
  breakdowns would make the cost story sharper for the README.

## Mid-term

- **Streaming responses** — OpenAI clients expect `stream=true` for chat.
  Adding it means the classifier still picks the model up-front but the
  response is forwarded chunk-by-chunk. The verifier still runs on the
  fully-rendered final answer.
- **Pluggable similarity scorer** — semantic similarity is fast but noisy.
  A small LLM-as-Judge backend (one Ollama call with a fixed rubric) would
  cut false-positive corrections at the cost of one extra round-trip per
  divergent request.
- **Live-reload anchors** — `ANCHORS` is a module constant today. A simple
  YAML loader + endpoint (`POST /v1/admin/anchors`) would let an operator
  tune the routing distribution without redeploying.

## Long-term / speculative

- **Tier-rewriting fallback** — if the chosen model is unhealthy (Ollama
  returns 5xx or times out), `route()` could rewrite the tier upward and
  log the override. Currently we just propagate a 502.
- **Per-tenant routing policies** — `AUTOPILOT_STAGE1_THRESHOLD` is
  process-wide. Multi-tenant deployments might want different cost/quality
  curves per workspace.
- **Export to OpenTelemetry** — `structlog` is fine for local dev but
  production observability wants traces with span attributes for tier,
  confidence, cost. Out of scope for the MVP.

## Won't do (decided)

- **Switch to LiteLLM** — `llm-regression-detector` uses LiteLLM for
  provider portability. Autopilot is intentionally Ollama-only; the
  decision engine doesn't need a provider abstraction layer to make its
  point.
- **Retrain the classifier** — the project's entire pitch is zero-shot. A
  fine-tune would be a different (interesting) project.
- **Replace SQLite with Postgres** — the correction dataset is meant to
  be portable, inspectable, and committable. SQLite is the right tool.
