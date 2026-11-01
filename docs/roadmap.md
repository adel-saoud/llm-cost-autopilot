# Roadmap

Open work, in rough priority order. Items here are **not** committed: they
exist so future-me (or a contributor) doesn't have to re-discover them.

## Near-term

- **Shadow-verify escalated prompts**: once a correction escalates a
  prompt, it goes to the standard model and is never verified again, so a
  false correction only expires with retention. Verifying a small sample of
  escalated prompts on the cheap model, and retiring a correction after N
  agreements, would let the memory unlearn.
- **Erasure by tenant**: retention deletes by age; GDPR erasure needs
  `delete_by_tenant` on both tables plus a memory rebuild.
- **Multi-worker memory**: refresh the in-memory index from SQLite
  (`WHERE id > last_seen`) on an interval, storing the embedding in the
  `corrections` row so a refresh does not re-encode.
- **Graceful shutdown drains background tasks**: `lifespan` should
  `gather()` any `verifier:*` tasks before yielding shutdown so the
  correction dataset isn't truncated when uvicorn cycles.
- **Persist verifier outcome on `routing_log`**: accuracy currently uses
  `1 - corrections / verifiable`. Adding a nullable `verification_outcome`
  column would let us distinguish `AGREED` from `FAILED` (currently both
  count as "not a failure"), and the dashboard could break out per-tier
  accuracy.
- **Per-tier savings panel**: Panel 1 shows aggregate savings; per-tier
  breakdowns would make the cost story sharper for the README.

## Mid-term

- **Streaming responses**: OpenAI clients expect `stream=true` for chat.
  Adding it means the classifier still picks the model up-front but the
  response is forwarded chunk-by-chunk. The verifier still runs on the
  fully-rendered final answer.
- **Pluggable similarity scorer**: semantic similarity is fast but noisy.
  A small LLM-as-Judge backend (one Ollama call with a fixed rubric) would
  cut false-positive corrections at the cost of one extra round-trip per
  divergent request.
- **Live-reload anchors**: `ANCHOR_PROMPTS` is a module constant today. A simple
  YAML loader + endpoint (`POST /v1/admin/anchors`) would let an operator
  tune the routing distribution without redeploying.

## Long-term / speculative

- **Tier-rewriting fallback**: if the chosen model is unhealthy (Ollama
  returns 5xx or times out), the endpoint could rewrite the tier upward and
  log the override. Currently we just propagate a 502.
- **Outcome reporting from llm-gateway**: an additive `POST /v2/route/{route_id}/outcome`
  so gateway traffic feeds the memory too. Blocked on who pays for the extra
  standard-model call and on streaming responses.
- **Per-tenant routing policies**: corrections are already per tenant, but
  `AUTOPILOT_STAGE1_THRESHOLD` is process-wide. Multi-tenant deployments might want different cost/quality
  curves per workspace.
- **Export to OpenTelemetry**: `structlog` is fine for local dev but
  production observability wants traces with span attributes for tier,
  confidence, cost. Out of scope for the MVP.

## Won't do (decided)

- **Switch to LiteLLM**: `llm-regression-detector` uses LiteLLM for
  provider portability. Autopilot is intentionally Ollama-only; the
  decision engine doesn't need a provider abstraction layer to make its
  point.
- **Retrain the classifier**: the project's entire pitch is zero-shot. A
  fine-tune would be a different (interesting) project.
- **Replace SQLite with Postgres**: the correction dataset is meant to
  be portable, inspectable, and committable. SQLite is the right tool.
