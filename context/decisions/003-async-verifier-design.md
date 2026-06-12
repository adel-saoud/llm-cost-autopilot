# ADR-003 — Async, fire-and-forget verifier

**Status:** Accepted (2026-06-10)

## Context

The point of routing to a cheap model is that the user gets their response
faster *and* cheaper. The point of the verifier is to catch routing
mistakes and feed them back into a correction dataset. If the verifier
blocks the response, the latency win is gone.

Options considered:

1. **Inline verification** — call the standard model before returning to
   the user, compare, return both responses or rewrite. Defeats the
   latency win and doubles the cost on every request.
2. **Sampling verifier** — run the verifier on N% of traffic only. Cheaper
   but the correction dataset grows N× slower and the dashboard's "learning
   over time" story takes proportionally longer to materialise.
3. **Async fire-and-forget** — schedule the verifier as an `asyncio.Task`
   *after* the user response is built. The user never waits; the verifier
   competes with foreground traffic for the same Ollama process but never
   on the critical path.

## Decision

Async fire-and-forget. `Verifier.verify_in_background(...)` calls
`asyncio.create_task(coro, name=f"verifier:{request_id}")` and returns the
task. The FastAPI endpoint **does not await** the task — it discards the
reference. The task has its own error handling: every exception inside the
coroutine is caught, logged via `structlog`, and absorbed. The user
response is built and sent before the task even starts running.

The task name is intentionally prefixed `verifier:` so future shutdown
code can drain only those tasks (`asyncio.all_tasks()` filtered by name).

## Consequences

- **Zero latency impact** on the routed request.
- **Verifier cost is hidden** in the savings number — every divergent
  request implicitly costs one extra standard-model call. For now the
  default cost simulation absorbs this; the dashboard could later show
  "savings net of verifier cost".
- **Tasks are dropped on the floor** — if uvicorn shuts down mid-task,
  the correction is lost. Acceptable for an MVP; the roadmap notes a
  shutdown-drain step.
- **Unbounded concurrency** — every cheap-routed request spawns a task.
  Under sustained load this competes with foreground traffic on the
  Ollama process. Mitigated by a semaphore in the roadmap.
- **Errors are silent by default** — the verifier never raises into the
  user response path. All failures go to structlog with
  `error_type` + `error` fields so an operator can grep for them.
