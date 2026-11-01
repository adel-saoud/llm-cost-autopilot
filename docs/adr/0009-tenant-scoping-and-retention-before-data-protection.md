---
status: proposed
date: 2026-10-31
---

# Tenant scoping and retention, without erasure, encryption or audit yet

## Context and Problem Statement

The routing log and the corrections table both store raw prompts and
answers (`storage/db.py`), the correction memory holds their embeddings, and
the admin endpoints return them ([docs/api.md](../api.md)). Several callers
can share one autopilot, and one tenant's mistakes must not change another
tenant's routing. Which data-protection controls ship now, and which are
deliberately left for later?

## Considered Options

- Share corrections across all callers.
- Scope corrections per tenant.
- Keep raw prompts with a retention period.
- Store only hashes or embeddings of prompts.
- Add erasure by tenant, encryption at rest and an audit log now.

## Decision Outcome

Chosen: corrections are scoped per tenant (`tenant_id` on both tables since
schema v2, `X-Autopilot-Tenant-Id` on `/v1/chat/completions`, required
`tenant_id` on `/v2/route`), raw rows older than `AUTOPILOT_RETENTION_DAYS`
(default 30) are deleted at startup, and every `/v1` endpoint requires a
bearer token when `AUTOPILOT_API_TOKEN` is set
([docs/configuration.md](../configuration.md)). Erasure, encryption and audit
are not built.

Draft reasoning, from the code and docs: the corrections table is meant to
be read by a person debugging routing and exported as a corpus
([ADR 0004](0004-correction-dataset.md)), which hashes alone would not
allow; retention bounds how long raw prompts live; per-tenant scope keeps
one caller's traffic from steering another's routing.

### Consequences

- One tenant's corrections never change another tenant's routing (CHANGELOG,
  `tests/test_correction_memory.py`).
- With `AUTOPILOT_API_TOKEN` unset the API starts open and logs
  `app.lifespan.api_open` ([docs/architecture.md](../architecture.md)), and the
  admin endpoints then expose raw prompts to anyone who can reach the port.
- Retention deletes by age only. Erasure on request needs a
  `delete_by_tenant` on both tables plus a memory rebuild, listed on the
  roadmap ([docs/roadmap.md](../roadmap.md)).
- There is no encryption at rest and no audit log of who read the admin
  endpoints. These have to land before real personal data goes through the
  service.
- `/v2/route` keeps the matched correction id out of its response, since it
  points at another request's prompt
  ([ADR 0006](0006-http-sidecar-tier-contract-v2.md)).
