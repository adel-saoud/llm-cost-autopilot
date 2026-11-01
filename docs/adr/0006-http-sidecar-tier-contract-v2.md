---
status: accepted
date: 2026-10-31
---

# HTTP sidecar that returns a tier, under a versioned route contract

## Context and Problem Statement

[llm-gateway](https://github.com/adel-saoud/llm-gateway) serves requests with
its own providers and wants autopilot's routing decision without autopilot
serving the request. The two repos release separately, and the gateway is
not the only possible caller. How does the gateway get a decision, what does
the decision contain, and how does the contract change without breaking the
caller? This ADR owns the producer side; llm-gateway's ADR covers the caller.

## Considered Options

- In-process Python plugin: the gateway imports the classifier.
- HTTP sidecar returning a model id.
- HTTP sidecar returning a tier, versioned by path, with a published JSON Schema.

## Decision Outcome

Chosen: an HTTP sidecar, `POST /v2/route` (`api/route.py`). The gateway
sends `{tenant_id, messages}` and gets `{route_id, tier, confidence, reason}`
back. No model id crosses the boundary. The contract is versioned by path,
`ROUTE_API_VERSION = "2"`, and published as
[docs/route-api.schema.json](../route-api.schema.json) (`api_version` "2"),
which a test keeps in step with the pydantic models and
`scripts/export_route_schema.py` regenerates. Within a version, changes are
additive: callers ignore unknown fields and unknown `reason` values
([docs/api.md](../api.md)).

Draft reasoning, from the code and docs: an HTTP boundary keeps the
gateway free of torch and the HuggingFace models; a tier lets each side
map it to its own providers and prices; path versioning plus a schema file
lets the gateway pin a copy and fail its own CI when the producer changes
(llm-gateway `scripts/sync_schemas.py`).

### Consequences

- v2 replaced v1 when the `complex` tier was removed
  ([ADR 0005](0005-two-tiers-one-model-family-list-prices.md)). v1 is no
  longer served and has no alias, since the tier set it promised no longer
  exists (`api/route.py`, CHANGELOG).
- One extra network hop per gateway request. The endpoint makes no model
  call, so its cost is one classification (median 28 ms in stage 2,
  `stage_2_median_latency_ms` in `results/manifest.json`, measured on a
  shared machine). When the call fails the gateway routes as if autopilot
  were absent (README, Part of a system).
- Traffic on `/v2/route` is never verified, so it adds no corrections; it
  only benefits from corrections stored for its tenant ([docs/api.md](../api.md)).
- The matched correction id stays in autopilot's logs, because it points at
  another request's prompt (`api/route.py`).
- Not built: outcome reporting from the gateway. An additive
  `POST /v2/route/{route_id}/outcome` is on the roadmap, blocked on who pays
  for the extra standard-model call and on streaming responses
  ([docs/roadmap.md](../roadmap.md)).
