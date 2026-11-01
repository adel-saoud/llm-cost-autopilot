# API surface

OpenAI-compatible. Any client that talks to `/v1/chat/completions` works.

```
POST  /v1/chat/completions       autopilot picks the model and serves the request
POST  /v2/route                  routing decision only: tier, confidence, reason
GET   /v1/stats                  cost savings %, routing accuracy %, correction count
GET   /v1/corrections?limit=100  routing failure dataset (inspect or export)
GET   /v1/explain/{request_id}   why this request was routed where it was
GET   /health                    liveness (no auth)
```

## Auth and tenants

When `AUTOPILOT_API_TOKEN` is set, every `/v1` endpoint needs
`Authorization: Bearer <token>`. The admin endpoints return raw prompts and
answers, so set it anywhere other than your own laptop. Send
`X-Autopilot-Tenant-Id` on `/v1/chat/completions` to scope corrections per
tenant (default `default`).

## Router API v2 (`POST /v2/route`)

This is how [`llm-gateway`](https://github.com/adel-saoud/llm-gateway) uses
autopilot as a sidecar: it sends `{tenant_id, messages}`, gets a tier back,
and serves the request with its own providers. No model id crosses the
boundary. The contract is versioned by path and published as
[`route-api.schema.json`](route-api.schema.json) (`"api_version": "2"`), which
a test keeps in step with the pydantic models
(`scripts/export_route_schema.py` regenerates it).

Within a version, changes are additive: callers ignore unknown fields and
unknown `reason` values. v2 replaced v1 because the tier set shrank to
`simple` and `moderate`; v1 is no longer served.

Requests served this way are not verified, so they never add a correction.
They do benefit from the corrections already stored for their tenant.

## Decision headers

Every request served on `/v1/chat/completions` carries its decision trail:

```
x-autopilot-request-id           chatcmpl-...
x-autopilot-complexity-tier      simple | moderate
x-autopilot-model-used           ollama/qwen3.5:0.8b
x-autopilot-confidence           0.9100
x-autopilot-routing-reason       stage_1_embedding | stage_2_zero_shot | correction_memory
x-autopilot-cost-usd             0.000021
x-autopilot-savings-usd          0.000121
```

## Explain

`GET /v1/explain/{request_id}` returns the classification, the chosen model,
the cost against the baseline and, when the verifier disagreed, the standard
model's answer and the similarity score. For an escalated request it names
the correction that caused the escalation.
