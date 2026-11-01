---
status: accepted
date: 2026-10-31
---

# Two tiers on one model family, priced at hosted list prices

## Context and Problem Statement

v0.1 had three tiers, `simple`, `moderate` and `complex`, but `complex` was
served by the same model as `moderate`, so it never changed what a request
cost or which model answered. Costs came from four hand-set
`AUTOPILOT_*_COST_PER_1K_*` settings, 30x to 75x apart, which made any
routing look like a large saving (CHANGELOG). The two default models,
`qwen2.5:0.5b` and `llama3.2:3b`, came from different families, so a quality
gap mixed model size with model family. How many tiers should the router
have, which models serve them, and where do the prices come from?

## Considered Options

- Keep three tiers and give `complex` its own, larger model.
- Two tiers, `simple` and `moderate`, on two sizes of one model family.
- Keep the hand-set prices.
- Price each local model at a hosted provider's list price for the same model.

## Decision Outcome

Chosen: two tiers on two sizes of one family, `qwen3.5:0.8b` for `simple`
and `qwen3.5:4b` for `moderate`, with thinking off on every call
(`AUTOPILOT_OLLAMA_THINK`, default `false`). Prices come from one table,
`src/llm_cost_autopilot/router/pricing.py`: DeepInfra's list prices for
`Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B`, read on 2026-10-24
([docs/configuration.md](../configuration.md#simulated-prices)). A model
without a row stops startup instead of costing nothing.

Draft reasoning, from the code and docs: a third tier is only worth its
classifier label when it maps to a different model and price; a strong and
weak pair is the usual setup for two-model routing; one family isolates
size as the variable; list prices make the savings figure something a
reader can check. Thinking is off so tokens, cost and latency cover the
served answer only (`docs/configuration.md`).

### Consequences

- Breaking: the `complex` tier is gone with no alias, ledger schema v3
  rewrites stored `complex` rows to `moderate` (`storage/db.py`), and the
  route API moved to v2 ([ADR 0006](0006-http-sidecar-tier-contract-v2.md)).
- Savings fell from 6.9% to 1.3% on the same evaluation (CHANGELOG, then
  `run_a_cost_reduction` in `results/manifest.json`, n=1000). The list prices
  differ by 3x (`router/pricing.py`), against 30x to 75x before, and the
  classifier sends almost every factual question to `moderate`
  (`run_a_cheap_share_qa` = 2.4%, n=500).
- On the same 500 questions `qwen3.5:0.8b` scores 16.8% and `qwen3.5:4b`
  35.6% gold accuracy (`simple_model_accuracy_qa`, `moderate_model_accuracy_qa`,
  n=500), so a request routed to the cheap model trades a large quality drop
  for a two-thirds price cut.
- Timed alone on 20 questions each, the simple tier answers in 792 ms
  median and the moderate tier in 3,437 ms (`results/tier_latency.json`).
- Prices go stale. `PRICES_RETRIEVED_ON` dates them, and a price change is a
  one-line edit plus a rerun of `scripts/reproduce.py`.
- Adding a tier back means a third model with its own price row, a new
  classifier label and a route API version bump.
