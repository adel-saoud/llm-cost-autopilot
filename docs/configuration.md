# Configuration

Every setting is an environment variable with the `AUTOPILOT_` prefix, read
by `src/llm_cost_autopilot/config.py` through pydantic-settings. Startup stops
on an `AUTOPILOT_*` name that is not in this table, in the environment or in
`.env`, and names the closest valid one: pydantic-settings alone would ignore
a typo and keep the default.

| Variable | Default | Effect |
|:--|:--|:--|
| `AUTOPILOT_STAGE1_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace id for stage-1 embeddings; the verifier's scorer and the correction memory use it too. |
| `AUTOPILOT_STAGE2_MODEL` | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | HuggingFace id for the stage-2 zero-shot pipeline. |
| `AUTOPILOT_STAGE1_THRESHOLD` | `0.75` | Cosine confidence below which stage 2 runs. Lowering it puts stage 2 on the hot path. |
| `AUTOPILOT_VERIFIER_ENABLED` | `true` | Master switch; `false` means no background verification at all. |
| `AUTOPILOT_VERIFIER_DIVERGENCE_THRESHOLD` | `0.70` | Similarity below this is recorded as a routing failure. |
| `AUTOPILOT_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon URL. |
| `AUTOPILOT_SIMPLE_MODEL` | `qwen3.5:0.8b` | Ollama id for the SIMPLE tier. Needs a row in the price table below. |
| `AUTOPILOT_MODERATE_MODEL` | `qwen3.5:4b` | Ollama id for the MODERATE tier, the cost baseline and the verifier's reference. Needs a row in the price table below. |
| `AUTOPILOT_OLLAMA_THINK` | `false` | Ollama's `think` flag on every call. `false` turns off Qwen3.5's reasoning trace, so tokens and costs cover the served answer only. |
| `AUTOPILOT_SQLITE_PATH` | `autopilot.db` | Ledger and correction-dataset file. |
| `AUTOPILOT_CORRECTION_FEEDBACK_ENABLED` | `true` | Whether past corrections escalate similar prompts. |
| `AUTOPILOT_CORRECTION_MATCH_THRESHOLD` | `0.80` | Minimum cosine similarity to a past correction for it to apply. |
| `AUTOPILOT_CORRECTION_MAX_PER_TENANT` | `500` | Corrections kept in memory per tenant; oldest dropped first. |
| `AUTOPILOT_RETENTION_DAYS` | `30` | Routing-log and correction rows (raw prompts) older than this are deleted at startup. |
| `AUTOPILOT_API_TOKEN` | unset | Bearer token for every `/v1` endpoint. Unset leaves the API open. |
| `AUTOPILOT_VERIFIER_MAX_CONCURRENCY` | `2` | Standard-model verifier calls at once. |
| `AUTOPILOT_VERIFIER_MAX_PENDING` | `100` | Verifications allowed to wait; more are skipped. |
| `AUTOPILOT_OLLAMA_TEMPERATURE` / `AUTOPILOT_OLLAMA_SEED` | unset | Pin sampling for reproducible runs. |

## Simulated prices

Ollama is free, so each model is priced at the list price of the same model
on DeepInfra ([model list](https://api.deepinfra.com/models/list), read on
2026-10-24). The table lives in `src/llm_cost_autopilot/router/pricing.py`;
a model without a row stops startup instead of costing nothing.

| Ollama model | Priced as | USD per 1M input tokens | USD per 1M output tokens |
|:--|:--|--:|--:|
| `qwen3.5:0.8b` | `Qwen/Qwen3.5-0.8B` | 0.01 | 0.05 |
| `qwen3.5:4b` | `Qwen/Qwen3.5-4B` | 0.03 | 0.15 |
