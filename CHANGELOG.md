# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-01

### Added
- Correction memory: each verifier correction is embedded and kept per
  tenant; a later prompt that is close enough to a past misroute, and closer
  to it than to any anchor of the tier it was misrouted to, is sent one tier
  up. Decisions made this way carry
  `classifier_stage = correction_memory` and the id of the matched correction.
- `POST /v2/route`: routing decision only (tier, confidence, reason), for an
  upstream gateway that serves the request with its own providers. JSON
  Schema published in `docs/route-api.schema.json` (`api_version` "2") and
  kept in sync by a test.
- Multi-tenant scoping: `tenant_id` on the routing log and corrections;
  `X-Autopilot-Tenant-Id` header on `/v1/chat/completions`, required field on
  `/v2/route`. One tenant's corrections never change another tenant's routing.
- `AUTOPILOT_API_TOKEN`: bearer token required on every `/v1` endpoint when set.
- `AUTOPILOT_RETENTION_DAYS` (default 30): routing-log and correction rows,
  which hold raw prompts, are deleted at startup once older than this.
- `X-Autopilot-Routing-Reason` response header.
- Verifier limits: `AUTOPILOT_VERIFIER_MAX_CONCURRENCY` and
  `AUTOPILOT_VERIFIER_MAX_PENDING`; overflow is skipped and logged as
  `skipped_overloaded` instead of queueing without bound.
- `AUTOPILOT_OLLAMA_TEMPERATURE` and `AUTOPILOT_OLLAMA_SEED` for reproducible runs.
- `AUTOPILOT_OLLAMA_THINK` (default `false`): Ollama's `think` flag, sent on
  every call so Qwen3.5 answers without a reasoning trace.
- `router/pricing.py`: one table of simulated prices per Ollama model, taken
  from DeepInfra's list prices for the same models and dated.
- `scripts/reproduce.py` runs the evaluation and writes `results/manifest.json`
  with every README number, its sample size, package versions, hardware,
  Ollama and Hugging Face revisions, seed and dataset hash.
- The evaluation prefetches every answer it can need into the response cache
  with bounded concurrency (`--prefetch-concurrency`, default 4), then replays
  runs A and B one request at a time from the cache, so learning order is
  unchanged. A stalled Ollama call now times out after 120 s (5 s to connect).
- `scripts/reproduce.py --timing N` measures per-tier Ollama latency on
  uncached calls, one at a time, into `results/tier_latency.json`, and
  records whether other models were loaded.
- Feedback-loop evaluation on NQ-Open questions with model-written
  paraphrases, scored against gold answers (`scripts/build_feedback_eval_set.py`,
  `scripts/eval_feedback_loop.py`, report in `docs/eval-feedback-loop.md`).
- Gitleaks, pyright and commitizen pre-commit hooks; `cz bump` writes the
  version and the tag; this changelog stays hand-written.
- Dependabot for uv, GitHub Actions and pre-commit hooks, monthly and grouped.
- CI job that builds the wheel, installs it in a clean venv and builds the app.
- `docs/api.md`, `docs/configuration.md`, `docs/project-structure.md`, and
  failure modes plus a component diagram in `docs/architecture.md`.

### Changed
- **BREAKING:** Python 3.12 is now the minimum (was 3.11). Development and the
  Docker image use 3.14.
- Build backend is `uv_build` (was hatchling). Dev tools moved to the `dev`
  dependency group, so `uv sync` installs them without `--extra dev`.
- Ruff uses the shared rule set (E, F, W, I, B, UP, S, SIM, RUF); `S` covers
  the bandit checks. Coverage gate raised from 85% to 90%.
- CI: top-level `contents: read`, separate pre-commit, test and build jobs,
  every action pinned to a commit SHA.
- `context/` moved to `docs/`: `docs/architecture.md`, `docs/adr/`,
  `docs/roadmap.md`. README follows the shared section order.
- Em-dashes removed from code, docs and comments.
- **BREAKING:** two tiers, `simple` and `moderate`. `complex` is removed with
  no alias: it was served by the same model as `moderate`. Ledger schema v3
  rewrites stored `complex` rows to `moderate`; v2 had added tenants
  (existing rows become tenant `default`).
- **BREAKING:** router API v2. `POST /v1/route` is now `POST /v2/route` and
  the tier enum is `simple | moderate`. `/v1/route` is gone, with no alias.
- **BREAKING:** `AUTOPILOT_BUDGET_MODEL` is now `AUTOPILOT_SIMPLE_MODEL` and
  `AUTOPILOT_STANDARD_MODEL` is now `AUTOPILOT_MODERATE_MODEL`, named after the
  tiers they serve. The old names stop startup, with no alias.
- **BREAKING:** default models are `qwen3.5:0.8b` (simple) and `qwen3.5:4b`
  (moderate), were `qwen2.5:0.5b` and `llama3.2:3b`.
- **BREAKING:** `AUTOPILOT_BUDGET_COST_PER_1K_INPUT`,
  `AUTOPILOT_BUDGET_COST_PER_1K_OUTPUT`, `AUTOPILOT_STANDARD_COST_PER_1K_INPUT`
  and `AUTOPILOT_STANDARD_COST_PER_1K_OUTPUT` are removed. Prices come from
  `router/pricing.py`, per million tokens; a model without a price stops
  startup.
- **BREAKING:** startup stops on an `AUTOPILOT_*` variable that is not a
  setting, in the environment or `.env`, and names the closest valid one.
- **BREAKING:** startup stops when Ollama is older than 0.17.5, the first
  release with the small Qwen3.5 sizes, or its version cannot be read.
- Verifier tasks are held by reference until they finish, so the event loop
  cannot garbage-collect one mid-run.
- `numpy` is now a core dependency (correction-memory search).
- Internal renames for explicit names: `classifier/_real_backends.py` is now
  `classifier/huggingface_backends.py`, `verifier/_real_scorer.py` is now
  `verifier/embedding_scorer.py`, `ANCHORS` is `ANCHOR_PROMPTS`, `Stats` is
  `LedgerStats`, `ModelRegistry.standard` is `ModelRegistry.standard_model`,
  `CorrectionMemory.size` is `CorrectionMemory.correction_count` and
  `OllamaClient(max_retries=...)` is `OllamaClient(max_attempts=...)`.
  `router/decision.py` is folded into `router/models.py` and the `route()`
  wrapper is gone (`ModelRegistry.for_tier` does the lookup).
- `scripts/build_feedback_eval_set.py` paraphrases with the moderate-tier
  model by default, thinking off (was `llama3.2:3b`). The published eval set
  stays the frozen file written by `llama3.2:3b`, recorded in the manifest.
- The frozen eval set is committed in `eval_data/` with its CC BY-SA 3.0
  attribution, so a fresh clone can run `scripts/reproduce.py`; it was in the
  gitignored `data/` folder. `scripts/reproduce.py` no longer rebuilds a
  missing set, since a rebuild gives a different set.
- The load-test pool of multi-step prompts is named `multi_step` (was
  `complex`), so it is not mistaken for the removed tier.
- Dashboard uses Streamlit's `width="stretch"` instead of the deprecated
  `use_container_width`.

### Docs
- `docs/DECISIONS.md` is replaced by Architecture Decision Records in MADR 4
  format: the four existing ADRs keep their text and gain dated updates, five
  new ones are added, all accepted, and `docs/adr/README.md` indexes them.

### Removed
- The `autopilot` console script: it pointed at a `llm_cost_autopilot.cli`
  module that never existed. Run the API with `uvicorn llm_cost_autopilot.main:app`.

### Fixed
- Ollama 5xx responses and connection errors that outlast the retries now
  raise `OllamaError`, so `/v1/chat/completions` returns 502 as documented
  instead of an unhandled 500.
- `scripts/eval_feedback_loop.py --out-dir` now also moves the response cache.
- README stated a 0.65 stage-1 threshold; the code default is 0.75.
- README no longer presents the v0.1 synthetic-pool figures (14.6% savings,
  94.6% agreement, stage 1 on ~75% of traffic) as measured results. Every
  README number now comes from `results/manifest.json`.
- Results regenerated on the new models and prices. Old run (`qwen2.5:0.5b`
  and `llama3.2:3b`, simulated prices 30x to 75x apart) vs new run
  (`qwen3.5:0.8b` and `qwen3.5:4b`, list prices 3x apart), run A:
  savings 6.9% to 1.3%; gold accuracy on reworded questions 42.8% to 36.4%
  and on unseen questions 38.0% to 32.4%; QA prompts on the cheap model
  13.8% to 2.4%; corrections 30 to 1; verifier recall 29% to 5.3%. The
  smaller price gap and a classifier that now sends almost every factual
  question to the moderate tier explain the drop in savings. On the same 500
  questions `qwen3.5:0.8b` scores 16.8% and `qwen3.5:4b` 35.6%. The feedback
  loop still does not meet the pre-registered bar. Measured alone, the simple tier answers
  in 792 ms median and the moderate tier in 3,437 ms (n=20 each).

## [0.1.0] - 2026-06-17

### Added
- Two-stage zero-shot complexity classifier (embedding similarity, then
  DeBERTa-v3-xsmall zero-shot) routing to a budget or standard Ollama model.
- Async verifier that compares cheap and standard answers and records
  disagreements in a SQLite correction dataset.
- OpenAI-compatible `/v1/chat/completions`, `/v1/stats`, `/v1/corrections`,
  `/v1/explain/{request_id}`, Streamlit dashboard and 1,000-request load test.

[Unreleased]: https://github.com/adel-saoud/llm-cost-autopilot/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/adel-saoud/llm-cost-autopilot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/adel-saoud/llm-cost-autopilot/releases/tag/v0.1.0
