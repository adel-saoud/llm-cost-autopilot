# ADR-001 — Two-stage zero-shot classifier

**Status:** Accepted (2026-06-10)

## Context

The routing decision needs to be fast (the user is waiting), accurate
enough that the savings story is real, and require zero labelled training
data — the project's pitch is that you can drop it in front of any LLM
workload without first building a dataset.

Three options were on the table:

1. **scikit-learn classifier** (logistic regression on TF-IDF, or a small
   transformer fine-tune). Cheap at inference but needs a labelled corpus,
   which contradicts the zero-shot pitch.
2. **Single zero-shot model** (e.g. `facebook/bart-large-mnli`). Works
   without training data but every request pays ~200 ms of NLI inference,
   regardless of how obvious the request is.
3. **Two-stage pipeline** — embedding similarity (stage 1) handles the
   confident cases, escalating to a zero-shot NLI model (stage 2) only when
   stage 1 is unsure. This mirrors what production systems like vLLM
   Semantic Router do, validated by Red Hat / Google / NVIDIA research.

## Decision

Use the two-stage pipeline. Stage 1 (`all-MiniLM-L6-v2`) embeds the prompt
once and compares against pre-encoded anchor prompts (5 per tier). If the
top tier's cosine similarity is `>= AUTOPILOT_STAGE1_THRESHOLD` (default
0.75), the routing decision is made immediately. Otherwise stage 2
(`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`) takes over with a
zero-shot classification head.

Backends are exposed via `Protocol`s so the orchestrator (`TwoStageClassifier`)
doesn't depend on HuggingFace at import time — tests inject deterministic
fakes and the heavy ML imports live in `classifier/_real_backends.py`, lazy
inside the FastAPI lifespan.

## Consequences

- **Latency** — stage 1 is <5 ms post-init; stage 2 is ~50 ms. Empirically
  stage 1 handles ~75% of traffic so the average is well under the single
  zero-shot baseline.
- **Tuning lever** — `stage1_threshold` is the single knob ops can pull.
  Lowering it makes stage 2 a hot path and raises CPU usage; raising it
  risks routing mistakes on borderline prompts. The README documents this.
- **No training data needed** — the only "supervision" is the curated
  anchor list, which is editable as a constant. Adding/removing anchors
  shifts routing without retraining.
- **Tied to HuggingFace ecosystem** — both stages depend on HF models being
  downloadable on first run. The Docker image caches them in a volume.
