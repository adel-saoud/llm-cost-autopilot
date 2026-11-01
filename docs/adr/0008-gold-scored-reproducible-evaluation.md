---
status: proposed
date: 2026-10-31
---

# Gold-scored evaluation on a frozen set, replayed from a response cache

## Context and Problem Statement

v0.1's figures (14.6% savings, 94.6% verifier agreement) came from 30
hand-written prompts per tier and did not hold on real questions (CHANGELOG).
Agreement with the bigger model is not quality: escalation raises it by
construction (`scripts/eval_feedback_loop.py`). The feedback loop is order
dependent, since each request can teach the memory, so its replay must be
sequential, and on a shared laptop a sequential run of about 1,500 model
calls took hours. What do the published numbers measure, and how are they
made reproducible?

## Considered Options

- Hand-written prompts per tier, scored by agreement with the standard model.
- A public chat log.
- NQ-Open questions with gold answers, plus model-written paraphrases, with
  paraphrase and whole-intent hold-outs.
- For runtime: a sequential run, or concurrent generation into a cache
  followed by a sequential replay.

## Decision Outcome

Chosen: NQ-Open questions with gold answers and three model-written
paraphrases each, frozen in
[eval_data/nq_open_paraphrases.jsonl](../../eval_data/nq_open_paraphrases.jsonl)
(CC BY-SA 3.0, [eval_data/README.md](../../eval_data/README.md)). Half the
intents are seen in warm-up and measured on other paraphrases; the other
half are never seen. Quality is gold accuracy of the answer served,
normalised containment match. `scripts/reproduce.py` first asks both models
every prompt, 4 calls at a time, into a disk cache keyed on the full
request, then replays runs A and B one request at a time from that cache,
and writes every README number with its n to `results/manifest.json`.

Draft reasoning, from the code and docs: gold answers measure what the user
gets; two hold-outs separate "stops repeating a mistake" from "escalates
without reason"; model-written paraphrases keep the author from shaping the
test set; the cache split keeps learning order while removing sequential
model time.

### Consequences

- Prefetch ran at about 56 Ollama calls per minute against about 13 for the
  sequential run, and the replay made no model calls (CHANGELOG).
- Concurrent inference is not bit-identical to one-at-a-time inference, so
  the manifest records `prefetch_concurrency` and names the cache on the
  producing machine as the reproducibility source. A fresh clone regenerates
  the cache and can differ slightly.
- The frozen set was paraphrased by `llama3.2:3b` (`dataset.paraphrase_model`
  in the manifest). Rebuilding it with the current default model gives a
  different set, so `scripts/reproduce.py` stops when the file is missing
  instead of rebuilding it.
- Containment scoring is lenient and favours long answers; the same rule
  applies to both runs, so A/B differences are unaffected
  (`scripts/eval_feedback_loop.py`).
- Latency is measured separately and alone (`scripts/reproduce.py --timing`,
  `results/tier_latency.json`), never in the same run as quality.
- The stage 1 and verifier embedding model, `all-MiniLM-L6-v2`, lists Natural
  Questions among its training data on its model card, so its similarity
  scores on these questions may not transfer to other traffic.
