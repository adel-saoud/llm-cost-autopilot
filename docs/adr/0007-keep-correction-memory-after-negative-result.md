---
status: accepted
date: 2026-10-31
---

# Keep the correction memory behind a flag after a negative result

## Context and Problem Statement

The correction memory embeds each verifier correction per tenant and sends a
later prompt one tier up when it is at least
`AUTOPILOT_CORRECTION_MATCH_THRESHOLD` (0.80) similar to a correction and
closer to it than to the anchors of the tier it was misrouted to
([docs/architecture.md](../architecture.md)). Its value was tested with a
claim bar fixed before any run (`scripts/eval_feedback_loop.py`): a gain may
be claimed only if verifier precision is at least 0.70, the paraphrase
hold-out gain has a 95% CI low above 0, the intent hold-out gain CI low is
above -0.02, and run B costs at most 10% more than run A. Both measured runs
missed the bar. What happens to the feature?

## Considered Options

- Remove the correction memory.
- Keep it, on by default, and claim nothing from it.
- Keep it behind `AUTOPILOT_CORRECTION_FEEDBACK_ENABLED` and document the
  negative result in the README.

## Decision Outcome

Chosen: keep it in the code behind `AUTOPILOT_CORRECTION_FEEDBACK_ENABLED`
(default `true`, [docs/configuration.md](../configuration.md)), claim no gain
from it, and report the result in the README's Results and Limits.

Draft reasoning, from the docs: the measured failure is upstream of the
memory. It can only learn from corrections, and the verifier produced few
of them, so the experiment says more about the verifier and the classifier
than about nearest-neighbour escalation. The evaluation harness that can
retest it with a different verifier stays useful
([ADR 0008](0008-gold-scored-reproducible-evaluation.md)).

### Consequences

- Earlier run, `qwen2.5:0.5b` and `llama3.2:3b`: 30 corrections, verifier
  recall 29%, accuracy on reworded questions unchanged between runs
  (CHANGELOG).
- Current run, `qwen3.5:0.8b` and `qwen3.5:4b`: 1 correction
  (`run_a_corrections`), verifier precision 100% (n=1) and recall 5.3%
  (n=19), 0 escalations in run B (`run_b_escalations`, n=1000), and a
  paraphrase gain of +0.0 points, CI [+0.0, +0.0] over 125 intents
  (`paraphrase_gain_points`), all in `results/manifest.json`. Verdict: claim
  not supported ([docs/eval-feedback-loop.md](../eval-feedback-loop.md)).
- The memory costs one embedding search per request and holds raw-prompt
  corrections per tenant, capped by `AUTOPILOT_CORRECTION_MAX_PER_TENANT`.
- A false correction is not undone: escalated prompts are not verified again,
  so it stays until retention or the cap drops it. Shadow verification is on
  the roadmap ([docs/roadmap.md](../roadmap.md)).
- The next experiment is a stronger verifier, not a stronger memory: the
  roadmap lists an LLM-judge scorer as the pluggable alternative to
  similarity.
