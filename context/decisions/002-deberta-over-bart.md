# ADR-002 — DeBERTa-v3-xsmall over BART-large-MNLI for stage 2

**Status:** Accepted (2026-06-10)

## Context

The brief's original sketch named `facebook/bart-large-mnli` as the zero-shot
classifier — the historical default, well-known and well-documented. But
BART is a 2022 sequence-to-sequence model adapted to classification via the
NLI trick; it was not designed for the task. DeBERTa-v3 was, and the small
NLI-tuned checkpoints from MoritzLaurer have consistently outperformed BART
on the canonical NLI benchmarks since 2023.

Candidates considered:

- `facebook/bart-large-mnli` — 407 M params, 2022, well-known.
- `MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33` — 435 M params,
  highest accuracy, slowest.
- `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` — **70.8 M
  params**, MIT licence, ONNX-shippable, CPU-viable. Beats BART on every NLI
  benchmark I checked.

## Decision

Use `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` at stage 2.

## Consequences

- **6× smaller than BART**, fits comfortably on a CPU-only deployment,
  trims the Docker image and the cold-start time.
- **Better accuracy** on the NLI-style task this codebase actually uses it
  for.
- **MIT licence** — clean for any downstream user. BART is also MIT, so
  this is a tie, but DeBERTa wins on every other axis.
- **Less Stack-Overflow recognition** — most "zero-shot classification"
  tutorials still demo BART, so the README explicitly calls out the choice
  for anyone surprised by the model id.
- **Tied to a community checkpoint** — `MoritzLaurer` is an individual
  maintainer. If the checkpoint disappears we'd need to re-pin to another
  DeBERTa-v3 zero-shot checkpoint. Acceptable risk given the maintainer's
  publication history.
