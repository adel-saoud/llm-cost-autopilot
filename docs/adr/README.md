# Architecture Decision Records

MADR 4.0.0 format. `proposed` means the reasoning is a draft from the code and
docs, waiting for the maintainer to confirm it; `accepted` means confirmed.

| ADR | Title | Status |
|:--|:--|:--|
| [0001](0001-two-stage-classifier.md) | Two-stage zero-shot classifier | accepted |
| [0002](0002-deberta-over-bart.md) | DeBERTa-v3-xsmall over BART-large-MNLI for stage 2 | accepted |
| [0003](0003-async-verifier-design.md) | Async, fire-and-forget verifier | accepted |
| [0004](0004-correction-dataset.md) | SQLite as the correction-dataset store | accepted |
| [0005](0005-two-tiers-one-model-family-list-prices.md) | Two tiers on one model family, priced at hosted list prices | accepted |
| [0006](0006-http-sidecar-tier-contract-v2.md) | HTTP sidecar that returns a tier, under a versioned route contract | accepted |
| [0007](0007-keep-correction-memory-after-negative-result.md) | Keep the correction memory behind a flag after a negative result | accepted |
| [0008](0008-gold-scored-reproducible-evaluation.md) | Gold-scored evaluation on a frozen set, replayed from a response cache | accepted |
| [0009](0009-tenant-scoping-and-retention-before-data-protection.md) | Tenant scoping and retention, without erasure, encryption or audit yet | accepted |
