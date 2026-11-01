# Eval data

`nq_open_paraphrases.jsonl` is the frozen evaluation set behind every number in
the main README and `results/manifest.json`. Its SHA-256 is recorded in the
manifest. Do not edit it: a changed file is a different eval set.

## Contents

250 lines, one intent each:

- `intent_id`: `nq-val-N`, where N is the question's position among the
  validation rows kept by the short-question filter.
- `question`, `answers`: one question and its gold answers from NQ-Open.
- `paraphrases`: three rewrites of the question, written by `llama3.2:3b` in
  Ollama at temperature 0 with seed 42 by `scripts/build_feedback_eval_set.py`.

## Source and licence

The questions and answers come from NQ-Open
([google-research-datasets/nq_open](https://huggingface.co/datasets/google-research-datasets/nq_open),
validation split), the open-domain version of Natural Questions:

- Kwiatkowski et al., "Natural Questions: a Benchmark for Question Answering
  Research", TACL 2019.
- Lee, Chang and Toutanova, "Latent Retrieval for Weakly Supervised Open
  Domain Question Answering", ACL 2019.

NQ-Open is licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). This file is
an adaptation (filtered rows plus model-written paraphrases) and is shared
under the same licence. The repository's MIT licence covers the code, not
this file.
