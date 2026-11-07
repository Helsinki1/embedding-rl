# QUEST OR-only benchmark, version 1

A frozen, training-free evaluation of OR query decomposition and dense retrieval.
This package is independent of the older research code elsewhere in the project.

**Completed baseline:** OR decomposition macro F1 **20.56%** on 438 test queries. [Full results and predictions](runs/baseline-v1/README.md).

## Frozen dataset

| Split | Official queries | Official OR templates | Retained |
|---|---:|---:|---:|
| Validation | 323 | 85 | 81 |
| Test | 1,727 | 512 | 438 |

Use `data/validation.jsonl` and `data/test.jsonl` for future runs. Do not regenerate
their decompositions unless you intend to create a new benchmark version. Each
line contains a stable source ID, the original natural-language `query`, unchanged
gold document titles in `docs`, cached LLM `components`, source metadata, and the
cleaning rationale. `original_query` and metadata contain privileged benchmark
information: use them only for auditing and evaluation, never as model inputs.

All four original domains are retained: films, books, plants, and animals.
The test set contains 117 film, 110 book, 101 plant, and 110 animal queries.
Original union templates comprise 226 two-category and 212 three-category queries.
The LLM produced 240 two-component and 198 three-component decompositions; these
differences are retained as part of the decomposer baseline, not corrected with
gold decomposition metadata.

## Filtering rules

1. Keep only official `_ or _` and `_ or _ or _` templates.
2. Give only the natural-language query to GPT-5.4 mini. Exclude queries judged
   mixed, malformed, or ambiguous, and obtain self-contained OR components for
   accepted queries. The prompt and every response are saved in `provenance/`.
3. Apply an additional conservative text guard: require explicit `or` and exclude
   conjunction/negation words outside a documented proper-name allowlist.
   Thus names such as "Saint Kitts and Nevis" are preserved. Ordinary compound
   category descriptions remain atomic, as in QUEST's own construction.

Filtering never uses retrieval quality or test labels. The original query text
and answer sets are not rewritten. Of the 512 OR-template test queries, 60 failed
the query-only LLM audit and 14 more failed the text guard. All exclusions,
including non-OR templates, are recorded with reasons. This is a conservative,
LLM-audited subset; it is not a claim that every excluded query is inherently
invalid or that every retained decomposition is perfect.

## Baseline protocol

- **Decomposer:** `gpt-5.4-mini-2026-03-17`, low reasoning, strict JSON schema,
  query text only, no web tools or answer generation. Calls are cached.
- **Encoder:** frozen `Qwen/Qwen3-Embedding-8B`, revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
- **Corpus:** all 325,427 unique QUEST document titles. The source has 325,505
  rows, including 78 exact duplicate title/text records. No distractor documents
  are removed based on query type, relevance labels, or domain.
- **Document encoding:** title, newline, document text; last-token pooling,
  normalized vectors, up to 32,768 tokens, BF16 inference and FP16 vector storage.
- **Query encoding:** Qwen's retrieval instruction applied to each standalone
  component; normalized FP32 cosine search over the entire corpus.
- **OR prediction:** retrieve the top k documents for every component, then union
  them by exact document title. Duplicates count once. No final cap or reranker.
- **Cutoff:** choose k from 1 through 20 using validation macro F1, with ties
  resolved toward smaller k. Freeze it before test evaluation.
- **Control:** original whole query with the same frozen encoder; choose a
  top-k cutoff from 1 through 60 using validation only.
- **Training:** none. Validation cutoff selection does not update model weights.

These are zero-fine-tuning results with validation-selected retrieval cutoffs,
not a claim of a completely untuned zero-shot protocol. Results on this subset
are not directly comparable to the original paper's full-test results.

## Scoring and reuse

Predictions should be JSONL records with `query` and a `docs` list of exact QUEST
document titles. IDs may also be included. Every gold query must have exactly one
prediction record; an empty `docs` list is permitted. Score a future run with:

```bash
python3 benchmarks/quest_or/scripts/evaluate.py \
  --gold benchmarks/quest_or/data/test.jsonl \
  --predictions path/to/predictions.jsonl
```

The scorer deduplicates document titles and computes per-query precision, recall,
and F1, then averages each metric equally over queries. It also reports domain
and source-template breakdowns. F1 is `2 * correct / (predicted_count + gold_count)`.
Document order and embedding similarity do not directly affect this metric.

```bash
python3 -m unittest discover -s benchmarks/quest_or/scripts -p 'test_*.py'
python3 benchmarks/quest_or/scripts/validate_dataset.py
```

`manifest.json` records the immutable dataset checksums. `provenance/` includes
source checksums, exclusions, all decomposition responses, corpus duplicate audit,
and the exact cleaning prompt. `runs/baseline-v1/` stores the frozen protocol,
selected cutoffs, aggregate metrics, and individual predictions when complete.
Raw corpus downloads and large embedding arrays stay on the SSH compute server;
no API keys or SSH credentials belong in this package.

## Reproduce retrieval on the compute server

The original work directory is `/home/ubuntu/quest-or-baseline-20261106` on
`ubuntu@68.209.73.124`. Its cached corpus embeddings can be reused without
re-encoding the corpus. The Python runtime used is
`/home/ubuntu/venvs/vllm/bin/python`; exact package versions are saved with the run.
Set `QUEST_OR_WORK_DIR` to use another work directory. Scripts expect `data/`
for the downloaded source corpus and `results/` for intermediate arrays and runs.

For a fresh work directory, use `prepare.py` to download the pinned model and
original corpus. Copy the frozen validation/test JSONL files to
`results/val_clean.jsonl` and `results/test_clean.jsonl`, and the saved protocol to
`results/protocol.json`. The two corpus workers are `embed.py --rank 0` and
`embed.py --rank 1`. After both finish, run `embed.py --mode queries --rank 1`,
then `retrieve_score.py`. This path reuses the frozen LLM decompositions and
requires no API key. It requires two CUDA GPUs for the two corpus shards.

`decompose.py` and `freeze.py` are archived to document the original construction,
not as required steps for future evaluation. Creating a different decomposition
or filtering policy should produce a new version, preserving this v1 package.

Sources: [paper](https://arxiv.org/abs/2305.11694),
[official dataset](https://github.com/google-research/language/tree/master/language/quest),
[official scoring implementation](https://github.com/google-research/language/blob/master/language/quest/eval/run_eval.py).
