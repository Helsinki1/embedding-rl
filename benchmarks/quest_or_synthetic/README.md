# Synthetic OR training data

This package adds **100,000 synthetic training queries** and **67,713 distinct
passages** to the current QUEST training setup. The combined training file has
**100,363 examples**; the combined corpus has **393,140 documents**. The existing
78 development and 78 test query/label rows are preserved exactly.

The source is the `train` split of
[sentence-transformers/natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions),
revision `f9e894e1081e206e577b4eaa9ee6de2b06ae6f17`. Its query–positive passage
pairs are used directly; no LLM generates or judges labels, and no model is
trained as part of dataset preparation. The upstream dataset card is saved in
`provenance/source_README.md`; this package does not assign a new license to source text.

## Construction

1. Read the pinned Parquet file and verify its published SHA256. Only source
   training data is used.
2. Exclude empty rows, queries longer than 256 characters, and queries containing
   explicit `and`, `or`, negation, or exclusion words. This is a conservative
   lexical filter: it can exclude proper names containing those words, and does
   not claim to identify every implicit logical constraint.
3. Group queries by casefolded, whitespace-normalized text and union all supplied
   positives for each group. Preserve the original query and passage text.
4. Deduplicate passages by exact text. Stable document IDs are `nq:` followed by
   the passage's full SHA256, kept in the corpus `title` field for compatibility
   with the existing retrieval and scoring code. These IDs are not encoder input.
5. With seed 42, create **50,000 two-component** and **50,000 three-component**
   examples. Queries are joined literally as `(q1) OR (q2)` or
   `(q1) OR (q2) OR (q3)`. Labels are the deduplicated union of component positives.
   Sample distinct component queries; reject repeated unordered combinations
   and combinations with fewer distinct positives than components. A shuffled
   anchor schedule includes every eligible source query at least once.

The source contains 100,231 rows. The lexical filter excludes 10,941, leaving
89,290 distinct queries, all represented in the synthetic examples. The final
examples have two or three supplied positives, respectively. Passage IDs may be
shared across different training examples. `component_docs` retains the exact
label provenance, and `source_queries.jsonl` maps component IDs to source row indices.

Per the requested assumption, no overlap audit against existing QUEST queries or
documents is performed; the datasets are treated as disjoint. The source labels
are **known positives, not exhaustive relevance judgments**. Other passages may
also answer a query. These are synthetic training examples, not additional
held-out QUEST benchmark cases. Randomly paired topics also differ from QUEST's
curated category queries.

## Files

| File | Contents |
|---|---|
| `data/train.jsonl` | Existing 363 training rows followed by all 100,000 synthetic rows |
| `data/train_synthetic.jsonl` | All synthetic training rows |
| `data/train_or2.jsonl` | 50,000 two-component OR rows |
| `data/train_or3.jsonl` | 50,000 three-component OR rows |
| `data/source_queries.jsonl` | Eligible component queries, positive IDs, source row indices |
| `data/corpus_added.jsonl` | 67,713 new passage records, with `title` IDs and `text` |
| `data/validation.jsonl`, `data/test.jsonl` | Existing evaluation rows, unchanged |
| `manifest.json` | Source revision, hashes, filters, seed, counts, file checksums |
| `provenance/base_split.json` | Exact existing QUEST partition used for the merge |

The corpus is additive: the existing 325,427 QUEST documents plus
`data/corpus_added.jsonl`. A full combined raw corpus is materialized on the
compute server at `/home/ubuntu/quest-or-synthetic-20261106/corpus/documents.jsonl`.
Its identity and count report is copied into `provenance/combined_corpus.json`.
Large document-vector caches stay on compute.

## Reuse and reproduce

From the project root:

```bash
python -m benchmarks.quest_or_synthetic.validate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q benchmarks/quest_or_synthetic/test_synthetic.py

# Rebuild into a fresh directory; the frozen package is never overwritten.
python -m benchmarks.quest_or_synthetic.build \
  --output /tmp/quest-or-nq-rebuild --pairs 50000 --triples 50000 --seed 42
```

Building requires `pyarrow`. Validation uses the Python standard library.
Training uses the existing `quest_or_training/requirements.txt` environment.

## Frozen document embeddings and training integration

The added passages use the same pinned `Qwen/Qwen3-Embedding-8B`, BF16 inference,
last-token pooling, L2 normalization, 4096 dimensions, and FP16 vector storage as
the original corpus. The source passage already includes its title text, so the
new encoder input is the unmodified passage text, without the synthetic ID or a
query instruction. The context limit is 32,768 tokens. All synthetic queries are
also checked against the existing training encoder's 512-token limit.

The dedicated entry point below loads the **combined training file and both
corpus vector sets**. It reuses the existing RL and multi-positive InfoNCE
implementations. The original QUEST-only entry point and cached runs are left
intact. Training must be explicitly launched; dataset preparation does not launch it.

On compute, from `/home/ubuntu/quest-or-synthetic-20261106/project`:

```bash
# Already prepared for this release; the encoder resumes only matching data.
/home/ubuntu/venvs/vllm/bin/python -m benchmarks.quest_or_synthetic.embed \
  --output /home/ubuntu/quest-or-synthetic-20261106/embeddings --gpu 1

# Example future training invocation; choose a fresh output directory.
/home/ubuntu/venvs/quest-training-v2/bin/python -m benchmarks.quest_or_synthetic.train \
  --method both \
  --embeddings /home/ubuntu/quest-or-synthetic-20261106/embeddings \
  --output /home/ubuntu/quest-or-synthetic-20261106/training/new-run
```

The inherited default configuration performs 200 updates and samples uniformly
over the combined training rows. That is a short pilot, not a full pass over this
larger dataset; use `--config` for a deliberate training budget. With uniform
sampling, synthetic rows comprise approximately 99.64% of examples. Development
still selects checkpoints and cutoffs using the unchanged QUEST development set.
The expanded corpus changes retrieval difficulty, so future scores require
baselines evaluated against that same expanded corpus.

The augmented training rerun uses `config.epoch.json`: one complete shuffled pass,
batch size 32, 3,137 updates including the final partial batch, and development
evaluation every 300 updates plus the final update. Each method sees every one
of the 100,363 training examples once. Both start from the original Qwen checkpoint;
learning rate, LoRA setup, losses, and 4096-dimensional output match the earlier
pilot. Specify this configuration explicitly with `--config`.

The trainer saves both the best development checkpoint including step zero and
the best checkpoint after at least one update. `evaluate_completed.py` scores the
untrained encoder, the best trained checkpoint, and the final checkpoint against
the expanded corpus, with cutoffs determined from development only. This makes
it explicit when development prefers retaining the untrained encoder. Prior
scores on the smaller corpus and the 200-update pilot are not matched controls
for this larger-data, longer-budget run.

## Completed one-pass rerun

Saved records are under `runs/augmented-v2/`. Each method completed 3,137 updates
and saw all 100,363 training rows once, with seed 42. Scores below use the same
78 QUEST test queries and all 393,140 documents; trained checkpoints are selected
by development F1 among checkpoints after at least one update.

| Encoder | Selected update | Test F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Untrained whole-query Qwen3 | 0 | 19.00% | 17.39% | 22.80% |
| RL | 2,400 | 18.86% | 16.82% | 23.47% |
| InfoNCE | 3,137 | 19.30% | 17.79% | 22.91% |

When selection also permits the initial checkpoint, development retains the
untrained encoder for InfoNCE. Its trained score above is explicitly reported
separately. RL's final update scores 18.93% F1. Paired bootstrap intervals for
both trained methods' F1 differences versus the expanded-corpus baseline include
zero. These single-seed results do not show a clear improvement.

`results/comparison.json` contains full metrics, checkpoint choices, confidence
intervals, source provenance, and remote checkpoint paths. All eight prediction
files were independently checked for the exact test-query set and their macro
precision, recall, and F1 recomputed successfully. Model weights remain on compute
under `/home/ubuntu/quest-or-training-augmented-20261106-v2/results/`.

## Verification for this release

All 100,000 union labels and file hashes passed validation. Repeating the full
generation produced byte-identical synthetic data, and all five focused tests
passed. The independent CPU integration check loaded all **393,140 × 4096**
vectors, checked finite values and unit norms, and resolved every training,
development, and test positive ID. All 67,713 added passages were encoded without
truncation; the longest synthetic query is 83 tokens. Reports are saved in
`provenance/validation.json`, `provenance/embedding_progress.json`, and
`provenance/integration_check.json`.

The initial vLLM process reported an interpreter-shutdown abort after all vectors
were saved and checksummed. Independent verification passed, GPU memory returned
to its prior level, and rerunning the completed job exited successfully without
re-encoding. This runtime detail is recorded in `provenance/runtime_note.json`.
