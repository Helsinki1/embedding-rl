# QUEST single-vector training baselines

Two independent runs start from the same pinned `Qwen/Qwen3-Embedding-8B` checkpoint
and train a query encoder with LoRA (rank 16, attention q/k/v/o projections). Each
whole natural-language query produces **one normalized 4096-dimensional vector**,
using the existing retrieval instruction and last-token pooling. The document
encoder and all 325,427 cached corpus vectors remain frozen. There is no query
decomposition, dimension change, projection head, reranker, or new LLM labeling.

This is query-only LoRA fine-tuning, not full-parameter or joint query/document
fine-tuning. Both methods use the same adapter architecture, initialization seed,
training examples, update count, validation rules, and full-corpus search.
The implementation is separate from the frozen `../quest_or/` package and older
BrowseComp experiments.

## Fixed data protocol

`split.json` version 2 freezes a seed-42 partition of **all 519 cleaned queries**:
**363 training, 78 development, and 78 test**. Sorted source IDs are shuffled once;
the first 78 become test, the next 78 development, and the remainder training.
This split was chosen without consulting retrieval outcomes. The source file
hashes are checked at startup; source files are never rewritten. `query` is the
only encoder input. `docs` supplies supervision and evaluation labels;
decomposition and privileged source metadata are not encoder inputs.

This is a new experimental split that repurposes the original QUEST validation
and test queries. Its 78 test queries are held out from these training runs, but
the frozen methods have previously been evaluated on the source dataset. Scores
must not be presented as official QUEST test results or compared directly to
the older 438-query scores. All four methods now select cutoffs using the same
78 development queries and are scored on the same 78 test queries.

## Objectives and selection

**RL:** The policy samples a latent vector `z ~ Normal(mu(query), sigma² I)`, where
`sigma = exploration_radius / sqrt(4096)`. The environment normalizes this one
vector, retrieves top-k from the entire corpus, and returns QUEST answer-set F1:
`2 * correct / (k + number_of_gold_answers)`. Actions and rewards are detached.
Eight samples per query use the other seven samples' mean reward as the baseline.
The loss is `mean(-(reward - baseline) * log_prob(z)) + beta * KL`, with exact
Gaussian KL to that query's cached initial encoder output. KL is on the latent
Gaussian policies, not on the normalized vectors or a document softmax. The
reward k is selected on development before training and held fixed thereafter.
Inference uses the deterministic mean embedding; no exploration noise is used.

**InfoNCE:** Every gold answer is a positive. The loss averages `-log p(document |
query)` over all positives, with a temperature-scaled full-corpus softmax. This
is multi-positive InfoNCE with equal positive target weights, so an OR query's
other correct answers are not treated as negatives. No candidate mining or
sampled negative approximation is used.

For both methods, development macro F1 selects the checkpoint and top-k cutoff
from 1 through 60. Cutoff ties prefer smaller k, checkpoint ties retain the
earlier checkpoint, and the untrained step-0 checkpoint is eligible. The chosen
checkpoint and cutoff are saved before test scoring. The original QUEST package's
evaluator produces macro precision, recall, F1, and domain/template breakdowns.
These are two single-seed baselines; results do not establish statistical
significance. A selected step-0 checkpoint means training did not improve
development F1.

The final 200-update checkpoint is also reported as a secondary result. Its
cutoff is selected from that checkpoint's development curve, without test-based
checkpoint selection. If development already selected update 200, this is the
same result. Otherwise use `python -m benchmarks.quest_or_training.evaluate_last
--run PATH_TO_METHOD_RUN --work-dir PATH_TO_CORPUS_WORK_DIR` to score it separately.

## Run after corpus preparation

Use a **separate training environment** with the dependencies in `requirements.txt`.
Do not alter the environment used by an active corpus worker. From a repository
checkout on the compute server:

```bash
python -m pip install -r benchmarks/quest_or_training/requirements.txt
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.quest_or_training.train \
  --method both \
  --work-dir /home/ubuntu/quest-or-baseline-20261106 \
  --output /home/ubuntu/quest-or-training-runs/single-vector-v2 \
  --wait
```

`--wait` checks both corpus progress files before allocating model/GPU memory.
Without it, an incomplete corpus is an error. A completed corpus does not imply
that a GPU is idle: the original workflow may proceed to query encoding/search.
Launch training after that workflow releases the GPU, or use a free GPU.
This command runs RL and InfoNCE **sequentially**, each from a fresh base model.
Use `--method rl` or `--method infonce` for an individual run. Existing run folders
are rejected to prevent accidental overwrites. There is no automatic resume.
Individual methods can run concurrently on separate GPUs with a shared output
parent; each writes its own method directory and method-specific results file.

Rescore the existing frozen whole-query and decomposed OR-union baselines from
their completed retrieval cache without re-encoding or invoking an API:

```bash
python -m benchmarks.quest_or_training.frozen_baselines \
  --work-dir /home/ubuntu/quest-or-baseline-20261106 \
  --output /home/ubuntu/quest-or-training-runs/single-vector-v2/frozen_baselines
```

Frozen whole-query retrieval selects k from 1–60; OR-union selects a per-component
k from 1–20 and deduplicates the union. Cached frozen query vectors use the
original vLLM/BF16 inference and FP16 storage, whereas training evaluates query
vectors with Transformers/BF16 inference and FP32 normalized outputs. The model,
prompt, pooling, dimension, and frozen document vectors are the same; small
numerical differences between inference implementations may remain.

The cached FP16 corpus is re-normalized in FP32, retaining the existing baseline's
cosine-search convention. The corpus tensor uses approximately 5.3 GB on the GPU.
The model uses BF16, FP32 adapter optimization, and gradient checkpointing; each
run targets one H100 80 GB. Query length is capped at 512 tokens by an explicit
error, not silent truncation. The baseline corpus retains its 32,768-token limit.
The local model directory must be the pinned checkpoint downloaded by the
original preparation workflow.

Each method writes `run.json`, per-step metrics, development curves, `best.pt`,
`last.pt`, the selected PEFT `adapter/`, `selection.json`, test predictions,
per-query details, and `summary.json`. Method-specific `*-results.json` files collect
the scores for the requested methods. `run.json` records package versions, code hashes,
split provenance, corpus provenance, and training settings. Scores are only
generated by an actual completed run; implementation tests use a tiny synthetic
encoder. No OpenAI or W&B credentials are required; metrics are logged locally.

## Verify locally

```bash
python -m pytest -q benchmarks/quest_or_training/test_training.py
python -m benchmarks.quest_or_training.train --help
```

Tests cover policy-gradient direction, detached rewards/actions, exact KL,
multi-positive InfoNCE gradients, official answer-set F1, shard ordering and
completion checks, fixed split isolation, cutoff ties, overwrite protection,
cached OR deduplication, both complete training/evaluation paths using a small
differentiable model, and Qwen3/LoRA pooling and backpropagation with a miniature
Qwen model.

Model and adapter references:
[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) and
[PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora).
