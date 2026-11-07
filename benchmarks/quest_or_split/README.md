# Three-piece Qwen retrieval baseline

Use the original instructed whole-query Qwen3-Embedding-8B encoder and last-token
pooling, without a projection head, learned slots, or query decomposition. Split
its 4096 coordinates into contiguous ranges `[0:1366]`, `[1366:2731]`, and
`[2731:4096]`. Split every cached document embedding along the **same ranges**.
L2-normalize each query and document piece separately, retrieve top k per piece,
then union and deduplicate by document ID. All coordinates are retained.

Coordinate thirds are not guaranteed to represent separate semantic aspects.
This is an empirical baseline; it does not assume that arbitrary coordinate
slices have the same quality as Qwen's supported prefix truncation.

## Data and comparison

Reuse the exact frozen split pinned in `../quest_or_balanced/shared_protocol.json`:
3,600 training queries, 400 validation queries, and 1,000 test queries, with all
393,140 existing documents. The training portion totals 4,000 including validation.
The canonical loader verifies the manifest and query-file hashes before model
allocation. The current shared split holds out complete query combinations;
component questions and synthetic positive passages can recur across partitions.
It is distinct from the retired component-disjoint split. Synthetic positives are
known but not exhaustive. Only `query` is an encoder input.

Evaluate the untrained split encoder and two fresh query-only LoRA runs:

- **InfoNCE:** logsumexp across the three cosine scores divided by temperature,
  followed by a full-corpus softmax; average negative log probability over all
  known positives. The three slices jointly cover the target answer set.
- **RL:** independently sample a Gaussian latent action around each normalized
  query piece, with sigma `0.2 / sqrt(piece_dimension)`. Normalize each sampled
  action for retrieval; reward the deduplicated union with answer-set F1. Eight
  joint actions per query use a leave-one-out reward baseline. Sum the three
  Gaussian log probabilities for REINFORCE. Add 0.01 times their summed exact KL
  to the original query's initial piece policies. Actions and rewards are detached;
  gradients flow through the means and slice normalization. Evaluation is deterministic.

Both runs use rank **32**, alpha **64**, q/k/v/o projections, seed 42, learning rate
1e-4, effective batch size 32, microbatch size 8, temperature 0.05, gradient clipping
at 1, and one identical shuffled epoch: 113 updates. Alpha/rank remains 2, matching
the previous rank-16/alpha-32 setup. Document vectors and base weights stay frozen.
Every training example is seen once; the final partial batch is weighted by its size.
Full-corpus retrieval uses FP32 and disables TF32.

## Selection and artifacts

Development macro F1 selects k from 1 to 20 per piece. Evaluate steps 0, 40, 80,
and 113. Ties prefer smaller k and earlier checkpoints.
RL's reward k is chosen at step 0 and fixed throughout training. Select the best
checkpoint including step 0, the best trained checkpoint, and the final checkpoint
before evaluating any trained method on test labels. Save all three results so a
step-0 selection cannot be mistaken for training progress.

Each run saves configurations, source hashes, corpus and dataset provenance,
adapter checkpoints, losses and gradients, validation curves, selection records,
test predictions, official-compatible metrics, and per-piece retrieval diagnostics.
Separate run directories and an isolated remote code snapshot protect other jobs.
No credentials or external API calls are needed.

The launcher applies GNU `timeout` at 1,790 seconds with a five-second kill grace
period to each method, including data/model loading and final evaluation. A timeout
is reported as incomplete. Every method starts from fresh base weights; adapters
from the superseded 100,363-query run are never reused.

The launcher runs these three methods sequentially on GPU 1, requires at least
40 GiB free before each start, and the training process caps its PyTorch GPU
allocator at 40% of device memory (about 32 GiB). Other GPU jobs keep running;
sharing the GPU can increase their elapsed time. No other processes are stopped
and no existing caches, environments, or run directories are changed.

```bash
python -m pytest -q benchmarks/quest_or_split/test_split.py
python -m benchmarks.quest_or_split.train --method frozen \
  --dataset /home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic \
  --work-dir /home/ubuntu/quest-or-baseline-20261106 \
  --embeddings /home/ubuntu/quest-or-synthetic-20261106/embeddings \
  --output /path/to/new-run/frozen
```

Use `--method infonce` or `--method rl` and a fresh output directory for training.
Dependencies are the existing `../quest_or_training/requirements.txt`; do not
upgrade environments used by other running jobs.
