# Shared 10,000-query experiment

This version implements the November 6 Google Doc specification and the user's
choice to score final merged results against each gold component. Historical
packages and results remain unchanged. All eleven approaches now use `run.py`.

| Source | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| QUEST | 441 | 26 | 52 | 519 |
| MS MARCO | 4,029 | 237 | 474 | 4,740 |
| NQ | 4,030 | 237 | 474 | 4,741 |
| Total | 8,500 | 500 | 1,000 | 10,000 |

`manifest.json` pins data hashes and source revisions. `protocol.json` pins that
manifest and every shared training setting. `data.load_splits` verifies hashes,
source counts, union labels, and split isolation before any run.
Only `data/train.jsonl` supplies training examples. The compatibility file named
`data/train_synthetic.jsonl` contains all queries solely for the existing corpus
encoder's query-length audit; it is never passed to a training objective. Always
use the shared loader rather than globbing data files.

QUEST retains the 519 previously accepted OR queries and union labels. Gold
components come from the official category mapping intersected with those labels.
Two components have no accepted positive documents, both in training: their
queries remain present; those components are unscorable rather than fabricated
negatives. Every validation/test component has positive labels.

MS MARCO uses the pinned Microsoft v1.1 train source, selected-passage labels,
and all candidate passages attached to sampled atomic questions. NQ uses the
previously pinned sentence-transformers natural-questions source. Two or three
atomic questions are composed with explicit OR, balanced as closely as possible.
Each synthetic atom is used once. Connected groups of duplicate questions or
identical positive passages stay within one split, including cross-source
duplicates. QUEST category groups also stay within one split. These are custom
research splits, not the official MS MARCO/NQ leaderboard evaluations.
QUEST documents may satisfy different categories across partitions: 61 distinct
positive documents occur in both training and test labels, 23 in training and
validation, and 2 in validation and test. The split isolates atomic categories,
not every QUEST entity. NQ/MS MARCO identical positive passage texts do not cross
partitions. `source/data_audit.json` records this and the source/arity statistics.

The common corpus has 489,188 documents: all 325,427 existing QUEST documents,
all 67,713 existing NQ passages, and 96,048 MS MARCO candidate passages. The new
passages use the same frozen Qwen3-Embedding-8B revision, document prompt,
4096-dimensional normalized vectors, and 32,768-token context cap. The source
provides incomplete relevance judgments; unjudged results score as nonrelevant.
This is not a run against MS MARCO's complete 8.8M-passage collection.

## Approaches and training

| Current approach | Historical implementation reused |
|---|---|
| Single frozen / InfoNCE / RL | `quest_or_training` encoder and objectives |
| Split-thirds frozen / InfoNCE / RL | `quest_or_split` normalization and objectives |
| One-pass slots InfoNCE / RL | `quest_or_multivector` one-pass encoder |
| Three-pass slots InfoNCE / RL | `quest_or_multivector` differentiable feedback encoder |
| Frozen LLM decomposition + union | `quest_or_llm_union`, pinned query-only GPT-5.4-mini prompt |

Earlier `quest_or`, `quest_or_training`, `quest_or_synthetic`, `quest_or_balanced`,
and augmented/rank-32 variants represent earlier data sizes or protocols, rather
than additional model approaches. Root BrowseComp/Modal experiments are a separate
task and are not part of this eleven-approach specification.

All eight trained approaches start from the original frozen base checkpoint,
use rank-32/alpha-64 LoRA on q/k/v/o projections, batch 32, microbatch 8, seed 42,
learning rate 1e-4, and one epoch of 266 updates including the final partial batch.
Slot methods additionally learn three slot embeddings. Every document encoder
stays frozen. Training logs record cumulative examples by source, loss, gradient
norm, slot gradients, and validation metrics; RL also records reward F1, reward
spread, zero-advantage fraction, policy loss and KL.
Gradient norms are recorded before clipping to 1.0. Large logged norms therefore
describe the incoming gradient, not an unclipped optimizer update.

InfoNCE averages negative log-probability over all known positive documents, with
the full corpus denominator. For multiple vectors, logsumexp combines their
document scores before the corpus softmax. RL samples eight Gaussian actions per
query, uses union F1 with a leave-one-out baseline, and adds exact Gaussian KL to
initialization (coefficient .01, radius .2). The joint KL sums policy coordinates
and slots, so its magnitude can differ by architecture; it is observable in logs.
References use the exact shuffled microbatches and padding used in training.

Validation happens at initialization, steps 89 and 178, and the final step 266.
The primary trained result selects the best updated checkpoint, with initialization
reported separately. Cutoffs maximize validation union F1: single k=1..60,
multiple vectors k=1..20 per vector, followed by deduplication. Ties retain the
earliest checkpoint/smaller k. Test labels never select checkpoints or cutoffs.
The launcher enforces a 30-minute maximum per run including evaluation.

## Metrics and comparison

Union precision/recall/F1 compare the final deduplicated result set to all union
positives. Component metrics compare that same final set to each component's gold
set, average components within a query, then average queries. Thus a result useful
only to component A lowers component B's precision, exactly as requested.

Primary results use validation-selected cutoffs and show actual result counts.
Fixed-budget comparisons at 10, 20, 60 and 100 merge ranked lists round-robin,
deduplicating before the budget is counted. These compare identical final result
counts across single and multiple vectors. Coverage@100 is the fraction of gold
components with at least one positive in the merged top 100; it is not precision
or complete recall. Results include source and arity breakdowns, individual query
predictions/scores, timing and peak allocated GPU memory.

The LLM receives query text only. Cached calls must match the pinned prompt and
model; rejected/invalid decompositions fall back to the whole query and never
remove test cases. The decomposition cache and fallbacks are saved separately.

W&B receives all eight training histories and three frozen controls in project
`quest-or-spec-v2`, grouped as `10000-queries-rank32`. `wandb_sync.py` streams logs
from the remote snapshot using the desktop credential, including steps completed
before syncing began. It records validation source/component metrics, source
counts, gradient norms, RL diagnostics and final fixed-budget test scores. A final
comparison run adds bar charts and the verified score artifact. Credentials never
enter the remote code snapshot. LLM retrieval timings exclude the offline LLM
decomposition calls; `decomposition/usage.json` records token usage separately.

## Reproduction

From the repository root, `python -m benchmarks.quest_or_spec_v2.build` builds only
when no frozen manifest exists. Source download URLs/revisions are in the manifest.
Use `python -m benchmarks.quest_or_spec_v2.decompose` to fill the resumable LLM cache.
The shared additive-corpus encoder accepts this package as `--dataset`.

On the existing compute server, the versioned snapshot is
`/home/ubuntu/quest-spec-v2-20261106/code`; the two GPU queues use
`python -m benchmarks.quest_or_spec_v2.launch --gpu 0` and `--gpu 1`. They wait for
completed embeddings and available memory, preserve unrelated jobs, refuse to
overwrite results, and save status for failures/timeouts. All training commands
and exact source hashes are recorded in each run.

Use `collect.py` after all eleven runs finish to recompute saved metrics, verify
the full training mixture, and write the comparative score table. Do not compare
these scores numerically with the old 5,000-query table as if they used the same
test set or retrieval corpus.
