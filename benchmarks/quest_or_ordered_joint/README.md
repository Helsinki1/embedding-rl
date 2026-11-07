# Ordered joint InfoNCE experiment

This isolated follow-up trains four fresh Qwen3-Embedding-8B query encoders:
pooled-output MLP, final-token-sequence attention plus MLP, one-pass causal slots,
and three-pass completed-state feedback. It does not change running or completed
experiments. It reuses the exact 8,500/500/1,000 `quest_or_spec_v2` split and full
489,188-document corpus through the shared loaders and evaluator.

## Fixed output priority

Output 1 is supervised against stored component 1, output 2 against component 2,
and output 3 against component 3. There is no permutation search, best matching,
or Hungarian assignment. Priority means fixed semantic slot order, not unequal
loss weights, inferred importance, or a larger retrieval budget for earlier slots.
Synthetic query component order exactly follows mention order. QUEST uses the
original template/category order; a natural-language paraphrase can reorder the
mentions. This limitation is recorded rather than silently inventing importance.

The source dataset already contains explicit OR decompositions, so no extra LLM
calls are needed. `prepare.py` encodes only training components with the original
frozen Qwen and the same retrieval instruction. It caches normalized FP32 vectors,
ordered row IDs, unique clause texts, and checksums. No held-out decomposition
targets enter training.

## Objective and training

`loss = document_InfoNCE + 1.0 * ordered_clause_InfoNCE`, jointly from the first
update, using temperature .05 for both terms. The document objective is the
existing collective multi-positive InfoNCE: logsumexp combines the three slot
scores, then a full-corpus softmax is averaged over all known positive documents.
Clause InfoNCE pairs each output with its fixed-position frozen clause target.
Its denominator uses all unique training-clause embeddings, excluding the query's
other valid clauses. Identical texts share one teacher-bank entry. Each query has
equal weight regardless of its two/three-clause arity. For a two-clause query,
slot 3 has no clause supervision but continues to receive document supervision.
Unjudged document labels retain the shared protocol's limitations.

Every architecture gets fresh rank-32/alpha-64 LoRA on q/k/v/o, seed 42, learning
rate 1e-4, batch 32/microbatch 8, identical shuffled one-epoch data, 266 planned
updates, and gradient clipping to 1.0. The autoregressive model uses its own
predictions during all training and evaluation; no teacher forcing or warm-up is
introduced in this controlled comparison. These settings deliberately isolate
the added ordered objective from changes to the existing training schedule.
F1 is a validation/test metric, not an RL reward. No RL is used.

## New output heads

Both heads learn a residual correction to the normalized final-token embedding,
with small nonzero initialization, then normalize each of three output vectors.
The pooled MLP is LayerNorm -> Linear(4096,512) -> GELU -> Linear(512,12288).
The attention head uses three learned 256-dimensional readers, eight attention
heads over all valid final token representations (including the final token),
then a shared Linear(256,512) -> GELU -> Linear(512,4096) for each reader.
The residual initializes both near Qwen's existing embedding. It does not enforce
different aspects by construction; the ordered clause loss supplies that signal.
The existing one/three-pass architectures are imported without modifications.

## Selection, metrics, and limits

Select checkpoint and top-k per slot only on validation, among updated checkpoints.
Primary union and merged-set component metrics remain exactly the shared protocol.
Additionally report each slot against its same-position gold component, with no
matching, and retrieval using only the first 1/2/3 slots under an equal final
20-document budget. These are diagnostics, not replacements for primary metrics.
Keep fixed final budgets of 10/20/60/100, source/arity breakdowns, predictions,
separate loss terms, clause accuracy/cosines, slot/reader gradients, runtime and
source counts. Missing positive component labels remain unscorable.

Each run has a launcher-enforced 1,800-second limit including final evaluation.
Teacher preparation is shared preprocessing and its time is recorded separately.
The queues wait for available GPU memory/utilization and never stop other jobs.

Run CPU tests with `python -m pytest -q benchmarks/quest_or_ordered_joint/test_ordered.py`.
Remote run snapshot: `/home/ubuntu/quest-ordered-joint-20261106/code`.
Commands: `python -m benchmarks.quest_or_ordered_joint.prepare --output ...`, then
`python -m benchmarks.quest_or_ordered_joint.launch --gpu 0` and `--gpu 1`.
