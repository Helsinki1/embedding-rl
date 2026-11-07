# Three-vector InfoNCE and RL comparisons

Two independent runs start from the pinned Qwen3-Embedding-8B checkpoint, with
rank-16 LoRA on all attention q/k/v/o projections and three directly trained
4096-dimensional FP32 placeholder vectors. Documents and base model weights stay
frozen. Each run returns three normalized 4096-dimensional query vectors.

## Architectures

**one_pass:** append three learned slots to the instructed query, perform one
causal forward pass, and read their three final hidden states.

**three_pass:** first evaluate the query followed by slot 1. Append slot 2 plus
the preceding completed hidden state, normalized to the initialization token's
L2 norm, and evaluate that prefix. Append slot 3 plus the preceding completed
hidden state in the same way and evaluate again. Earlier slot inputs stay in the
prefix. There is no detach between passes. This preserves gradients from later
outputs through earlier completed outputs. The initial slots, model parameters,
LoRA initialization and output dimensions match one_pass exactly.

Slots start from Qwen's EOS input embedding with independently sampled Gaussian
noise of standard deviation 1% of that embedding's RMS. Their parameters are
shared across examples; the output states depend on each input query. Feedback
scaling is fixed and adds no parameters. The current implementation recomputes
prefixes with gradient checkpointing during training. Evaluation of three_pass
reuses the unchanged causal prefix through a KV cache; tests compare cached and
uncached outputs. It does not report optimized serving latency.

## Data and loss

Reuse the canonical `../quest_or_balanced` split: 5,000 queries, comprising
3,600 gradient-training, 400 validation, and 1,000 test queries. The 400 validation
queries are included in the 4,000-query training portion of the outer 80/20 split.
The split is stratified by domain/source and OR arity to match the source pool.
Exact files, IDs, exclusions, and hashes are frozen in its manifest; every new
approach must reuse them. All 393,140 document vectors are reused. Prior source
data and results remain separate. Both variants start from fresh base weights.
The synthetic data provides known, non-exhaustive positives; it dominates the
uniformly shuffled training mixture. Only `query` is an encoder input.

For InfoNCE and a document d, combine the three scores as:

    score(d) = logsumexp_i(cosine(embedding_i, document_d) / temperature)

Apply the full-corpus softmax to these combined scores and average negative log
probability over every known positive. This trains collective answer coverage;
the loss is not applied independently to each slot. There is no component
supervision, hard assignment of meanings to slots, or diversity penalty.

With `--method rl`, replace InfoNCE with Gaussian REINFORCE using the F1 of the
deduplicated union of all three retrieval sets, plus KL to each architecture's
frozen initial outputs. Sample eight joint actions per query, with independent
coordinate noise of standard deviation `0.2 / sqrt(4096)` on each output vector.
These sampled vectors are normalized for retrieval. Noise is added after all
three means are computed; the three-pass model still feeds its preceding mean
representation into the next pass. Use detached actions and a leave-one-out
reward baseline. Sum log-probabilities and exact Gaussian KL over all three
slots and dimensions; average over queries, with KL weight 0.01. This is the
joint-policy extension of the single-vector RL baseline. Select reward k on
initial validation once, before updates. Recompute references without a KV cache
in the exact training microbatch order, including matching padding, and keep them
fixed throughout training. Different BF16 padding can otherwise create a spurious
KL penalty even with identical weights. Abort if the first update's KL exceeds 0.001.
`config.rl.json` preserves the InfoNCE optimizer, architecture, and query budget.

Both runs use seed 42, LoRA alpha 32, learning rate 1e-4, temperature 0.05,
effective batch size 32 (microbatches of 16), and one identical shuffled pass
over all 3,600 training examples: 113 optimizer steps. Gradients are clipped to norm 1.
The last partial batch is weighted by its actual size. Full-corpus scoring uses
FP32 vectors, with TF32 disabled, matching the existing training convention.

## Evaluation and artifacts

Retrieve top k per vector and union by document ID. Development macro F1 selects
k from 1 through 20 and the checkpoint, including step 0. Evaluate every 40
updates and at the last update. Cutoff ties prefer smaller k; checkpoint ties
prefer the earlier step. Save selection before test scoring. The original
QUEST scorer computes precision, recall, and F1 on deduplicated answer sets.

Each fresh run writes provenance/configuration, trainable parameter counts,
loss and gradient norms, per-slot gradients, validation curves, slot cosine
similarities, duplicate retrieval counts, unique relevant contributions, update
times, progress, best/last/best-trained checkpoints, final selection and test
predictions. Checkpoints contain the LoRA and placeholder parameters together;
load with the matching model/configuration using the shared `restore_weights`.

One GPU is assigned per variant. `launch_when_free.py` waits until the existing
augmented experiments finish training and evaluation, then requires at least
48 GiB free and utilization at most 20% for three checks. It never stops another
process. Each run has an outer 1,800-second wall-clock limit, including startup
and evaluation. The trainer reserves the final three minutes for evaluation and
stops updates early if needed. A timeout is reported as a timeout, not a completed
run. A queue timeout or training failure is recorded in launcher status.
The remote launch uses an isolated source snapshot and the existing training
environment without upgrading it. Dataset/vector caches are read in place.

## Verification

Run `python -m pytest -q benchmarks/quest_or_multivector/test_multivector.py`.
Tests exercise actual miniature Qwen3 models with rank-16 LoRA, one-versus-three
forward calls, causality, padding, completed-state feedback gradients, updates
to every placeholder, frozen base weights, collective multi-positive loss,
deduplication, gradient accumulation, checkpoint selection and final scoring.

Use `python -m benchmarks.quest_or_multivector.train --help` for the entry point.
The environment requirements are the existing `../quest_or_training/requirements.txt`.
