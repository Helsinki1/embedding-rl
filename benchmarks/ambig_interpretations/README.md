# Corpus-free interpretation embedding distillation

This experiment implements the user's November 6 request to train three output
embeddings directly against embeddings of human-labeled interpretations. It is
a separately authorized protocol, independent of the historical QUEST retrieval
experiments and their F1/InfoNCE objectives. No document corpus is opened or mounted.

The AMER paper reports 5,044 training questions, but the downloaded data release
contains 5,323 training rows. Its answer groups are not interpretation text.
We join question IDs to the original AmbigNQ human-written disambiguated questions.
The user selected questions with two or three interpretations, masking the third
output when only two labels exist. We do not generate labels for missing rows.

The frozen `manifest.json` records source and split checksums and exclusions.
There are 2,634 train, 293 validation and 706 test queries. We remove 54 question
overlaps between the AMER train and evaluation releases. Train/validation splitting
groups exact shared query/interpretation text. Test comes from the filtered AMER
evaluation release. These are custom splits, not reproduction of published scores.

Teacher: original Qwen/Qwen3-Embedding-8B, revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, frozen, last-token pooling,
4,096-dimensional unit vectors. The same retrieval query instruction is used for
original ambiguous queries and interpretation text. The teacher encodes labeled
interpretations once. Student inputs contain only the original ambiguous query.

Both students start from the same base and seed, using rank-32/alpha-64 LoRA on
q/k/v/o projections plus three learned slots, learning rate 1e-4, batch 32,
microbatch 8 and three epochs. Maximum sequence length is 256; overlength text
raises an error rather than being truncated. No reinforcement learning, retrieval
metrics, negatives or contrastive document objective is used.

* `mtp`: append three learned slots after the query and read their final hidden
  states in one causal forward pass. This is a learned output-slot placement
  experiment, not an additional pretrained multi-token-prediction head.
* `autoregressive`: make three forward passes, adding the previous completed,
  normalized output (scaled to token-embedding norm) to the next learned slot.
  Feedback remains differentiable; no gold interpretation is fed into the student.

The loss is the minimum average `1 - cosine(prediction, teacher_target)` over all
one-to-one assignments. For two targets, only two distinct slots contribute; the
unmatched third is masked. This is equivalent to half the squared Euclidean
distance between matched unit vectors. Epoch checkpoints are selected by validation
loss. Initialization and the original ambiguous-query vector repeated three times
are reported as controls. Final test runs only after selection is saved. Prediction
and target pairwise cosine expose collapse; two-label examples do not supervise
the extra output and no claim is made that it is a valid third interpretation.

Run on Modal with an absolute Unix deadline covering the complete experiment:

```bash
modal run benchmarks/ambig_interpretations/modal_run.py --run-id RUN_ID --deadline UNIX_TIMESTAMP
```

The isolated Modal volume is `ambig-interpretation-distillation`. Teacher encoding
runs first, followed by simultaneous MTP and autoregressive training on two H100s.
The launcher terminates workers before the deadline; training reserves time for
checkpoint saving and final evaluation. Each architecture saves initial, best and
last trainable weights, metrics, configuration and held-out prediction embeddings.
The weights retain the original Qwen embedding dimension for later retrieval.

Export vectors for new JSONL queries (each row has a `query` field):

```bash
python -m benchmarks.ambig_interpretations.encode --architecture mtp \
  --checkpoint /path/to/mtp/best.pt --input /path/to/queries.jsonl \
  --output /path/to/query_vectors.pt
```

This writes `[queries, 3, 4096]` normalized vectors. Retrieval can later search
each vector and merge the resulting document lists. Training itself does not
access the corpus.

The separate `retrieval_eval.py` evaluates the frozen best checkpoints on the
existing `quest_or_spec_v2` corpus and its unchanged 1,000 test queries: 474
MS MARCO, 474 NQ and 52 QUEST. It reruns the frozen single-vector baseline,
verifies the original document-vector checksums and ID ordering, and uses exact
cosine search over all 489,188 documents. Checkpoints are not updated. The
separate 500-query validation split selects only the retrieval cutoff: 1–60
documents for the single vector or 1–20 per output for the three-vector models.
Every selected cutoff is saved before test inference. Results include union and
component metrics, source/arity breakdowns and fixed budgets 10/20/60/100.

For reproduction, copy the original cached shards into the Modal volume under
`/shared-corpus-v2/{base/results,nq,msmarco}` with their ID/progress files, and
include `source_checksums.json` mapping the four vector paths to their source
server SHA-256 checksums. Then run:

```bash
python -m benchmarks.ambig_interpretations.prepare_retrieval_bundle
modal run benchmarks/ambig_interpretations/modal_retrieval.py --run-id FRESH_RUN_ID
```

The bundle freezes the shared dataset, scorer, retrieval policy and code. A
30-minute Modal timeout bounds this evaluation. The original 1,000-query test
set is retained even when an NQ component matches AmbigQA training text; an
overlap audit and secondary score excluding those queries are saved separately.

Sources: [AMER code/data](https://github.com/timchen0618/amer),
[AmbigNQ annotation schema](https://github.com/shmsw25/AmbigQA#dataset-contents).
