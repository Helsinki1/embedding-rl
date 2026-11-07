# Query embedding RL on BrowseComp-Plus

For the current **Qwen3 QUEST / MS MARCO / NQ OR-retrieval comparison**, use
[`benchmarks/quest_or_spec_v2`](benchmarks/quest_or_spec_v2/README.md): the shared
10,000-query split, rank-32 LoRA, eleven approaches, component metrics, and W&B
logging. The BrowseComp-Plus workflow documented below is a separate experiment.

This repository publishes the experiment source code, tests, configuration,
documentation, and standalone utilities under `work/`. Downloaded datasets,
corpora, generated decompositions, run artifacts, model checkpoints, and local
credentials are excluded. Benchmark manifests record the original data hashes;
data-dependent commands and tests require those datasets to be restored or
prepared separately using the corresponding benchmark's instructions. Some
utility scripts retain paths and service identifiers from the original runs and
need local configuration before reuse.

**Benchmark evidence labels → full-corpus nDCG@10 reward → REINFORCE updates the query encoder.**

This is a research exercise using a custom benchmark split. The default workflow uses binary
evidence labels; `modal_graded.py` adds the separately audited LLM-reward experiment described below.
Compare the unchanged encoder with RL on identical held-out queries before claiming gains.
InfoNCE is an optional contrastive-learning comparison; it is not part of the RL update.

| Part | Choice |
| --- | --- |
| Dataset | BrowseComp-Plus: 830 queries and 100,195 web documents |
| Split | Custom approximately 581 train / 83 validation / 166 test; evidence groups stay together |
| Encoder | `BAAI/bge-small-en-v1.5`, normalized CLS vectors and query instruction prefix |
| Trainable part | Query-encoder weights; corpus vectors stay frozen |
| Action | Gaussian vector around the query embedding, normalized for retrieval |
| Retrieval | Exact cosine search over the entire frozen corpus, including during training |
| Reward | Evidence nDCG@10, using the benchmark's binary relevance labels |
| Loss | REINFORCE with leave-one-out baseline, plus KL to the original query policy |
| Validation | Evidence nDCG@10, Evidence Recall@100, Gold nDCG@10 |
| Final test | Original encoder versus best validation checkpoint on untouched test queries |

This is **custom benchmark adaptation**, not the official zero-shot benchmark protocol.
The source dataset calls all queries `test`; we create our own train/validation/test split.
Queries sharing evidence documents or identical normalized text stay together. Counts are targets;
groups are never split to meet a quota. Every query is assigned once. Other semantic near-duplicates
are not automatically detected. The split and resolved source revisions are saved in the manifest/bundle.

The corpus is shared by all splits. Training only reads training-query labels; checkpoint selection
uses validation. Test queries are not encoded or evaluated until the explicit final-test command.
The pilot uses the same split, caps training at 20 and validation at 10 queries, and retains the
full corpus and test set. No synthetic examples are generated in this version.

## Run on Modal

Use Python 3.10 or 3.11. Install from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

Authenticate Modal with `modal setup` if needed. For the first pilot, offline W&B needs no
W&B secret, and the pipeline needs no OpenAI credentials:

```bash
modal run modal_prepare.py --config config.pilot.json
EMBEDDING_RL_OFFLINE=1 modal run modal_train.py --config config.pilot.json --method baseline
EMBEDDING_RL_OFFLINE=1 modal run modal_train.py --config config.pilot.json --method rl
```

Preparation downloads the pinned corpus, benchmark labels and encoder, then encodes all documents
on a T4. Budget for several GB of downloads and a full-corpus encoding job even for the pilot.
The preparation function has four CPUs for parallel tokenization, 16 GiB host memory and a
four-hour timeout. Training uses one T4. These commands incur Modal compute costs.
See `VERIFICATION.md` for executed checks and cloud runs.

Data and checkpoints persist in the `embedding-rl-data` Modal Volume. Do not prepare the same
configuration concurrently. The old SciFact judge rejects BrowseComp-Plus before making a request;
the separate `modal_graded.py` workflow supports BrowseComp-Plus grading explicitly.

For online W&B, set your username/team in `wandb_entity`, add `WANDB_API_KEY` to a Modal secret
named `wandb-secret`, and omit `EMBEDDING_RL_OFFLINE=1`. `WANDB_SECRET_NAME` overrides its name;
`WANDB_ENTITY` in the secret overrides the config. Local metrics are always written as JSONL.
Benchmark query text is omitted from ranking tables so it is not uploaded to W&B.

## Persistent results and reranker reuse

All cloud stages mount the named **`embedding-rl-data`** volume at `/data` and explicitly commit
their results. Completed files survive containers and app shutdowns:

| Volume path | Contents |
| --- | --- |
| `huggingface/` | Downloaded encoder and pinned full scraped-page corpus |
| `prepared/<fingerprint>/` | Corpus vectors, original query vectors, document IDs/URLs, queries, labels, splits and provenance |
| `runs/<run-id>/` | Best and final query-encoder weights (`best.pt`, `last.pt`), config, metrics, validation results and summary |
| `exports/reranker/<run-id>/top-25/` | Reusable reranker examples, full candidate page text and checksums |

Changing the learning rate or number of steps already reuses the same prepared directory.
Changing query limits or split settings now also reuses document vectors from compatible completed
preparations, checking corpus/model revisions, maximum input length and document order/content metadata.
Only the new training/validation query vectors are encoded. Changing the corpus or encoder requires
new document vectors. Preparation writes its completion manifest last; an unfinished first encoding
job is not treated as a usable cache. Do not delete this volume while its artifacts are needed.

After a run completes, export its best checkpoint's natural top-25 results:

```bash
modal run modal_export.py --run-id REPLACE_WITH_RUN_ID --k 25
modal volume ls embedding-rl-data /runs
modal volume get embedding-rl-data exports/reranker/REPLACE_WITH_RUN_ID ./reranker-data
```

The equivalent local command is `embedding-rl export --run-id REPLACE_WITH_RUN_ID --k 25`.
`train.jsonl` and `validation.jsonl` contain queries and ranked candidates with document IDs,
retrieval scores and evidence relevance labels. Join each candidate's `document_id` to
`documents.jsonl` for the full scraped page text and URL. These private exports contain benchmark
text: keep them on your volume or local disk, out of public repositories and W&B uploads.
`manifest.json` records model/data provenance, checkpoint and file SHA-256 hashes, split sizes and
how many queries retrieved any known evidence. Repeating an export verifies the checksums and
reuses it. Test queries are never exported or evaluated by this command.

The pilot exports 20 training and 10 validation examples. Relevant documents are not injected into
the candidate list, and some queries can have zero relevant candidates. Unjudged documents receive
zero for the benchmark metric; that does not make them verified negatives. `best.pt` and `last.pt`
save reusable model weights, not optimizer state for exact resumption of an interrupted training step.
The summary reports both best and final validation scores, including regressions at the final step.

## Audited LLM rewards and anchor experiments

`modal_graded.py` uses a persistent 0–3 evidence rubric: irrelevant, topical, useful partial
evidence, decisive evidence. Positive grades must cite supplied page and question passages;
uncertain grades are capped at 1, and every positive receives a second independent judgment.
The lower grade is cached. A training-only audit checks repeatability and a stronger model before
training starts. Agreement is not human accuracy. The judge sees excerpts of at most 9,000 characters,
so it can miss evidence elsewhere on a page.

The frozen corpus and train/validation/test split are reused. Every sampled top-10 result is judged;
unjudged results are not silently treated as irrelevant. Graded training uses **fixed-max DCG@10**
(DCG divided by the all-grade-3 ideal), not pooled nDCG. Evaluation uses a common judged pool
and reports ordinary graded nDCG, strict graded nDCG (topical pages receive zero), and the original
benchmark metrics separately. Changing reward source also changes its scale.

The three new conditions are binary reward with normalization, LLM reward without normalization,
and LLM reward with normalization, each at seeds 42–44 and 500 updates. Normalization divides
within-query leave-one-out advantages by reward standard deviation floored at 0.05 and clips at ±5;
it is a stabilization heuristic, not the original unbiased RLOO estimator. Tied groups remain zero.
The old unnormalized binary controls provide the fourth condition. No InfoNCE is used in these runs.

Run these stages **sequentially**; one process owns the shared API budget ledger:

```bash
modal run modal_graded.py --stage prepare
modal run modal_graded.py --stage calibrate
modal run modal_graded.py --stage grade
modal run modal_graded.py --stage binary-controls
modal run modal_graded.py --stage llm-training
modal run modal_graded.py --stage rankings
modal run modal_graded.py --stage evaluate
```

The default experiment is `graded-20261106`. It requires the Modal secret `embedding-rl-openai`
with `OPENAI_API_KEY`; the local key is named `OPENAI_KEY` and is never embedded in source files.
Calls, frozen query/page labels, calibration, model checkpoints and comparisons persist on the
same volume. The shared ledger is `experiments/graded-20261106/judge/budget.json`, capped at a
conservative $95 estimate against the authorized $100 total API budget. Preliminary grader trials
count toward that total. Do not reset the ledger or run another writer concurrently. Modal compute
is separate. Repeated completed stages reuse their cached artifacts.

`modal_anchor.py` separately tests weaker/no KL anchoring and learning rate with the original
binary reward and **no OpenAI calls**. For 384-dimensional unit vectors and radius 0.15, the original
Gaussian KL coefficient 0.01 equals a cosine-distance penalty coefficient of about 170.67.
The softer variant uses coefficient 0.5, motivated by
[RELER's fixed-index setup](https://arxiv.org/html/2610.07731v1). This is a targeted ablation,
not a reproduction of RELER's vMF policy or complete training recipe.

```bash
modal run --detach modal_anchor.py
modal run --detach modal_anchor.py --selected no_anchor_lr5e-6 --repeat
```

Its protocol and run markers persist under `experiments/anchor-20261106/`. The first command
screens four settings at seed 42; the second repeats the selected setting at seeds 43 and 44.
All selection remains exploratory validation; the 166 test queries stay untouched.

When the pilot works, use `--config config.json` for preparation and training. Choose settings on
validation, then run the selected run ID on the final test once:

```bash
EMBEDDING_RL_OFFLINE=1 modal run modal_train.py --run-id REPLACE_WITH_SELECTED_RUN_ID
```

The run supplies its saved config. `test.json` contains the selected model's metrics, original-model
`before_metrics`, and their `delta`. Results are cached. Do not use test scores to tune the next run.

The equivalent local stages are:

```bash
embedding-rl prepare --config config.pilot.json --device cuda
embedding-rl train --config config.pilot.json --method rl --device cuda
embedding-rl final-test --run-id REPLACE_WITH_SELECTED_RUN_ID --device cuda
```

For local training without W&B credentials, set `wandb_mode` to `offline` or `disabled` in the config.
Unlike the Modal wrapper, the local CLI does not read `EMBEDDING_RL_OFFLINE`.

## Sampling and training-length follow-up

`config.followup.json` uses the full 581/83 train/validation split and the existing corpus cache.
The follow-up runs a 2x2 grid of 8/32 samples and radius 0.15/0.45 for 500 updates, evaluating at
steps 0, 10, 50, 100, 250 and 500. It first probes the four settings on the same 64 training queries,
then repeats the control and the selected variant at two additional training seeds.

```bash
WANDB_SECRET_NAME=wandb-secret modal run modal_followup.py --config config.followup.json
```

`training_seed` controls optimization separately from `seed`, which fixes the data split.
The larger radius uses a proportionally larger KL coefficient to preserve the penalty on encoder
movement. Selection uses final-step validation nDCG; ties follow condition order. Counts are matched
on optimizer updates, not compute. No final-test queries are used.

The completed experiment `knobs-a5f18dcb93` found more informative rewards with wider/more sampling,
but no validation advantage. Control and 32-sample training were repeated across three seeds;
the wider and combined settings each have one seed. See `VERIFICATION.md` for results.
The [W&B comparison](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/brfek36w)
links all eight runs. Reports and source snapshot persist in the `embedding-rl-data` volume under
`experiments/knobs-a5f18dcb93`; each run retains its checkpoints under `runs/<run_id>`.

Add `--experiment-id knobs-a5f18dcb93` with unchanged source and config to reuse completed stages.
The resume check validates configuration, not source equality; use the saved source snapshot when
reproducing this experiment. Omit the experiment ID to launch a fresh grid.

## Reward and loss

For each query, `m = normalize(encoder(query))` and
`sigma = exploration_radius / sqrt(embedding_dimension)`. Sample eight raw latent actions
`z ~ Normal(m, sigma² I)`, normalize each for search, and rank the full corpus.

Reward is binary-label Evidence nDCG@10. Its ideal ranking uses **all known evidence documents**,
including those outside the retrieved top ten. Unjudged documents score zero; judgments may be
incomplete. Gold labels are a separate diagnostic, not an extra training reward.

For each sample, `advantage = reward - mean(other samples' rewards)`:

```text
loss = -mean(advantage * log_prob(z)) + kl_weight * KL(policy || initial_policy)
KL   = ||m - m_initial||² / (2 * sigma²)
```

Rewards and sampled actions are detached. Backpropagation flows through the raw Gaussian action's
log-probability into the encoder; it never passes through search, sorting, or reward computation.
The default full-corpus recipe retains the Gaussian policy and does not use RELER's
von Mises–Fisher sampler or shortlist projection. The KL is exact for these equal-variance Gaussian policies.
Dropout remains off while gradients remain enabled.

The InfoNCE comparison also searches/trains against the full frozen corpus, treating every known
evidence document as a positive. Each method starts from the same pretrained model. The best
checkpoint includes step zero, so validation regressions retain the original model.

Watch `train/reward_std` and `train/zero_advantage_fraction`: identical rewards across samples mean
no REINFORCE signal for that query. All-zero rewards can mean the relevant pages are too far away;
more training steps alone cannot fix missing exploration. Use validation retrieval and sampling
diagnostics to check whether a reward change is useful.

## Limitations and verification

BGE-small reads only the first 512 tokens of each web document. BrowseComp-Plus pages are often
much longer, and some are multilingual; missing evidence and weak initial recall are likely limits.
There is no chunking or multilingual encoder in this first exercise. Public pretraining contamination
is also unknown. The first 10-step RL pilot completed with unchanged validation nDCG@10
(0.0652554); it demonstrates execution, not retrieval improvement. See `VERIFICATION.md`.

```bash
python -m pytest -q
python -m ruff check .
```

Tests cover full-corpus rewards, the policy gradient, exact KL, binary InfoNCE positives, dataset
schema/de-obfuscation, grouped splits, training without a judge, frozen documents, and final-test
separation using a tiny differentiable encoder. They require no API credentials or model downloads.

`browsecomp.py` loads labels and splits; `data.py` freezes corpus vectors; `objectives.py` holds the
losses; `train.py` runs optimization; `evaluate.py` reports retrieval metrics. `work/` is gitignored;
keep decrypted benchmark data local and do not commit it.

- [BrowseComp-Plus dataset and label definitions](https://huggingface.co/datasets/Tevatron/browsecomp-plus)
- [Corpus schema](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)
- [Authors' de-obfuscation script](https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_build_index/decrypt_dataset.py)
- [Official retrieval evaluation](https://github.com/texttron/BrowseComp-Plus#evaluating-retrieval-only-results)
- [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5)

## Cached candidate-pool RL experiments

The completed `modal_candidate.py` experiments reuse frozen training pools (original top25
plus known evidence). `CachedPoolReward` requires complete cached LLM grades; the explicit
`benchmark` control uses binary benchmark labels on the identical lists. All evaluation still
searches the full corpus, and final-test queries are excluded.

With `--pool-every 10`, nine updates use full-corpus binary rewards and one uses the pool.
Projection averages away Gaussian noise outside the fixed pool span only on pool updates.
It changes the policy-gradient estimator, not the sampled actions used for rewards. Applying
shortlist projection to a full-corpus reward would discard relevant retrieval directions.

The pool-only recipe overfit its lists and regressed on full-corpus validation. Mixed training
and identical-list binary controls were compared across three seeds; see `VERIFICATION.md`.
The optional `--grade-source benchmark --repeat-all` runs all six binary control jobs.
Always use a new experiment ID for a changed configuration, label source or implementation.
Existing markers validate settings and supervision before reuse.

The training loss remains RL throughout these experiments; no InfoNCE baseline was run.
The test suite covers projection, label-source isolation, budget persistence and split boundaries.

## Frozen inference controls and reusable indexes

`modal_feedback.py` tests fixed centroid feedback settings without updating the encoder.
`modal_candidate_audit.py` measures natural top-25 coverage and the diagnostic ceiling from
perfectly reordering those candidates. Its oracle uses benchmark labels only for evaluation;
it is not a deployable ranking algorithm.

`modal_lexical.py` builds first-512-token and full-text BM25 indexes, then evaluates fixed RRF
fusion with the original and trained dense models. Interrupted indexing resumes from saved
text/token batches. Both indexes, provenance and validation score arrays persist in the Modal
volume under `experiments/lexical-20261106`. Check saved indexes independently with:

```bash
modal run modal_lexical.py --verify-only
```

`modal_reranker_control.py` scores natural top-25 candidates with a frozen, revision-pinned
BGE cross-encoder at 512 and 2048 tokens per query/document pair. It deduplicates pairs and
persists logits after each scoring block. The default handles the dense candidate systems;
`--include-lexical` additionally requires the completed lexical artifacts and reuses existing
scores. It performs no RL updates or paid API calls. The longer text limit is a separate
context budget, not an encoder-training improvement.

These controls use the same 83 validation queries; all 166 final-test queries remain untouched.
Their validation caches must not become reranker training data. The judge cache at
`experiments/graded-20261106/judge-terra-v1` contains judgments keyed by query/document;
select only the 581 prepared training-query IDs when reusing it for reranker training.

`modal_transfer.py` checks the original and six final trained query encoders on fixed
SciFact/NFCorpus external test sets without tuning. It saves vectors, pinned dataset
revisions and official BEIR metrics under `experiments/transfer-20261106`.

`modal_passages.py` reuses the full lexical token cache to index 512-token passages
with stride 384 and aggregates their scores by maximum per original page. These are
retained lexical tokens after stemming and stopword removal, not BGE tokens. The
fixed recipe saves its index, passage boundaries, page mapping and validation scores
under `experiments/passages-20261106`. Run with `--verify-only` to independently
reload and check saved scores. It does not evaluate either external transfer set.

`modal_reranker_control.py --stronger-hybrids` evaluates the same frozen reranker
on prefix-BM25 and passage-max fusion candidates from all three final full-corpus
encoders. It stores this adaptive follow-up separately under
`experiments/hybrid-reranker-20261106` and reuses matching earlier logits.

`modal_feature_reranker.py` trains a small one-step RL reranker: four linear weights
combine per-query normalized hybrid RRF, dense cosine, log-BM25 and frozen BGE scores.
It samples top-10 rankings from the natural top-25 list without replacement, using a
Plackett–Luce policy and an unscaled leave-one-out policy-gradient loss. The reward
is benchmark evidence nDCG with the whole-corpus ideal denominator. This updates
four weights, not either pretrained model, and uses no contrastive loss or API calls.
The fixed upstream retriever is the seed-42 final full-corpus encoder plus prefix
BM25; three policy-training seeds each run 500 updates on the 581 training queries.
Validation is evaluated at the final checkpoint. Feature tensors, candidate indices,
frozen logits, protocol and policy weights persist in `experiments/feature-reranker-20261106`.
Use `--train-only` to reuse completed feature preparation after an interruption.

`modal_rerank_credit.py` compares placement credit and conditional marginalization
with the original whole-ranking RLOO estimator, keeping the same four features and
additive nDCG objective. Exact-enumeration tests verify the expected gradients.
Both passed a training-only variance gate; the six resulting policies persist in
`experiments/rerank-credit-20261106`.

`modal_neural_reranker.py` reuses those natural candidates and caches the frozen
BGE classifier's 1,024-dimensional penultimate representation in FP32, with TF32
disabled. The original four scalar features are retained unchanged. A learned linear
residual adds to the original four-feature policy, starting at zero. The cache
checks that the frozen final projection reproduces the current FP32 logits, saves
resumable blocks, and keeps training and validation query IDs separate. Three
500-update RL policies use scalar/residual learning rates of 0.01/0.0001. All
pretrained weights stay frozen; only 1,028 scalar parameters change. Artifacts
persist in `experiments/neural-reranker-fp32-20261106`; `--train-only` resumes after
feature preparation. This experiment also leaves the final test untouched.

The earlier `neural-reranker-20261106` and `neural-reranker-fp16-20261106` directories preserve stopped numerical-check attempts and training-only FP16/FP32 diagnostics. No policies were trained in those attempts.

## Reranker KL control

`modal_reranker_anchor.py` reuses the scalar and FP32 neural caches to train two predeclared first-choice KL strengths, 0.01 and 0.1, with three seeds each. The fixed reference is the initial hybrid distribution; this is not full-ranking KL. Artifacts persist under `experiments/reranker-anchor-20261106`. All six final policies are reported, with no final-test evaluation.

## Feature conditioning control

`modal_reranker_conditioning.py` fits a training-only projection of frozen neural features onto the four scalar features, removes that component, and divides residuals by training RMS. This is an invertible change of ranking-policy coordinates, with no new information. It runs three seeds each at KL 0 and 0.1. Transform parameters, cached transformed features, and equivalent original-coordinate policy weights persist under `experiments/reranker-conditioning-20261106`.

## Feature component ablation

`modal_reranker_components.py` reuses the saved transform to test removal-only and scaling-only, each at KL 0 and 0.1 with three seeds. Both coordinate systems are saved under `experiments/reranker-components-20261106`; no new fitting or retrieval occurs.
