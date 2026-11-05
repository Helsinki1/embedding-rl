# Query embedding RL, on Modal with W&B

**An OpenAI judge grades relevance → nDCG scores sampled rankings → a policy gradient updates the query encoder.**

This is a small research exercise, not a demonstrated improvement yet. Run the pilot first,
inspect the judgments, then compare RL with the unchanged encoder and an InfoNCE baseline.

## What is implemented

| Part | Choice |
| --- | --- |
| Benchmark | SciFact: 5,183 scientific abstracts |
| Split | 200 train / 50 validation from official train; all 300 official test queries held back |
| Encoder | `BAAI/bge-small-en-v1.5`; normalized CLS vectors; query instruction prefix |
| Trainable part | All query-encoder weights; document vectors stay frozen |
| State | Query text |
| Action | A sampled vector around the query embedding; normalized before retrieval |
| Training environment | Fixed pool of 100 candidates/query, mixing dense retrieval and BM25 |
| Judge | `gpt-5.4-mini-2026-03-17`, low reasoning; cached grades 0–3 and short reasons |
| Reward | nDCG@10 within the complete judged candidate pool |
| Update | REINFORCE with a leave-one-out baseline over 8 samples, plus reference-policy KL |
| Evaluation | Deterministic full-corpus retrieval, using benchmark labels: nDCG@10 and recall@25 |
| Comparisons | Unchanged encoder, multi-positive InfoNCE, RL |

Evidence that **contradicts** a claim is relevant too. The rubric measures evidence usefulness,
not whether a claim is true. Training uses only judge grades; benchmark relevance labels are
used for grouping the development split and for validation/final testing.

Training and validation keep claims sharing positive evidence in separate groups. The official
test split remains intact. The corpus is shared across all splits, as in ordinary retrieval
evaluation. A public pretrained encoder may have encountered this public benchmark previously;
the split prevents leakage introduced by this exercise, not unknown pretraining contamination.

## Local setup

Use Python 3.10 or 3.11. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m ruff check .
```

The automated tests use a tiny differentiable encoder and a mocked judge, require no API keys,
and exercise actual offline W&B logging. They test the policy gradient against a known analytic
gradient, reward normalization, split boundaries, cache resumption, and final-test separation.

The local CLI also runs each real stage:

```bash
embedding-rl prepare --config config.pilot.json
embedding-rl judge --config config.pilot.json
embedding-rl train --config config.pilot.json --method rl
embedding-rl final-test --run-id REPLACE_WITH_SELECTED_RUN_ID
```

These real commands download models/data or call paid APIs as appropriate. `prepare` defaults
to CPU locally; add `--device cuda` on a GPU machine. The full corpus is easier to encode on Modal.
Credentials are read from environment variables; `.env.example` documents their names and is
not loaded automatically. Never put API keys in either JSON config.

## Modal setup

Modal is already authenticated on this machine. On a new machine run `modal setup`.
In the [Modal Secrets dashboard](https://modal.com/secrets), use:

| Secret name | Required fields |
| --- | --- |
| `openai-keys` | `OPENAI_API_KEY` |
| `wandb-secret` | `WANDB_API_KEY` |

An `openai-keys` secret already exists in this Modal workspace; its value has not been inspected.
The judge checks that it contains `OPENAI_API_KEY`. Override secret names with the local environment
variables `OPENAI_SECRET_NAME` and `WANDB_SECRET_NAME` if needed.

Create your W&B account, then replace `YOUR_WANDB_ENTITY` in the config you use with your W&B
username or team name. Keep project `embedding-rl`; W&B creates the project on the first online
run. An entity supplied as `WANDB_ENTITY` in the Modal secret overrides the config.

Start with the pilot: **20 training queries × 50 candidates = 1,000 paid relevance judgments**,
in batches of 10. This is a pair limit, not a dollar limit; cost depends on token usage.
The full config uses 20,000 pairs. Responses include token usage in the judgment cache.

```bash
modal run modal_prepare.py --config config.pilot.json
modal run --detach modal_judge.py --config config.pilot.json
```

Wait for the judge stage to finish before training. Preparation needs neither OpenAI nor W&B
credentials; judging runs on CPU and needs only OpenAI. Stages use the same persistent
`embedding-rl-data` Volume. Do not run two preparation/judging jobs on the same config concurrently.

Before training, inspect a few cached grades and reasons:

```bash
modal volume get embedding-rl-data /prepared ./downloaded-prepared
```

Judgments live under `prepared/<preparation-id>/judgments/<judge-id>/<query-id>.json`.
Check roughly 20 examples, especially grade 1 versus 2 and contradicting evidence.
Changing the judge model, rubric, or key judge settings creates a separate cache.

Then run the three methods, each starting from the same pretrained encoder:

```bash
modal run modal_train.py --config config.pilot.json --method baseline
modal run modal_train.py --config config.pilot.json --method infonce
modal run modal_train.py --config config.pilot.json --method rl
```

To try training before creating a W&B account:

```bash
EMBEDDING_RL_OFFLINE=1 modal run modal_train.py --config config.pilot.json --method rl
```

This skips the W&B secret and writes offline logs to the Volume. After creating your account,
download them and sync using your actual entity/project:

```bash
modal volume get embedding-rl-data /runs ./downloaded-runs
wandb sync --entity YOUR_ACTUAL_ENTITY --project embedding-rl PATH_TO_OFFLINE_RUN_DIRECTORY
```

Preparation and training use one T4 GPU. Judge calls run separately on CPU, with completed
batches committed to the Volume so rerunning reuses grades. Training checkpoints persist;
interrupted training starts a new run (optimizer resumption is not implemented).

Once the pilot works, use `--config config.json` for **every** stage. Changing the data split or
pool size creates a distinct prepared dataset and judgment cache. Reuse that prepared dataset
when comparing learning rates or exploration settings.

## What to watch in W&B

- `validation/ndcg@10`: the main hillclimbing signal, measured over the full corpus.
- `validation/recall@25`: whether useful evidence is among the returned 25 documents.
- `train/reward_mean` and `train/reward_std`: sampled ranking quality and learning signal.
- `train/kl`, `train/query_drift`, `train/grad_norm`: how much the query encoder is moving.
- `rankings`: before/after validation examples with document titles and benchmark relevance.

The best checkpoint is selected by validation nDCG, including the initial model at step zero.
If training makes things worse, the saved best checkpoint remains the initial model. Every run
records config, model/dataset revisions, code fingerprint, judge fingerprint, and local JSONL
metrics. Results are under `/data/runs/<run-id>` on the Volume.

For your first comparison, change only learning rate (`1e-6` → `3e-6`) or exploration radius
(`0.15` → `0.3`) one at a time. If reward spread is zero, sampled rankings are identical and
REINFORCE has no learning signal. If training reward rises while validation falls, you may be
overfitting the candidate pools or judge. Improvement is not guaranteed.

Choose your configuration using validation results. Only then evaluate the selected run on test:

```bash
modal run modal_train.py --run-id REPLACE_WITH_SELECTED_RUN_ID
```

The saved run supplies its original config and best checkpoint. Final-test results are cached in
`test.json` and logged as a separate W&B run in the same experiment group. Do not use test scores
to choose the next hyperparameter change. For a serious result, predeclare repeat seeds and
comparisons before looking at test results.

## The policy gradient, precisely

Let `m = normalize(encoder(query))`, and `sigma = exploration_radius / sqrt(embedding_dimension)`.
Sample latent vector `z ~ Normal(m, sigma² I)`. The environment normalizes `z`, computes cosine
similarities with frozen candidate vectors, and returns nDCG@10. IDCG always uses all judged
candidates, so a bad retrieved subset cannot redefine its own ideal ranking.

The update uses `-(reward - mean_of_other_rewards) * log_prob(z)` and averages over samples.
Sampled `z` is detached; the gradient flows through `log_prob`, not through sorting or KNN.
The Gaussian log probability is for the **raw latent vector**, not its normalized version.
The exact latent-policy KL to the initial encoder is `||m - m_initial||² / (2 sigma²)`.
Dropout stays disabled while autograd remains enabled.

This version trains on fixed candidate pools; validation/test search the full corpus. It does
not yet optimize retrieval from changing full-corpus KNN results during training. Doing that
would require judging newly retrieved documents and defining a consistent relevance reference.
SciFact is a manageable scientific-retrieval exercise, not a scraped-web benchmark or a reranker.

## Code map and references

`data.py` prepares the benchmark; `judge.py` defines the rubric/cache; `objectives.py` contains
the RL/InfoNCE math; `train.py` runs updates; `evaluate.py` performs full-corpus evaluation.
The three `modal_*.py` entrypoints call exactly the same Python code as the local CLI.

- [BGE model card and encoding instructions](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [SciFact benchmark relevance labels](https://huggingface.co/datasets/BeIR/scifact-qrels)
- [OpenAI GPT-5.4 mini model](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Modal Volumes](https://modal.com/docs/guide/volumes) and [Secrets](https://modal.com/docs/guide/secrets)
