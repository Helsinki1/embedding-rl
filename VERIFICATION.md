# Anchor and learning-rate ablation — 2026-11-06

- Completed four 500-step screening runs plus two seed repeats on the same 581 train / 83
  validation queries and 100,195-document corpus. Binary evidence nDCG remains the reward.
  No LLM API calls or InfoNCE; all 166 test queries remain untouched.
- Original Gaussian KL coefficient 0.01 implies cosine coefficient 170.67 at radius 0.15 in
  384 dimensions. Tested cosine coefficient 0.5 and no anchor at learning rates 1e-6 and 5e-6.
- Seed-42 final validation nDCG: soft anchor/1e-6 **0.0391871**, no anchor/1e-6 **0.0446225**,
  soft anchor/5e-6 **0.0437987**, no anchor/5e-6 **0.0869234**. Original encoder **0.0373637**;
  prior strong-anchor controls **0.0374285**. The largest change combines two settings.
- No anchor/5e-6 repeated at seeds 43 and 44: **0.1003535**, **0.0941197**. Three-seed mean
  **0.0937989**, sample standard deviation **0.0067208**. Seed-averaged per-query improvement
  versus the original encoder is **+0.0564352**, with a descriptive paired-query bootstrap
  95% interval **[+0.0353862, +0.0792419]**. Scores improve for 32 validation queries and worsen
  for 4. This interval does not account for screening four variants or repeated validation use.
- Mean Recall@100 **0.2654135** and gold nDCG@10 **0.0875626**. These are promising validation
  results, not an official benchmark score or an untouched final-test result.
- W&B selected runs: [42](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/e027lgj6),
  [43](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/uysre18q),
  [44](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/zywf55er).
  Protocol, records and source persist under `experiments/anchor-20261106`; weights remain in
  the associated `runs/` directories. Ruff and diff whitespace checks passed.
- The matched sampling follow-up `unanchored-knobs-20261106` completed. With the same no-anchor,
  5e-6 setup, 32 samples and radius 0.45 scored **0.1234384**, **0.1105335**, **0.1235386** across
  seeds 42–44: mean **0.1191702**, versus matched 8-sample/radius-0.15 mean **0.0937989**.
  All three seed-paired differences were positive. The seed-averaged query-bootstrap difference
  was **+0.0253713**, 95% interval **[+0.0128258, +0.0387198]**, before any tuning correction.
- Wider sampling alone scored **0.0848998** at seed 42, below its **0.0869234** control; more
  samples alone scored **0.1157160**. Those two settings each have only one seed. Informative
  training groups increased from **29.87%** for the matched control to **60.47%** with both knobs.
  [W&B comparison](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/y9ch4j9i).
- Three fresh 2,000-step runs completed under `longer-20261106`. Final validation nDCG values
  were **0.1351010**, **0.1404990**, **0.1397076**; mean **0.1384358**, sample standard deviation
  **0.0029150**. Their actual within-run means at steps 500/1000/1500/2000 were
  **0.1191394 / 0.1279092 / 0.1295320 / 0.1384358**. All three final values exceeded their own
  step-500 scores. Only the update budget/evaluation schedule changed.
- Fresh CUDA runs were not perfectly bitwise reproducible: seed 42's step-500 nDCG differed
  from its earlier run by **-0.0000923**; seeds 43/44 matched. Reported training-length gains use
  the actual within-run readings. No final-test queries or new API calls were used.

## LLM grading experiment — completed, 2026-11-06

- Added grounded 0–3 page grading with positive confirmation, immutable cached labels, a
  shared dollar ledger, and explicit train-only reward access. Ten new tests cover grading,
  cached paid responses, concurrent budget reservations, normalization, evaluation and integration;
  the full suite passed 41 tests before launch. Ruff and diff checks passed.
- A weaker preliminary grader failed the preselected consistency threshold. GPT-6.1 Terra's
  training-only audit passed: 128 pairs across 32 queries, **92.97%** exact repeat agreement,
  **95.31%** exact agreement with GPT-6.1 Sol. Agreement is not accuracy; many pairs are negative,
  and both judges come from the same model family.
- Warmed **17,821** cached pairs, grading new sampled pages as needed. Evaluated 19 checkpoints
  on a common pool of **1,360 validation pairs**, with a stronger-judge audit on 12 fixed queries.
  Conservative total API estimate **$87.7507**, including preliminary trials and evaluation,
  against a $95 ledger ceiling and $100 authorization. Modal compute is separate.
- Three-seed means on original evidence labels: previous binary control **0.0374285**,
  binary normalized **0.0374285**, LLM raw **0.0374069**, LLM normalized **0.0369537**.
  The original pilot, re-evaluated on all 83 validation queries, scored **0.0374285**.
- Shared-pool graded nDCG means: previous binary control **0.0739737**, binary normalized
  **0.0741842**, LLM raw **0.0741982**, LLM normalized **0.0735934**. Every paired-query
  bootstrap interval for graded improvement versus the binary control included zero.
  The stronger 12-query audit also gave no support for an improvement.
- Informative training groups increased from **18.58%** in binary controls to **29.20%** with
  LLM raw reward. This additional signal did not produce a meaningful retrieval gain under
  the original strong anchor. The later binary-reward ablations are separate interventions.
  [W&B comparison](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/3qyiyi3w).
- LLM training uses fixed-max DCG, while final graded evaluation uses a shared-pool nDCG.
  This reward-scale difference is recorded explicitly. The 166 test queries remain untouched.
- Persistent experiment: `experiments/graded-20261106`. The continuation pipeline completed
  training, rankings, evaluation and report collection. The overnight heartbeat checks current
  artifacts and the shared ledger before further experiments.
- A separate synthetic fixed-pool Gaussian projection probe preserved every candidate score
  to floating-point precision and all rewards exactly. Across 4,000 groups it reduced gradient
  variance **6.90×**, with no resolved paired mean shift (maximum component z **2.99**).
  This is a numerical estimator check on eight synthetic documents, not a retrieval result;
  a matched cached-candidate training comparison is the next planned experiment.

## Controlled sampling and training-length follow-up — 2026-11-06

- Completed eight 500-step RL runs on a T4 with the full 100,195-document corpus and cached
  581 training / 83 validation split. The 166 final-test queries remain untouched.
- Compared 8 versus 32 samples and exploration radius 0.15 versus 0.45. Wider sampling scales
  the KL coefficient from 0.01 to 0.09, holding the mean-displacement penalty constant.
  Batch size 4, learning rate 1e-6, and split seed 42 are fixed. No InfoNCE or LLM judge used.
- Paired probe on 64 training queries: reward variation appeared for 9/64 queries with the
  control, 10/64 with more samples, 11/64 with wider sampling, and 17/64 with both. Best-of-sample
  rewards improve with more exploration, but require labels to select and are not deployment scores.
- All four seed-42 conditions finished at validation nDCG@10 **0.03742847964167595**, versus
  **0.037363700568675995** initially. This tiny gain changes the nDCG of just one validation query.
  It is not evidence of a useful generalization gain. The earlier 10-query pilot is not comparable.
- Final-score selection tied, so the stable condition-order tie-break chose `more_samples` for
  repeats against control at seeds 43 and 44. Control finished at **0.03742847964167595** for all
  three seeds. More samples tied it for seeds 42 and 43 and returned to baseline at seed 44.
  Its mean paired difference was **-0.000021593024333318073**. Wider/both have only one seed each.
- Evaluations at 0, 10, 50, 100, 250 and 500 updates show no sustained validation gain from more
  steps. The fixed 64-query training monitor improved in some runs; it is an in-sample diagnostic.
  Comparisons use equal update counts, not equal compute budgets.
- [W&B comparison](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/brfek36w) and
  [paired sampling probe](https://wandb.ai/yx2948-columbia-university/embedding-rl/runs/sl22kmrk).
  Volume `embedding-rl-data` retains `/experiments/knobs-a5f18dcb93` with the comparison, source
  snapshot and report, and `/runs/<run_id>` with metrics, provenance, best and final model weights.
- 31 automated tests, Ruff, and `git diff --check` passed before launch. New checks cover split/cache
  stability across training seeds, matched KL strength, paired probes, fixed training monitors,
  explicit evaluation checkpoints, final-vs-best metrics and seed-paired comparisons.

## Real Modal pilot and persistence verification — 2026-11-06

- Encoded all 100,195 BrowseComp-Plus documents on a T4 and committed the prepared artifacts to
  `embedding-rl-data/prepared/2df840e50dfc76be`. Fixed a PyArrow/NumPy import incompatibility by
  pinning PyArrow 20.0.0, and enabled four CPU tokenizer threads for preparation.
- Ran the unchanged encoder (`baseline-b303413bc2`) and 10 RL steps (`rl-4f5e40ad1b`) with
  20 training queries and 10 validation queries, searching the full corpus. No InfoNCE run or
  LLM judge was used. W&B authenticated successfully and both runs synced online.
- Baseline, step 5 and step 10 validation scores were identical: evidence nDCG@10
  **0.06525542587041855**, Recall@100 **0.17722222208976746**, gold nDCG@10
  **0.10000000149011612**. Best checkpoint remains step 0. This pilot does not show improvement.
  Across 40 sampled query occurrences, 34 (85%) had no reward variation across the eight actions.
- Both `best.pt` and `last.pt` are saved with the run's config, metrics and validation results.
  A separate job successfully reused all document vectors to prepare the larger query split at
  `prepared/79f6ff30a298db92`, without re-encoding the corpus.
- A fresh export job loaded the persisted best checkpoint and vectors, then saved natural top-25
  reranker examples to `exports/reranker/rl-4f5e40ad1b/top-25`: 20 training queries, 10 validation
  queries, and 719 unique pages with full scraped text. Known evidence appears in the candidate
  sets for 8 training queries and 4 validation queries. No test queries were exported or evaluated.
- 26 automated tests, Ruff, and `git diff --check` passed. New checks cover corpus reuse across
  query settings, cache incompatibility, saved final weights, checkpoint-based reranker export,
  split boundaries, unchanged natural candidates, full page text, and export checksum validation.
- W&B: [baseline](https://wandb.ai/davidxionghl-columbia-university/embedding-rl/runs/xlpb6ghx),
  [RL](https://wandb.ai/davidxionghl-columbia-university/embedding-rl/runs/luv5uxmr).

The named volume survives app shutdowns. The checkpoint files contain reusable model weights;
they do not include optimizer state for exact resumption of an interrupted training step.

## BrowseComp-Plus migration verification — 2026-11-05

- 22 automated tests passed. Added checks for the dataset adapter, three-way evidence-grouped
  split, pilot holdout stability, full-corpus preparation, binary relevance rewards/InfoNCE,
  training without any judge or candidate pool, frozen document vectors, and before/after testing.
- Ruff and `git diff --check` passed. Local CLI and Modal prepare/train entrypoints show help.
- Checked one real hosted query record without printing benchmark text: decoding succeeded;
  the record had six evidence IDs and five gold IDs, with gold a subset of evidence.
- Read the official evidence qrels: 830 queries and 5,064 positive query/document pairs.
  Grouping these labels with seed 42 yielded 581 train / 83 validation / 166 test queries.
  The loader also groups identical normalized query text, so actual saved counts remain authoritative.
- Resolved and pinned both BrowseComp-Plus Hugging Face dataset revisions. The full dataset,
  model encoding job, Modal image build, GPU training, and retrieval improvements have not
  been validated end to end. Tests use a tiny differentiable encoder and synthetic fixtures.
- Used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid an unrelated system-installed pytest plugin
  with a missing dependency. No paid API, Modal job, or online W&B upload was launched.

The exercise retains Gaussian latent-action sampling and exact Gaussian reference KL.
It does not implement RELER's vMF sampler or CMP. Read README for commands and limitations.

## Earlier SciFact verification — 2026-11-04

- 14 automated tests passed, including a numeric check of the REINFORCE gradient,
  fixed-reference nDCG, KL, InfoNCE, split isolation, interrupted judge caching,
  all three training methods, explicit final testing, and real offline W&B log creation.
- Ruff passed; local CLI and all three Modal entrypoints import and show CLI help.
- Downloaded and read the real SciFact dataset: 5,183 documents; the default selection
  yields 200 training, 50 validation, and 300 official test queries.
- Loaded the real BGE-small checkpoint on CPU. A forward/backward check produced a
  384-dimensional vector with unit norm and nonzero encoder gradients.
- Installed this project into the task-local environment at
  `../../work/embedding-rl-venv`. The environment reuses existing system packages;
  use the clean-venv instructions in README for a fresh installation.

Tests used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because this machine has unrelated pytest
plugins with missing dependencies. The project tests require no extra pytest plugins.

No Modal container build, paid GPU job, paid OpenAI judgment, online W&B upload, or real
benchmark training run has been executed. The tests establish pipeline correctness on
small fixtures; they do not establish retrieval improvement or cloud-runtime success.

Start with `config.pilot.json` after adding the account settings in README.

## Candidate-pool projection and matched label controls — 2026-11-06

- 49 tests, Ruff and whitespace checks passed before the binary-pool controls launched. Tests cover conditional score gradients, Monte Carlo variance, redundant candidate spans, complete immutable LLM labels, training-only access, full-corpus gradient preservation, and identical-candidate binary label controls.
- All candidate runs use one-step Gaussian RLOO. Conditional projection is restricted to candidate-pool updates. Actual rewards always use the original sampled actions.
- The unanchored fixed-list screen collapsed on full-corpus validation despite improving its training-list metric. A cosine0.5 anchor prevented collapse but projection did not beat its matched final checkpoint. Both are single-seed negative screens.
- Mixed runs use exactly450 full-corpus updates and50 pool updates. Each label source and estimator was repeated at seeds42/43/44; binary controls were all predeclared.
- alternating-pool-20261106/unprojected: final validation mean **0.1246725**, seed standard deviation **0.0058984**.
- alternating-pool-20261106/projected: final validation mean **0.1292035**, seed standard deviation **0.0006103**.
- alternating-binary-pool-20261106/unprojected: final validation mean **0.1260114**, seed standard deviation **0.0024231**.
- alternating-binary-pool-20261106/projected: final validation mean **0.1320837**, seed standard deviation **0.0049792**.
- The12-query stronger-judge audit favors the earlier2000-step binary models in mean (.1091→.1816), but its query-bootstrap interval includes zero. Total conservative API estimate **$90.6006**; candidate training used no API calls.
- All166 test queries remain untouched. Repeated validation selection prevents treating these as final-test claims. See results.md and candidate-research-summary.json in the private artifact directory for matched bootstrap intervals and run links.

## Longer mixed training and inference controls — 2026-11-06

- Three predeclared mixed2,000-update runs completed, mean nDCG **0.1452072**, versus matched full-corpus-only **0.1384358**.
- Each run performs exactly1,800 full-corpus and200 projected binary-pool updates. Settings and reward provenance are checked against completed controls before launch.
- Fixed-setting inference controls completed: soft-centroid feedback, first512/full-textBM25, and RRF60/top1000 fusion with original and three prior binary2000 models, followed by a frozenBGE top25 reranker at512/2048token pair limits. They are separate from RL training and use no labels to transform queries or rank documents.
- 57 tests, Ruff and whitespace checks passed. Three new tests check feedback identity, normalization, deterministic retrieved-vector use and protocol validation. A lexical retrieval smoke check passed with the pinned dependency. An additional interrupted-tokenization test verified resumption, discarded partial writes, and score equality to ordinaryBM25 tokenization. Two candidate-coverage tests check the global ideal denominator and inability to recover relevant documents outside the candidate pool. Two reranking tests check candidate preservation, tie handling, duplicate rejection and the global denominator. Both persisted lexical indexes were reloaded in a fresh container and reproduced all validation score arrays within1e-6.
- All166 test queries remain untouched;83 validation queries were reused for exploration. This batch uses no API calls. Detailed query intervals and every condition are in results.md.

## External transfer check — 2026-11-06

- Evaluated the original encoder and all six predeclared trained encoders on 300 SciFact and 323 NFCorpus test queries, with frozen original document embeddings and no tuning.
- Pinned corpus/query/qrels revisions and text schemas were checked. The official evaluator passed its linear-gain numerical smoke check; per-query IDs and all three seeds align.
- The 166 BrowseComp-Plus final-test queries remain untouched. No API calls or training updates were made. Cached external vectors and results persist in Modal.

## Lexical passage control — 2026-11-06

- 59 tests passed, including complete within-page chunk coverage, empty/short pages, native integer iteration, and max aggregation without duplicate pages. Ruff and whitespace checks passed.
- One fixed passage recipe, no training or API calls. Cached dense vectors reproduce earlier per-query baselines within 1e-6. A fresh container reload reproduces every passage-aggregated score within 1e-6.
- BrowseComp final-test and external transfer datasets were not evaluated in this experiment.

## Stronger hybrid reranking — 2026-11-06

- All 12 conditions and three seeds completed; every source baseline matched its earlier per-query metrics.
- After a local client connection error, a fresh CPU container recomputed before/after/oracle metrics from saved logits with zero error. No rescoring was required.
- Both limits contain 3,977 unique pairs, including 2,514 reused scores and 1,463 new scores each. No final-test or external transfer evaluation.

## Small feature-policy reranker — 2026-11-06

- Before launch, all 62 tests, Ruff and whitespace checks passed. Exact enumeration of all three-item rankings verifies normalized Plackett–Luce probabilities and an unbiased leave-one-out gradient. A deterministic Monte Carlo check verifies Gumbel sampling probabilities and no duplicate placements. Feature normalization is checked within each query, including constant signals.
- A synthetic end-to-end training smoke check moved a known relevant item from outside the top 10 to rank 1, improving nDCG from 0 to 1 in 250 updates. This is a code check, not a benchmark result.
- The predeclared experiment uses one fixed upstream encoder, all three policy seeds, 500 final-checkpoint updates and natural candidate lists. Training receives only the 581 training-query tensors. Validation baseline and frozen-BGE metrics must match the earlier seed-42 controls.

- The completed feature policies were independently reconstructed in NumPy from saved features and weights; maximum metric error 4.2e-08. All three final checkpoints and split IDs match.

## Ranking credit assignment — 2026-11-06

- 65 tests passed. Exact small-space enumeration verifies placement and marginalization gradients; marginalizing the first position has the exact gradient without sampling noise.
- Both estimators passed the predeclared training-only variance gate and all six policy runs completed. Independent NumPy reconstruction verifies final validation metrics from saved weights.
- Removed unused full-sort backing storage from the feature candidate tensors without changing values or provenance, reducing their combined files from about 1.07 GB to 1.09 MB. The compaction record is persisted.

## Neural residual ranking policy — 2026-11-06

- 67 tests passed, including zero-residual logits/probability/gradient equivalence and frozen-residual training equivalence.
- All 16,600 captured representations reproduce the frozen head output; final metric reconstruction in NumPy differs by at most 4.2e-08.
- All three 500-update policies persist with protocol digests; the training function receives only training tensors. No final-test or external-transfer evaluations were made.

## First-choice KL control — 2026-11-06

- 69 tests passed. Zero strength reproduces unregularized updates; KL tests cover its direction, detached reference, nonnegativity and score-shift invariance.
- All six final checkpoints independently reconstruct in NumPy: metric error 6.3e-08, KL error 8.6e-08.
- Only cached training tensors enter optimization. No new page scoring, GPU training, paid API calls, final-test or external-transfer evaluation.

## Feature conditioning — 2026-11-06

- 71 tests passed, including exact policy-family equivalence and fixed training-only fitting.
- Independent NumPy least squares and scaling reproduce the saved transform. All six metrics reconstruct within 8.2e-08; mapped original-coordinate metrics differ by at most 0.
- No new page scoring, GPU training, paid API calls, final-test or external-transfer evaluation.

## Feature component ablation — 2026-11-06

- 73 tests passed, including equivalent policy-family tests for both component transforms.
- All twelve saved policies reconstruct independently in NumPy within 6.3e-08; mapped original-coordinate metrics differ by at most 0.
- No new fitting, page scoring, GPU training, API calls, final-test or external-transfer evaluation.

## Final storage cleanup and closeout — 2026-11-06

- Compacted the training-only projection artifact from 119,014,387 to 58,463 bytes; tensor values, dtypes, metadata and hashes match the earlier local and remote artifact. Added a minimal-storage regression assertion; both conditioning tests pass. No policy changed.
- All 36 small reranker checkpoints have 500 updates and a saved protocol digest. The new registry records hashes and volume paths. The 42 encoder runs and 84 encoder checkpoint records remain preserved.
- All final training apps are stopped; no more experiments are queued. API estimate remains $90.60059215. All 166 BrowseComp final-test queries remain untouched.
