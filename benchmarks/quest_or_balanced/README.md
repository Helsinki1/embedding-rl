# Compact 80/20 baseline rerun

This experiment replaces the previous 100,363-row training set and QUEST-only
78-query test set with a fixed, representative 5,000-query sample. It does not
overwrite the original benchmark, previous splits, or results.

| Partition | Total |
|---|---:|
| Gradient training | 3,600 |
| Validation | 400 |
| Test | 1,000 |

The outer split is **80% training pool / 20% test**. Ten percent of the training
pool is reserved for validation, so the actual gradient/validation/test split is
72/8/20. The canonical split is produced by the shared `build.py` and verified
by `data.load_splits`; exact counts, strata, exclusions, and hashes are recorded
in its manifest. Every new approach must reuse these exact files and IDs, not
generate its own 5,000-row sample. The original split proposed in this conversation
was retired before training, in favor of the other task's shared split.

The shared builder stratifies by domain/source and OR template and excludes
the recorded prior exposures from new holdouts. The held-out unit is a complete
query combination; component questions and answer documents may recur across
partitions. All 393,140 documents remain in the search corpus, including test
positives. Each approach starts from fresh base weights. Test relevance labels
are excluded from training and model selection.

Both methods start from fresh Qwen3-Embedding-8B weights, with one normalized
4096-dimensional query embedding and frozen document vectors. Each uses one
epoch, batch size 32, and 113 updates. The RL Gaussian REINFORCE/F1/KL objective
and the full-corpus multi-positive InfoNCE objective are unchanged. LoRA rank 16,
learning rate 0.0001, and all other hyperparameters are inherited from the earlier
run. Validation occurs at updates 40, 80, and 113. Cutoff k and the best updated
checkpoint are selected only on validation. The primary trained result always
uses a checkpoint after an update; `summary.json` also preserves the existing
selection rule that allows the untrained initialization to win. Frozen and
trained comparisons are written under `evaluations/`.

The two jobs run on separate GPUs. Each is enclosed by GNU `timeout` at 1,790
seconds with a five-second kill grace period, covering data/model loading,
training, and test evaluation. A timeout is recorded as a timeout, not reported
as a completed run. No unrelated processes are stopped.

The split is generated once by its owning task and then frozen. Do not rebuild
it for each approach. Verify the existing package before starting a run:

```bash
python -c 'from benchmarks.quest_or_balanced.data import load_splits; load_splits("benchmarks/quest_or_balanced", "benchmarks/quest_or_synthetic")'
```

The manifest stores exact hashes, source revisions, counts, and holdout exclusions.
`launch.py` is the launcher for the existing compute server and expects a code
snapshot under `<run-root>/code`. It refuses to launch the same directory twice.

Metrics are macro-averaged per-query set precision, recall, and F1. NQ labels are
unions of supplied positives, not exhaustive relevance judgments. Scores on this
new mixed test set are not directly comparable to earlier QUEST-only scores;
source and arity breakdowns accompany the overall scores.
