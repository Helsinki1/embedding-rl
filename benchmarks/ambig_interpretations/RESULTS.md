# Interpretation embedding experiment: completed

Both models completed three epochs (249 updates, 7,902 training-example exposures)
on two Modal H100s. Setup through completed evaluation took 18.42 minutes;
the remote job took 11.75 minutes. Neither hit the time limit.
No document corpus was mounted, read, or searched.

| Method | Held-out cosine matching loss ↓ | Matched cosine ↑ | Output-to-output cosine | Train/eval wall time |
|---|---:|---:|---:|---:|
| Original query embedding repeated three times | 0.110977 | 0.889023 | 1.000000 | Frozen control |
| Three output slots in one pass (MTP placement) | 0.101523 | 0.898477 | 0.995716 | 216.7 s |
| Three autoregressive output embeddings | 0.101533 | 0.898467 | 0.995948 | 595.3 s |

Both lower loss by about 8.5% relative to the frozen original-query control.
However, the outputs mostly collapse to a shared direction: pairwise cosine is
about 0.996, whereas the teacher interpretation vectors average
0.854666. On the three-interpretation subset,
the same pattern holds (MTP 0.995529,
autoregressive 0.995805,
teacher 0.833250). This run demonstrates
improved teacher matching, but does not demonstrate useful separation into distinct
interpretations. The methods are essentially tied; there is only one seed each.
Retrieval quality was not evaluated.

The teacher is the unchanged, revision-pinned Qwen3-Embedding-8B. The loss directly
matches the three predicted unit vectors to human-written interpretation vectors
using minimum one-to-one cosine distance. For two labels, the unmatched third
output is masked. Both students use identical rank-32 LoRA initialization, batch
order, learning rate 1e-4 and three epochs. Epoch 3 was selected for both using
validation loss before test evaluation.

The user-approved 2–3-interpretation subset has 2,634 training, 293 validation and
706 test questions. The downloaded AMER release has 5,323 training rows, rather
than the paper's reported 5,044. Rows missing original human rewrites or eligible
annotations were excluded. Fifty-four train/evaluation query overlaps were removed.
Held-out question IDs are disjoint; two exact interpretation strings recur under
different training and test questions. The frozen manifest records all exclusions.

Four tests passed in the pinned Modal environment: assignment/masking, causal
output slots, differentiable autoregressive feedback, and split isolation.
The saved predictions are finite unit vectors with shape [706, 3, 4096].

* [MTP best checkpoint](/home/davidx/Downloads/embedding-rl/benchmarks/ambig_interpretations/runs/ambig-20261106-v1/mtp/best.pt)
* [Autoregressive best checkpoint](/home/davidx/Downloads/embedding-rl/benchmarks/ambig_interpretations/runs/ambig-20261106-v1/autoregressive/best.pt)
* [Machine-readable comparison](/home/davidx/Downloads/embedding-rl/benchmarks/ambig_interpretations/runs/ambig-20261106-v1/comparison.json)
* [Checkpoint and prediction audit](/home/davidx/Downloads/embedding-rl/benchmarks/ambig_interpretations/runs/ambig-20261106-v1/analysis.json)
* [Training and inference instructions](/home/davidx/Downloads/embedding-rl/benchmarks/ambig_interpretations/README.md)
* [Modal run](https://modal.com/apps/davidxionghl/main/ap-dVggIqb9rogAm3TjqCPOnp)

Initial, best and final weights and all training logs remain in Modal volume
`ambig-interpretation-distillation/ambig-20261106-v1`. Best weights, test predictions,
metrics, source snapshot, teacher metadata and checksums are also saved locally.
