# Retrieval benchmark versions

The current specification is implemented in **[quest_or_spec_v2](quest_or_spec_v2/README.md)**.
It provides one dataset loader, trainer and evaluator for eleven approaches, with
10,000 queries from QUEST/MS MARCO/NQ and rank-32 LoRA. Use its manifest and
protocol for all new comparisons. W&B project: [quest-or-spec-v2](https://wandb.ai/yx2948-columbia-university/quest-or-spec-v2).

| Package | Role |
|---|---|
| `quest_or_spec_v2` | Current shared 8,500/500/1,000 protocol; eight trained methods and three frozen controls |
| `quest_or` | Historical original QUEST-only frozen single/LLM baselines; source labels and decomposition prompt |
| `quest_or_training` | Historical single-vector training; reusable encoder, checkpoint helpers and objectives |
| `quest_or_synthetic` | Historical NQ augmentation; pinned NQ source and frozen corpus cache |
| `quest_or_balanced` | Historical 5,000-query QUEST/NQ split and single-vector runs |
| `quest_or_llm_union` | Historical LLM union control and reusable query-only decomposition logic |
| `quest_or_split` | Historical rank-32 split-third runs; reusable coordinate splitting and objectives |
| `quest_or_multivector` | Historical rank-16 slot runs; reusable one-pass/three-pass architectures and objectives |

Historical datasets, completed results and execution snapshots are retained for
reproducibility. Their old entry points intentionally retain their old protocol;
the current runner reuses their model/objective modules with the new shared inputs.
Do not mix their scores into the new test table. No MLP expansion-head run is
listed in the supplied specification or these completed OR-retrieval experiments.
