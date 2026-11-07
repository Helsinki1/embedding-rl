# Shared training and evaluation protocol

The user's November 6 Google Doc specification supersedes the earlier 5,000-query
QUEST/NQ protocol for new comparisons. The canonical package is now
`quest_or_spec_v2/`, containing 10,000 queries: 8,500 training, 500 validation,
1,000 test; 519 QUEST, 4,740 MS MARCO, and 4,741 NQ in total.

Use `quest_or_spec_v2.data.load_splits`, its frozen manifest, and `protocol.json`
for every architecture/objective. Do not make method-specific samples. Record
manifest, protocol, code, model, and corpus provenance. All learned approaches
use rank-32 LoRA, one shared epoch and batch order, frozen document vectors,
and either collective multi-positive InfoNCE or union-F1 REINFORCE plus KL.

Report union precision/recall/F1 and per-component precision/recall/F1. Per the
user's explicit choice, score each method's final merged result set against
each gold component. Never match output slots to labels for the primary metrics.
Also retain per-query predictions, source/arity breakdowns, actual training-source
counts, fixed-budget diagnostics, and component coverage@100. Do not fabricate
positive labels for unjudged components.

Select checkpoints and retrieval cutoffs only on validation; freeze those choices
before evaluating test. For trained rows select among updated checkpoints and
report initialization validation separately. Start each approach from fresh base weights unless the
user explicitly asks for another initialization and its test exposure is audited.
Use the full existing corpus plus the pinned MS MARCO candidate extension for
every approach. Keep every training run within the user's 30-minute
wall-clock budget, including final evaluation, with a launcher-enforced limit.

Do not modify historical datasets, completed results, or running job snapshots to
apply this policy retroactively. `quest_or_balanced/` remains the historical
5,000-query benchmark. The current version was explicitly authorized by the
user's Google Doc migration request; any further split/protocol change requires
another explicit user change. Do not edit that Google Doc.
