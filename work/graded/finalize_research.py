"""Close the research batch without changing experiments or consuming API calls."""
from datetime import datetime, timezone
import hashlib

import modal
import torch

from publish_mixed_results import OUT, ROOT, publish_artifacts, read, write


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    checkpoints = sorted(OUT.rglob('final.pt'))
    assert len(checkpoints) == 36
    registry = []
    for path in checkpoints:
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint['step'] == 500
        registry.append({'path': str(path.relative_to(OUT)),
            'volume_path': 'experiments/' + str(path.relative_to(OUT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'protocol_digest': checkpoint['protocol_digest'], 'step': checkpoint['step']})
    write('reranker-checkpoints.json', registry)
    compaction = read('reranker-conditioning-20261106/compaction-verification.json')
    assert compaction['values_and_metadata_unchanged']
    log = read('research-log.json')
    log['status'] = 'Complete: all planned training, independent metric checks, and persistence checks finished'
    log['finalized_utc'] = datetime.now(timezone.utc).isoformat()
    log['next_steps'] = [
        'No further experiments are queued. Pause the overnight heartbeat after final publication.',
        'Future work only: compare feature rescaling against a residual-learning-rate sweep using training/inner-validation, then replicate across upstream retriever seeds or new data.',
        'Freeze one final recipe and evaluation protocol before ever opening the 166-query BrowseComp-Plus final test. Do not tune or reevaluate the already-checked external test sets.',
        'Reuse saved candidates, representations, labels, policies and source archives. The shared API ledger remains authoritative.']
    log['storage_compaction'] = compaction
    log['completed_artifacts'] = {'encoder_runs': 42, 'encoder_checkpoint_files': 84,
        'small_reranker_policies': 36, 'small_policy_registry': 'reranker-checkpoints.json',
        'latest_source': 'source-snapshot-final-20261106.zip', 'final_test_evaluated': False}
    log['final_literature'] = [
        {'title': 'KL-Regularized Reinforcement Learning for Generative Modelling is Designed to Mode Collapse',
         'url': 'https://proceedings.iclr.cc/paper_files/paper/2026/hash/5306f6ce1082d29e4968acdacab84fd7-Abstract-Conference.html',
         'lesson': 'KL regularization does not by itself guarantee useful diversity. Its target depends on reward, reference and penalty. Our first-choice marginal penalty is not the full-action objective analyzed there; no MARA implementation or reproduction claim.'},
        {'title': 'Rethinking Reasoning in Document Ranking: Why Chain-of-Thought Falls Short',
         'url': 'https://arxiv.org/html/2510.08985v1',
         'lesson': 'Direct ranking outperformed explicit reasoning variants in their controlled BRIGHT/BEIR comparisons. Extra reasoning is not automatically the next priority. This does not test our model or rule out acquiring new evidence by browsing.'},
        {'title': 'Structured Preconditioners in Adaptive Optimization: A Unified Analysis',
         'url': 'https://proceedings.mlr.press/v267/xie25j.html',
         'lesson': 'Optimization geometry merits its own control. Our fixed least-squares/RMS transform is not Shampoo, and its empirical gain could partly reflect changed effective learning rates.'}]
    write('research-log.json', log)
    overview = '''## Current findings

The clearest encoder gains came from allowing meaningful updates: relaxing the original strong anchor, increasing the learning rate, then using more/wider samples and longer training. The latest reranker result points mainly to **feature rescaling**, which changes how easily its existing features can be learned.

| Retrieval system | Validation evidence nDCG@10 |
|---|---:|
| Original encoder | 0.037364 |
| Original ten-step pilot, rescored on all 83 validation queries | 0.037428 |
| Full-corpus binary-reward RL, 2,000 updates | 0.138436 |
| Mixed full-corpus/candidate-pool RL, 2,000 updates | 0.145207 |
| Full-corpus RL + prefix BM25 fusion | 0.170628 |

The full-corpus, mixed-training and fusion rows average three upstream encoder seeds; the original pilot is one run. The mixed-training advantage over full-corpus RL is inconclusive. BM25 fusion is an inference change, not an additional RL training gain.

The following **separate, matched reranker comparison** fixes the upstream encoder to seed 42 and its natural top-25 prefix-hybrid results. Each learned row averages three policy seeds, all trained for 500 updates. The KL penalty covers only the first-choice distribution.

| Reranker on this fixed candidate set | No KL | KL strength 0.1 |
|---|---:|---:|
| Unchanged hybrid | 0.167248 | 0.167248 |
| Neural features, original coordinates | 0.168399 | 0.172815 |
| Remove scalar-predictable components only | 0.168889 | 0.173926 |
| Rescale features only | 0.173951 | 0.178116 |
| Remove components and rescale | 0.177322 | 0.180074 |

Rescaling explains most of the observed improvement. Removal alone and the extra benefit of removal on top of rescaling remain inconclusive. The transform adds no information or model capacity: it changes optimization coordinates and effective step sizes. A simple learning-rate sweep is still a missing control. Do not compare 0.180074 directly with the earlier 0.170628 mean as if the upstream seeds were matched.

**What did not establish a gain:** LLM grades provided more varied rewards but did not beat matched binary supervision; fixed-pool encoder training overfit its candidate lists; lower-variance ranking gradients and richer raw neural features alone did not improve validation convincingly. Frozen reranking failed to improve the stronger hybrid lists, and its 2,048-token variant was worse. These are results for the tested recipes, not general impossibility claims.

An external check found a specialization cost: SciFact fell from 0.71271 to 0.68648 after full-corpus RL. NFCorpus changes were inconclusive. Those external test sets were checked once and were not used for subsequent tuning.

The current BrowseComp comparisons reuse **83 validation queries** after training on 581; **all 166 final-test queries remain untouched**. Adaptive comparisons and unadjusted query-bootstrap intervals are exploratory, even when positive. Replicated policy seeds do not measure uncertainty over different upstream retrievers or fresh datasets.

All planned training is complete. **42 encoder runs / 84 encoder checkpoints and 36 small reranker policies** persist in Modal. The conservative OpenAI API estimate remains **$90.60**; the final KL, feature-conditioning and component-ablation batch used no paid API calls. Modal compute is separate. Details, failures, sources and per-seed results follow below.
'''
    report_path = OUT / 'results.md'
    report = report_path.read_text()
    start, end = '<!-- current-findings -->', '<!-- /current-findings -->'
    before, rest = report.split(start, 1)
    _, after = rest.split(end, 1)
    report = before + start + '\n' + overview + end + after
    final_marker = '## Final literature lessons and reuse handoff'
    report = report.split(final_marker)[0].rstrip()
    report += '''

## Final literature lessons and reuse handoff

[KL-Regularized Reinforcement Learning for Generative Modelling is Designed to Mode Collapse](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5306f6ce1082d29e4968acdacab84fd7-Abstract-Conference.html) cautions that KL does not automatically preserve useful diversity: reward, reference and penalty shape the optimum together. Our penalty covers the first-selection marginal of a ranking policy, so this is a conceptual lesson rather than a reproduction of its full-action analysis or MARA method.

[Rethinking Reasoning in Document Ranking](https://arxiv.org/html/2510.08985v1) found direct ranking stronger than explicit reasoning variants in its controlled BRIGHT/BEIR study. That supports testing optimization and scoring before assuming a longer reasoning process will help. Reading additional pages could supply new evidence; this paper does not show that such information acquisition is useless.

[Structured Preconditioners in Adaptive Optimization](https://proceedings.mlr.press/v267/xie25j.html) studies how optimization geometry affects learning. Our fixed feature transformation is not its algorithm. Here rescaling helped under fixed nominal learning rates; the next clean control is a learning-rate sweep inside the training split, followed by replication across retriever seeds or new data. This is a future handoff, not another queued experiment.

Reusable artifacts live in Modal volume `embedding-rl-data`, under `experiments/`. The local copy is the directory containing this report. Preserve natural candidate membership and the saved query IDs when reusing these files:

- `feature-reranker-20261106/features.pt` and `candidates.pt`: four normalized features, evidence gains, global ideal denominators, and natural top-25 IDs for the fixed 581/83 train/validation split.
- `neural-reranker-fp32-20261106/neural.pt`: all 16,600 frozen 1,024-dimensional representations; pinned BGE revision and 512-token input protocol are saved alongside it.
- `reranker-conditioning-20261106/transform.pt`: training-only projection and residual RMS scales. These must remain fixed when applying the policy to new queries.
- `reranker-conditioning-20261106/beta-0.1-seed-{42,43,44}/final.pt`: the three combined-treatment checkpoints. They contain both transformed-coordinate weights and equivalent `original_theta` / `original_delta` weights. On the same four normalized scalar features F and raw neural features H, the latter score candidates as `F @ original_theta + H @ original_delta`. Score offsets shared by a query do not affect ranking. This is a reranker over the given candidates, not a retriever that finds missing pages.
- `reranker-checkpoints.json` indexes all 36 small policies with hashes, paths and protocol digests; `research-checkpoints.json` retains the 42 encoder runs. Follow the saved protocol for each policy rather than mixing feature versions.
- Existing judged-pair caches include both training and validation IDs. Filter to the 581 training IDs before training any new model; judge agreement is not a guarantee that labels are correct.
- `source-snapshot-final-20261106.zip` captures the final implementation without credentials. Earlier named snapshots preserve the exact producer source for prior runs. `artifact-index.json` records local artifact hashes.

A storage-only fix cloned the least-squares solution, removing unused backing storage from `transform.pt`: 119,014,387 bytes became 58,463 bytes. Tensor hashes, values, dtypes and provenance were checked before and after, including against the previous local file. No weights or rankings changed. The original producer source remains in `source-snapshot-reranker-conditioning-20261106.zip`; the final source adds the clone for future fits.

The 73-test suite passed before final storage cleanup; both conditioning tests passed again after the storage assertion was added. Independent NumPy reconstruction verifies the final component policies within 6.4e-08 nDCG; mapping them to original coordinates changes the metrics by zero. All training and compaction apps have stopped. No additional final-test or external-transfer evaluation was performed. Freeze the chosen recipe and evaluation protocol before any future final-test run.
'''
    report_path.write_text(report)
    verification_path = ROOT / 'VERIFICATION.md'
    marker = '## Final storage cleanup and closeout — 2026-11-06'
    verification = verification_path.read_text().split(marker)[0].rstrip()
    verification_path.write_text(verification + '\n\n' + marker + '\n\n'
        '- Compacted the training-only projection artifact from 119,014,387 to 58,463 bytes; tensor values, dtypes, metadata and hashes match the earlier local and remote artifact. Added a minimal-storage regression assertion; both conditioning tests pass. No policy changed.\n'
        '- All 36 small reranker checkpoints have 500 updates and a saved protocol digest. The new registry records hashes and volume paths. The 42 encoder runs and 84 encoder checkpoint records remain preserved.\n'
        '- All final training apps are stopped; no more experiments are queued. API estimate remains $90.60059215. All 166 BrowseComp final-test queries remain untouched.\n')
    with volume.batch_upload(force=True) as batch:
        batch.put_file(OUT / 'reranker-conditioning-20261106/compaction-verification.json',
            'experiments/reranker-conditioning-20261106/compaction-verification.json')
    publish_artifacts('source-snapshot-final-20261106.zip', extra_names=(
        'reranker-checkpoints.json', 'reranker-anchor-summary.json', 'reranker-anchor-results.png',
        'reranker-anchor-results.pdf', 'reranker-conditioning-summary.json',
        'reranker-conditioning-results.png', 'reranker-conditioning-results.pdf',
        'reranker-components-summary.json', 'reranker-components-results.png', 'reranker-components-results.pdf'))


if __name__ == '__main__':
    main()
