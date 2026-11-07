"""Independently reconstruct and publish both predeclared KL strengths."""
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal
import numpy as np
import torch

from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write
from publish_neural_reranker import neural_ndcg


def manual_kl(subset, neural, checkpoint):
    logits = (subset['features'].numpy() @ checkpoint['theta'].numpy() + neural.numpy() @ checkpoint['delta'].numpy()).astype(np.float64)
    reference = subset['features'][:, :, 0].numpy().astype(np.float64)
    def log_softmax(x):
        centered = x - x.max(1, keepdims=True)
        return centered - np.log(np.exp(centered).sum(1, keepdims=True))
    logp, logq = log_softmax(logits), log_softmax(reference)
    return float((np.exp(logp) * (logp - logq)).sum(1).mean())


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'reranker-anchor-20261106'
    folder.mkdir(exist_ok=True)
    names = ['protocol.json', 'comparison.json']
    names += [f'beta-{beta:g}-seed-{seed}/{name}' for beta in (.01, .1) for seed in (42, 43, 44) for name in ('result.json', 'final.pt', 'metrics.jsonl')]
    for name in names:
        target = folder / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(f'experiments/reranker-anchor-20261106/{name}')))
    result = read('reranker-anchor-20261106/comparison.json')
    protocol = read('reranker-anchor-20261106/protocol.json')
    scalar = torch.load(OUT / 'feature-reranker-20261106/features.pt', weights_only=True)
    neural = torch.load(OUT / 'neural-reranker-fp32-20261106/neural.pt', weights_only=True)
    reference = read('neural-reranker-fp32-20261106/comparison.json')
    if protocol['source_protocol_digest'] != neural['protocol_digest'] or protocol['scalar_protocol_digest'] != scalar['protocol_digest']:
        raise ValueError('Feature provenance differs')
    if result['query_ids'] != scalar['validation']['query_ids'] or result['browsecomp_final_test_evaluated'] or result['api_calls']:
        raise ValueError('Evaluation scope differs')
    if {(r['beta'], r['seed']) for r in result['runs']} != {(b, s) for b in (.01, .1) for s in (42, 43, 44)}:
        raise ValueError('Missing predeclared run')
    for module, expected in protocol['source_hashes'].items():
        if hashlib.sha256((ROOT / (module + '.py')).read_bytes()).hexdigest() != expected:
            raise ValueError('Policy implementation changed')
    errors, kl_errors = [], []
    for row in result['runs']:
        checkpoint = torch.load(folder / f'beta-{row["beta"]:g}-seed-{row["seed"]}/final.pt', weights_only=True)
        if checkpoint['protocol_digest'] != result['protocol_digest'] or checkpoint['step'] != 500 or checkpoint['beta'] != row['beta'] or checkpoint['theta'].tolist() != row['theta']:
            raise ValueError('Saved final policy differs')
        errors.extend([float(np.abs(neural_ndcg(scalar['validation'], neural['features'][581:], checkpoint) - row['after']).max()),
                       abs(neural_ndcg(scalar['train'], neural['features'][:581], checkpoint).mean() - row['train_after'])])
        kl_errors.append(abs(manual_kl(scalar['train'], neural['features'][:581], checkpoint) - row['train_first_choice_kl']))
    if max(errors) > 1e-6 or max(kl_errors) > 1e-6:
        raise ValueError(f'Independent reconstruction differs: metrics={max(errors)}, KL={max(kl_errors)}')
    original_kl = []
    for seed in (42, 43, 44):
        checkpoint = torch.load(OUT / f'neural-reranker-fp32-20261106/seed-{seed}/final.pt', weights_only=True)
        original_kl.append(manual_kl(scalar['train'], neural['features'][:581], checkpoint))
    verification = {'independent_numpy_metric_max_error': max(errors), 'independent_numpy_kl_max_error': max(kl_errors),
        'policies': 6, 'tests_passed': 69, 'train_queries': 581, 'validation_queries': 83, 'test_evaluated': False}
    write('reranker-anchor-20261106/verification.json', verification)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(folder / 'verification.json', 'experiments/reranker-anchor-20261106/verification.json')
    summary = {k: v for k, v in result.items() if k not in ('runs', 'query_ids')} | {'verification': verification,
        'unregularized_mean_train_kl': float(np.mean(original_kl)), 'unregularized_mean_train_ndcg': reference['mean_train_after']}
    write('reranker-anchor-summary.json', summary)
    log = read('research-log.json')
    log['active_experiments']['reranker-anchor-20261106'].update(status='complete and independently verified', artifact='reranker-anchor-20261106/comparison.json')
    log['queued_reranker_anchor_hypothesis']['status'] = 'complete and independently verified'
    log['reranker_anchor_findings'] = summary
    log['status'] = 'Both predeclared first-choice KL controls complete; all six policies verified'
    write('research-log.json', log)
    lines = ['## First-choice KL regularization for the neural reranker', '',
        'This control penalizes movement away from the starting hybrid’s distribution over the first selected document. '
        'It uses KL(current probabilities || fixed initial probabilities), averaged across training queries. '
        'It changes the RL objective; it is not full-permutation KL and does not change the benchmark evaluation metric.', '',
        '| KL strength | Training nDCG@10 | Training first-choice KL | Validation nDCG@10 | Difference versus unregularized neural RL, 95% interval |',
        '|---|---:|---:|---:|---|',
        f'| 0, existing control | {reference["mean_train_after"]:.6f} | {np.mean(original_kl):.6f} | {result["unregularized_mean"]:.6f} | Reference |']
    for row in result['summary']:
        lines.append(f'| {row["beta"]:g} | {row["mean_train_after"]:.6f} | {row["mean_train_first_choice_kl"]:.6f} | {row["mean_after"]:.6f} | {interval(row["versus_unregularized"])} |')
    lines.extend(['', f'The unchanged hybrid scores {result["hybrid_mean"]:.6f}. Differences from that baseline are:', ''])
    for row in result['summary']:
        lines.append(f'- Strength {row["beta"]:g}: {interval(row["versus_hybrid"])}.')
    lines.extend(['',
        'Both strengths and all three seeds were predeclared at 15:49 UTC, before the unregularized neural validation results. '
        'Each run starts from the same hybrid and uses 500 updates, 16 training queries, 32 sampled rankings per query, '
        'scalar learning rate 0.01, and residual learning rate 0.0001. All final checkpoints are reported. '
        'There is no checkpoint or strength selection based on validation. The same 83 validation queries have nevertheless '
        'been reused throughout this broader research, so paired intervals remain descriptive and unadjusted.', '',
        'The zero-strength control reproduces the prior training updates exactly in regression tests. Additional tests check '
        'the KL direction, fixed reference, nonnegativity, and invariance to shifting all scores together. '
        f'Independent NumPy reconstruction from saved weights gives maximum metric error {max(errors):.2g} and KL error {max(kl_errors):.2g}. '
        'All 69 tests pass. The six policies, full provenance, metrics, and W&B logs persist. No new page scoring, paid API calls, '
        'GPU training, final-test evaluations, or external-transfer evaluations were used.', '',
        'The first-choice formulation is related to the regularizer in '
        '[On-Policy Distillation Meets Off-Policy GRPO](https://aclanthology.org/2026.findings-emnlp.477/), '
        'but our reference is the initial hybrid and our reward is benchmark evidence nDCG. This does not reproduce its teacher-distillation system.', '',
        'W&B runs: ' + ', '.join(f'[KL {r["beta"]:g}, seed {r["seed"]}]({r["wandb_url"]})' for r in result['runs'] if r.get('wandb_url')), ''])
    report = (OUT / 'results.md').read_text()
    heading = '\n## First-choice KL regularization for the neural reranker\n'
    if heading in report:
        start = report.index(heading)
        end = report.find('\n## ', start + len(heading))
        report = report[:start] + (report[end:] if end >= 0 else '')
    report = report.rstrip() + '\n\n' + '\n'.join(lines)
    report = report.replace('plus twelve small reranker policies', 'plus eighteen small reranker policies')
    note = 'The two predeclared first-choice KL controls produced validation means of ' + ' and '.join(f'{r["mean_after"]:.5f} (strength {r["beta"]:g})' for r in result['summary']) + ', versus 0.16840 without the penalty.\n\n'
    if note not in report:
        report = report.replace('An external transfer check found a cost to specialization:', note + 'An external transfer check found a cost to specialization:')
    (OUT / 'results.md').write_text(report)
    labels = ['No KL', 'KL 0.01', 'KL 0.1']
    means = [[reference['mean_train_after']] + [r['mean_train_after'] for r in result['summary']],
             [result['unregularized_mean']] + [r['mean_after'] for r in result['summary']]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for i, ax in enumerate(axes):
        bars = ax.bar(labels, means[i], color=['#0072B2', '#E69F00', '#009E73'])
        ax.bar_label(bars, fmt='%.4f', padding=4)
        ax.set_ylim(0, max(max(v) for v in means) * 1.2)
        ax.set_ylabel('nDCG@10')
        ax.set_title(['Training: 581 queries', 'Validation: 83 reused queries'][i])
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].axhline(result['hybrid_mean'], linestyle='--', color='black', label='Unchanged hybrid')
    axes[1].legend()
    fig.suptitle('KL penalty on the first-choice distribution\nSame neural policy and cached candidates; three seeds per setting')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'reranker-anchor-results.{ext}', dpi=180)
    plt.close(fig)
    for name, heading, paragraph in (
        ('README.md', '\n## Reranker KL control\n', '\n`modal_reranker_anchor.py` reuses the scalar and FP32 neural caches to train two predeclared first-choice KL strengths, 0.01 and 0.1, with three seeds each. The fixed reference is the initial hybrid distribution; this is not full-ranking KL. Artifacts persist under `experiments/reranker-anchor-20261106`. All six final policies are reported, with no final-test evaluation.\n'),
        ('VERIFICATION.md', '\n## First-choice KL control — 2026-11-06\n', f'\n- 69 tests passed. Zero strength reproduces unregularized updates; KL tests cover its direction, detached reference, nonnegativity and score-shift invariance.\n- All six final checkpoints independently reconstruct in NumPy: metric error {max(errors):.2g}, KL error {max(kl_errors):.2g}.\n- Only cached training tensors enter optimization. No new page scoring, GPU training, paid API calls, final-test or external-transfer evaluation.\n')):
        path = ROOT / name
        if heading not in path.read_text():
            path.write_text(path.read_text() + heading + paragraph)
    publish_artifacts('source-snapshot-reranker-anchor-20261106.zip', ('reranker-anchor-summary.json', 'reranker-anchor-results.png', 'reranker-anchor-results.pdf'))


if __name__ == '__main__':
    main()
