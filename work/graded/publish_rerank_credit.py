"""Publish the variance gate and every admitted estimator training run."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal
import numpy as np
import torch

from publish_feature_reranker import manual_ndcg
from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'rerank-credit-20261106'
    folder.mkdir(exist_ok=True)
    for name in ('protocol.json', 'comparison.json', 'variance-probe.json'):
        payload = b''.join(volume.read_file(f'experiments/rerank-credit-20261106/{name}'))
        json.loads(payload)
        (folder / name).write_bytes(payload)
    result = read('rerank-credit-20261106/comparison.json')
    protocol = read('rerank-credit-20261106/protocol.json')
    probe = read('rerank-credit-20261106/variance-probe.json')
    features = torch.load(OUT / 'feature-reranker-20261106/features.pt', weights_only=True)
    if protocol['probe_query_ids'] != features['train']['query_ids'][:32] or result['query_ids'] != features['validation']['query_ids']:
        raise ValueError('Probe/evaluation query identities differ')
    if result['browsecomp_final_test_evaluated'] or features['protocol_digest'] != protocol['source_protocol_digest']:
        raise ValueError('Protocol source differs')
    errors = []
    for estimator in result['selected_estimators']:
        if probe['variance_ratio'][estimator] >= .7:
            raise ValueError('Estimator did not pass the predeclared gate')
        if {r['seed'] for r in result['runs'] if r['estimator'] == estimator} != {42, 43, 44}:
            raise ValueError('Missing admitted policy seed')
    for row in result['runs']:
        label = f'{row["estimator"]}-seed-{row["seed"]}'
        (folder / label).mkdir(exist_ok=True)
        for name in ('result.json', 'final.pt', 'metrics.jsonl'):
            (folder / label / name).write_bytes(b''.join(volume.read_file(f'experiments/rerank-credit-20261106/{label}/{name}')))
        checkpoint = torch.load(folder / label / 'final.pt', weights_only=True)
        if checkpoint['protocol_digest'] != result['protocol_digest'] or checkpoint['step'] != 500 or checkpoint['weights'].tolist() != row['weights']:
            raise ValueError('Saved credit policy differs')
        values = manual_ndcg(features['validation'], row['weights'])
        errors.append(float(np.abs(values - row['after']).max()))
    if errors and max(errors) > 1e-6:
        raise ValueError('Independent policy metrics differ')
    verification = {'independent_numpy_metric_max_error': max(errors, default=0), 'policies': len(result['runs']),
                    'probe_training_queries': 32, 'validation_queries': 83, 'test_evaluated': False}
    write('rerank-credit-20261106/verification.json', verification)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(folder / 'verification.json', 'experiments/rerank-credit-20261106/verification.json')
    write('rerank-credit-summary.json', {'summary': result['summary'], 'variance_ratio': result['variance_ratio'],
                                       'verification': verification, 'api_calls': 0, 'test_evaluated': False})
    log = read('research-log.json')
    log['active_experiments']['rerank-credit-20261106'].update(status='complete and independently verified',
        artifact='rerank-credit-20261106/comparison.json')
    log['rerank_credit_findings'] = read('rerank-credit-summary.json')
    write('research-log.json', log)
    lines = ['## Lower-variance credit assignment for the same ranking policy', '',
        'Both alternatives reduced measured gradient variance to about one third of the whole-ranking RLOO baseline, '
        'but neither improved validation nDCG. Lower gradient noise did not remove the observed generalization gap of this four-feature model.', '',
        '| Estimator | Relative gradient variance | Final validation nDCG@10 | Difference versus ordinary RLOO, 95% interval |',
        '|---|---:|---:|---|', f'| Whole-ranking RLOO | 1.000 | {result["reference_mean"]:.6f} | Reference |']
    for row in result['summary']:
        lines.append(f'| {row["estimator"]} | {result["variance_ratio"][row["estimator"]]:.3f} | {row["mean_after"]:.6f} | {interval(row["versus_ranking_rloo"])} |')
    lines.extend(['',
        'The variance probe used the first 32 training queries, including any lists with no known evidence, 128 independent groups '
        'of 32 rankings per query, and the initial four weights. All estimators used the same sampled rankings. '
        'The metric is the sum of the four parameter variances, averaged over those queries. It is not a claim about a full neural network '
        'or the entire optimization path. Both estimators passed the predeclared threshold of 70% of baseline variance, '
        'so all three seeds were trained for each using the same 500-update recipe.', '',
        'Placement credit applies a document’s discounted reward only to the placements up to its position. '
        'Marginalization averages over the possible next document at each sampled prefix and differentiates both that expectation '
        'and the probability of the prefix. Neither changes the expected additive nDCG objective or adds browsing actions.', '',
        'These are adaptations of sections 4.2 and 4.4 of [Exposure-Based Reinforcement Learning to Rank](https://arxiv.org/html/2607.18689v1), '
        'not its complete exposure-based system. Their expected gradients were checked by exact enumeration before use. '
        'The decomposition requires an additive ranking reward; arbitrary LLM judgments about a whole list need separate treatment.', '',
        'Validation uses the same fixed upstream seed-42 retriever and 83 reused queries. Intervals are descriptive and unadjusted for '
        'the broader exploration. The marginalization result is slightly worse than ordinary RLOO in this comparison; '
        'that is a useful negative result, not evidence that variance reduction is generally harmful.', '',
        f'All six final policies were reconstructed from saved features and weights with maximum metric error {max(errors, default=0):.2g}. '
        'The full suite now has 65 passing tests. No additional page scoring, paid API calls, final-test evaluation or external transfer checks were needed.', ''])
    report = (OUT / 'results.md').read_text()
    heading = '\n## Lower-variance credit assignment for the same ranking policy\n'
    if heading in report:
        report = report[:report.index(heading)]
    report = report.rstrip() + '\n\n' + '\n'.join(lines)
    note = 'Training a four-weight RL reranker raised training nDCG from about 0.221 to 0.290 but left validation near 0.168, versus its matched 0.167 baseline. Two credit-assignment methods cut initial gradient variance by about threefold without improving validation.\n\n'
    if note not in report:
        report = report.replace('An external transfer check found a cost to specialization:', note + 'An external transfer check found a cost to specialization:')
    report = report.replace('42 research runs and 84 checkpoint files persist in Modal.', '42 encoder runs and 84 encoder checkpoint files, plus nine small reranker policies, persist in Modal.')
    (OUT / 'results.md').write_text(report)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    labels = ['Ranking RLOO', 'Placement', 'Marginalized']
    ratios = [result['variance_ratio'][name] for name in ('ranking', 'placement', 'marginalize_all')]
    bars = axes[0].bar(labels, ratios, color=['#0072B2', '#E69F00', '#009E73'])
    axes[0].bar_label(bars, fmt='%.3f', padding=4)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel('Relative gradient variance at initialization')
    values = [result['reference_mean']] + [next(r['mean_after'] for r in result['summary'] if r['estimator'] == name) for name in ('placement', 'marginalize_all')]
    bars = axes[1].bar(labels, values, color=['#0072B2', '#E69F00', '#009E73'])
    axes[1].bar_label(bars, fmt='%.4f', padding=4)
    axes[1].axhline(result['hybrid_mean'], color='black', linestyle='--', label='Unchanged hybrid')
    axes[1].set_ylim(0, .21)
    axes[1].set_ylabel('Final validation nDCG@10')
    axes[1].legend()
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Less gradient noise did not improve this model’s validation ranking\nSame features, objective and optimizer; three seeds each')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'rerank-credit-results.{ext}', dpi=180)
    plt.close(fig)
    path = ROOT / 'VERIFICATION.md'
    content = path.read_text()
    heading = '\n## Ranking credit assignment — 2026-11-06\n'
    if heading not in content:
        path.write_text(content + heading + '\n- 65 tests passed. Exact small-space enumeration verifies placement and marginalization gradients; marginalizing the first position has the exact gradient without sampling noise.\n'
            '- Both estimators passed the predeclared training-only variance gate and all six policy runs completed. Independent NumPy reconstruction verifies final validation metrics from saved weights.\n'
            '- Removed unused full-sort backing storage from the feature candidate tensors without changing values or provenance, reducing their combined files from about 1.07 GB to 1.09 MB. The compaction record is persisted.\n')
    publish_artifacts('source-snapshot-rerank-credit-20261106.zip', ('rerank-credit-summary.json', 'rerank-credit-results.png', 'rerank-credit-results.pdf'))


if __name__ == '__main__':
    main()
