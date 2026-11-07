"""Independently verify and publish the small RL feature reranker."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal
import numpy as np
import torch

from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write


def manual_ndcg(subset, weights):
    features, gains, ideal = (subset[k].numpy() for k in ('features', 'gains', 'ideal'))
    order = np.argsort(-(features @ np.asarray(weights, dtype=np.float32)), axis=1, kind='stable')[:, :10]
    dcg = (np.take_along_axis(gains, order, axis=1) / np.log2(np.arange(10) + 2)).sum(1)
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'feature-reranker-20261106'
    folder.mkdir(exist_ok=True)
    names = ['protocol.json', 'comparison.json', 'data-summary.json', 'compaction-verification.json', 'features.pt', 'candidates.pt', 'bge-scores.json']
    names += [f'seed-{seed}/{name}' for seed in (42, 43, 44) for name in ('result.json', 'final.pt', 'metrics.jsonl')]
    for name in names:
        data = b''.join(volume.read_file(f'experiments/feature-reranker-20261106/{name}'))
        target = folder / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    result = read('feature-reranker-20261106/comparison.json')
    protocol = read('feature-reranker-20261106/protocol.json')
    if protocol['steps'] != 500 or protocol['samples'] != 32 or protocol['retriever_seed'] != 42 or not protocol['final_checkpoint_only']:
        raise ValueError('Policy protocol differs from the predeclared experiment')
    prepared = read('feature-reranker-20261106/data-summary.json')
    data = torch.load(folder / 'features.pt', weights_only=True)
    if {r['seed'] for r in result['runs']} != {42, 43, 44} or result['browsecomp_final_test_evaluated']:
        raise ValueError('Unexpected policy experiment scope')
    if len(data['train']['query_ids']) != 581 or len(data['validation']['query_ids']) != 83 or set(data['train']['query_ids']) & set(data['validation']['query_ids']):
        raise ValueError('Policy training split differs')
    if data['validation']['query_ids'] != result['query_ids'] or data['protocol_digest'] != result['protocol_digest']:
        raise ValueError('Saved policy provenance differs')
    before = manual_ndcg(data['validation'], [1, 0, 0, 0])
    frozen = manual_ndcg(data['validation'], [0, 0, 0, 1])
    errors = [abs(before.mean() - result['mean_before']), abs(frozen.mean() - result['frozen_bge_mean'])]
    for row in result['runs']:
        checkpoint = torch.load(folder / f'seed-{row["seed"]}/final.pt', weights_only=True)
        if checkpoint['step'] != 500 or checkpoint['protocol_digest'] != result['protocol_digest']:
            raise ValueError('Final policy checkpoint differs')
        weights = checkpoint['weights'].tolist()
        if weights != row['weights']:
            raise ValueError('Saved policy weights differ')
        values = manual_ndcg(data['validation'], weights)
        errors.extend([np.max(np.abs(values - row['after'])), abs(manual_ndcg(data['train'], weights).mean() - row['train_after'])])
    if max(errors) > 1e-6:
        raise ValueError('Independent NumPy metric reconstruction differs')
    raw = data['raw_features'].numpy()
    standardized = (raw - raw.mean(1, keepdims=True)) / np.maximum(raw.std(1, keepdims=True), 1e-6)
    normalization_error = float(np.abs(standardized - torch.cat([data['train']['features'], data['validation']['features']]).numpy()).max())
    if normalization_error > 2e-5:
        raise ValueError('Feature normalization differs')
    verification = {'independent_numpy_metric_max_error': float(max(errors)), 'normalization_max_error': normalization_error,
                    'policy_checkpoints': 3, 'train_queries': 581, 'validation_queries': 83, 'api_calls': 0, 'test_evaluated': False}
    write('feature-reranker-20261106/verification.json', verification)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(folder / 'verification.json', 'experiments/feature-reranker-20261106/verification.json')
    write('feature-reranker-summary.json', {k: v for k, v in result.items() if k not in ('query_ids', 'runs')} | {'verification': verification,
        'weights_by_seed': {r['seed']: r['weights'] for r in result['runs']}, 'data': prepared})
    log = read('research-log.json')
    log['active_experiments']['feature-reranker-20261106'].update(status='complete and independently verified',
        artifact='feature-reranker-20261106/comparison.json')
    log['feature_reranker_findings'] = read('feature-reranker-summary.json')
    write('research-log.json', log)
    lines = ['## One-step RL for a four-feature reranker', '',
        'This policy learns four weights over the normalized hybrid score, dense cosine, log-BM25 and frozen BGE score. '
        'It samples an ordered top-10 list from 25 natural candidates without replacement, gets benchmark nDCG as reward, '
        'and updates with an unscaled leave-one-out policy gradient. Both pretrained models stay frozen. This is RL on ranking actions, '
        'not contrastive training or full cross-encoder fine-tuning.', '',
        f'The matched upstream retriever is fixed at seed 42, so its baseline is {result["mean_before"]:.6f}. '
        'The three repeats below vary only the small policy’s training seed; they are distinct from the earlier three-encoder averages.', '',
        '| System | Validation nDCG@10 |', '|---|---:|',
        f'| Unchanged prefix hybrid | {result["mean_before"]:.6f} |',
        f'| Frozen BGE alone on the same candidates | {result["frozen_bge_mean"]:.6f} |',
        f'| RL feature policy, mean of three policy seeds | {result["mean_after"]:.6f} |', '',
        f'RL minus unchanged hybrid: **{interval(result["paired_query_bootstrap"])}**. '
        'This is a descriptive paired query interval on the reused 83-query validation set.', '',
        '| Policy seed | Train before | Train after | Validation after | Weights: hybrid / dense / keyword / BGE |',
        '|---|---:|---:|---:|---|']
    for row in result['runs']:
        weights = ' / '.join(f'{w:.3f}' for w in row['weights'])
        lines.append(f'| {row["seed"]} | {row["train_before"]:.6f} | {row["train_after"]:.6f} | {row["mean_after"]:.6f} | {weights} |')
    lines.extend(['',
        'All policies start from the original hybrid ranking and run exactly 500 Adam updates, using 16 training queries and 32 sampled '
        'rankings per query per update. The learning rate is 0.01. There is no intermediate validation selection, KL penalty, '
        'reward standardization or hyperparameter sweep. Each feature is standardized separately within the same query’s candidate list. '
        'Correlated feature weights should not be read as independent causal importance.', '',
        f'Natural candidates contain known evidence for {prepared["train_with_evidence"]}/581 training queries and '
        f'{prepared["validation_with_evidence"]}/83 validation queries. No evidence was injected. '
        'The reward uses the whole-corpus ideal denominator, so reranking cannot earn credit for finding missing pages. '
        'Unjudged pages count as zero for this benchmark metric; they are not verified negatives.', '',
        '[Neural PG-RANK](https://arxiv.org/html/2310.04407) motivates direct metric optimization with ranking policies. '
        '[Exposure-based RL for ranking](https://arxiv.org/html/2607.18689v1) motivates careful variance reduction. '
        'Our estimator was checked against exact enumeration on a small ranking problem; this experiment does not reproduce either full system.', '',
        f'Independent NumPy reconstruction from saved features and final weights matched metrics within {max(errors):.2g}. '
        'All feature data, 16,600 frozen scores and three policy checkpoints persist under `experiments/feature-reranker-20261106`. '
        'The 14,525 training scores are reusable; 2,075 validation scores were reused. Keep these splits separate in future training.', '',
        'No paid API calls, upstream encoder updates, BrowseComp final-test evaluations or external-transfer evaluations were made.', '',
        'W&B runs: ' + ', '.join(f'[policy seed {r["seed"]}]({r["wandb_url"]})' for r in result['runs'] if r.get('wandb_url')), ''])
    report = (OUT / 'results.md').read_text()
    heading = '\n## One-step RL for a four-feature reranker\n'
    if heading in report:
        report = report[:report.index(heading)]
    (OUT / 'results.md').write_text(report.rstrip() + '\n\n' + '\n'.join(lines))
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    means = [result['mean_before'], result['frozen_bge_mean'], result['mean_after']]
    bars = ax.bar(['Unchanged hybrid', 'Frozen BGE', 'RL feature policy'], means, color=['#0072B2', '#999999', '#E69F00'])
    ax.bar_label(bars, fmt='%.3f', padding=5)
    ax.scatter([2] * 3, [r['mean_after'] for r in result['runs']], color='black', s=20, zorder=3, label='Three policy seeds')
    ax.set_ylim(0, max(means) * 1.25)
    ax.set_ylabel('Validation nDCG@10')
    ax.set_title('One fixed upstream retriever; same natural top-25 candidates\n83 reused validation queries; final checkpoints only')
    ax.legend(loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'feature-reranker-results.{ext}', dpi=180)
    plt.close(fig)
    path = ROOT / 'VERIFICATION.md'
    content = path.read_text()
    line = f'\n- The completed feature policies were independently reconstructed in NumPy from saved features and weights; maximum metric error {max(errors):.2g}. All three final checkpoints and split IDs match.\n'
    if line not in content:
        path.write_text(content + line)
    publish_artifacts('source-snapshot-feature-reranker-20261106.zip', ('feature-reranker-summary.json', 'feature-reranker-results.png', 'feature-reranker-results.pdf'))


if __name__ == '__main__':
    main()
