"""Publish every component ablation with independent metric reconstruction."""
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal
import numpy as np
import torch

from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write
from publish_reranker_conditioning import metric_from_scores


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'reranker-components-20261106'
    folder.mkdir(exist_ok=True)
    names = ['protocol.json', 'comparison.json']
    names += [f'{mode}-beta-{beta:g}-seed-{seed}/{name}' for mode in ('remove_only', 'scale_only') for beta in (0., .1) for seed in (42, 43, 44) for name in ('result.json', 'final.pt', 'metrics.jsonl')]
    for name in names:
        target = folder / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(f'experiments/reranker-components-20261106/{name}')))
    result = read('reranker-components-20261106/comparison.json')
    protocol = read('reranker-components-20261106/protocol.json')
    scalar = torch.load(OUT / 'feature-reranker-20261106/features.pt', weights_only=True)
    original = torch.load(OUT / 'neural-reranker-fp32-20261106/neural.pt', weights_only=True)
    fitted = torch.load(OUT / 'reranker-conditioning-20261106/transform.pt', weights_only=True)
    combined = read('reranker-conditioning-20261106/comparison.json')
    if protocol['source_protocol_digest'] != original['protocol_digest'] or protocol['parent_protocol_digest'] != fitted['protocol_digest'] or fitted['query_ids'] != scalar['train']['query_ids']:
        raise ValueError('Input provenance differs')
    expected = {(mode, beta, seed) for mode in ('remove_only', 'scale_only') for beta in (0., .1) for seed in (42, 43, 44)}
    if {(r['mode'], r['beta'], r['seed']) for r in result['runs']} != expected or result['query_ids'] != scalar['validation']['query_ids'] or result['browsecomp_final_test_evaluated'] or result['api_calls']:
        raise ValueError('Experiment scope differs')
    for module, expected_hash in protocol['source_hashes'].items():
        if hashlib.sha256((ROOT / (module + '.py')).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Source implementation changed')
    f = torch.cat([scalar['train']['features'], scalar['validation']['features']]).numpy().astype(np.float64)
    h = original['features'].numpy().astype(np.float64)
    centered = h - h.mean(1, keepdims=True)
    errors, mapped_errors = [], []
    for mode in ('remove_only', 'scale_only'):
        transformed = (centered - f @ fitted['projection'].numpy()) if mode == 'remove_only' else (centered / fitted['scale'].numpy())
        transformed = transformed.astype(np.float32)
        for row in (r for r in result['runs'] if r['mode'] == mode):
            checkpoint = torch.load(folder / f'{mode}-beta-{row["beta"]:g}-seed-{row["seed"]}/final.pt', weights_only=True)
            if checkpoint['protocol_digest'] != result['protocol_digest'] or checkpoint['step'] != 500 or checkpoint['mode'] != mode or checkpoint['beta'] != row['beta']:
                raise ValueError('Saved checkpoint differs')
            theta, delta = checkpoint['theta'].numpy(), checkpoint['delta'].numpy()
            scores = f.astype(np.float32) @ theta + transformed @ delta
            mapped = f @ checkpoint['original_theta'].numpy() + h @ checkpoint['original_delta'].numpy()
            for split, sl in (('train', slice(None, 581)), ('validation', slice(581, None))):
                values = metric_from_scores(scalar[split], scores[sl])
                equivalent = metric_from_scores(scalar[split], mapped[sl])
                mapped_errors.append(float(np.abs(values - equivalent).max()))
                errors.append(float(np.abs(values - row['after']).max()) if split == 'validation' else abs(values.mean() - row['train_after']))
    if max(errors + mapped_errors) > 1e-6:
        raise ValueError(f'Independent reconstruction differs: {max(errors)}, {max(mapped_errors)}')
    verification = {'independent_numpy_metric_max_error': max(errors), 'mapped_original_metric_max_error': max(mapped_errors),
        'policies': 12, 'tests_passed': 73, 'train_queries': 581, 'validation_queries': 83, 'test_evaluated': False}
    write('reranker-components-20261106/verification.json', verification)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(folder / 'verification.json', 'experiments/reranker-components-20261106/verification.json')
    write('reranker-components-summary.json', {k: v for k, v in result.items() if k not in ('runs', 'query_ids')} | {'verification': verification})
    log = read('research-log.json')
    log['active_experiments']['reranker-components-20261106'].update(status='complete and independently verified', artifact='reranker-components-20261106/comparison.json')
    log['queued_reranker_components_hypothesis']['status'] = 'complete and independently verified'
    log['reranker_components_findings'] = read('reranker-components-summary.json')
    log['status'] = 'Final planned training ablation complete; all twelve policies verified'
    write('research-log.json', log)
    lines = ['## Separating component removal from feature rescaling', '',
        'The final training ablation separates the two changes in the promising conditioning result. '
        'Both variants reuse exactly the saved training-only projection and residual RMS scales. '
        'Removal-only subtracts the scalar-predictable component without rescaling. Scaling-only divides the query-centered '
        'original neural features by the same residual RMS scales used in the combined treatment, without subtracting anything. '
        'This holds the scaling factors fixed across the comparison; it is not a separate fit of ordinary feature standard deviations.', '',
        '| KL strength | Neither | Remove only | Scale only | Both |', '|---|---:|---:|---:|---:|']
    table = []
    for beta in (0., .1):
        remove = next(r for r in result['summary'] if r['beta'] == beta and r['mode'] == 'remove_only')
        scale = next(r for r in result['summary'] if r['beta'] == beta and r['mode'] == 'scale_only')
        both = next(r for r in combined['summary'] if r['beta'] == beta)
        values = [remove['matched_control_mean'], remove['mean_after'], scale['mean_after'], both['mean_after']]
        table.append(values)
        lines.append(f'| {beta:g} | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines.extend(['', '| Variant | KL | Difference versus neither, 95% interval | Difference versus both, 95% interval |', '|---|---:|---|---|'])
    for row in result['summary']:
        lines.append(f'| {row["mode"]} | {row["beta"]:g} | {interval(row["versus_matched_control"])} | {interval(row["versus_combined"])} |')
    lines.extend(['',
        'All twelve final policies are reported: two interventions, two KL strengths, and three seeds each. '
        'The training recipe remains 500 updates, 16 queries, 32 rankings, and learning rates 0.01/0.0001. '
        'The variants retain the same representable ranking family and are saved both in transformed and original coordinates. '
        'Rescaling changes effective step sizes; no learning-rate sweep was added.', '',
        'This ablation was specified after the combined result, so it is adaptive research on the same 83 reused validation queries. '
        'The intervals are descriptive and unadjusted. It uses one fixed upstream retriever and cannot establish production gains '
        'or replace a later frozen evaluation on untouched data.', '',
        f'Independent NumPy reconstruction agrees within {max(errors):.2g}; original-coordinate mapped policies differ by '
        f'at most {max(mapped_errors):.2g} in nDCG. All 73 tests pass. All twelve policies and their W&B logs persist. '
        'No additional API calls, GPU training, feature fitting, page scoring, final-test evaluation or external-transfer evaluation were used.', '',
        'W&B runs: ' + ', '.join(f'[{r["mode"]}, KL {r["beta"]:g}, seed {r["seed"]}]({r["wandb_url"]})' for r in result['runs'] if r.get('wandb_url')), ''])
    report = (OUT / 'results.md').read_text()
    heading = '\n## Separating component removal from feature rescaling\n'
    if heading in report:
        start = report.index(heading)
        end = report.find('\n## ', start + len(heading))
        report = report[:start] + (report[end:] if end >= 0 else '')
    report = report.rstrip() + '\n\n' + '\n'.join(lines)
    report = report.replace('plus twenty-four small reranker policies', 'plus thirty-six small reranker policies')
    (OUT / 'results.md').write_text(report)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    xs = np.arange(2)
    for i, (name, color) in enumerate(zip(['Neither', 'Remove only', 'Scale only', 'Both'], ['#0072B2', '#E69F00', '#009E73', '#CC79A7'], strict=True)):
        bars = ax.bar(xs + (i - 1.5) * .2, [row[i] for row in table], width=.19, label=name, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.axhline(result['hybrid_mean'], linestyle='--', color='black', linewidth=1, label='Unchanged hybrid')
    ax.set_xticks(xs, ['No KL', 'KL 0.1'])
    ax.set_ylim(0, max(max(row) for row in table) * 1.3)
    ax.set_ylabel('Validation nDCG@10')
    ax.set_title('Which feature-coordinate change helped?\nSame cached features and training recipe; three policy seeds')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'reranker-components-results.{ext}', dpi=180)
    plt.close(fig)
    for name, heading, paragraph in (
        ('README.md', '\n## Feature component ablation\n', '\n`modal_reranker_components.py` reuses the saved transform to test removal-only and scaling-only, each at KL 0 and 0.1 with three seeds. Both coordinate systems are saved under `experiments/reranker-components-20261106`; no new fitting or retrieval occurs.\n'),
        ('VERIFICATION.md', '\n## Feature component ablation — 2026-11-06\n', f'\n- 73 tests passed, including equivalent policy-family tests for both component transforms.\n- All twelve saved policies reconstruct independently in NumPy within {max(errors):.2g}; mapped original-coordinate metrics differ by at most {max(mapped_errors):.2g}.\n- No new fitting, page scoring, GPU training, API calls, final-test or external-transfer evaluation.\n')):
        path = ROOT / name
        if heading not in path.read_text():
            path.write_text(path.read_text() + heading + paragraph)
    publish_artifacts('source-snapshot-reranker-components-20261106.zip', ('reranker-components-summary.json', 'reranker-components-results.png', 'reranker-components-results.pdf'))


if __name__ == '__main__':
    main()
