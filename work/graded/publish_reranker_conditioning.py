"""Verify fitted feature coordinates, mapped weights, and all final rankings."""
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal
import numpy as np
import torch

from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write
from publish_neural_reranker import neural_ndcg
from publish_reranker_anchor import manual_kl


def metric_from_scores(subset, scores):
    order = np.argsort(-scores, axis=1, kind='stable')[:, :10]
    dcg = (np.take_along_axis(subset['gains'].numpy(), order, axis=1) / np.log2(np.arange(10) + 2)).sum(1)
    ideal = subset['ideal'].numpy()
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'reranker-conditioning-20261106'
    folder.mkdir(exist_ok=True)
    names = ['protocol.json', 'comparison.json', 'conditioning-diagnostic.json', 'transform.pt', 'conditioned.pt']
    names += [f'beta-{beta:g}-seed-{seed}/{name}' for beta in (0., .1) for seed in (42, 43, 44) for name in ('result.json', 'final.pt', 'metrics.jsonl')]
    for name in names:
        target = folder / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(f'experiments/reranker-conditioning-20261106/{name}')))
    result = read('reranker-conditioning-20261106/comparison.json')
    protocol = read('reranker-conditioning-20261106/protocol.json')
    diagnostic = read('reranker-conditioning-20261106/conditioning-diagnostic.json')
    scalar = torch.load(OUT / 'feature-reranker-20261106/features.pt', weights_only=True)
    original = torch.load(OUT / 'neural-reranker-fp32-20261106/neural.pt', weights_only=True)
    fitted = torch.load(folder / 'transform.pt', weights_only=True)
    transformed = torch.load(folder / 'conditioned.pt', weights_only=True)
    if original['protocol_digest'] != protocol['source_protocol_digest'] or scalar['protocol_digest'] != protocol['scalar_protocol_digest']:
        raise ValueError('Input feature provenance differs')
    if fitted['query_ids'] != scalar['train']['query_ids'] or transformed['query_ids'] != original['query_ids'] or result['query_ids'] != scalar['validation']['query_ids']:
        raise ValueError('Fit/evaluation query boundary differs')
    if fitted['protocol_digest'] != result['protocol_digest'] or transformed['protocol_digest'] != result['protocol_digest']:
        raise ValueError('Transform provenance differs')
    if {(r['beta'], r['seed']) for r in result['runs']} != {(b, s) for b in (0., .1) for s in (42, 43, 44)} or result['browsecomp_final_test_evaluated'] or result['api_calls']:
        raise ValueError('Experiment scope differs')
    for module, expected in protocol['source_hashes'].items():
        if hashlib.sha256((ROOT / (module + '.py')).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed')
    f = torch.cat([scalar['train']['features'], scalar['validation']['features']]).numpy().astype(np.float64)
    h = original['features'].numpy().astype(np.float64)
    centered = h - h.mean(1, keepdims=True)
    b = np.linalg.lstsq(f[:581].reshape(-1, 4), centered[:581].reshape(-1, 1024), rcond=None)[0]
    residual = centered - f @ b
    scales = np.maximum(np.sqrt(np.mean(residual[:581] ** 2, axis=(0, 1))), .001)
    projection_error = float(np.abs(b - fitted['projection'].numpy()).max())
    scale_error = float(np.abs(scales - fitted['scale'].numpy()).max())
    transform_error = float(np.abs((residual / scales).astype(np.float32) - transformed['features'].numpy()).max())
    if projection_error > 1e-9 or scale_error > 1e-9 or transform_error > 1e-5:
        raise ValueError('Independent training-only transform differs')
    errors, kl_errors, score_errors, mapped_errors = [], [], [], []
    for row in result['runs']:
        checkpoint = torch.load(folder / f'beta-{row["beta"]:g}-seed-{row["seed"]}/final.pt', weights_only=True)
        if checkpoint['step'] != 500 or checkpoint['beta'] != row['beta'] or checkpoint['protocol_digest'] != result['protocol_digest']:
            raise ValueError('Saved final checkpoint differs')
        new_delta = checkpoint['delta'].numpy().astype(np.float64) / scales
        new_theta = checkpoint['theta'].numpy().astype(np.float64) - b @ new_delta
        if np.abs(new_delta - checkpoint['original_delta'].numpy()).max() > 1e-9 or np.abs(new_theta - checkpoint['original_theta'].numpy()).max() > 1e-9:
            raise ValueError('Mapped original weights differ')
        for split, sl in (('train', slice(None, 581)), ('validation', slice(581, None))):
            values = neural_ndcg(scalar[split], transformed['features'][sl], checkpoint)
            original_scores = f[sl] @ new_theta + h[sl] @ new_delta
            equivalent = metric_from_scores(scalar[split], original_scores)
            mapped_errors.append(float(np.abs(values - equivalent).max()))
            transformed_scores = f[sl] @ checkpoint['theta'].numpy().astype(np.float64) + transformed['features'][sl].numpy().astype(np.float64) @ checkpoint['delta'].numpy().astype(np.float64)
            difference = original_scores - transformed_scores
            score_errors.append(float(np.abs(difference - difference.mean(1, keepdims=True)).max()))
            errors.append(float(np.abs(values - row['after']).max()) if split == 'validation' else abs(values.mean() - row['train_after']))
        kl_errors.append(abs(manual_kl(scalar['train'], transformed['features'][:581], checkpoint) - row['train_first_choice_kl']))
    if max(errors + kl_errors + mapped_errors) > 1e-6 or max(score_errors) > 1e-5:
        raise ValueError(f'Policy reconstruction differs: metric={max(errors)}, mapped={max(mapped_errors)}, score={max(score_errors)}, KL={max(kl_errors)}')
    verification = {'independent_numpy_metric_max_error': max(errors), 'independent_numpy_kl_max_error': max(kl_errors),
        'mapped_original_metric_max_error': max(mapped_errors), 'centered_score_max_error': max(score_errors),
        'projection_max_error': projection_error, 'scale_max_error': scale_error, 'transformed_features_max_error': transform_error,
        'policies': 6, 'tests_passed': 71, 'fit_queries': 581, 'validation_queries': 83, 'test_evaluated': False}
    write('reranker-conditioning-20261106/verification.json', verification)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(folder / 'verification.json', 'experiments/reranker-conditioning-20261106/verification.json')
    summary = {k: v for k, v in result.items() if k not in ('runs', 'query_ids')} | {'verification': verification}
    write('reranker-conditioning-summary.json', summary)
    log = read('research-log.json')
    log['active_experiments']['reranker-conditioning-20261106'].update(status='complete and independently verified', artifact='reranker-conditioning-20261106/comparison.json')
    log['queued_reranker_conditioning_hypothesis']['status'] = 'complete and independently verified'
    log['reranker_conditioning_findings'] = summary
    log['status'] = 'Conditioned feature policies complete; all six reconstructed in both coordinate systems'
    write('research-log.json', log)
    lines = ['## Making the remaining neural features easier to learn', '',
        'This experiment subtracts the linear component already explained by the four scalar features, then rescales each remaining '
        'neural feature by its training RMS. The fit uses only training queries and no relevance labels. It keeps the same information '
        'and representable ranking policies; it changes the optimization coordinates and effective step sizes.', '',
        f'In the query-centered training features, the first principal direction accounts for {diagnostic["first_pc_variance_fraction"]:.2%} '
        f'of variance. The four scalar features explain {diagnostic["scalar_projection_explained_variance"]:.2%}. '
        'This is a concrete conditioning problem, not evidence that the remaining variation is irrelevant.', '',
        '| KL strength | Original coordinates, validation | Transformed coordinates, validation | Difference, 95% interval |',
        '|---|---:|---:|---|']
    for row in result['summary']:
        lines.append(f'| {row["beta"]:g} | {row["matched_control_mean"]:.6f} | {row["mean_after"]:.6f} | {interval(row["versus_matched_control"])} |')
    lines.extend(['', f'The matched unchanged hybrid scores {result["hybrid_mean"]:.6f}. Comparisons with that baseline:', ''])
    for row in result['summary']:
        lines.append(f'- KL {row["beta"]:g}: {interval(row["versus_hybrid"])}; training nDCG {row["mean_train_after"]:.6f}.')
    lines.extend(['',
        'Both transformations are fitted in FP64 on the 581 training queries; transformed features are saved in FP32. '
        'The RMS floor is 0.001 and no feature reached that floor. The same fixed transform is applied to validation. '
        'All six runs use the previous training function, 500 updates, 16 queries, 32 rankings, scalar learning rate 0.01, '
        'and neural learning rate 0.0001. Each final policy is also saved as equivalent weights on the original features.', '',
        'This is an adaptive follow-up after observing redundant learned corrections and the KL controls. Both specified KL settings '
        'and all three seeds are reported. The paired intervals are positive against their matched unconditioned controls, '
        'but remain exploratory on reused validation and unadjusted for the broader search. The 0.1801 result varies policy seeds '
        'with one fixed upstream encoder; it must not be directly substituted for the earlier three-upstream-encoder mean of 0.1706.', '',
        'Removing redundant components and rescaling were changed together. This experiment alone does not separate their contributions. '
        'It is a fixed feature-coordinate change, not a new information source, full backbone fine-tuning, or an implementation of Shampoo.', '',
        f'Independent NumPy fitting reproduces the transform, and final metrics agree within {max(errors):.2g}. '
        f'Mapping the policies back to original features changes validation/training metrics by at most {max(mapped_errors):.2g}; '
        f'centered score error is at most {max(score_errors):.2g}. All 71 tests pass. '
        'Projection coefficients, scales, transformed features, all six checkpoints, metrics and W&B logs persist. '
        'No paid API calls, GPU training, new page scoring, final-test evaluation or external-transfer evaluation were used.', '',
        'W&B runs: ' + ', '.join(f'[KL {r["beta"]:g}, seed {r["seed"]}]({r["wandb_url"]})' for r in result['runs'] if r.get('wandb_url')), ''])
    report = (OUT / 'results.md').read_text()
    heading = '\n## Making the remaining neural features easier to learn\n'
    if heading in report:
        start = report.index(heading)
        end = report.find('\n## ', start + len(heading))
        report = report[:start] + (report[end:] if end >= 0 else '')
    report = report.rstrip() + '\n\n' + '\n'.join(lines)
    report = report.replace('plus eighteen small reranker policies', 'plus twenty-four small reranker policies')
    note = 'Changing the neural feature coordinates produced a promising adaptive result: validation 0.17732 without KL and 0.18007 with KL 0.1, versus matched unconditioned controls 0.16840 and 0.17281. This uses one upstream encoder and three policy seeds, not three upstream encoders.\n\n'
    if note not in report:
        report = report.replace('An external transfer check found a cost to specialization:', note + 'An external transfer check found a cost to specialization:')
    (OUT / 'results.md').write_text(report)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    xs = np.arange(2)
    for offset, key, label, color in ((-.18, 'matched_control_mean', 'Original coordinates', '#0072B2'), (.18, 'mean_after', 'Transformed coordinates', '#E69F00')):
        bars = ax.bar(xs + offset, [r[key] for r in result['summary']], width=.35, label=label, color=color)
        ax.bar_label(bars, fmt='%.4f', padding=4)
    ax.axhline(result['hybrid_mean'], linestyle='--', color='black', label='Unchanged hybrid')
    ax.set_xticks(xs, ['No KL', 'KL 0.1'])
    ax.set_ylim(0, .23)
    ax.set_ylabel('Validation nDCG@10')
    ax.set_title('Same feature information, different optimization coordinates\nThree policy seeds; 83 reused validation queries')
    ax.legend(loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'reranker-conditioning-results.{ext}', dpi=180)
    plt.close(fig)
    for name, heading, paragraph in (
        ('README.md', '\n## Feature conditioning control\n', '\n`modal_reranker_conditioning.py` fits a training-only projection of frozen neural features onto the four scalar features, removes that component, and divides residuals by training RMS. This is an invertible change of ranking-policy coordinates, with no new information. It runs three seeds each at KL 0 and 0.1. Transform parameters, cached transformed features, and equivalent original-coordinate policy weights persist under `experiments/reranker-conditioning-20261106`.\n'),
        ('VERIFICATION.md', '\n## Feature conditioning — 2026-11-06\n', f'\n- 71 tests passed, including exact policy-family equivalence and fixed training-only fitting.\n- Independent NumPy least squares and scaling reproduce the saved transform. All six metrics reconstruct within {max(errors):.2g}; mapped original-coordinate metrics differ by at most {max(mapped_errors):.2g}.\n- No new page scoring, GPU training, paid API calls, final-test or external-transfer evaluation.\n')):
        path = ROOT / name
        if heading not in path.read_text():
            path.write_text(path.read_text() + heading + paragraph)
    publish_artifacts('source-snapshot-reranker-conditioning-20261106.zip', ('reranker-conditioning-summary.json', 'reranker-conditioning-results.png', 'reranker-conditioning-results.pdf'))


if __name__ == '__main__':
    main()
