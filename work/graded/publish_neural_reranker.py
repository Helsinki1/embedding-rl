"""Verify frozen neural representations and publish all three final RL policies."""
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal
import numpy as np
import torch

from publish_feature_reranker import manual_ndcg
from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write


def neural_ndcg(subset, neural, checkpoint):
    scores = subset['features'].numpy() @ checkpoint['theta'].numpy() + neural.float().numpy() @ checkpoint['delta'].numpy()
    order = np.argsort(-scores, axis=1, kind='stable')[:, :10]
    dcg = (np.take_along_axis(subset['gains'].numpy(), order, axis=1) / np.log2(np.arange(10) + 2)).sum(1)
    ideal = subset['ideal'].numpy()
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'neural-reranker-fp32-20261106'
    folder.mkdir(exist_ok=True)
    names = ['protocol.json', 'comparison.json', 'cache-verification.json', 'neural.pt']
    names += [f'seed-{seed}/{name}' for seed in (42, 43, 44) for name in ('result.json', 'final.pt', 'metrics.jsonl')]
    for name in names:
        payload = b''.join(volume.read_file(f'experiments/neural-reranker-fp32-20261106/{name}'))
        target = folder / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(payload)
    for name in ('protocol.json', 'diagnostics/NVIDIA-A10.json'):
        target = OUT / 'neural-reranker-20261106' / name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(b''.join(volume.read_file(f'experiments/neural-reranker-20261106/{name}')))
    target = OUT / 'neural-reranker-fp16-20261106' / 'protocol.json'
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b''.join(volume.read_file('experiments/neural-reranker-fp16-20261106/protocol.json')))
    result = read('neural-reranker-fp32-20261106/comparison.json')
    protocol = read('neural-reranker-fp32-20261106/protocol.json')
    cache = read('neural-reranker-fp32-20261106/cache-verification.json')
    scalar = torch.load(OUT / 'feature-reranker-20261106/features.pt', weights_only=True)
    neural = torch.load(folder / 'neural.pt', weights_only=True)
    reference = read('feature-reranker-20261106/comparison.json')
    headroom = read('reranker-headroom-diagnostic.json')
    if protocol['policy_source_sha256'] != hashlib.sha256((ROOT / 'rerank_neural.py').read_bytes()).hexdigest():
        raise ValueError('Policy source changed after execution')
    if neural['protocol_digest'] != result['protocol_digest'] or scalar['protocol_digest'] != protocol['source_protocol_digest']:
        raise ValueError('Neural cache provenance differs')
    if neural['query_ids'] != scalar['train']['query_ids'] + scalar['validation']['query_ids'] or result['query_ids'] != scalar['validation']['query_ids']:
        raise ValueError('Saved neural query order differs')
    if neural['features'].shape != (664, 25, 1024) or neural['features'].dtype != torch.float32 or not torch.isfinite(neural['features']).all():
        raise ValueError('Neural cache shape or values differ')
    if protocol['dtype'] != 'float32' or protocol['tf32'] or cache['max_head_reconstruction_error'] > .0001:
        raise ValueError('Frozen representation equivalence check failed')
    if {r['seed'] for r in result['runs']} != {42, 43, 44} or result['browsecomp_final_test_evaluated'] or result['api_calls'] or result['backbone_updates']:
        raise ValueError('Neural policy experiment scope differs')
    before = manual_ndcg(scalar['validation'], [1, 0, 0, 0])
    train_before = manual_ndcg(scalar['train'], [1, 0, 0, 0]).mean()
    errors = [abs(before.mean() - result['mean_before'])]
    for row in result['runs']:
        checkpoint = torch.load(folder / f'seed-{row["seed"]}/final.pt', weights_only=True)
        if checkpoint['step'] != 500 or checkpoint['protocol_digest'] != result['protocol_digest'] or checkpoint['theta'].tolist() != row['theta']:
            raise ValueError('Saved final neural policy differs')
        if checkpoint['delta'].shape != (1024,) or not torch.isfinite(checkpoint['delta']).all():
            raise ValueError('Saved residual head differs')
        errors.extend([float(np.abs(neural_ndcg(scalar['validation'], neural['features'][581:], checkpoint) - row['after']).max()),
                       abs(neural_ndcg(scalar['train'], neural['features'][:581], checkpoint).mean() - row['train_after'])])
    if max(errors) > 1e-6:
        raise ValueError(f'Independent NumPy reconstruction differs: {max(errors)}')
    verification = {'independent_numpy_metric_max_error': max(errors), 'policy_checkpoints': 3, 'train_queries': 581,
                    'validation_queries': 83, 'cache': cache, 'api_calls': 0, 'test_evaluated': False}
    write('neural-reranker-fp32-20261106/verification.json', verification)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(folder / 'verification.json', 'experiments/neural-reranker-fp32-20261106/verification.json')
    summary = {k: v for k, v in result.items() if k not in ('query_ids', 'runs')} | {'verification': verification,
        'train_before': float(train_before), 'runs': [{k: v for k, v in row.items() if k not in ('before', 'after')} for row in result['runs']]}
    write('neural-reranker-summary.json', summary)
    log = read('research-log.json')
    log['active_experiments']['neural-reranker-fp32-20261106'].update(status='complete and independently verified', artifact='neural-reranker-fp32-20261106/comparison.json')
    log['queued_neural_reranker_hypothesis']['status'] = 'complete and independently verified'
    log['neural_reranker_findings'] = summary
    log['status'] = 'Frozen neural-feature residual reranker complete; three final policies verified'
    write('research-log.json', log)
    lines = ['## A richer frozen-feature RL reranker', '',
        'This experiment adds a learned linear correction over 1,024 frozen neural features for each query/page pair. '
        'It retains the previous four scalar features, making 1,028 trainable parameters. The correction starts at zero, '
        'so the initial ranking is exactly the same hybrid baseline. No pretrained backbone weights change.', '',
        'The original cache attempt stopped before training: a maximum logit difference of 0.015625 exceeded the initial 0.01 guard. '
        'A diagnostic on the first 256 training pairs found zero change from attaching the hook; both cached and new FP16 outputs '
        'were within 0.016014 of an FP32 reference. A second attempt with a 0.05 maximum guard stopped at a 0.0546875 outlier '
        'after 5,888 cached pairs; its block mean difference was 0.002695. Neither attempt reached policy training. '
        'The completed experiment instead computes every new neural feature in FP32 with TF32 disabled. Differences from the old '
        'FP16 scores are recorded, without claiming numerical equivalence between precisions. The old four scalar features and '
        'initial ranking remain unchanged. This revision used no neural validation outcomes; both stopped protocols and diagnostics persist.', '',
        '| Policy | Train nDCG@10 | Validation nDCG@10 |', '|---|---:|---:|',
        f'| Unchanged seed-42 hybrid | {train_before:.6f} | {result["mean_before"]:.6f} |',
        f'| Four-feature RL, three policy seeds | {np.mean([r["train_after"] for r in reference["runs"]]):.6f} | {result["scalar_reference_mean"]:.6f} |',
        f'| Neural-feature RL, three policy seeds | {result["mean_train_after"]:.6f} | {result["mean_after"]:.6f} |', '',
        f'Neural policy minus unchanged hybrid: **{interval(result["versus_hybrid"])}**. '
        f'Neural minus four-feature RL: **{interval(result["versus_scalar_policy"])}**. '
        'These are descriptive, unadjusted paired query intervals on the same reused 83 validation queries.', '',
        '| Policy seed | Train after | Validation after | Residual weight norm |', '|---|---:|---:|---:|']
    for row in result['runs']:
        lines.append(f'| {row["seed"]} | {row["train_after"]:.6f} | {row["mean_after"]:.6f} | {row["delta_norm"]:.4f} |')
    lines.extend(['',
        f'For context, perfectly ordering only the existing candidates would score {headroom["train"]["oracle_mean"]:.6f} on training '
        f'and {headroom["validation"]["oracle_mean"]:.6f} on validation. The four-feature policy closed about '
        f'{headroom["train"]["fraction_headroom_closed"]:.1%} of its available training improvement and '
        f'{headroom["validation"]["fraction_headroom_closed"]:.1%} on validation. Candidate coverage differs between splits, '
        'but missing candidates alone do not explain the unchanged validation ranking. These oracles are diagnostics using incomplete benchmark labels.', '',
        'The same natural top-25 candidates are used for 581 training and 83 validation queries. The upstream encoder remains '
        'fixed at seed 42. The BGE reranker uses the same pinned revision and 512-token input as the scalar '
        'control, with the explicitly revised FP32 precision. A hook captures the input to its final output projection; 16,600 feature vectors are saved in resumable blocks.', '',
        f'Maximum absolute disagreement with cached scalar logits was {cache["max_cached_logit_error"]:.6g}; '
        f'reapplying the frozen final projection gave error {cache["max_head_reconstruction_error"]:.6g}. '
        'Zero residual preserves the old logits, ranking probabilities and scalar gradients. Freezing the residual reproduces '
        'the previous training updates in the regression test.', '',
        'Each of three seeds runs 500 Adam updates with 16 training queries and 32 sampled rankings per query. The scalar '
        'learning rate is 0.01; the neural correction uses 0.0001. The reward and whole-ranking RLOO estimator are unchanged. '
        'There is no contrastive loss, KL penalty or intermediate validation checkpoint selection. This tests one specified '
        'linear head and learning rate, not all neural rerankers or full-model fine-tuning.', '',
        f'Independent NumPy reconstruction of saved policies reproduced metrics within {max(errors):.2g}. '
        'All 67 automated tests pass. Feature blocks, provenance, all three final policies, metrics and W&B logs persist. '
        'No paid API calls or final-test/external-dataset evaluations were used.', '',
        'A new literature check, [On-Policy Distillation Meets Off-Policy GRPO](https://aclanthology.org/2026.findings-emnlp.477/), '
        'uses student-sampled Plackett–Luce rankings rewarded by a fixed trained teacher. Its teacher-derived nDCG, '
        'first-selection KL/entropy regularizers and roughly 89,000 instruction-following training examples differ from this experiment. '
        'Its human judge audit is also stronger evidence than model-to-model agreement alone. This suggests a future controlled '
        'teacher-reward study with audited labels; it does not establish that more grading or regularization would help here.', '',
        'W&B runs: ' + ', '.join(f'[seed {r["seed"]}]({r["wandb_url"]})' for r in result['runs'] if r.get('wandb_url')), ''])
    report = (OUT / 'results.md').read_text()
    heading = '\n## A richer frozen-feature RL reranker\n'
    if heading in report:
        start = report.index(heading)
        end = report.find('\n## ', start + len(heading))
        report = report[:start] + (report[end:] if end >= 0 else '')
    report = report.rstrip() + '\n\n' + '\n'.join(lines)
    report = report.replace('plus nine small reranker policies', 'plus twelve small reranker policies')
    note = f'The richer frozen-neural-feature policy reached {result["mean_after"]:.5f} validation nDCG, versus {result["scalar_reference_mean"]:.5f} for four features and {result["mean_before"]:.5f} for the same unchanged hybrid. Its training score was {result["mean_train_after"]:.5f}.\n\n'
    report = report.replace('An external transfer check found a cost to specialization:', note + 'An external transfer check found a cost to specialization:')
    (OUT / 'results.md').write_text(report)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    labels = ['Unchanged\nhybrid', 'Four-feature\nRL', 'Neural-feature\nRL']
    values = [[train_before, np.mean([r['train_after'] for r in reference['runs']]), result['mean_train_after']],
              [result['mean_before'], result['scalar_reference_mean'], result['mean_after']]]
    for i, (ax, vals) in enumerate(zip(axes, values, strict=True)):
        bars = ax.bar(labels, vals, color=['#0072B2', '#E69F00', '#009E73'])
        ax.bar_label(bars, fmt='%.4f', padding=4)
        ax.set_ylim(0, max(max(v) for v in values) * 1.2)
        ax.set_title(['Training: 581 queries', 'Validation: 83 reused queries'][i])
        ax.set_ylabel('nDCG@10')
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].scatter([2] * 3, [r['mean_after'] for r in result['runs']], color='black', s=18, zorder=3, label='Three policy seeds')
    axes[1].legend(loc='upper left')
    fig.suptitle('Adding frozen neural features to the same RL ranking policy\nOne fixed upstream encoder; final checkpoints only')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'neural-reranker-results.{ext}', dpi=180)
    plt.close(fig)
    path = ROOT / 'VERIFICATION.md'
    heading = '\n## Neural residual ranking policy — 2026-11-06\n'
    if heading not in path.read_text():
        path.write_text(path.read_text() + heading + '\n- 67 tests passed, including zero-residual logits/probability/gradient equivalence and frozen-residual training equivalence.\n'
            f'- All 16,600 captured representations reproduce the frozen head output; final metric reconstruction in NumPy differs by at most {max(errors):.2g}.\n'
            '- All three 500-update policies persist with protocol digests; the training function receives only training tensors. No final-test or external-transfer evaluations were made.\n')
    path = ROOT / 'README.md'
    content = path.read_text().replace('persist in `experiments/neural-reranker-20261106`; `--train-only` resumes after',
        'persist in `experiments/neural-reranker-fp32-20261106`; `--train-only` resumes after')
    note = '\nThe earlier `neural-reranker-20261106` and `neural-reranker-fp16-20261106` directories preserve stopped numerical-check attempts and training-only FP16/FP32 diagnostics. No policies were trained in those attempts.\n'
    if note not in content:
        content += note
    path.write_text(content)
    publish_artifacts('source-snapshot-neural-reranker-fp32-20261106.zip', ('neural-reranker-summary.json', 'neural-reranker-results.png', 'neural-reranker-results.pdf', 'source-neural-fp16-attempt.zip', 'reranker-headroom-diagnostic.json'))


if __name__ == '__main__':
    main()
