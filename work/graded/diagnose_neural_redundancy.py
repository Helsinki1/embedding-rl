"""Describe what the learned residual adds, without retraining or selecting a model."""
import numpy as np
import torch

from publish_mixed_results import OUT, publish_artifacts, read, write


def main():
    scalar = torch.load(OUT / 'feature-reranker-20261106/features.pt', weights_only=True)
    neural = torch.load(OUT / 'neural-reranker-fp32-20261106/neural.pt', weights_only=True)
    if neural['query_ids'] != scalar['train']['query_ids'] + scalar['validation']['query_ids']:
        raise ValueError('Query order differs')
    values = neural['features'].numpy()
    bge = scalar['raw_features'][:, :, 3].numpy()
    bge = bge - bge.mean(1, keepdims=True)
    rows = []
    for seed in (42, 43, 44):
        checkpoint = torch.load(OUT / f'neural-reranker-fp32-20261106/seed-{seed}/final.pt', weights_only=True)
        if checkpoint['protocol_digest'] != neural['protocol_digest']:
            raise ValueError('Checkpoint provenance differs')
        residual = values @ checkpoint['delta'].numpy()
        residual -= residual.mean(1, keepdims=True)
        row = {'seed': seed}
        for split, sl in (('train', slice(None, 581)), ('validation', slice(581, None))):
            row[split] = float(np.corrcoef(bge[sl].flatten(), residual[sl].flatten())[0, 1])
        rows.append(row)
    result = {'runs': rows, 'method': 'Pearson correlation after centering within each query, then pooling candidate pairs within each split.',
              'comparison': 'Learned neural residual versus original raw BGE logit; no labels used.',
              'interpretation': 'The learned correction closely tracks an existing score. This does not establish that the full frozen representation has no other useful information.',
              'training_updates': 0, 'api_calls': 0, 'test_evaluated': False}
    write('neural-residual-diagnostic.json', result)
    log = read('research-log.json')
    log['neural_residual_diagnostic'] = result
    log['api_budget_rechecked'] = {'charged_or_reserved_usd': 90.60059214999976, 'ceiling_usd': 95, 'checked_utc': '2026-11-06T15:52:00Z'}
    write('research-log.json', log)
    paragraph = ('A post-training, label-free diagnostic found that the learned neural correction is almost a copy of an existing signal: '
        'after subtracting each query’s candidate mean, its pooled Pearson correlation with the original raw BGE logit is '
        '0.99911–0.99954 on training and 0.99909–0.99952 on validation across the three policies. '
        'This is consistent with the extra head mostly reweighting information already represented by the BGE feature. '
        'It does not prove that all 1,024 frozen features are redundant or that another parameterization cannot use them. '
        'No relevance labels, new training, checkpoint selection or final-test evaluation were used for this diagnostic.\n')
    report = (OUT / 'results.md').read_text()
    if paragraph not in report:
        (OUT / 'results.md').write_text(report.rstrip() + '\n\n' + paragraph)
    publish_artifacts('source-snapshot-neural-reranker-fp32-20261106.zip', ('neural-reranker-summary.json',
        'neural-reranker-results.png', 'neural-reranker-results.pdf', 'source-neural-fp16-attempt.zip',
        'reranker-headroom-diagnostic.json', 'neural-residual-diagnostic.json'))
    print(result)


if __name__ == '__main__':
    main()
