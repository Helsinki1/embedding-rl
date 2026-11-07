"""Collect the predeclared transfer check and append it to the existing research report."""
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal

from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'transfer-20261106'
    folder.mkdir(exist_ok=True)
    for name in ('comparison.json', 'protocol.json', 'scifact/runs.json', 'nfcorpus/runs.json'):
        data = b''.join(volume.read_file(f'experiments/transfer-20261106/{name}'))
        json.loads(data)
        target = folder / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    result = read('transfer-20261106/comparison.json')
    protocol = read('transfer-20261106/protocol.json')
    if len(result['runs']) != 14 or result['browsecomp_final_test_evaluated'] or result['training_updates']:
        raise ValueError('Unexpected transfer experiment scope')
    for dataset in ('scifact', 'nfcorpus'):
        rows = [r for r in result['runs'] if r['dataset'] == dataset]
        qids = rows[0]['query_ids']
        if len(qids) != protocol['datasets'][dataset]['queries'] or any(r['query_ids'] != qids for r in rows):
            raise ValueError('External query alignment differs')
        for family in ('full2000', 'mixed2000'):
            if {r['seed'] for r in rows if r['family'] == family} != {42, 43, 44}:
                raise ValueError('Missing transfer seed')
    write('transfer-summary.json', {'summary': result['summary'], 'api_calls': 0, 'training_updates': 0,
        'browsecomp_final_test_evaluated': False, 'external_test_queries': 623,
        'runner_sha256': hashlib.sha256((ROOT / 'modal_transfer.py').read_bytes()).hexdigest(),
        'evaluator_linear_gain_smoke_error': result['evaluator_linear_gain_smoke_error']})
    log = read('research-log.json')
    hypothesis = log.pop('queued_post_controls_hypothesis', log.get('completed_transfer_hypothesis', {}))
    hypothesis['status'] = 'complete; no external training or tuning'
    log['completed_transfer_hypothesis'] = hypothesis
    log['transfer_findings'] = result['summary']
    log['active_experiments']['transfer-20261106'] = {'status': 'complete', 'api_calls': 0,
        'training_updates': 0, 'artifact': 'transfer-20261106/comparison.json',
        'datasets': ['SciFact', 'NFCorpus'], 'models': 'original plus three full2000 and three mixed2000 encoders'}
    log['status'] = 'Inference controls and predeclared external transfer checks complete'
    log['next_steps'] = [
        'Read the transfer results before choosing another hypothesis. Do not tune on these external checks.',
        'Preserve the 166 untouched BrowseComp-Plus final-test queries and the shared API ledger.',
        'Continue bounded research only before 17:00 UTC, then pause the heartbeat and report findings.']
    write('research-log.json', log)

    def metric(dataset, family, name):
        return next(r for r in result['summary'] if (r['dataset'], r['family'], r['metric']) == (dataset, family, name))

    marker = '## External transfer check'
    report = (OUT / 'results.md').read_text().split(marker)[0].rstrip()
    lines = [marker, '',
        'The datasets and seven encoders were fixed before evaluating this check. Each external '
        'corpus is encoded once with the original BGE model, using the first 512 tokens of title '
        'plus text. Only the query encoder differs. No feedback, reranking, training or tuning was '
        'performed on these datasets. The comparison uses their official test query IDs: 300 SciFact '
        'queries and 323 NFCorpus queries.', '',
        '| Dataset | Original nDCG@10 | Full-corpus RL | Mixed RL |', '|---|---:|---:|---:|']
    for dataset in ('scifact', 'nfcorpus'):
        values = [metric(dataset, family, 'ndcg_cut_10')['mean'] for family in ('original', 'full2000', 'mixed2000')]
        lines.append(f'| {dataset} | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines += ['', '| Dataset | Original recall@100 | Full-corpus RL | Mixed RL |', '|---|---:|---:|---:|']
    for dataset in ('scifact', 'nfcorpus'):
        values = [metric(dataset, family, 'recall_100')['mean'] for family in ('original', 'full2000', 'mixed2000')]
        lines.append(f'| {dataset} | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines += ['', '| Dataset | Query encoder | nDCG difference versus original and descriptive 95% query interval |',
              '|---|---|---|']
    changes = []
    for dataset in ('scifact', 'nfcorpus'):
        for family, label in (('full2000', 'Full-corpus RL'), ('mixed2000', 'Mixed RL')):
            delta = metric(dataset, family, 'ndcg_cut_10')['paired_query_bootstrap']
            lines.append(f'| {dataset} | {label} | {interval(delta)} |')
            changes.append((dataset, label, delta))
    lines += ['', 'Trained-model means average all three seeds. Intervals average seeds per query '
              'before resampling queries; the individual seed results are retained in the JSON files. '
              'This is a diagnostic of specialization, not a claim of contamination-free zero-shot '
              'evaluation: these public datasets may have appeared in the base model’s pretraining. '
              'Compare models within each dataset; raw scores across corpora are not comparable.', '',
              'The [BEIR evaluation implementation](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py) '
              'uses pytrec_eval. We used pinned pytrec-eval-terrier 0.5.11 with its linear relevance '
              'gains for graded nDCG, verified by a numerical smoke check. All model and data revisions '
              'are pinned in protocol.json. The broader motivation is '
              '[BEIR’s heterogeneous retrieval evaluation](https://arxiv.org/abs/2104.08663).', '',
              '![Query-encoder transfer differences](transfer-results.png)', '',
              'All 166 BrowseComp-Plus final-test queries remain untouched. This check made no paid '
              'API calls, and the conservative total API estimate remains $90.60. Document vectors, '
              'query rankings and reports persist under /data/experiments/transfer-20261106.', '']
    (OUT / 'results.md').write_text(report + '\n\n' + '\n'.join(lines))

    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    for i, (dataset, label, value) in enumerate(changes):
        mean = value['mean_delta']
        ax.errorbar(mean, i, xerr=[[mean - value['lower_95']], [value['upper_95'] - mean]],
                    fmt='o', color='#0072B2' if label == 'Full-corpus RL' else '#D55E00', capsize=4)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set_yticks(range(len(changes)), [f'{d}: {label}' for d, label, _ in changes])
    ax.invert_yaxis()
    ax.set(title='Transfer after training the query encoder on BrowseComp-Plus',
           xlabel='Change in nDCG@10 versus the original query encoder')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    fig.suptitle('Frozen original document encoder; three seeds; descriptive query-bootstrap intervals', fontsize=9)
    fig.savefig(OUT / 'transfer-results.png', dpi=180)
    fig.savefig(OUT / 'transfer-results.pdf')
    plt.close(fig)
    marker = '## External transfer check — 2026-11-06'
    verification = (ROOT / 'VERIFICATION.md').read_text().split(marker)[0].rstrip()
    (ROOT / 'VERIFICATION.md').write_text(verification + '\n\n' + marker + '\n\n'
        '- Evaluated the original encoder and all six predeclared trained encoders on 300 SciFact '
        'and 323 NFCorpus test queries, with frozen original document embeddings and no tuning.\n'
        '- Pinned corpus/query/qrels revisions and text schemas were checked. The official evaluator '
        'passed its linear-gain numerical smoke check; per-query IDs and all three seeds align.\n'
        '- The 166 BrowseComp-Plus final-test queries remain untouched. No API calls or training '
        'updates were made. Cached external vectors and results persist in Modal.\n')
    publish_artifacts('source-snapshot-transfer-20261106.zip',
                      ('transfer-summary.json', 'transfer-results.png', 'transfer-results.pdf'))


if __name__ == '__main__':
    main()
