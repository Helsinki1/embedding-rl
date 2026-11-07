"""Publish every condition from the stronger-candidate frozen reranker follow-up."""
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal

from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'hybrid-reranker-20261106'
    folder.mkdir(exist_ok=True)
    for name in ('protocol.json', 'comparison.json', 'complete-candidates.json', 'verification.json', 'scores-512.json', 'scores-2048.json'):
        data = b''.join(volume.read_file(f'experiments/hybrid-reranker-20261106/{name}'))
        json.loads(data)
        (folder / name).write_bytes(data)
    result = read('hybrid-reranker-20261106/comparison.json')
    verification = read('hybrid-reranker-20261106/verification.json')
    if len(result['runs']) != 12 or len(result['query_ids']) != 83 or result['test_evaluated'] or result['training_updates']:
        raise ValueError('Unexpected frozen reranker experiment scope')
    if verification['fresh_container_recomputed_metric_max_error'] > 1e-6:
        raise ValueError('Cached reranking metrics did not reproduce')
    for variant in ('rrf_first512', 'rrf_passage_max'):
        for length in (512, 2048):
            if {r['seed'] for r in result['runs'] if r['variant'] == variant and r['max_length'] == length} != {42, 43, 44}:
                raise ValueError('Missing reranker condition')
    write('hybrid-reranker-summary.json', {'summary': result['summary'], 'context_comparisons': result['context_comparisons'],
                                         'verification': verification, 'api_calls': 0, 'test_evaluated': False})
    log = read('research-log.json')
    log['active_experiments']['hybrid-reranker-20261106'].update(status='complete and independently verified',
        client_error='The local client reported a Connection._transport error after scoring; the remote comparison was saved. All metrics were subsequently recomputed from cached scores in a fresh container with zero error.')
    log['hybrid_reranker_findings'] = result['summary']
    write('research-log.json', log)
    names = {'rrf_first512': 'Prefix BM25 + RL', 'rrf_passage_max': 'Passage BM25 + RL'}
    lines = ['## Frozen reranking of the stronger hybrid candidates', '',
        'The frozen reranker did not improve the stronger hybrid lists on average. The 512-token version reduced the prefix hybrid and '
        'left the passage hybrid approximately unchanged; both intervals include zero. The 2,048-token version reduced both hybrids, '
        'with descriptive paired intervals below zero. This remains exploratory evidence after repeated validation selection.', '',
        '| Candidate source | Pair token limit | Before nDCG@10 | After nDCG@10 | Difference, 95% query interval |',
        '|---|---:|---:|---:|---|']
    for row in result['summary']:
        lines.append(f'| {names[row["variant"]]} | {row["max_length"]} | {row["mean_before"]:.6f} | {row["mean_after"]:.6f} | {interval(row["paired_query_bootstrap"])} |')
    lines.extend(['',
        'Each row averages all three final full-corpus encoders on the same 83 validation queries. Natural top-25 candidates are preserved; '
        'no relevant pages are injected. The pinned BGE-v2-m3 reranker reads the same scraped-text input as before, with pair truncation. '
        'It does not receive the passage selected by keyword retrieval. These are frozen inference comparisons, not reranker training.', '',
        '| Candidate source | Longer minus shorter context, 95% query interval |', '|---|---|'])
    for row in result['context_comparisons']:
        lines.append(f'| {names[row["variant"]]} | {interval(row["longer_minus_shorter"])} |')
    lines.extend(['',
        'There are 3,977 unique query/page pairs per text limit: 2,514 previously scored pairs were reused after provenance checks, '
        'and 1,463 new pairs were scored. A client connection error occurred after scoring, but the remote results were saved. '
        'An independent fresh-container check recomputed all before/after/oracle values exactly from cached logits. '
        'Both per-query source baselines also match their earlier completed experiments.', '',
        'No paid API calls or training updates were made. All 166 BrowseComp-Plus final-test queries and the external transfer datasets '
        'were left untouched in this follow-up. Private artifacts persist under `experiments/hybrid-reranker-20261106`.', ''])
    report = (OUT / 'results.md').read_text()
    heading = '\n## Frozen reranking of the stronger hybrid candidates\n'
    if heading in report:
        report = report[:report.index(heading)]
    report = report.rstrip() + '\n\n' + '\n'.join(lines)
    overview = 'Frozen reranking did not improve the stronger hybrids: prefix fusion moved from 0.17063 to 0.15967 at 512 tokens, and passage fusion from 0.16568 to 0.16481. Both 2,048-token versions fell to about 0.132.\n\n'
    if overview not in report:
        report = report.replace('An external transfer check found a cost to specialization:', overview + 'An external transfer check found a cost to specialization:')
    (OUT / 'results.md').write_text(report)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, variant in zip(axes, names, strict=True):
        rows = sorted([r for r in result['summary'] if r['variant'] == variant], key=lambda r: r['max_length'])
        oracle = statistics.mean(statistics.mean(r['oracle']) for r in result['runs'] if r['variant'] == variant and r['max_length'] == 512)
        values = [rows[0]['mean_before'], rows[0]['mean_after'], rows[1]['mean_after']]
        bars = ax.barh(['Before', 'Reranker 512', 'Reranker 2048'], values, color=['#0072B2', '#E69F00', '#999999'])
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.set_xlim(0, .22)
        ax.invert_yaxis()
        ax.set_title(f'{names[variant]}\nCandidate oracle: {oracle:.3f}')
        ax.set_xlabel('Validation nDCG@10')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen reranking: all three seeds; 83 reused validation queries')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'hybrid-reranker-results.{ext}', dpi=180)
    plt.close(fig)
    path = ROOT / 'VERIFICATION.md'
    text = path.read_text()
    note = '\n## Stronger hybrid reranking — 2026-11-06\n'
    if note not in text:
        path.write_text(text + note + '\n- All 12 conditions and three seeds completed; every source baseline matched its earlier per-query metrics.\n'
            '- After a local client connection error, a fresh CPU container recomputed before/after/oracle metrics from saved logits with zero error. No rescoring was required.\n'
            '- Both limits contain 3,977 unique pairs, including 2,514 reused scores and 1,463 new scores each. No final-test or external transfer evaluation.\n')
    publish_artifacts('source-snapshot-hybrid-reranker-20261106.zip', ('hybrid-reranker-summary.json', 'hybrid-reranker-results.png', 'hybrid-reranker-results.pdf'))


if __name__ == '__main__':
    main()
