"""Publish the fixed lexical passage experiment without replacing older findings."""
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal

from publish_mixed_results import OUT, ROOT, interval, publish_artifacts, read, write


def main():
    volume = modal.Volume.from_name('embedding-rl-data')
    folder = OUT / 'passages-20261106'
    folder.mkdir(exist_ok=True)
    for name in ('comparison.json', 'protocol.json', 'verification.json'):
        data = b''.join(volume.read_file(f'experiments/passages-20261106/{name}'))
        json.loads(data)
        (folder / name).write_bytes(data)
    result = read('passages-20261106/comparison.json')
    protocol = read('passages-20261106/protocol.json')
    verification = read('passages-20261106/verification.json')
    previous = read('lexical-20261106/comparison.json')
    if protocol['size'] != 512 or protocol['stride'] != 384 or protocol['external_datasets_evaluated']:
        raise ValueError('Passage protocol differs from the predeclared recipe')
    if len(result['runs']) != 5 or result['query_ids'] != previous['query_ids'] or result['browsecomp_final_test_evaluated']:
        raise ValueError('Unexpected passage comparison scope')
    if {r['seed'] for r in result['runs'] if r['model'] == 'binary2000'} != {42, 43, 44}:
        raise ValueError('Missing passage-fusion seed')
    if verification['fresh_container_reload_max_error'] > 1e-6 or verification['queries'] != 83:
        raise ValueError('Saved passage index verification failed')
    write('passages-summary.json', {'summary': result['summary'], 'verification': verification,
        'runner_sha256': hashlib.sha256((ROOT / 'modal_passages.py').read_bytes()).hexdigest(),
        'api_calls': 0, 'training_updates': 0, 'browsecomp_final_test_evaluated': False})
    log = read('research-log.json')
    hypothesis = log.pop('active_experiment', {})
    if hypothesis.get('name') == 'passages-20261106':
        hypothesis['status'] = 'complete and reloaded independently'
        log['completed_passage_hypothesis'] = hypothesis
    log['active_experiments']['passages-20261106'] = {'status': 'complete', 'artifact': 'passages-20261106/comparison.json'}
    log['passage_findings'] = result['summary']
    write('research-log.json', log)
    names = {'bm25': 'BM25 alone', 'original': 'Original encoder + BM25', 'binary2000': 'Full-corpus RL + BM25'}
    lines = ['## Lexical passages instead of whole-page scoring', '',
        f'This single predeclared recipe indexes {result["chunks"]:,} overlapping passages from {result["documents"]:,} pages. '
        'Each window contains up to 512 retained lexical tokens, with stride 384, after English stemming and stopword removal. '
        'These are not BGE tokens. Each page gets its highest passage score; pages are deduplicated before ranking. '
        'BM25 parameters and equal-weight RRF (constant 60, depth 1,000) match the earlier controls.', '',
        '| System | Prefix BM25 | Whole-page BM25 | Passage-max BM25 | Passage minus prefix, 95% interval |',
        '|---|---:|---:|---:|---|']
    old = {(r['model'], r['variant']): r['mean_ndcg'] for r in previous['summary']}
    for row in result['summary']:
        family = row['model']
        lines.append(f'| {names[family]} | {old[(family, "first512")]:.6f} | {old[(family, "full")]:.6f} | '
                     f'{row["mean_ndcg"]:.6f} | {interval(row["comparisons"]["first512"])} |')
    lines.extend(['',
        'All rows use the same 83 validation queries; trained rows average all three final checkpoints. '
        'Intervals resample paired queries after averaging seeds. They are descriptive and unadjusted for repeated exploration. '
        'The 166 BrowseComp-Plus final-test queries and both external transfer datasets were not evaluated in this experiment. '
        'No training or API calls were made.', '',
        'This changes the scoring unit, corpus term frequencies and length normalization together; it does not isolate context length. '
        'Long pages also have more opportunities to produce a high-scoring passage. Passage scores do not establish that an answer is present.', '',
        '[MaxP research](https://cs.uwaterloo.ca/~jimmylin/publications/ZhangXinyu_etal_ECIR2021.pdf) motivates the maximum-score aggregation rule; '
        'this lexical adaptation is not a reproduction of its transformer experiments. '
        '[Positional-bias research](https://arxiv.org/abs/2207.01262v4) shows why prefix baselines can be competitive, '
        'but our prefix result alone does not establish that mechanism here. '
        '[DAPR](https://arxiv.org/abs/2305.13915v4) highlights the opposing risk: isolated passages can lose necessary document context.', '',
        f'Fresh-container reload reproduced every saved page score: maximum error {verification["fresh_container_reload_max_error"]:.2g}. '
        'Index, passage boundaries, page mapping, scores and protocol persist under `experiments/passages-20261106` in the private Modal volume.', ''])
    text = (OUT / 'results.md').read_text()
    heading = '\n## Lexical passages instead of whole-page scoring\n'
    if heading in text:
        text = text[:text.index(heading)]
    (OUT / 'results.md').write_text(text.rstrip() + '\n\n' + '\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, family in zip(axes, ('bm25', 'binary2000'), strict=True):
        row = next(r for r in result['summary'] if r['model'] == family)
        values = [old[(family, 'first512')], old[(family, 'full')], row['mean_ndcg']]
        bars = ax.barh(['Prefix', 'Whole page', 'Best passage'], values, color=['#0072B2', '#999999', '#E69F00'])
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.set_xlim(0, max(values) * 1.25)
        ax.invert_yaxis()
        ax.set_title(names[family])
        ax.set_xlabel('Validation nDCG@10')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('One fixed passage recipe; 83 reused validation queries\nDifferent tokenization and scoring units; exploratory comparison')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'passages-results.{ext}', dpi=180)
    plt.close(fig)
    verification_path = ROOT / 'VERIFICATION.md'
    section = '\n## Lexical passage control — 2026-11-06\n'
    content = verification_path.read_text()
    if section not in content:
        verification_path.write_text(content + section + '\n- 59 tests passed, including complete within-page chunk coverage, empty/short pages, native integer iteration, and max aggregation without duplicate pages. Ruff and whitespace checks passed.\n'
            '- One fixed passage recipe, no training or API calls. Cached dense vectors reproduce earlier per-query baselines within 1e-6. A fresh container reload reproduces every passage-aggregated score within 1e-6.\n'
            '- BrowseComp final-test and external transfer datasets were not evaluated in this experiment.\n')
    publish_artifacts('source-snapshot-passages-20261106.zip', ('passages-summary.json', 'passages-results.png', 'passages-results.pdf'))


if __name__ == '__main__':
    main()
