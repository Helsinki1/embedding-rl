"""Append the next completed research batch without rewriting earlier findings."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal

ROOT = Path('/home/davidx/Downloads/embedding-rl')
OUT = Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def interval(value):
    return f"{value['mean_delta']:+.6f} [{value['lower_95']:+.6f}, {value['upper_95']:+.6f}]"


def main():
    mixed = read('mixed-longer-summary.json')
    feedback = read('feedback-20261106/comparison.json')
    lexical = read('lexical-20261106/comparison.json')
    lexical_verification = read('lexical-20261106/verification.json')
    lexical_tokenization = read('lexical-20261106/tokenization-verification.json')
    if not lexical_tokenization['all_term_counts_equal']:
        raise ValueError('Independent lexical tokenization check failed')
    coverage = read('candidate-coverage-20261106/comparison.json')
    coverage_feedback = read('candidate-coverage-20261106/feedback-comparison.json')
    reranker = read('frozen-reranker-20261106/comparison.json')
    reranker_verification = read('frozen-reranker-20261106/verification.json')
    log = read('research-log.json')
    for name in ('alternating-longer-20261106', 'feedback-20261106', 'lexical-20261106'):
        log['active_experiments'][name]['status'] = 'complete'
    log['mixed_longer_findings'] = mixed
    log['feedback_findings'] = feedback['summary']
    log['lexical_findings'] = lexical['summary']
    log['lexical_verification'] = lexical_verification
    log['lexical_tokenization_verification'] = lexical_tokenization
    log['candidate_coverage_findings'] = coverage['summary']
    log['frozen_reranker_findings'] = reranker['summary']
    log['active_experiments']['frozen-reranker-20261106']['status'] = 'complete'
    log['active_experiments']['candidate-coverage-20261106'] = {
        'status': 'complete', 'api_calls': 0, 'training_updates': 0,
        'artifact': 'candidate-coverage-20261106/comparison.json'}
    log['completed_mixed_longer_hypothesis'] = log.pop('next_no_api_hypothesis', log.get('completed_mixed_longer_hypothesis', {}))
    log['status'] = 'Mixed training-length, inference feedback, lexical and frozen reranker controls complete'
    log['next_steps'] = [
        'Read the newest three completed experiment summaries before selecting the next bounded hypothesis; do not duplicate them.',
        'Keep learned-encoder gains separate from inference fusion and additional-document-context gains. All166 test queries remain untouched.',
        'The API ledger remains$90.60059215/$95; preserve it and favor cached/no-API work.',
        'Continue only before17:00UTC on2026-11-06, then pause the overnight heartbeat and report.']
    write('research-log.json', log)
    marker = '## Longer mixed training and inference controls'
    report = (OUT / 'results.md').read_text().split(marker)[0].rstrip()
    for old_count in (49, 52, 53, 55):
        report = report.replace(f'{old_count} automated tests passed', '57 automated tests passed')
    lines = [marker, '',
        'The mixed recipe was selected from the previous four 500-step conditions by highest three-seed '
        'mean validation nDCG. Three fresh 2,000-update runs were then predeclared. All use binary '
        'evidence labels, 32 samples, radius0.45, LR5e-6, no anchor, and projection only on each tenth '
        'candidate-pool update. Each completed exactly1,800 full-corpus and200 pool updates.', '',
        '| Seed | Full-corpus only, 2,000 steps | Mixed, 2,000 steps | Difference |',
        '|---|---:|---:|---:|',
    ]
    for row in mixed['runs']:
        lines.append(f'| {row["seed"]} | {row["control_final_ndcg"]:.6f} | {row["final_ndcg"]:.6f} | {row["seed_delta"]:+.6f} |')
    lines += [f'| Mean | {mixed["control_mean_ndcg"]:.6f} | {mixed["mean_ndcg"]:.6f} | '
              f'{mixed["mean_ndcg"] - mixed["control_mean_ndcg"]:+.6f} |', '',
              'The descriptive seed-averaged query-bootstrap difference is '
              + interval(mixed['paired_query_bootstrap']) + '. Final checkpoints are used, not each seed’s '
              'best validation step. Actual within-run500/1000/1500/2000 means are '
              + ' / '.join(f'{mixed["mean_validation_by_step"][str(step)]:.6f}' for step in (500, 1000, 1500, 2000))
              + '. These remain exploratory validation comparisons.', '',
              '### Extra retrieval without extra learning', '',
              'A separate frozen-model control uses a softmax-weighted centroid of initial results to '
              'adjust the query and search again. The method uses no relevance labels at inference, '
              'no extra encoder calls and no API calls. One refinement requires two corpus searches; '
              'two refinements require three. We tested four settings chosen before evaluation on '
              'the original encoder and all three earlier full-corpus2,000-step checkpoints.', '',
              '| Model | Feedback variant | Mean nDCG@10 | Difference versus same encoder |',
              '|---|---|---:|---|']
    for row in feedback['summary']:
        lines.append(f'| {row["model"]} | {row["variant"]} | {row["mean_ndcg"]:.6f} | {interval(row["paired_query_bootstrap"])} |')
    lines += ['', 'The paper-inspired default uses top3, interpolation0.5 and temperature0.05. '
              'The gentle variant uses interpolation0.2; top10 changes only candidate count; two_rounds '
              'repeats the default with fresh retrieval. Our implementation normalizes the weighted '
              'centroid before interpolation. This is a vector-only adaptation of '
              '[Test-Time Compute for Dense Retrieval](https://arxiv.org/html/2605.11374v4), '
              'not a reproduction of the paper’s whole program search. Its reported task dependence '
              'motivated evaluating the original and trained encoders separately.', '',
              '### What can a top-25 reranker recover?', '',
              'The oracle below uses known benchmark labels to reorder only the naturally retrieved '
              '25 candidates. It is a diagnostic upper bound under these incomplete labels, not a '
              'trained reranker result. Hit@25 is the fraction of queries with any known evidence '
              'among those candidates; recall@25 is the fraction of known evidence recovered.', '',
              '| Encoder | Inference | nDCG@10 | Hit@25 | Recall@25 | Oracle rerank-25 nDCG@10 |',
              '|---|---|---:|---:|---:|---:|']
    for row in coverage['summary']:
        lines.append(f'| {row["family"]} | {row["variant"]} | {row["ndcg10"]:.6f} | '
                     f'{row["hit25"]:.3%} | {row["recall25"]:.3%} | {row["oracle_rerank25_ndcg10"]:.6f} |')
    mixed_feedback = next(r['feedback_minus_plain'] for r in coverage_feedback['comparisons']
                          if r['family'] == 'mixed2000' and r['metric'] == 'ndcg10')
    lines += ['', 'The fixed feedback setting applied to the mixed2000 models gives an additional '
              + interval(mixed_feedback) + ' nDCG. That new composition was evaluated after the earlier '
              'feedback screen, so this unadjusted interval is exploratory. It also reduces hit@25 '
              'from56.63% to51.41%: higher mean ranking quality does not mean more queries have evidence '
              'available to a reranker. Full2000 plain has an oracle ceiling near0.293 versus actual0.138, '
              'but roughly43% of queries have no known evidence in their top25.', '',
              '### Keyword search and document context', '',
              'BM25 uses pinnedbm25s0.3.16, English stemming/stopwords, Lucene scoring,k1=1.5,b=0.75. '
              'The first512 index uses decoded text visible to the BGE tokenizer at its512-token limit; '
              'the full index uses the entire scraped text. Neither uses title/URL enrichment. '
              'The shortened index approximates equal visible content, not identical tokenization.', '',
              'Hybrid retrieval combines the top1,000 dense and lexical results with equal-weight '
              'reciprocal-rank fusion and fixedconstant60. The parameters were fixed before evaluation. '
              'Full-text lexical retrieval has a larger context budget than dense retrieval, so its gain '
              'must not be attributed to RL. Indexes and score arrays persist for reuse.', '',
              '| System | Text context / fusion | Mean nDCG@10 | Difference and95% query interval |',
              '|---|---|---:|---|']
    for row in lexical['summary']:
        name = row['model'] if row['model'] == 'bm25' or row['variant'] == 'baseline' else row['model'] + ' + BM25 RRF'
        lines.append(f'| {name} | {row["variant"]} | {row["mean_ndcg"]:.6f} | {interval(row["paired_query_bootstrap"])} |')
    lines += ['', 'For BM25, differences compare full text against first512. For hybrid rows, differences '
              'compare against that same dense encoder without fusion. Seed-based models are averaged '
              'per query first; all evaluations use the same83 validation queries. None of these intervals '
              'correct for repeated validation use or choosing among variants.', '',
              'Modal preempted the first indexing attempt. The resumed implementation saves text/token '
              'batches and index provenance, so a restart reuses completed work. A separate fresh '
              'container reloaded both saved indexes and reproduced all83 query score arrays exactly. '
              f'An independent functional-tokenizer check matched every term count on{lexical_tokenization["documents"]} '
              f'documents ({lexical_tokenization["tokens_checked"]:,} tokens), including random and high-scoring pages.', '',
              '### Frozen pretrained reranker', '',
              'The revision-pinned [BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) '
              'scored each natural top25 using query/document pairs truncated at512 or2048tokens. '
              'It received no relevance labels and had no parameter updates. Scores are cached by '
              'query/document pair and context limit; ties retain the original candidate order. '
              'No missing evidence was inserted into evaluation lists.', '',
              '| Retriever | Inference | Pair token limit | Before | After reranking | Difference and95% query interval |',
              '|---|---|---:|---:|---:|---|']
    for row in reranker['summary']:
        lines.append(f'| {row["family"]} | {row["variant"]} | {row["max_length"]} | '
                     f'{row["mean_before"]:.6f} | {row["mean_after"]:.6f} | {interval(row["paired_query_bootstrap"])} |')
    lines += ['', f'This scored{reranker["unique_pairs"]:,} unique validation query/document pairs '
              'at each context limit. The score caches and natural candidate lists persist separately '
              'from training data. All before-reranking values reproduce the earlier dense or lexical '
              f'controls within{reranker_verification["max_baseline_error"]:.2g}; reranked scores stay '
              'below their candidate oracles. The2048-token variant has a larger text budget than '
              'the dense index, and the tokenizers differ. These are inference comparisons, not gains from '
              'additional RL training.', '',
              '| Retriever | Inference | 2048-token minus512-token reranker nDCG and95% query interval |',
              '|---|---|---|']
    for row in reranker['context_comparisons']:
        lines.append(f'| {row["family"]} | {row["variant"]} | {interval(row["longer_minus_shorter"])} |')
    lines += ['', '![Training-length and inference-control results](mixed-inference-results.png)', '',
              '### Additional literature and verification', '',
              '[ARHN](https://arxiv.org/html/2604.11092v1) studies answer-grounded relabeling and filtering '
              'for contrastive retrievers. Its weaker labeler can reduce retrieval quality, which supports '
              'checking disagreements rather than assuming more labels are better. We did not adopt its '
              'contrastive loss or alter our immutable judge labels.', '',
              '[Expert-Guided Rubrics](https://arxiv.org/html/2610.26932v1) blends rubric and retrieval '
              'rewards for query rewriting and validates against expert judgments. That differs from '
              'our embedding policy; our repeated same-family judge agreement does not supply equivalent '
              'external validation. These papers motivate diagnostics, not claims that our judges are accurate.', '',
              '[Late Chunking](https://arxiv.org/abs/2409.04701) pools chunks after a long-context '
              'transformer has processed the document. Our512-token BGE encoder cannot reproduce it '
              'by simply averaging separately encoded windows; any such future control must be labeled '
              'ordinary chunking.', '',
              'Recent reranker literature adds a distinct question: does the selected set jointly '
              'cover the information need? [RubricRanker](https://arxiv.org/html/2608.03527v1) uses '
              'query-specific criteria for relevance, redundancy, consistency and source properties, '
              'with supervised cold start followed by RL. [AdaTutoRank](https://arxiv.org/html/2609.32472v1) '
              'adds adaptive guidance to address credit assignment within a selected set. Neither '
              'result validates our current pointwise judge; they motivate a separately versioned '
              'set-coverage experiment rather than silently changing existing labels.', '',
              '[RRPO](https://arxiv.org/html/2604.02091v1) samples documents without replacement from '
              'reranker scores and rewards downstream answers from a frozen reader. This provides '
              'a concrete future reranking action space, but reader-based rewards and its sequential '
              'optimization are different from our one-step Gaussian query policy. We have not '
              'implemented or reproduced it.', '',
              'The full57-test suite, Ruff and whitespace checks passed. The lexical dependency also '
              'passed a real retrieval smoke check before indexing. This batch made zero API calls; '
              'the total conservative API estimate remains$90.60. All166 final-test queries remain untouched.', '']
    complete_report = report + '\n\n' + '\n'.join(lines)
    start_marker, end_marker = '<!-- current-findings -->', '<!-- /current-findings -->'
    if start_marker in complete_report:
        before, rest = complete_report.split(start_marker, 1)
        complete_report = before + rest.split(end_marker, 1)[1].lstrip('\n')
    lexical_score = next(r['mean_ndcg'] for r in lexical['summary'] if r['model'] == 'bm25' and r['variant'] == 'full')
    lexical_short = next(r['mean_ndcg'] for r in lexical['summary'] if r['model'] == 'bm25' and r['variant'] == 'first512')
    hybrid_score = next(r['mean_ndcg'] for r in lexical['summary'] if r['model'] == 'binary2000' and r['variant'] == 'full')
    hybrid_short = next(r['mean_ndcg'] for r in lexical['summary'] if r['model'] == 'binary2000' and r['variant'] == 'first512')
    reranked_short = next(r['mean_after'] for r in reranker['summary'] if (r['family'], r['variant'], r['max_length']) == ('full2000', 'plain', 512))
    reranked_long = next(r['mean_after'] for r in reranker['summary'] if (r['family'], r['variant'], r['max_length']) == ('full2000', 'plain', 2048))
    overview = [start_marker, '## Current findings', '',
        'The clearest RL improvement came from relaxing the original strong anchor, increasing '
        'the learning rate, then using more/wider samples and more updates. LLM grading increased '
        'reward variation but has not beaten matched binary supervision. Pure fixed-pool encoder '
        'training overfit its lists and hurt whole-corpus retrieval.', '',
        '| System | Validation evidence nDCG@10 |', '|---|---:|',
        '| Original encoder | 0.037364 |',
        f'| Full-corpus RL, 2000 updates | {mixed["control_mean_ndcg"]:.6f} |',
        f'| Mixed-candidate RL, 2000 updates | {mixed["mean_ndcg"]:.6f} |',
        f'| BM25 using the first 512 tokens | {lexical_short:.6f} |',
        f'| Full-text BM25 | {lexical_score:.6f} |',
        f'| Full-corpus RL + first-512-token BM25 fusion | {hybrid_short:.6f} |',
        f'| Full-corpus RL + full-text BM25 fusion | {hybrid_score:.6f} |',
        f'| Full-corpus RL + frozen reranker, 512 tokens | {reranked_short:.6f} |',
        f'| Full-corpus RL + frozen reranker, 2048 tokens | {reranked_long:.6f} |', '',
        'The lexical and reranker rows change the inference system or document context; they are not '
        'additional RL gains. Trained systems average three seeds. All results reuse 83 validation '
        'queries; 166 final-test queries remain untouched. Small differences are exploratory. '
        'The detailed tables below include failures, both reranker context limits, and paired intervals.', '',
        '42 research runs and 84 checkpoint files persist in Modal. Conservative OpenAI API estimate: '
        '**$90.60**; this new batch used no paid API calls. Modal compute is separate.', end_marker, '']
    title, body = complete_report.split('\n', 1)
    (OUT / 'results.md').write_text(title + '\n\n' + '\n'.join(overview) + '\n' + body.lstrip('\n'))

    verification_marker = '## Longer mixed training and inference controls — 2026-11-06'
    verification = (ROOT / 'VERIFICATION.md').read_text().split(verification_marker)[0].rstrip()
    (ROOT / 'VERIFICATION.md').write_text(verification + '\n\n' + verification_marker + '\n\n'
        f'- Three predeclared mixed2,000-update runs completed, mean nDCG **{mixed["mean_ndcg"]:.7f}**, '
        f'versus matched full-corpus-only **{mixed["control_mean_ndcg"]:.7f}**.\n'
        '- Each run performs exactly1,800 full-corpus and200 projected binary-pool updates. Settings '
        'and reward provenance are checked against completed controls before launch.\n'
        '- Fixed-setting inference controls completed: soft-centroid feedback, first512/full-textBM25, '
        'and RRF60/top1000 fusion with original and three prior binary2000 models, followed by a '
        'frozenBGE top25 reranker at512/2048token pair limits. They are separate '
        'from RL training and use no labels to transform queries or rank documents.\n'
        '- 57 tests, Ruff and whitespace checks passed. Three new tests check feedback identity, '
        'normalization, deterministic retrieved-vector use and protocol validation. A lexical retrieval '
        'smoke check passed with the pinned dependency. An additional interrupted-tokenization test '
        'verified resumption, discarded partial writes, and score equality to ordinaryBM25 tokenization. '
        'Two candidate-coverage tests check the global ideal denominator and inability to recover '
        'relevant documents outside the candidate pool. Two reranking tests check candidate preservation, '
        'tie handling, duplicate rejection and the global denominator. Both persisted lexical indexes were reloaded '
        'in a fresh container and reproduced all validation score arrays within1e-6.\n'
        '- All166 test queries remain untouched;83 validation queries were reused for exploration. '
        'This batch uses no API calls. Detailed query intervals and every condition are in results.md.\n')

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    earlier = read('longer-summary.json')['mean_validation_by_step']
    steps = [500, 1000, 1500, 2000]
    axes[0].plot(steps, [earlier[str(s)] for s in steps], marker='o', label='Full corpus only')
    axes[0].plot(steps, [mixed['mean_validation_by_step'][str(s)] for s in steps], marker='o', label='90% corpus +10% pool')
    axes[0].set(title='Training: means across three seeds', xlabel='Updates', ylabel='Validation nDCG@10')
    axes[0].legend(fontsize=8)
    plot_rows = [('Original', .0373637006), ('RL, 2000', mixed['control_mean_ndcg'])]
    plot_rows += [('RL + feedback', next(r['mean_ndcg'] for r in feedback['summary'] if r['model'] == 'binary2000' and r['variant'] == 'paper_default'))]
    for model, variant, name in [('bm25', 'first512', 'BM25, first512'), ('bm25', 'full', 'BM25, full'),
                                 ('binary2000', 'first512', 'RL + first512 BM25'),
                                 ('binary2000', 'full', 'RL + full BM25')]:
        plot_rows.append((name, next(r['mean_ndcg'] for r in lexical['summary'] if r['model'] == model and r['variant'] == variant)))
    axes[1].barh([r[0] for r in plot_rows], [r[1] for r in plot_rows], color='#0072B2')
    axes[1].invert_yaxis()
    axes[1].set(title='Inference controls; context budgets differ', xlabel='Validation nDCG@10')
    families = [('original', 'plain', 'Original'), ('full2000', 'plain', 'RL corpus'),
                ('mixed2000', 'plain', 'RL mixed'), ('bm25', 'full', 'BM25 full'),
                ('full2000', 'rrf_full', 'RL + full BM25')]
    for offset, length, label, color in ((-.2, 512, 'Reranker512', '#0072B2'), (.2, 2048, 'Reranker2048', '#E69F00')):
        chosen = [next(r for r in reranker['summary'] if (r['family'], r['variant'], r['max_length']) == (f, v, length)) for f, v, _ in families]
        axes[2].barh([i + offset for i in range(len(families))], [r['mean_after'] for r in chosen], height=.35, label=label, color=color)
    axes[2].set_yticks(range(len(families)), [name for _, _, name in families])
    axes[2].invert_yaxis()
    axes[2].set(title='Frozen reranking of the same top25', xlabel='Validation nDCG@10')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('83 validation queries; untouched final test; exploratory findings', fontsize=10)
    fig.savefig(OUT / 'mixed-inference-results.png', dpi=170)
    fig.savefig(OUT / 'mixed-inference-results.pdf')
    plt.close(fig)

    publish_artifacts()


def publish_artifacts(snapshot_name='source-snapshot-mixed-longer-20261106.zip', extra_names=()):
    """Refresh the source archive and integrity manifest after report updates."""
    files = sorted({p for pattern in ('*.py', 'config*.json', 'requirements*.txt', 'pyproject.toml',
        'README.md', 'VERIFICATION.md', 'embedding_rl/*.py', 'tests/*.py', 'work/graded/*.py')
        for p in ROOT.glob(pattern) if p.is_file()})
    with zipfile.ZipFile(OUT / snapshot_name, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(ROOT)))
    shutil.copyfile(OUT / snapshot_name, OUT / 'source-snapshot.zip')
    write('source-snapshot-info.json', {'scope': 'Latest source; historical runs retain code_sha256; earlier source archives are preserved.',
        'files': len(files), 'secrets_included': False, 'current_training_code_sha256': hashlib.sha256(b''.join(
            p.name.encode() + p.read_bytes() for p in sorted((ROOT / 'embedding_rl').glob('*.py')))).hexdigest()})
    checkpoints = read('research-checkpoints.json')
    excluded = {'artifact-index.json', 'research-artifact-index.json'}
    index = {'verified_runs': len(checkpoints), 'checkpoint_files': 2 * len(checkpoints),
        'files': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded}}
    write('research-artifact-index.json', index)
    write('artifact-index.json', index)
    volume = modal.Volume.from_name('embedding-rl-data')
    with volume.batch_upload(force=True) as batch:
        for name in ('results.md', 'research-log.json', 'research-checkpoints.json', 'mixed-longer-summary.json',
                     'mixed-longer-records.json', 'mixed-inference-results.png', 'mixed-inference-results.pdf',
                     'source-snapshot.zip', snapshot_name, 'source-snapshot-info.json',
                     'research-artifact-index.json', 'artifact-index.json', *extra_names):
            batch.put_file(OUT / name, f'experiments/graded-20261106/{name}')
        for name in ('README.md', 'VERIFICATION.md'):
            batch.put_file(ROOT / name, f'experiments/graded-20261106/{name}')
    print(json.dumps({'published': str(OUT / 'results.md'), 'verified_runs': len(checkpoints), 'source_files': len(files)}))


if __name__ == '__main__':
    main()
