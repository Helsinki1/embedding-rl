"""Publish completed candidate controls, judge audit, and reproducibility artifacts."""
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
    summary = read('candidate-research-summary.json')
    audit = read('winner-audit-summary.json')
    label_audit = read('training-label-agreement.json')
    budget = read('budget.json')['charged_or_reserved_usd']
    short_names = {
        'candidate-cmp-20261106': 'Fixed judged pool only, no anchor',
        'candidate-cmp-anchor05-20261106': 'Fixed judged pool only, cosine anchor 0.5',
        'alternating-pool-20261106': '90% full corpus + 10% judged pool',
        'alternating-binary-pool-20261106': '90% full corpus + 10% binary pool',
    }
    lines = [
        '## Overnight follow-up: candidate-pool experiments', '',
        'All runs below use 500 updates, 32 Gaussian samples, radius 0.45 and learning rate 5e-6. '
        'They still use one-step RL. The fixed candidate lists contain the original top 25 plus known '
        'training evidence (mean 30.62 documents); deployment evaluation searches all 100,195 documents '
        'for the same 83 validation queries. All 166 final-test queries remain untouched.', '',
        '| Training setup | Estimator | Seeds | Final full-corpus nDCG@10 | Fixed training-pool nDCG after |',
        '|---|---|---:|---:|---:|',
        f'| Full-corpus binary control | Ordinary RLOO | 3 | {summary["full_only_500_mean"]:.6f} | — |',
    ]
    for name, data in summary['experiments'].items():
        for row in data['comparison']:
            lines.append(f'| {short_names[name]} | {row["condition"]} | {len(row["seeds"])} | '
                         f'{row["mean_ndcg"]:.6f} | {row["fixed_training_pool_after"]:.6f} |')
    lines += [
        '', 'Candidate-pool training scores use that experiment’s labels and denominator. They are '
        'in-sample diagnostics, not comparable to full-corpus validation or across label sources.', '',
        'The unanchored fixed-pool models learned their small training lists but collapsed on full-corpus '
        'retrieval: ordinary/projected final scores 0.0153/0.0039, despite pool scores 0.6424/0.7285. '
        'A softer anchor prevented that collapse (0.1044/0.0896), but projection still did not improve '
        'the final result. Both were single-seed screens; neither met its predeclared repeat criterion.', '',
        'In the numerical probe, conditional projection reduced variance of query-embedding tangent '
        'gradients by 13.19× across 64 training queries. This does not measure full-network gradient '
        'variance or guarantee better retrieval. Projection changes only the score-function gradient; '
        'training rewards use the original sampled actions. It is never applied to full-corpus reward updates.', '',
        'Mixed training performs exactly 450 full-corpus binary updates and 50 candidate-pool updates. '
        'Both judged variants improved over their matched full-corpus controls in all three seeds. '
        'The query-bootstrap intervals below still include zero; this is a promising validation trend, '
        'not a confirmed generalization gain. The projected variant beat the ordinary mixed variant '
        'in two of three seeds. Training reward logs alternate two objectives and should not be treated '
        'as one comparable reward curve.', '',
        '| Matched comparison | Mean difference and descriptive 95% query interval |',
        '|---|---|',
    ]
    for label, data in summary['alternating_vs_full_only'].items():
        lines.append(f'| Mixed judged {label} − full only | {interval(data["query_bootstrap_vs_full_only"])} |')
    for label, data in summary['matched_label_source_comparison'].items():
        lines.append(f'| Mixed judged − mixed binary, {label} | {interval(data["llm_minus_binary_query_bootstrap"])} |')
        lines.append(f'| Mixed binary {label} − full only | {interval(data["binary_minus_full_only_query_bootstrap"])} |')
    lines += [
        '', 'The binary-pool controls use identical candidate lists, schedules, seeds and estimators; '
        'only pool labels change. Benchmark-unjudged pages explicitly receive zero in that control. '
        'This benchmark convention is never used as a fallback for missing LLM judgments. All six '
        'binary controls were scheduled before observing their outcomes. These comparisons test whether '
        'label source helps beyond the mixed training setup; they do not isolate granularity from every '
        'other difference between benchmark and LLM judgments.', '',
        'A training-only audit of all 17,789 candidate pairs found 3,519 known evidence pages: '
        '472 were graded irrelevant, 163 topical, 1,831 useful partial evidence and 1,053 decisive. '
        'Thus 18.0% of known evidence pages receive grade0/1. Among 14,270 benchmark-unjudged pages, '
        'the judge assigns useful grades2/3 to260. Only4 of581 training queries lack any known evidence '
        'with grade2+. These counts reveal differing supervision, not judge accuracy: unjudged does '
        'not mean irrelevant, and the grader sees selected excerpts rather than complete pages.', '',
        'All intervals average seeds per query before bootstrapping queries. They do not account for '
        'the repeated use of validation data to choose experiments. The earlier 2,000-step binary '
        'mean of 0.138436 remains a separate, longer-budget result.', '',
        '![Candidate training and validation comparison](candidate-results.png)', '',
        'The projection idea is adapted from [RELER](https://arxiv.org/html/2610.07731v1), using a '
        'Gaussian policy rather than its vMF policy. The paper explicitly excludes shortlist projection '
        'from its full-index setting. [ANCE](https://arxiv.org/abs/2007.00808) motivates checking whether '
        'training candidates represent the competing documents encountered at retrieval time. ANCE '
        'uses contrastive learning; our experiments borrow this diagnostic insight while retaining RL. '
        'Candidate mismatch is a plausible explanation for our collapse, not an isolated causal finding.', '',
        '## Audit of the stronger binary models', '',
        'The same 12 preselected validation queries were audited on a shared expanded pool of 356 pages. '
        'The original encoder, earlier binary control and three final 2,000-step models were compared. '
        'Because the pool expanded, these nDCGs should be compared within this audit, not against the '
        'earlier 19-model audit.', '',
        '| Model | Primary judged nDCG@10 | Stronger judged nDCG@10 |',
        '|---|---:|---:|',
    ]
    for row in audit['comparison']:
        lines.append(f'| {row["condition"]} | {row["primary_ndcg12"]:.6f} | {row["stronger_ndcg12"]:.6f} |')
    lines += ['', 'The stronger judge’s mean gain for the 2,000-step models versus the original encoder was '
              + interval(audit['paired_query_bootstrap']['stronger_audit/llm_ndcg@10'])
              + '. Its interval includes zero. Twelve queries and two judges from the same model family '
              'are insufficient to establish relevance improvement. The strict metric excluding topical-only '
              'pages points in the same direction, with similarly wide uncertainty.', '',
              f'This audit added a conservative ${audit["api_increment"]:.2f}; the task-wide API estimate '
              f'is now **${budget:.2f}** against the shared $95 ledger ceiling and $100 authorization. '
              'All candidate training experiments made zero API calls. Modal compute is separate.', '']
    marker = '## Overnight follow-up: candidate-pool experiments'
    report = (OUT / 'results.md').read_text().split(marker)[0].rstrip()
    report = report.replace('**$87.75**', f'**${budget:.2f}**').replace('41 automated tests passed', '49 automated tests passed')
    (OUT / 'results.md').write_text(report + '\n\n' + '\n'.join(lines))

    log = read('research-log.json')
    log['status'] = 'Candidate and label-source controls complete; overnight continuation remains active'
    log['constraints']['last_api_upper_estimate_usd'] = budget
    log['literature'] = [r for r in log['literature'] if r['url'] != 'https://arxiv.org/abs/2007.00808']
    log['literature'].append({
        'title': 'Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval',
        'url': 'https://arxiv.org/abs/2007.00808',
        'lesson': 'Training candidates that miss current full-index competitors can mislead retrieval training. Motivated mixed full-corpus/pool RL after fixed-pool collapse. ANCE is contrastive; our experiments remain RL. Our observed mismatch explanation is an inference.'})
    for name in short_names:
        log['active_experiments'][name] = {'status': 'complete', 'artifact': f'{name}/comparison.json',
            'seeds': [42] if name.startswith('candidate-') else [42, 43, 44], 'steps': 500, 'api_calls': 0}
    log['active_experiments']['winner-audit-20261106'] = {'status': 'complete', 'validation_queries': 12,
        'incremental_api_upper_estimate': audit['api_increment'], 'artifact': 'winner-audit-summary.json'}
    log['candidate_findings'] = summary
    log['winner_audit'] = audit
    log['training_label_agreement'] = label_audit
    log['completed_candidate_hypothesis'] = log.pop('next_no_api_hypothesis', log.get('completed_candidate_hypothesis', {}))
    log['next_steps'] = [
        'Do not duplicate completed candidate or judge experiments. Compare matched binary/LLM pool controls before attributing any gain to the LLM.',
        'If continuing before17:00UTC, predeclare a bounded2000-step mixed-versus-existing-full-corpus comparison. Choose label source using completed controls and report the selection. Use three seeds and final checkpoints; no new API calls are needed.',
        'Retain the166 untouched test queries for a separately frozen final evaluation; do not tune on them.',
        'Pause the overnight heartbeat and produce a final report at17:00UTC on2026-11-06.']
    choices = [(row['mean_ndcg'], name, row['condition'])
               for name in ('alternating-pool-20261106', 'alternating-binary-pool-20261106')
               for row in summary['experiments'][name]['comparison']]
    score, selected_experiment, estimator = max(choices)
    log['next_no_api_hypothesis'] = {
        'name': 'Longer training of the selected mixed recipe',
        'selection_rule': 'Highest three-seed mean final500-update validation nDCG among the four completed mixed conditions; selection is exploratory.',
        'selected_experiment': selected_experiment, 'selected_estimator': estimator,
        'grade_source': 'benchmark' if 'binary' in selected_experiment else 'llm',
        'selected_mean_ndcg500': score, 'experiment_id': 'alternating-longer-20261106',
        'steps': 2000, 'seeds': [42, 43, 44], 'pool_every': 10, 'samples': 32,
        'exploration_radius': .45, 'learning_rate': 5e-6, 'kl_weight': 0,
        'reference': 'All three completed longer-20261106 full-corpus binary runs at2000updates; mean0.1384358406',
        'protocol': 'Fresh runs from the original encoder. Predeclare all three seeds and final-step comparison before launch. Save actual500/1000/1500/2000metrics, because freshCUDA runs may differ slightly. Do not project full-corpus reward updates.',
        'api_calls': 0, 'final_test_queries': 'untouched',
        'additional_diagnostic': 'If useful later, investigate judge/benchmark disagreements using the training-only label audit. Do not change immutable labels without a separately versioned rubric and experiment.'}
    write('research-log.json', log)

    verification_marker = '## Candidate-pool projection and matched label controls — 2026-11-06'
    verification = (ROOT / 'VERIFICATION.md').read_text().split(verification_marker)[0].rstrip()
    details = [verification_marker, '',
        '- 49 tests, Ruff and whitespace checks passed before the binary-pool controls launched. '
        'Tests cover conditional score gradients, Monte Carlo variance, redundant candidate spans, '
        'complete immutable LLM labels, training-only access, full-corpus gradient preservation, '
        'and identical-candidate binary label controls.',
        '- All candidate runs use one-step Gaussian RLOO. Conditional projection is restricted to '
        'candidate-pool updates. Actual rewards always use the original sampled actions.',
        '- The unanchored fixed-list screen collapsed on full-corpus validation despite improving its '
        'training-list metric. A cosine0.5 anchor prevented collapse but projection did not beat its '
        'matched final checkpoint. Both are single-seed negative screens.',
        '- Mixed runs use exactly450 full-corpus updates and50 pool updates. Each label source and '
        'estimator was repeated at seeds42/43/44; binary controls were all predeclared.',
    ]
    for name in ('alternating-pool-20261106', 'alternating-binary-pool-20261106'):
        for row in summary['experiments'][name]['comparison']:
            details.append(f'- {name}/{row["condition"]}: final validation mean **{row["mean_ndcg"]:.7f}**, '
                           f'seed standard deviation **{row["seed_std"]:.7f}**.')
    details += [f'- The12-query stronger-judge audit favors the earlier2000-step binary models in mean '
                f'(.1091→.1816), but its query-bootstrap interval includes zero. '
                f'Total conservative API estimate **${budget:.4f}**; candidate training used no API calls.',
                '- All166 test queries remain untouched. Repeated validation selection prevents treating '
                'these as final-test claims. See results.md and candidate-research-summary.json in the '
                'private artifact directory for matched bootstrap intervals and run links.', '']
    (ROOT / 'VERIFICATION.md').write_text(verification + '\n\n' + '\n'.join(details))
    readme_marker = '## Cached candidate-pool RL experiments'
    readme = (ROOT / 'README.md').read_text().split(readme_marker)[0].rstrip()
    readme = readme.replace('This retains the existing Gaussian policy, rather than implementing RELER\'s von Mises–Fisher\nsampler or conditional-mean projection.',
                            'The default full-corpus recipe retains the Gaussian policy and does not use RELER\'s\nvon Mises–Fisher sampler or shortlist projection.')
    readme = readme.replace('Use the InfoNCE baseline and validation\nresults to diagnose this before changing the reward.',
                            'Use validation retrieval and sampling\ndiagnostics to check whether a reward change is useful.')
    (ROOT / 'README.md').write_text(readme + '\n\n' + readme_marker + '\n\n'
        'The completed `modal_candidate.py` experiments reuse frozen training pools (original top25\n'
        'plus known evidence). `CachedPoolReward` requires complete cached LLM grades; the explicit\n'
        '`benchmark` control uses binary benchmark labels on the identical lists. All evaluation still\n'
        'searches the full corpus, and final-test queries are excluded.\n\n'
        'With `--pool-every 10`, nine updates use full-corpus binary rewards and one uses the pool.\n'
        'Projection averages away Gaussian noise outside the fixed pool span only on pool updates.\n'
        'It changes the policy-gradient estimator, not the sampled actions used for rewards. Applying\n'
        'shortlist projection to a full-corpus reward would discard relevant retrieval directions.\n\n'
        'The pool-only recipe overfit its lists and regressed on full-corpus validation. Mixed training\n'
        'and identical-list binary controls were compared across three seeds; see `VERIFICATION.md`.\n'
        'The optional `--grade-source benchmark --repeat-all` runs all six binary control jobs.\n'
        'Always use a new experiment ID for a changed configuration, label source or implementation.\n'
        'Existing markers validate settings and supervision before reuse.\n\n'
        'The training loss remains RL throughout these experiments; no InfoNCE baseline was run.\n'
        'The 49-test suite covers projection, label-source isolation, budget persistence and split boundaries.\n')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for experiment, short, style in [('candidate-cmp-20261106', 'Pool only', '-'),
                                      ('candidate-cmp-anchor05-20261106', 'Pool + anchor', '--')]:
        for r in read(f'{experiment}/comparison.json')['runs']:
            points = r['evaluations']
            axes[0].plot([v['step'] for v in points], [v['validation/ndcg@10'] for v in points],
                         style, marker='o', markersize=3, label=f'{short}, {r["label"]}')
    axes[0].axhline(.037363700568675995, color='gray', linewidth=1, label='Original encoder')
    axes[0].set(title='Fixed pools can fit while retrieval collapses', xlabel='Training updates',
                ylabel='Full-corpus validation nDCG@10', ylim=(0, .15))
    axes[0].legend(fontsize=7, loc='upper left')
    rows = [('Full only', [.12343839555978775, .11053347587585449, .12353862822055817])]
    for experiment, prefix in [('alternating-pool-20261106', 'LLM'),
                                ('alternating-binary-pool-20261106', 'Binary')]:
        rows.extend((f'{prefix}\n{r["condition"]}', r['final_values'])
                    for r in summary['experiments'][experiment]['comparison'])
    for i, (name, values) in enumerate(rows):
        axes[1].scatter([i - .08, i, i + .08], values, s=22, color='#0072B2')
        mean = sum(values) / len(values)
        axes[1].plot([i - .2, i + .2], [mean, mean], color='#D55E00', linewidth=2.5)
    axes[1].set(xticks=range(len(rows)), xticklabels=[r[0] for r in rows],
                title='Matched 500-update runs: dots are seeds', ylabel='Full-corpus validation nDCG@10')
    axes[1].tick_params(axis='x', labelsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('83 validation queries; final test untouched; exploratory comparisons', fontsize=11)
    fig.savefig(OUT / 'candidate-results.png', dpi=170)
    fig.savefig(OUT / 'candidate-results.pdf')
    plt.close(fig)

    checkpoint_runs = read('research-checkpoints.json')
    source_files = sorted({p for pattern in ('*.py', 'config*.json', 'requirements*.txt', 'pyproject.toml',
        'README.md', 'VERIFICATION.md', 'embedding_rl/*.py', 'tests/*.py', 'work/graded/*.py')
        for p in ROOT.glob(pattern) if p.is_file()})
    previous_source = OUT / 'source-snapshot-before-candidate-20261106.zip'
    if not previous_source.exists():
        shutil.copyfile(OUT / 'source-snapshot.zip', previous_source)
    with zipfile.ZipFile(OUT / 'source-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, str(path.relative_to(ROOT)))
    shutil.copyfile(OUT / 'source-snapshot.zip', OUT / 'source-snapshot-candidate-20261106.zip')
    write('source-snapshot-info.json', {'scope': 'Latest source after completed candidate experiments; historical runs retain their own code_sha256. This is not a claim that all earlier runs used identical source.',
        'files': len(source_files), 'secrets_included': False,
        'current_training_code_sha256': hashlib.sha256(b''.join(p.name.encode() + p.read_bytes()
            for p in sorted((ROOT / 'embedding_rl').glob('*.py')))).hexdigest()})
    excluded = {'artifact-index.json', 'research-artifact-index.json'}
    index = {'verified_runs': len(checkpoint_runs), 'checkpoint_files': 2 * len(checkpoint_runs),
             'files': {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded}}
    write('research-artifact-index.json', index)
    write('artifact-index.json', index)
    volume = modal.Volume.from_name('embedding-rl-data')
    with volume.batch_upload(force=True) as batch:
        for name in ('results.md', 'research-log.json', 'budget.json', 'research-checkpoints.json',
                     'candidate-research-summary.json', 'winner-audit-summary.json', 'source-snapshot.zip',
                     'training-label-agreement.json',
                     'source-snapshot-info.json', 'candidate-results.png', 'candidate-results.pdf',
                     'source-snapshot-before-candidate-20261106.zip', 'source-snapshot-candidate-20261106.zip',
                     'research-artifact-index.json', 'artifact-index.json'):
            batch.put_file(OUT / name, f'experiments/graded-20261106/{name}')
        for name in ('README.md', 'VERIFICATION.md'):
            batch.put_file(ROOT / name, f'experiments/graded-20261106/{name}')
    print(json.dumps({'published': str(OUT / 'results.md'), 'verified_runs': len(checkpoint_runs),
                      'source_files': len(source_files), 'api_upper_estimate': budget}))


if __name__ == '__main__':
    main()
