"""Collect completed, cached-label candidate experiments and matched controls."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics

import modal

from embedding_rl.graded_evaluate import paired_bootstrap

OUT = Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')
VOLUME = modal.Volume.from_name('embedding-rl-data')
EXPERIMENTS = ('candidate-cmp-20261106', 'candidate-cmp-anchor05-20261106',
               'alternating-pool-20261106', 'alternating-binary-pool-20261106')


def fetch(remote, local):
    target = OUT / local
    target.parent.mkdir(parents=True, exist_ok=True)
    content = b''.join(VOLUME.read_file(remote))
    if not content:
        raise ValueError(f'Empty artifact: {remote}; wait until the producing app has committed')
    value = json.loads(content) if target.suffix == '.json' else None
    target.write_bytes(content)
    return value


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def per_query(records):
    measurements = []
    for record in records:
        data = json.loads((OUT / record['summary']['run_id'] / 'validation.json').read_text())
        measurements.append({r['query_id']: r['ndcg@10'] for r in data['final']['per_query']})
    assert all(set(m) == set(measurements[0]) for m in measurements)
    return {q: statistics.mean(m[q] for m in measurements) for q in measurements[0]}


def main():
    comparisons = {}
    for experiment in EXPERIMENTS:
        for name in ('comparison.json', 'protocol.json', 'probe.json'):
            value = fetch(f'experiments/{experiment}/{name}', f'{experiment}/{name}')
            if name == 'comparison.json':
                comparisons[experiment] = value
    all_runs = [r for comparison in comparisons.values() for r in comparison['runs']]
    jobs = [(f'runs/{r["summary"]["run_id"]}/{name}', f'{r["summary"]["run_id"]}/{name}')
            for r in all_runs for name in
            ('run.json', 'summary.json', 'validation.json', 'training_probe.json', 'metrics.jsonl')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pair: fetch(*pair), jobs))
    reference = [json.loads(b''.join(VOLUME.read_file(
        f'experiments/unanchored-knobs-20261106/both-seed{seed}.json'))) for seed in (42, 43, 44)]
    base = per_query(reference)
    alternating = comparisons['alternating-pool-20261106']
    paired = {}
    for label in ('unprojected', 'projected'):
        selected = [r for r in alternating['runs'] if r['label'] == label]
        assert {r['settings']['training_seed'] for r in selected} == {42, 43, 44}
        for r in selected:
            assert r['update_counts'] == {'full_corpus': 450, 'candidate_pool': 50}
        values = per_query(selected)
        assert set(values) == set(base) and len(values) == 83
        controls = {r['settings']['training_seed']: r for r in reference}
        paired[label] = {
            'query_bootstrap_vs_full_only': paired_bootstrap(list(base.values()), [values[q] for q in base]),
            'seed_deltas_vs_full_only': {r['settings']['training_seed']:
                r['summary']['final_validation_ndcg'] - controls[r['settings']['training_seed']]['summary']['final_validation_ndcg']
                for r in selected},
        }
    summary = {'experiments': {name: {k: v for k, v in data.items() if k != 'runs'}
                               for name, data in comparisons.items()},
               'alternating_vs_full_only': paired,
               'full_only_500_mean': statistics.mean(r['summary']['final_validation_ndcg'] for r in reference),
               'new_training_runs': len(all_runs), 'api_training_calls': 0,
               'validation_queries': 83, 'test_evaluated': False,
               'caveat': 'Query intervals average seeds first and do not correct for validation selection. Fixed-pool and full-corpus rewards differ; alternating train/reward_mean is not one comparable objective curve.'}
    label_comparison = {}
    binary_runs = comparisons['alternating-binary-pool-20261106']['runs']
    for label in ('unprojected', 'projected'):
        binary = [r for r in binary_runs if r['label'] == label]
        judged = [r for r in alternating['runs'] if r['label'] == label]
        assert {r['settings']['training_seed'] for r in binary} == {42, 43, 44}
        assert all(r['update_counts'] == {'full_corpus': 450, 'candidate_pool': 50} for r in binary)
        first, second = per_query(binary), per_query(judged)
        assert first.keys() == second.keys() == base.keys()
        controls = {r['settings']['training_seed']: r for r in binary}
        label_comparison[label] = {
            'llm_minus_binary_query_bootstrap': paired_bootstrap(list(first.values()), [second[q] for q in first]),
            'binary_minus_full_only_query_bootstrap': paired_bootstrap(list(base.values()), [first[q] for q in base]),
            'llm_minus_binary_seed_deltas': {r['settings']['training_seed']:
                r['summary']['final_validation_ndcg'] - controls[r['settings']['training_seed']]['summary']['final_validation_ndcg']
                for r in judged},
        }
    summary['matched_label_source_comparison'] = label_comparison
    write('candidate-research-summary.json', summary)

    previous = {r['run_id']: r for r in json.loads((OUT / 'research-checkpoints.json').read_text())}
    for r in all_runs:
        run_id = r['summary']['run_id']
        weights = [{'path': e.path, 'size': e.size} for e in VOLUME.iterdir(f'runs/{run_id}')
                   if Path(e.path).name in ('best.pt', 'last.pt')]
        assert len(weights) == 2 and all(w['size'] > 1_000_000 for w in weights)
        previous[run_id] = {'run_id': run_id, 'weights': weights}
    write('research-checkpoints.json', list(previous.values()))
    budget = fetch('experiments/graded-20261106/judge/budget.json', 'budget.json')
    fetch('experiments/graded-20261106/training-label-agreement.json', 'training-label-agreement.json')
    print(json.dumps({'summary': summary, 'verified_runs': len(previous),
                      'api_upper_estimate': budget['charged_or_reserved_usd']}, indent=2))


if __name__ == '__main__':
    main()
