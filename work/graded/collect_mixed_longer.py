"""Collect final mixed-versus-full-corpus runs and the independent feedback control."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics

import modal

from embedding_rl.graded_evaluate import paired_bootstrap

OUT = Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')
VOLUME = modal.Volume.from_name('embedding-rl-data')


def fetch(remote, target):
    data = b''.join(VOLUME.read_file(remote))
    if not data:
        raise ValueError(f'Empty artifact: {remote}')
    parsed = json.loads(data) if target.endswith('.json') else None
    path = OUT / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return parsed


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


def queries(record):
    path = OUT / record['summary']['run_id'] / 'validation.json'
    data = json.loads(path.read_text())
    return {r['query_id']: r['ndcg@10'] for r in data['final']['per_query']}


def main():
    records = [fetch(f'experiments/alternating-longer-20261106/projected-seed{seed}.json',
                     f'alternating-longer-20261106/projected-seed{seed}.json') for seed in (42, 43, 44)]
    fetch('experiments/alternating-longer-20261106/protocol.json', 'alternating-longer-20261106/protocol.json')
    for name in ('comparison.json', 'protocol.json'):
        fetch(f'experiments/feedback-20261106/{name}', f'feedback-20261106/{name}')
    jobs = [(f'runs/{r["summary"]["run_id"]}/{name}', f'{r["summary"]["run_id"]}/{name}')
            for r in records for name in ('run.json', 'summary.json', 'metrics.jsonl', 'validation.json', 'training_probe.json')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pair: fetch(*pair), jobs))
    controls = json.loads((OUT / 'longer-records.json').read_text())
    controls = {r['settings']['training_seed']: r for r in controls}
    rows, old_queries, new_queries = [], [], []
    for record in records:
        seed = record['settings']['training_seed']
        control = controls[seed]
        if record['update_counts'] != {'full_corpus': 1800, 'candidate_pool': 200}:
            raise ValueError('Incorrect update counts')
        if json.dumps(record['settings'], sort_keys=True) != json.dumps(control['settings'], sort_keys=True):
            raise ValueError('Configuration mismatch')
        old, new = queries(control), queries(record)
        assert old.keys() == new.keys() and len(old) == 83
        old_queries.append(old)
        new_queries.append(new)
        steps = {r['step']: r for r in record['evaluations']}
        rows.append({'seed': seed, 'run_id': record['summary']['run_id'],
                     'control_run_id': control['summary']['run_id'],
                     'final_ndcg': record['summary']['final_validation_ndcg'],
                     'control_final_ndcg': control['summary']['final_validation_ndcg'],
                     'seed_delta': record['summary']['final_validation_ndcg'] - control['summary']['final_validation_ndcg'],
                     'step500_difference_from_prior': record['step500_difference'],
                     'validation_by_step': {str(s): steps[s]['validation/ndcg@10'] for s in (500, 1000, 1500, 2000)},
                     'fixed_train_by_step': {str(s): steps[s]['fixed_train/ndcg@10'] for s in (500, 1000, 1500, 2000)},
                     'paired_query_bootstrap': paired_bootstrap(list(old.values()), [new[q] for q in old]),
                     'wandb_url': record['summary']['wandb_url']})
    query_ids = list(old_queries[0])
    old = [statistics.mean(v[q] for v in old_queries) for q in query_ids]
    new = [statistics.mean(v[q] for v in new_queries) for q in query_ids]
    result = {'runs': rows, 'mean_ndcg': statistics.mean(r['final_ndcg'] for r in rows),
              'control_mean_ndcg': statistics.mean(r['control_final_ndcg'] for r in rows),
              'seed_std': statistics.stdev(r['final_ndcg'] for r in rows),
              'mean_validation_by_step': {str(s): statistics.mean(r['validation_by_step'][str(s)] for r in rows)
                                          for s in (500, 1000, 1500, 2000)},
              'paired_query_bootstrap': paired_bootstrap(old, new),
              'validation_queries': 83, 'test_evaluated': False, 'api_calls': 0,
              'selection': 'Recipe selected on earlier500-step validation means; all three2000-step seeds predeclared. Report final rather than peak score.',
              'caveat': 'Query bootstrap averages seeds first and does not correct for repeated validation selection. Equal updates are not exactly equal compute.'}
    write('mixed-longer-summary.json', result)
    write('mixed-longer-records.json', records)
    existing = {r['run_id']: r for r in json.loads((OUT / 'research-checkpoints.json').read_text())}
    for row in rows:
        run_id = row['run_id']
        weights = [{'path': e.path, 'size': e.size} for e in VOLUME.iterdir(f'runs/{run_id}')
                   if Path(e.path).name in ('best.pt', 'last.pt')]
        assert len(weights) == 2 and all(w['size'] > 1_000_000 for w in weights)
        existing[run_id] = {'run_id': run_id, 'weights': weights}
    write('research-checkpoints.json', list(existing.values()))
    with VOLUME.batch_upload(force=True) as batch:
        for name in ('mixed-longer-summary.json', 'mixed-longer-records.json'):
            batch.put_file(OUT / name, f'experiments/alternating-longer-20261106/{name}')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
