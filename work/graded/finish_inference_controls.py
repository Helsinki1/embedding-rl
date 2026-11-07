"""Finish dependent controls once producer artifacts are committed; never overlap scorers."""
import json
from pathlib import Path
import subprocess
import time

import modal

ROOT = Path('/home/davidx/Downloads/embedding-rl')
OUT = Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')
VOLUME = modal.Volume.from_name('embedding-rl-data')


def fetch(experiment, name):
    data = b''.join(VOLUME.read_file(f'experiments/{experiment}/{name}'))
    result = json.loads(data)
    folder = OUT / experiment
    folder.mkdir(exist_ok=True)
    (folder / name).write_bytes(data)
    return result


def wait_for(experiment, name):
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        try:
            return fetch(experiment, name)
        except FileNotFoundError:
            time.sleep(30)
    raise TimeoutError(f'Producer did not finish: {experiment}/{name}')


def command(args, name):
    print('START ' + name, flush=True)
    with (ROOT / 'work/graded' / f'{name}.log').open('w') as log:
        subprocess.run(args, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    print('DONE ' + name, flush=True)


def main():
    lexical = wait_for('lexical-20261106', 'comparison.json')
    command(['modal', 'run', '--detach', 'modal_lexical.py', '--verify-only'], 'lexical-verification')
    for name in ('protocol.json', 'verification.json'):
        fetch('lexical-20261106', name)
    wait_for('frozen-reranker-20261106', 'dense-comparison.json')
    command(['modal', 'run', '--detach', 'modal_reranker_control.py', '--include-lexical'], 'reranker-complete')
    reranker = fetch('frozen-reranker-20261106', 'comparison.json')
    for name in ('protocol.json', 'complete-candidates.json'):
        fetch('frozen-reranker-20261106', name)
    coverage = json.loads((OUT / 'candidate-coverage-20261106/comparison.json').read_text())
    expected = {(r['family'], r['variant'], r['seed']): r['per_query']['ndcg10'] for r in coverage['runs']}
    for row in lexical['runs']:
        if row['model'] == 'bm25' and row['variant'] == 'full':
            expected[('bm25', 'full', None)] = row['ndcg']
        if row['model'] == 'binary2000' and row['variant'] == 'full':
            expected[('full2000', 'rrf_full', row['seed'])] = row['ndcg']
    errors = []
    for row in reranker['runs']:
        baseline = expected[(row['family'], row['variant'], row['seed'])]
        if len(baseline) != 83 or len(row['before']) != 83:
            raise ValueError('Wrong query count')
        errors.extend(abs(a - b) for a, b in zip(baseline, row['before'], strict=True))
        if any(a > o + 1e-6 for a, o in zip(row['after'], row['oracle'], strict=True)):
            raise ValueError('Exceeded oracle')
    if max(errors) > 1e-6 or len(reranker['runs']) != 36 or not reranker['complete']:
        raise ValueError('Baseline mismatch or incomplete reranker control')
    verification = {'max_baseline_error': max(errors), 'systems': 18, 'contexts': [512, 2048],
                    'validation_queries': 83, 'all_candidate_oracle_checks_passed': True,
                    'test_evaluated': False, 'api_calls': 0}
    path = OUT / 'frozen-reranker-20261106/verification.json'
    path.write_text(json.dumps(verification, indent=2) + '\n')
    with VOLUME.batch_upload(force=True) as batch:
        batch.put_file(path, 'experiments/frozen-reranker-20261106/verification.json')
    command(['python3', 'work/graded/publish_mixed_results.py'], 'publish-mixed-results')
    print('INFERENCE_CONTROLS_PUBLISHED', flush=True)


if __name__ == '__main__':
    main()
