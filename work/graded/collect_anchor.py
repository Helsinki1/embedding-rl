"""Persist completed anchor results and reuse the exact matched sampling controls."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics

import modal

OUT = Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')
VOLUME = modal.Volume.from_name('embedding-rl-data')
conditions = [('cosine_anchor_0.5', 42), ('no_anchor', 42),
              ('cosine_anchor_0.5_lr5e-6', 42), ('no_anchor_lr5e-6', 42),
              ('no_anchor_lr5e-6', 43), ('no_anchor_lr5e-6', 44)]


def read(path):
    return json.loads(b''.join(VOLUME.read_file(path)))


records = [read(f'experiments/anchor-20261106/{label}-seed{seed}.json')
           for label, seed in conditions]
jobs = [(row['summary']['run_id'], name) for row in records for name in
        ('summary.json', 'run.json', 'validation.json', 'training_probe.json', 'metrics.jsonl')]


def download(job):
    run, name = job
    data = b''.join(VOLUME.read_file(f'runs/{run}/{name}'))
    path = OUT / run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'path': str(path.relative_to(OUT)), 'sha256': hashlib.sha256(data).hexdigest()}


with ThreadPoolExecutor(max_workers=4) as pool:
    files = list(pool.map(download, jobs))
selected = [r for r in records if r['label'] == 'no_anchor_lr5e-6']
values = [r['summary']['final_validation_ndcg'] for r in selected]
result = {'runs': records, 'selected_condition': 'no_anchor_lr5e-6',
          'seeds': [42, 43, 44], 'final_ndcg_mean': statistics.mean(values),
          'final_ndcg_sample_std': statistics.stdev(values), 'final_ndcg_values': values,
          'previous_control_mean': 0.03742847964167595,
          'original_encoder_ndcg': 0.037363700568675995,
          'validation_queries': 83, 'test_evaluated': False, 'api_calls': 0,
          'caveat': 'Exploratory validation selection; seed 42 screened four conditions. No final-test claim.',
          'files': files}
(OUT / 'anchor-comparison.json').write_text(json.dumps(result, indent=2)+'\n')

# These three 500-step runs have exactly the settings needed for the new grid control.
expected = json.loads(Path('config.unanchored-followup.json').read_text())
controls = []
for row in selected:
    settings = dict(expected, training_seed=row['settings']['training_seed'])
    if row['settings'] != settings:
        raise ValueError('Control reuse would mix training configurations')
    row = dict(row, label='control', reused_from='anchor-20261106/no_anchor_lr5e-6')
    path = OUT / f"unanchored-control-seed{settings['training_seed']}.json"
    path.write_text(json.dumps(row, indent=2)+'\n')
    controls.append((path, f"experiments/unanchored-knobs-20261106/control-seed{settings['training_seed']}.json"))
with VOLUME.batch_upload(force=True) as batch:
    for path, remote in controls:
        batch.put_file(path, remote)
    for name in ('anchor-comparison.json', 'research-log.json'):
        batch.put_file(OUT/name, f'experiments/anchor-20261106/{name}')
    batch.put_file('config.unanchored-followup.json', 'experiments/unanchored-knobs-20261106/config.json')
    batch.put_file('modal_anchor.py', 'experiments/anchor-20261106/modal_anchor.py')
print(json.dumps({k: v for k, v in result.items() if k not in {'runs', 'files'}}, indent=2))
