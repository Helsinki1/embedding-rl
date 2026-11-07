"""Collect all three fixed-horizon runs, including their actual step-500 measurements."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics

import modal

OUT = Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')
VOLUME = modal.Volume.from_name('embedding-rl-data')
EXPERIMENT = 'longer-20261106'
records = [json.loads(b''.join(VOLUME.read_file(f'experiments/{EXPERIMENT}/both-2000-seed{seed}.json')))
           for seed in (42,43,44)]


def download(job):
    run, name = job
    path = OUT/run/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b''.join(VOLUME.read_file(f'runs/{run}/{name}')))


jobs = [(r['summary']['run_id'],name) for r in records for name in
        ('summary.json','run.json','validation.json','training_probe.json','metrics.jsonl')]
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download,jobs))
rows=[]
for r in records:
    ev={e['step']:e for e in r['evaluations']}
    rows.append({'seed':r['settings']['training_seed'],'run_id':r['summary']['run_id'],
                 'validation_by_step':{str(s):ev[s]['validation/ndcg@10'] for s in (500,1000,1500,2000)},
                 'fixed_train_by_step':{str(s):ev[s]['fixed_train/ndcg@10'] for s in (500,1000,1500,2000)},
                 'within_run_500_to_2000':ev[2000]['validation/ndcg@10']-ev[500]['validation/ndcg@10'],
                 'step500_difference_from_earlier_run':r['step500_difference'],
                 'wandb_url':r['summary']['wandb_url']})
result={'runs':rows,'mean_validation_by_step':{str(s):statistics.mean(r['validation_by_step'][str(s)] for r in rows)
                                             for s in (500,1000,1500,2000)},
        'final_seed_std':statistics.stdev(r['validation_by_step']['2000'] for r in rows),
        'validation_queries':83,'test_evaluated':False,'api_calls':0,
        'caveat':'Fresh runs share configuration/seeds but are not guaranteed bitwise identical to earlier CUDA runs. Report actual within-run trajectories. Validation was used for exploration, not a final-test claim.'}
(OUT/'longer-summary.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'longer-records.json').write_text(json.dumps(records,indent=2)+'\n')
with VOLUME.batch_upload(force=True) as batch:
    for name in ('longer-summary.json','longer-records.json'):
        batch.put_file(OUT/name,f'experiments/{EXPERIMENT}/{name}')
print(json.dumps(result,indent=2))
