"""Collect completed sampling comparisons and draw validation curves."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modal

from embedding_rl.graded_evaluate import paired_bootstrap

OUT = Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')
VOLUME = modal.Volume.from_name('embedding-rl-data')
EXPERIMENT = 'unanchored-knobs-20261106'
data = json.loads(b''.join(VOLUME.read_file(f'experiments/{EXPERIMENT}/comparison.json')))
(OUT/'unanchored-comparison.json').write_text(json.dumps(data, indent=2)+'\n')


def download(job):
    run, name = job
    path = OUT/run/name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b''.join(VOLUME.read_file(f'runs/{run}/{name}')))


jobs = [(r['summary']['run_id'], name) for r in data['runs'] for name in
        ('summary.json', 'run.json', 'validation.json', 'training_probe.json', 'metrics.jsonl')]
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, jobs))


def by_query(label):
    runs = [r for r in data['runs'] if r['label']==label]
    values = []
    for run in runs:
        val = json.loads((OUT/run['summary']['run_id']/'validation.json').read_text())
        values.append({r['query_id']: r['ndcg@10'] for r in val['final']['per_query']})
    return {q:statistics.mean(r[q] for r in values) for q in values[0]}


base=by_query('control'); selected=by_query(data['selected_variant'])
paired=paired_bootstrap(list(base.values()),[selected[q] for q in base])
summary={'comparison':data['comparison'], 'paired_seed_deltas':data['paired_deltas'],
         'seed_averaged_paired_query_bootstrap':paired, 'validation_queries':83,
         'test_evaluated':False, 'wandb_url':data['wandb_url'],
         'caveat':'Exploratory validation selection; bootstrap does not correct for tuning. Wider-only and more-samples-only have one seed each.'}
(OUT/'unanchored-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

fig, ax=plt.subplots(figsize=(9,5.3),layout='constrained')
for label,color,title in [('control','#536878','8 samples, radius 0.15'),
                          ('both','#147d92','32 samples, radius 0.45')]:
    runs=[r for r in data['runs'] if r['label']==label]
    steps=[r['step'] for r in runs[0]['evaluations']]
    curves=[[r['validation/ndcg@10'] for r in run['evaluations']] for run in runs]
    for curve in curves: ax.plot(steps,curve,color=color,alpha=.3,linewidth=1)
    mean=[statistics.mean(c[i] for c in curves) for i in range(len(steps))]
    ax.plot(steps,mean,color=color,linewidth=2.5,label=title+' (3 seeds)')
ax.axhline(.03742847964167595,color='#b16446',linestyle='--',label='Previous strong-anchor runs: final mean')
ax.set(xlabel='RL updates',ylabel='Validation evidence nDCG@10',ylim=(0,.14),
       title='Exploration helps after relaxing the anchor')
ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15); ax.legend(loc='upper left',frameon=False)
fig.text(.01,-.015,'83 validation queries; 100,195 documents. New runs: KL=0, learning rate=5e-6. Final test untouched.',fontsize=9,color='#555555')
fig.savefig(OUT/'validation-progress.png',dpi=170,bbox_inches='tight'); plt.close(fig)
with VOLUME.batch_upload(force=True) as batch:
    for name in ('unanchored-comparison.json','unanchored-summary.json','validation-progress.png'):
        batch.put_file(OUT/name,f'experiments/{EXPERIMENT}/{name}')
print(json.dumps(summary,indent=2))
