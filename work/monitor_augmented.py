import json,time
from pathlib import Path
root=Path('/home/ubuntu/quest-or-training-augmented-20261106-v2')
for method in ('rl','infonce'):
    status=json.loads((root/f'{method}_status.json').read_text())
    path=root/f'results/{method}/metrics.jsonl'
    metrics=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try: metrics.append(json.loads(line))
            except json.JSONDecodeError: pass
    data={'method':method,'stage':status['stage'],'elapsed_seconds':round(time.time()-status.get('started',status.get('finished',time.time())))}
    if metrics:
        last=metrics[-1]
        data.update({key:last[key] for key in ('step','examples_seen','loss')})
        validation=[r for r in metrics if 'validation' in r]
        if validation:
            data['latest_validation']={'step':validation[-1]['step'],'f1':validation[-1]['validation']['f1'],'k':validation[-1]['validation']['k']}
            best=max(validation,key=lambda r:r['validation']['f1'])
            data['best_trained_validation']={'step':best['step'],'f1':best['validation']['f1'],'k':best['validation']['k']}
        if method=='rl':
            data['recent_reward_f1']=sum(r['reward_f1'] for r in metrics[-50:])/len(metrics[-50:])
            data['recent_zero_advantage_fraction']=sum(r['zero_advantage_fraction'] for r in metrics[-50:])/len(metrics[-50:])
    else:
        data['log_tail']=(root/f'logs/{method}.log').read_text().splitlines()[-2:]
    initial=root/f'results/{method}/initial_validation.json'
    if initial.exists(): data['initial_validation_f1']=json.loads(initial.read_text())['best']['f1']
    evaluation=root/f'{method}_evaluation_status.json'
    if evaluation.exists():data['evaluation_status']=json.loads(evaluation.read_text())
    if status['stage']=='failed':data['log_tail']=(root/f'logs/{method}.log').read_text().splitlines()[-20:]
    print(json.dumps(data))
