"""Download and verify only this experiment's completed artifacts."""
from pathlib import Path
import json,subprocess,sys,tarfile

project=Path('/home/davidx/Downloads/embedding-rl')
ssh_config='/home/davidx/Documents/Codex/2026-11-06/les/work/ssh/config'
remote='/home/ubuntu/quest-or-split-r32-20261106-v1'
root=project/'benchmarks/quest_or_split/runs/r32-v1'
work=project/'work/quest-split-launch'
command=['ssh','-F',ssh_config,'compute',
         f'tar -czf - --exclude="*.pt" --exclude="__pycache__" -C {remote}/results frozen infonce rl -C {remote} logs status.json launch.json initial_launcher_status.json supervise_parallel.py snapshot_manifest.json snapshot.tar.gz']
with (work/'artifacts.tar.gz').open('wb') as stream:
    subprocess.run(command,stdout=stream,check=True)
with tarfile.open(work/'artifacts.tar.gz') as archive:
    for member in archive.getmembers():
        target=(root/member.name).resolve()
        if not target.is_relative_to(root.resolve()) or member.issym() or member.islnk():
            raise ValueError('Unexpected archive path')
    archive.extractall(root)
sys.path.insert(0,str(project/'benchmarks/quest_or/scripts'))
from verify_official_score import verify
verification={}
entries=[]
for method in ('frozen','infonce','rl'):
    method_root=root/method
    progress=json.loads((method_root/'progress.json').read_text())
    if progress['stage']!='complete':raise ValueError(f'{method} is unfinished')
    metadata=json.loads((method_root/'run.json').read_text())
    if method!='frozen':
        assert progress['step']==3137 and progress['examples_seen']==100363
        assert metadata['config']['lora_rank']==32 and metadata['config']['lora_alpha']==64
    labels=(None,) if method=='frozen' else ('selected','best_trained','final')
    for label in labels:
        directory=method_root/label if label else method_root
        summary=json.loads((directory/'summary.json').read_text())
        name=method+('/'+label if label else '')
        score=verify(project/'benchmarks/quest_or_synthetic/data/test.jsonl',directory/'test_predictions.jsonl')
        for key,value in score.items():assert abs(summary['test']['overall'][key]-value)<1e-12
        verification[name]=score
        entries.append({'method':method,'evaluation':label or 'frozen','step':summary['checkpoint_step'],
                        'k_per_piece':summary['k'],'development_f1':summary['validation']['f1'],
                        **summary['test']['overall'],'diagnostics':summary['diagnostics']})
(root/'official_score_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
result={'experiment':'Three contiguous coordinate pieces of Qwen3-Embedding-8B',
        'dimensions':[1366,1365,1365],'document_count':393140,
        'training_queries':100363,'development_queries':78,'test_queries':78,
        'epochs':1,'steps':3137,'lora_rank':32,'lora_alpha':64,'seed':42,
        'remote_run':remote,'local_run':str(root),'results':entries,
        'notes':['The document encoder is frozen; document slices match query coordinates.',
                 'This is the custom 78-query held-out split, not official QUEST test.',
                 'Checkpoint and cutoff selection use development only.',
                 'Selected includes step zero; best_trained excludes step zero.',
                 'These are single-seed results on synthetic-augmented training data.']}
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
outputs=Path('/home/davidx/Documents/Codex/2026-11-06/les/outputs')
outputs.mkdir(exist_ok=True)
(outputs/'qwen-three-piece-rank32-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(entries,indent=2))
