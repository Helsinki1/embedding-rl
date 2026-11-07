import hashlib,json,tarfile
from pathlib import Path
import numpy as np
root=Path('/home/ubuntu/quest-or-training-augmented-20261106-v2')
dataset=Path('/home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic')
methods=('rl','infonce')
for method in methods:
    for suffix in ('status','evaluation_status'):
        status=json.loads((root/f'{method}_{suffix}.json').read_text())
        if status.get('stage')!='complete' or status.get('returncode')!=0:
            raise RuntimeError(f'{method} {suffix} is incomplete')
gold_rows=[json.loads(line) for line in (dataset/'data/test.jsonl').read_text().splitlines()]
gold={row['id']:row for row in gold_rows}
assert len(gold)==78
reports={}; per_query={}; coverage={}
for method in methods:
    metadata=json.loads((root/f'results/{method}/run.json').read_text())
    assert metadata['training_queries']==100363 and metadata['document_count']==393140
    assert metadata['config']['epochs']==1 and metadata['config']['steps']==3137
    steps=[json.loads(line) for line in (root/f'results/{method}/metrics.jsonl').read_text().splitlines()]
    assert len(steps)==3137 and steps[-1]['examples_seen']==100363
    assert [row['step'] for row in steps]==list(range(1,3138))
    coverage[method]={'updates':len(steps),'examples_seen':steps[-1]['examples_seen']}
    reports[method]={}
    for label,folder in (('selected',root/f'results/{method}'),*((label,root/f'results/{method}/extra_evaluation/{label}') for label in ('frozen','best_trained','final'))):
        report=json.loads((folder/'summary.json').read_text())
        predictions=[json.loads(line) for line in (folder/'test_predictions.jsonl').read_text().splitlines()]
        assert len(predictions)==78 and {row['id'] for row in predictions}==set(gold)
        values={}
        for row in predictions:
            desired=gold[row['id']]
            assert row['query']==desired['query']
            actual=set(row['docs']); expected=set(desired['docs'])
            assert len(actual)==len(row['docs'])
            correct=len(actual&expected)
            values[row['id']]={'precision':correct/len(actual) if actual else 0,'recall':correct/len(expected),'f1':2*correct/(len(actual)+len(expected))}
        for metric in ('precision','recall','f1'):
            actual=sum(value[metric] for value in values.values())/78
            assert abs(actual-report['test']['overall'][metric])<1e-12
        reports[method][label]=report
        per_query[(method,label)]={qid:result['f1'] for qid,result in values.items()}
assert per_query[('rl','frozen')]==per_query[('infonce','frozen')]
assert reports['rl']['frozen']['k']==reports['infonce']['frozen']['k']
frozen=reports['rl']['frozen']
rng=np.random.default_rng(42); intervals={}
for method in methods:
    delta=np.array([per_query[(method,'best_trained')][qid]-per_query[('rl','frozen')][qid] for qid in sorted(gold)])
    means=delta[rng.integers(0,78,size=(10000,78))].mean(axis=1)
    intervals[method]={'f1_difference':float(delta.mean()),'paired_bootstrap_95pct_interval':np.quantile(means,[.025,.975]).tolist(),'resamples':10000,'seed':42}
manifest=json.loads((dataset/'manifest.json').read_text())
report={'study':'QUEST OR + 100,000 synthetic NQ OR examples','training_examples':100363,'corpus_documents':393140,'validation_queries':78,'test_queries':78,'seed':42,'epochs':1,'batch_size':32,'updates_each':3137,'embedding_dimension':4096,'dataset_manifest':manifest,'config':metadata['config'],'frozen_baseline':frozen,'trained_results':{m:reports[m]['best_trained'] for m in methods},'selection_including_initial':{m:reports[m]['selected'] for m in methods},'final_epoch_results':{m:reports[m]['final'] for m in methods},'paired_differences_vs_frozen':intervals,'coverage':coverage,'independent_verification':'Precision, recall, and F1 recomputed for all eight prediction files; exact same 78 test queries; errors < 1e-12','remote_run_directory':str(root),'remote_checkpoints':{m:{'best_trained':str(root/f'results/{m}/best_trained.pt'),'final':str(root/f'results/{m}/last.pt'),'adapter_selected_including_initial':str(root/f'results/{m}/adapter')} for m in methods},'caveats':['Single seed and 78 held-out QUEST queries','The training budget and corpus differ from the earlier 200-update experiment','Synthetic positives are known rather than exhaustive; added passages and QUEST are assumed disjoint per dataset construction','Best-trained results select only updated checkpoints; selection including the untrained checkpoint is also reported']}
(root/'results/comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with tarfile.open(root/'result-records.tar.gz','w:gz') as archive:
    for path in sorted((root/'results').rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl') and 'adapter' not in path.parts:
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
    for path in sorted((root/'logs').glob('*.log')): archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
    for path in sorted(root.glob('*_status.json')): archive.add(path,arcname=path.name,recursive=False)
    archive.add(root/'launch.json',arcname='launch.json',recursive=False)
    for path in sorted((root/'code/benchmarks').rglob('*')):
        if path.is_file() and path.suffix in ('.py','.json','.md','.txt','.jsonl') and '__pycache__' not in path.parts:
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
print(json.dumps({'frozen':frozen['test']['overall'],'trained':{m:{'step':reports[m]['best_trained']['checkpoint_step'],'k':reports[m]['best_trained']['k'],**reports[m]['best_trained']['test']['overall']} for m in methods},'selected':{m:{'step':reports[m]['selected']['checkpoint_step'],**reports[m]['selected']['test']['overall']} for m in methods},'final':{m:reports[m]['final']['test']['overall'] for m in methods},'paired_differences':intervals,'archive_bytes':(root/'result-records.tar.gz').stat().st_size},indent=2))
