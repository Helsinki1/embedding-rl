"""Continue this experiment's existing InfoNCE process and launch RL separately."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path('/home/ubuntu/quest-or-split-r32-20261106-v1')
status_path=root/'status.json'
prior=json.loads(status_path.read_text())
assert prior['method']=='infonce' and prior['stage']=='running'
assert prior['launcher_pid']==3671245 and prior['pid']==3671300
(root/'initial_launcher_status.json').write_text(json.dumps(prior,indent=2)+'\n')
# Terminate only our waiting supervisor, never its active training child.
os.kill(prior['launcher_pid'],signal.SIGTERM)
processes={'infonce':prior['pid']}
started=time.time()
def write(value):
    p=status_path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(status_path)
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
try:
    stable=0
    while stable<3:
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        stable=stable+1 if free>=36864 else 0
        write({'stage':'rl_queued','infonce_pid':prior['pid'],'rl_gpu':0,'free_mib':free,'stable_checks':stable})
        if stable<3:time.sleep(15)
        if time.time()-started>86400:raise TimeoutError('GPU memory wait exceeded one day')
    command=['/home/ubuntu/venvs/quest-training-v2/bin/python','-m','benchmarks.quest_or_split.train','--method','rl','--dataset','/home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic','--work-dir','/home/ubuntu/quest-or-baseline-20261106','--embeddings','/home/ubuntu/quest-or-synthetic-20261106/embeddings','--output',str(root/'results/rl')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    with (root/'logs/rl.log').open('x') as log:
        child=subprocess.Popen(command,cwd=root/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
        processes['rl']=child.pid
        while True:
            complete=[method for method in processes if (root/'results'/method/'summary.json').exists()]
            failed=[method for method,pid in processes.items() if method not in complete and not alive(pid)]
            done=len(complete)+len(failed)==len(processes)
            write({'stage':('failed' if failed else 'complete') if done else 'running','parallel':True,'supervisor_pid':os.getpid(),'processes':processes,'gpus':{'infonce':1,'rl':0},'completed':['frozen']+complete,'failed':failed,'started_at':started,'updated_at':time.time()})
            if done:break
            time.sleep(20)
        child.wait()
except BaseException as error:
    write({'stage':'supervisor_failed','error':repr(error),'processes':processes})
    raise
