"""Finish the current comparison after the existing training coordinator exits."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/davidx/Downloads/embedding-rl')
WAIT_PID = 150550
deadline = time.monotonic() + 7 * 3600
while Path(f'/proc/{WAIT_PID}/cmdline').exists():
    command = Path(f'/proc/{WAIT_PID}/cmdline').read_bytes()
    if b'modal_graded.py' not in command:
        break
    if time.monotonic() > deadline:
        raise TimeoutError('Existing training has not finished after seven hours')
    time.sleep(20)

# Reuses persistent completed markers; resumes missing runs once if the first coordinator failed.
for stage in ('llm-training', 'rankings', 'evaluate'):
    print(json.dumps({'starting_stage': stage}), flush=True)
    with (ROOT / 'work/graded' / f'pipeline-{stage}.log').open('a') as output:
        subprocess.run(['modal', 'run', 'modal_graded.py', '--stage', stage], cwd=ROOT,
                       stdout=output, stderr=subprocess.STDOUT, check=True, timeout=5 * 3600)
print('COLLECTING', flush=True)
subprocess.run(['python3', 'work/graded/collect_results.py'], cwd=ROOT, check=True)
print('COMPLETE', flush=True)
