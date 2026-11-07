"""Isolated GPU queues; never interrupt other jobs; enforce 30 minutes per run."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.quest_or_spec_v2.build import write_json

PLANS = {0: ("attention_mlp", "one_pass"), 1: ("pooled_mlp", "three_pass")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/quest-ordered-joint-20261106"))
    p.add_argument("--architecture", choices=("one_pass", "three_pass", "pooled_mlp", "attention_mlp"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument("--clause-weight", type=float)
    args = p.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    for architecture in ((args.architecture,) if args.architecture else PLANS[args.gpu]):
        out = args.root / "results" / architecture
        state = args.root / "status" / (architecture + ".json")
        if out.exists() or state.exists():
            raise FileExistsError("Refusing to overwrite " + architecture)
        start, free_checks = time.monotonic(), 0
        while free_checks < 3:
            free, util = map(int, subprocess.check_output([
                "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip().split(","))
            ready = args.clause_weight == 0 or (args.root / "teachers/manifest.json").exists()
            free_checks = free_checks+1 if ready and free >= 50000 and util <= 20 else 0
            write_json(state, {"stage": "queued", "gpu": args.gpu, "free_mib": free,
                               "utilization": util, "teachers_ready": ready})
            if time.monotonic()-start > 7200:
                write_json(state, {"stage": "queue_timeout", "gpu": args.gpu})
                raise TimeoutError("GPU wait exceeded two hours")
            time.sleep(5)
        command = [sys.executable, "-m", "benchmarks.quest_or_ordered_joint.run",
                   "--architecture", architecture, "--root", str(args.root), "--output", str(out)]
        for flag, value in (("--epochs", args.epochs), ("--micro-batch-size", args.micro_batch_size),
                            ("--clause-weight", args.clause_weight)):
            if value is not None:
                command.extend((flag, str(value)))
        began = time.time()
        with (args.root / "logs" / (architecture + ".log")).open("w") as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            write_json(state, {"stage": "running", "gpu": args.gpu, "pid": child.pid,
                               "started": began, "command": command})
            try:
                code = child.wait(timeout=1800)
                stage = "complete" if code == 0 and (out / "summary.json").exists() else "failed"
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
                stage = "time_limit"
            write_json(state, {"stage": stage, "gpu": args.gpu, "returncode": code,
                               "started": began, "finished": time.time(), "command": command})
        print(json.dumps({"architecture": architecture, "stage": stage}), flush=True)


if __name__ == "__main__":
    main()
