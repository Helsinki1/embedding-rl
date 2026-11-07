"""Run the split baseline with a hard 30-minute budget per method."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--minimum-free-mib", type=int, default=40960)
    args = parser.parse_args()
    status = args.root / "status.json"
    if status.exists():
        raise FileExistsError("Refusing to duplicate an existing experiment")
    base = {"launcher_pid": os.getpid(), "gpu": args.gpu, "created_at": time.time(),
            "methods": ["frozen", "infonce", "rl"], "hard_limit_seconds": 1795}
    (args.root / "logs").mkdir(exist_ok=True)
    (args.root / "results").mkdir(exist_ok=True)
    completed, failures = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    for method in base["methods"]:
        try:
            deadline, stable = time.monotonic() + 86400, 0
            while stable < 3:
                free = int(subprocess.check_output([
                    "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], text=True).strip())
                stable = stable + 1 if free >= args.minimum_free_mib else 0
                write_status(status, {**base, "stage": "queued", "method": method,
                                      "free_mib": free, "stable_checks": stable,
                                      "completed": completed, "failures": failures})
                if time.monotonic() > deadline:
                    raise TimeoutError("Timed out waiting for GPU memory headroom")
                if stable < 3:
                    time.sleep(15)
            command = ["timeout", "--signal=TERM", "--kill-after=5s", "1790s",
                       sys.executable, "-m", "benchmarks.quest_or_split.train", "--method", method,
                       "--dataset", "/home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic",
                       "--split", str(args.root / "code/benchmarks/quest_or_balanced"),
                       "--work-dir", "/home/ubuntu/quest-or-baseline-20261106",
                       "--embeddings", "/home/ubuntu/quest-or-synthetic-20261106/embeddings",
                       "--output", str(args.root / "results" / method)]
            with (args.root / "logs" / f"{method}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=args.root / "code", env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                write_status(status, {**base, "stage": "running", "method": method,
                                      "pid": process.pid, "started_at": time.time(),
                                      "command": command, "completed": completed, "failures": failures})
                code = process.wait()
            if code:
                failures.append({"method": method, "returncode": code,
                                 "stage": "timed_out" if code in (124, 137) else "failed"})
            else:
                completed.append(method)
        except Exception as error:
            failures.append({"method": method, "error": repr(error)})
    write_status(status, {**base, "stage": "failed" if failures else "complete",
                          "completed": completed, "failures": failures, "finished_at": time.time()})
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
