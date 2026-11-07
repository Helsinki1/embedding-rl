"""Launch only these two isolated runs with a hard 30-minute wall-clock budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", choices=("rl", "infonce"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        method = args.worker
        status = root / f"{method}_status.json"
        command = ["timeout", "--signal=TERM", "--kill-after=5s", "1790s", sys.executable,
                   "-m", "benchmarks.quest_or_balanced.run", "--method", method,
                   "--corpus-dataset", "/home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic",
                   "--embeddings", "/home/ubuntu/quest-or-synthetic-20261106/embeddings",
                   "--work-dir", "/home/ubuntu/quest-or-baseline-20261106",
                   "--output", str(root / "results" / method)]
        started = time.time()
        state = {"stage": "running", "started": started, "command": command,
                 "hard_limit_seconds": 1795}
        status.write_text(json.dumps(state, indent=2) + "\n")
        code = subprocess.run(command, cwd=root / "code").returncode
        state.update({"stage": "complete" if code == 0 else "timed_out" if code in (124, 137) else "failed",
                      "returncode": code, "finished": time.time(), "elapsed_seconds": time.time() - started})
        status.write_text(json.dumps(state, indent=2) + "\n")
        return
    if (root / "launch.json").exists():
        raise FileExistsError("This run directory has already been launched")
    (root / "logs").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    launches = []
    for gpu, method in enumerate(("rl", "infonce")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        with (root / "logs" / f"{method}.log").open("xb") as log:
            worker = subprocess.Popen([sys.executable, "-m", "benchmarks.quest_or_balanced.launch",
                                       "--root", str(root), "--worker", method],
                                      cwd=root / "code", env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launches.append({"method": method, "gpu": gpu, "pid": worker.pid})
    payload = {"launched": time.time(), "workers": launches, "hard_limit_seconds": 1795}
    (root / "launch.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
