"""Sequential queue on one GPU, preserving unrelated jobs and enforcing run limits."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .build import HERE, write_json

PLANS = {
    0: [("single", "frozen"), ("single", "infonce"), ("single", "rl"),
        ("one_pass", "infonce"), ("one_pass", "rl"), ("llm", "frozen")],
    1: [("split", "frozen"), ("split", "infonce"), ("split", "rl"),
        ("three_pass", "infonce"), ("three_pass", "rl")],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ubuntu/quest-spec-v2-20261106"))
    args = parser.parse_args()
    status = args.root / "status"
    status.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    for architecture, method in PLANS[args.gpu]:
        name = architecture + "_" + method
        state_path = status / (name + ".json")
        output = args.root / "results" / name
        if state_path.exists() or output.exists():
            raise FileExistsError("Refusing to overwrite a prior run: " + name)
        wait_start = time.monotonic()
        while True:
            p = args.root / "embeddings/progress.json"
            ready = p.exists() and json.loads(p.read_text()).get("done") == 96048
            if architecture == "llm":
                ready = ready and (HERE / "decomposition/manifest.json").exists()
            free, utilization = map(int, subprocess.check_output([
                "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip().split(","))
            write_json(state_path, {"stage": "queued", "gpu": args.gpu, "free_mib": free,
                                     "utilization": utilization, "inputs_ready": ready})
            if ready and free >= 50000 and utilization <= 20:
                break
            if time.monotonic() - wait_start > 7200:
                raise TimeoutError("Input/GPU availability wait exceeded two hours")
            time.sleep(10)
        command = [sys.executable, "-m", "benchmarks.quest_or_spec_v2.run", "--architecture", architecture,
                   "--method", method, "--output", str(output), "--root", str(args.root)]
        started = time.time()
        with (args.root / "logs" / (name + ".log")).open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(state_path, {"stage": "running", "gpu": args.gpu, "pid": child.pid,
                                     "started": started, "command": command})
            try:
                code = child.wait(timeout=1800)
                stage = "complete" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
                stage = "time_limit"
            write_json(state_path, {"stage": stage, "gpu": args.gpu, "returncode": code,
                                     "started": started, "finished": time.time(), "command": command})
        print(json.dumps({"name": name, "stage": stage, "returncode": code}), flush=True)


if __name__ == "__main__":
    main()
