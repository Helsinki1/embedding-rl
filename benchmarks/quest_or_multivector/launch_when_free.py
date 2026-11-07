"""Wait for existing experiments and GPU headroom without changing other jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def write_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def dependencies_complete(root):
    for method in ("rl", "infonce"):
        states = {}
        for suffix in ("_status.json", "_evaluation_status.json"):
            path = root / (method + suffix)
            if path.exists():
                states[suffix] = json.loads(path.read_text()).get("stage")
        if states.get("_status.json") == "failed":
            continue
        if states.get("_evaluation_status.json") in ("complete", "failed"):
            continue
        return False, f"Waiting for existing {method} training/evaluation"
    return True, None


def gpu_ready(gpu, minimum_free_mib, maximum_utilization):
    response = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    free, utilization = map(int, response.strip().split(","))
    return free >= minimum_free_mib and utilization <= maximum_utilization, {
        "free_mib": free, "utilization_percent": utilization,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--variant", choices=("one_pass", "three_pass"), required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=49152)
    parser.add_argument("--maximum-utilization", type=int, default=20)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--timeout", type=float, default=86400)
    parser.add_argument("--max-run-seconds", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A training command is required after --")
    if args.max_run_seconds <= 0:
        parser.error("--max-run-seconds must be positive")
    args.status_root.mkdir(parents=True, exist_ok=True)
    status_path = args.status_root / f"{args.variant}-launch.json"
    if status_path.exists():
        raise FileExistsError("Launcher status already exists; refusing a duplicate run")
    base = {"variant": args.variant, "gpu": args.gpu, "launcher_pid": os.getpid(),
            "command": command, "queued_at": time.time(), "code_root": str(args.code_root)}
    stable_checks = 0
    last_reason = None
    deadline = time.monotonic() + args.timeout
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU queue timeout; no existing processes were interrupted")
            ready, reason = dependencies_complete(args.dependency_root)
            info = {}
            if ready:
                ready, info = gpu_ready(args.gpu, args.minimum_free_mib, args.maximum_utilization)
                reason = "Confirming GPU availability" if ready else "Waiting for GPU headroom/utilization"
            stable_checks = stable_checks + 1 if ready else 0
            write_status(status_path, {**base, "stage": "queued", "reason": reason,
                                       "stable_checks": stable_checks, "gpu_state": info,
                                       "updated_at": time.time()})
            if reason != last_reason:
                print(json.dumps({"stage": "queued", "reason": reason, "gpu": args.gpu}), flush=True)
                last_reason = reason
            if stable_checks >= 3:
                break
            time.sleep(args.interval)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        child = subprocess.Popen(command, cwd=args.code_root, env=env)
        running = {**base, "stage": "running", "training_pid": child.pid, "started_at": time.time()}
        write_status(status_path, running)
        print(json.dumps(running), flush=True)
        try:
            code = child.wait(timeout=args.max_run_seconds)
        except subprocess.TimeoutExpired:
            # The trainer normally finalizes early. This is the outer hard limit.
            child.kill()
            code = child.wait()
            write_status(status_path, {**running, "stage": "time_limit", "returncode": code,
                                      "finished_at": time.time(),
                                      "reason": "Hard run limit reached; preserved checkpoints remain available"})
            raise SystemExit(124)
        write_status(status_path, {**running, "stage": "complete" if code == 0 else "failed",
                                  "returncode": code, "finished_at": time.time()})
        raise SystemExit(code)
    except Exception as error:
        write_status(status_path, {**base, "stage": "failed", "error": repr(error),
                                  "finished_at": time.time()})
        raise


if __name__ == "__main__":
    main()
