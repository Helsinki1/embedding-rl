"""An isolated two-H100 experiment; only model and interpretation data are mounted."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("ambig-interpretation-distillation")
volume = modal.Volume.from_name("ambig-interpretation-distillation", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "peft==0.17.1",
                      "accelerate==1.11.0", "numpy==1.26.4", "huggingface_hub==0.36.0", "pytest==8.4.2")
         .env({"HF_HOME": "/experiment/huggingface", "TOKENIZERS_PARALLELISM": "false",
               "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(HERE, "/workspace/benchmarks/ambig_interpretations",
                        ignore=["source", "runs", "__pycache__", ".pytest_cache"]))


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536,
              volumes={"/experiment": volume}, timeout=1800, max_containers=1)
def run_remote(run_id, deadline):
    from huggingface_hub import snapshot_download
    from benchmarks.ambig_interpretations.model import MODEL_ID, REVISION
    root = Path("/experiment") / run_id
    root.mkdir(exist_ok=False)
    started = time.time()
    subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "/workspace/benchmarks/ambig_interpretations/test_interpretations.py"],
                   check=True, env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="4"))
    print(json.dumps({"stage": "model_download", "seconds_remaining": deadline-time.time()}), flush=True)
    model = snapshot_download(MODEL_ID, revision=REVISION)
    data = Path("/workspace/benchmarks/ambig_interpretations/data")
    env = dict(os.environ, OMP_NUM_THREADS="6")
    subprocess.run([sys.executable, "-m", "benchmarks.ambig_interpretations.prepare",
                    "--data", str(data), "--output", str(root / "teacher"), "--model", model,
                    "--deadline", str(deadline)], env=dict(env, CUDA_VISIBLE_DEVICES="0"),
                   check=True, timeout=max(1, deadline-time.time()-120))
    volume.commit()
    workers, handles, positions = {}, {}, {}
    try:
        for gpu, architecture in enumerate(("mtp", "autoregressive")):
            log = root / f"{architecture}.log"
            handles[architecture] = log.open("w")
            positions[architecture] = 0
            workers[architecture] = subprocess.Popen([
                sys.executable, "-m", "benchmarks.ambig_interpretations.train",
                "--architecture", architecture, "--model", model, "--data", str(data),
                "--teacher", str(root / "teacher"), "--output", str(root / architecture),
                "--deadline", str(deadline)], env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=handles[architecture], stderr=subprocess.STDOUT)
        while any(p.poll() is None for p in workers.values()):
            if time.time() >= deadline - 10:
                raise TimeoutError("Hard experiment wall-clock deadline")
            time.sleep(10)
            for architecture, p in workers.items():
                with (root / f"{architecture}.log").open() as stream:
                    stream.seek(positions[architecture])
                    for line in stream:
                        print(f"[{architecture}] {line.rstrip()}", flush=True)
                    positions[architecture] = stream.tell()
                if p.poll() not in (None, 0):
                    raise RuntimeError(f"{architecture} failed with {p.returncode}")
            volume.commit()
        results = {a: json.loads((root / a / "summary.json").read_text()) for a in workers}
        results["remote_wall_seconds"] = time.time()-started
        results["finished_at"] = time.time()
        (root / "comparison.json").write_text(json.dumps(results, indent=2))
        return results
    finally:
        for p in workers.values():
            if p.poll() is None:
                p.terminate()
        for h in handles.values():
            h.close()
        volume.commit()


@app.local_entrypoint()
def main(run_id: str, deadline: float):
    result = run_remote.remote(run_id, deadline)
    output = HERE / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
