"""Run the frozen-checkpoint corpus evaluation on an isolated Modal H100."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

app = modal.App("ambig-interpretation-retrieval")
volume = modal.Volume.from_name("ambig-interpretation-distillation")
bundle = Path(os.environ.get("AMBIG_EVAL_BUNDLE", "/tmp/ambig-retrieval-bundle"))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "peft==0.17.1",
                      "accelerate==1.11.0", "numpy==1.26.4", "huggingface_hub==0.36.0", "pytest==8.4.2")
         .env({"HF_HOME": "/experiment/huggingface", "TOKENIZERS_PARALLELISM": "false",
               "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONPATH": "/workspace", "PYTHONUNBUFFERED": "1"})
         .add_local_dir(bundle, "/workspace"))


@app.function(image=image, gpu="H100", cpu=16, memory=65536,
              volumes={"/experiment": volume}, timeout=1800, max_containers=1)
def run_remote(run_id, checkpoint_run):
    from huggingface_hub import snapshot_download
    from benchmarks.ambig_interpretations.model import MODEL_ID, REVISION
    started = time.time()
    model = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    try:
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "/workspace/benchmarks/quest_or_spec_v2/test_protocol.py"],
                       check=True, env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="4"))
        subprocess.run([sys.executable, "-m", "benchmarks.ambig_interpretations.retrieval_eval",
                        "--model", model, "--corpus", "/experiment/shared-corpus-v2",
                        "--checkpoints", f"/experiment/{checkpoint_run}",
                        "--output", f"/experiment/{run_id}", "--bundle", "/workspace"],
                       check=True, timeout=1650)
        result = json.loads((Path("/experiment") / run_id / "comparison.json").read_text())
        result["remote_wall_seconds"] = time.time()-started
        return result
    finally:
        volume.commit()


@app.local_entrypoint()
def main(run_id: str = "ambig-retrieval-20261106-v1", checkpoint_run: str = "ambig-20261106-v1"):
    result = run_remote.remote(run_id, checkpoint_run)
    output = Path(__file__).resolve().parent / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"run_id": run_id, "remote_wall_seconds": result["remote_wall_seconds"]}))
