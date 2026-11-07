"""Save reusable top-k reranker examples and full candidate page text to the volume."""
import json

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-export")


@app.function(image=image, gpu="T4", cpu=2, volumes={ROOT: volume}, memory=8192,
              timeout=1800, max_containers=1)
def export_remote(run_id, k):
    from embedding_rl.export import export_reranker
    try:
        return export_reranker(ROOT, run_id, k, "cuda")
    finally:
        volume.commit()


@app.local_entrypoint()
def main(run_id: str, k: int = 25):
    print(json.dumps(export_remote.remote(run_id, k), indent=2))
