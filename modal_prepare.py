"""Prepare frozen corpus embeddings and benchmark label/split manifests."""
import json
import os
import modal
from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-prepare")


@app.function(image=image, gpu="T4", cpu=4, volumes={ROOT: volume}, memory=16384,
              timeout=14400, max_containers=1)
def prepare_remote(settings):
    # Tokenize long web pages in parallel so CPU work can keep up with the GPU.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "4"
    from embedding_rl.config import Config
    from embedding_rl.data import prepare
    try:
        return prepare(ROOT, Config(**settings), "cuda")
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "config.json"):
    from embedding_rl.config import Config
    print(json.dumps(prepare_remote.remote(Config.load(config).to_dict()), indent=2))
