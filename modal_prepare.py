"""No credentials required: frozen corpus embeddings and training candidate pools."""
import json
import modal
from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-prepare")


@app.function(image=image, gpu="T4", volumes={ROOT: volume}, timeout=3600, max_containers=1)
def prepare_remote(settings):
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
