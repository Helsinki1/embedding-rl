"""Train one method, or explicitly evaluate a selected checkpoint on the final test."""
import json
import os
import modal
from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-train")
offline = os.environ.get("EMBEDDING_RL_OFFLINE") == "1"
secrets = [] if offline else [modal.Secret.from_name(
    os.environ.get("WANDB_SECRET_NAME", "wandb-secret"), required_keys=["WANDB_API_KEY"]
)]


@app.function(image=image, gpu="T4", volumes={ROOT: volume}, secrets=secrets,
              timeout=7200, max_containers=1)
def train_remote(settings, method, run_id):
    from embedding_rl.config import Config
    from embedding_rl.train import final_test, train
    try:
        if run_id:
            return final_test(ROOT, run_id, "cuda")
        return train(ROOT, Config(**settings), method, "cuda", on_checkpoint=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "config.json", method: str = "rl", run_id: str = ""):
    from embedding_rl.config import Config
    settings = Config.load(config).to_dict()
    if offline:
        settings["wandb_mode"] = "offline"
    print(json.dumps(train_remote.remote(settings, method, run_id), indent=2))
