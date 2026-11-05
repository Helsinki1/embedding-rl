"""CPU-only grading: no idle GPU while waiting for OpenAI."""
import json
import os
import modal
from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-judge")


@app.function(
    image=image, volumes={ROOT: volume}, timeout=86400, max_containers=1,
    secrets=[modal.Secret.from_name(os.environ.get("OPENAI_SECRET_NAME", "openai-keys"),
                                   required_keys=["OPENAI_API_KEY"])],
)
def judge_remote(settings):
    from embedding_rl.config import Config
    from embedding_rl.judge import judge
    try:
        return judge(ROOT, Config(**settings), on_batch=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "config.json"):
    from embedding_rl.config import Config
    print(json.dumps(judge_remote.remote(Config.load(config).to_dict()), indent=2))
