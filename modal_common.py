from pathlib import Path

import modal

ROOT = Path("/data")
volume = modal.Volume.from_name("embedding-rl-data", create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    .env({"HF_HOME": "/data/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("embedding_rl", "modal_common")
)
