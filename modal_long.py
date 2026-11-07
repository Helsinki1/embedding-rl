"""Bounded training-length check after the no-anchor exploration experiment."""
from dataclasses import replace
import json

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-longer")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume},
              timeout=7200, secrets=[modal.Secret.from_name("wandb-secret")], max_containers=1)
def longer(seed):
    from embedding_rl.config import digest, write_json
    from embedding_rl.followup import run_record
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.train import train

    source = ROOT / "experiments/unanchored-knobs-20261106/comparison.json"
    grid = json.loads(source.read_text())
    if grid["selected_variant"] != "both" or grid["mean_paired_delta"] <= 0:
        raise ValueError("The expected promising exploration result was not confirmed")
    matched = next(r for r in grid["runs"] if r["label"] == "both"
                   and r["settings"]["training_seed"] == seed)
    config = replace(base_config(), training_seed=seed, steps=2000,
                     eval_steps=(10, 50, 100, 250, 500, 1000, 1500, 2000),
                     samples_per_query=32, exploration_radius=0.45,
                     kl_weight=0.0, learning_rate=5e-6)
    old = {k: v for k, v in matched["settings"].items() if k not in {"steps", "eval_steps"}}
    new = {k: v for k, v in config.to_dict().items() if k not in {"steps", "eval_steps"}}
    if old != new:
        raise ValueError("Training-length comparison changed other settings")
    directory = ROOT / "experiments/longer-20261106"
    marker = directory / f"both-2000-seed{seed}.json"
    if marker.exists():
        existing = json.loads(marker.read_text())
        if digest(existing["settings"]) != digest(config.to_dict()):
            raise ValueError("Saved settings differ")
        return existing["summary"]
    write_json(directory / "protocol.json", {
        "seeds": [42, 43, 44], "steps": [500, 1000, 1500, 2000],
        "reward": "binary evidence nDCG; no API calls", "test_evaluated": False,
        "purpose": "Does longer training improve or overfit after relaxing the anchor and increasing exploration?",
        "start": "Fresh initialization per seed; saved earlier weights lack optimizer state",
        "comparison": "Same query/action sampling prefix through step 500; verify against the completed matched run",
        "selection": "Exploratory validation; report final 2000-step scores and trajectory, not only the peak",
    })
    try:
        print(f"START both-2000 seed={seed}", flush=True)
        summary = train(ROOT, config, "rl", "cuda", on_checkpoint=volume.commit)
        record = run_record(ROOT, "both-2000", config, summary)
        row500 = next(r for r in record["evaluations"] if r["step"] == 500)
        record["matched_500_run"] = matched["summary"]["run_id"]
        record["step500_difference"] = (row500["validation/ndcg@10"]
                                        - matched["summary"]["final_validation_ndcg"])
        write_json(marker, record)
        print("FINISHED " + json.dumps(summary), flush=True)
        return summary
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    for seed in (42, 43, 44):
        print(json.dumps(longer.remote(seed)), flush=True)
