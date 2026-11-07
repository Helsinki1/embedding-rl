"""No-API ablation of the Gaussian KL anchor and encoder learning rate."""
from dataclasses import replace
import json

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-anchor")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume},
              timeout=7200, secrets=[modal.Secret.from_name("wandb-secret")], max_containers=1)
def experiment(seed=42, selected=""):
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.followup import run_record
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.train import train

    base = base_config()
    _, manifest, _, tensors = load_prepared(ROOT, base)
    dimension = tensors["documents"].shape[-1]
    # For unit means: beta * GaussianKL = beta * dimension/radius^2 * (1-cos).
    soft_beta = 0.5 * base.exploration_radius**2 / dimension
    variants = [
        ("cosine_anchor_0.5", soft_beta, 1e-6),
        ("no_anchor", 0.0, 1e-6),
        ("cosine_anchor_0.5_lr5e-6", soft_beta, 5e-6),
        ("no_anchor_lr5e-6", 0.0, 5e-6),
    ]
    if selected:
        if selected not in {row[0] for row in variants}:
            raise ValueError("Unknown ablation")
        variants = [row for row in variants if row[0] == selected]
    directory = ROOT / "experiments/anchor-20261106"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", {
        "prepared_digest": digest(manifest), "dimension": dimension,
        "original_cosine_coefficient": base.kl_weight * dimension / base.exploration_radius**2,
        "soft_cosine_coefficient": 0.5, "soft_kl_weight": soft_beta,
        "reward": "original binary evidence nDCG@10; no API calls",
        "design": "2 anchor levels x 2 learning rates; same 500 steps, samples, radius and data",
        "screening_seed": 42, "test_evaluated": False,
        "followup_rule": "Repeat promising final-step validation gains on seeds 43 and 44; exploratory selection is not a final-test result",
        "source": "https://arxiv.org/html/2610.07731v1#A4.SS3",
    })
    volume.commit()
    try:
        for label, beta, lr in variants:
            config = replace(base, training_seed=seed, kl_weight=beta, learning_rate=lr)
            marker = directory / f"{label}-seed{seed}.json"
            if marker.exists():
                record = json.loads(marker.read_text())
                if record["settings"] != json.loads(json.dumps(config.to_dict())):
                    raise ValueError("Existing marker has different configuration")
            else:
                print(f"START {label} seed={seed}", flush=True)
                summary = train(ROOT, config, "rl", "cuda", on_checkpoint=volume.commit)
                record = run_record(ROOT, label, config, summary)
                validation = json.loads((ROOT / "runs" / summary["run_id"] / "validation.json").read_text())
                before = {r["query_id"]: r["ndcg@10"] for r in validation["before"]["per_query"]}
                after = {r["query_id"]: r["ndcg@10"] for r in validation["final"]["per_query"]}
                record["paired_vs_base"] = paired_bootstrap(list(before.values()), [after[q] for q in before])
                write_json(marker, record)
                volume.commit()
            print("FINISHED " + json.dumps({"label": label, "seed": seed,
                  **record["summary"], "paired": record["paired_vs_base"]}), flush=True)
        return {"experiment": str(directory), "seed": seed, "completed": len(variants)}
    finally:
        volume.commit()


@app.local_entrypoint()
def main(seed: int = 42, selected: str = "", repeat: bool = False):
    if repeat and not selected:
        raise ValueError("Choose a screened variant before repeating seeds")
    for training_seed in ((43, 44) if repeat else (seed,)):
        print(json.dumps(experiment.remote(training_seed, selected)))
