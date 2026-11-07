"""Probe exploration, run the 2x2 grid, then repeat control and selected variant."""
from dataclasses import replace
import json
import os
from pathlib import Path
import uuid

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-followup")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume},
              secrets=[modal.Secret.from_name(os.environ.get("WANDB_SECRET_NAME", "wandb-secret"),
                                               required_keys=["WANDB_API_KEY"])],
              timeout=7200, max_containers=1)
def experiment_remote(settings, experiment_id, stage, label="", records=None, selected=""):
    from embedding_rl.config import Config, digest, write_json
    from embedding_rl.followup import run_record, sampling_probe, summarize
    from embedding_rl.tracking import Tracker
    from embedding_rl.train import train

    config = Config(**settings)
    directory = ROOT / "experiments" / experiment_id
    if Path(experiment_id).name != experiment_id:
        raise ValueError("Invalid experiment ID")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if stage == "probe":
            path = directory / "probe" / "probe.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result["config_digest"] != digest(settings):
                    raise ValueError("Cannot resume probe with different settings")
                return result
            return sampling_probe(ROOT, config, directory / "probe", experiment_id, "cuda")
        if stage == "train":
            path = directory / f"{label}-seed{config.training_seed}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if digest(saved["settings"]) != digest(settings):
                    raise ValueError("Cannot resume a condition with different settings")
                return saved
            print(f"START {label} seed={config.training_seed}", flush=True)
            result = train(ROOT, config, "rl", "cuda", on_checkpoint=volume.commit)
            record = run_record(ROOT, label, config, result)
            write_json(path, record)
            print(f"FINISHED {label} seed={config.training_seed} {json.dumps(result)}", flush=True)
            return record
        if stage != "summary":
            raise ValueError("Unknown experiment stage")
        result = summarize(records, selected)
        result["experiment_id"] = experiment_id
        tracker = Tracker(directory, config, "followup-summary", {"experiment_id": experiment_id})
        try:
            if tracker.run:
                import wandb
                tracker.run.log({"runs": wandb.Table(
                    columns=["condition", "seed", "samples", "radius", "final_ndcg", "run_url"],
                    data=[[r["label"], r["settings"]["training_seed"],
                           r["settings"]["samples_per_query"], r["settings"]["exploration_radius"],
                           r["summary"]["final_validation_ndcg"], r["summary"]["wandb_url"]]
                          for r in records])})
            result["wandb_url"] = tracker.run.url if tracker.run else None
            write_json(directory / "comparison.json", result)
            tracker.finish({"mean_paired_delta": result["mean_paired_delta"],
                            "selected_variant": selected})
        except BaseException:
            tracker.finish(success=False)
            raise
        return result
    finally:
        volume.commit()


@app.local_entrypoint()
def main(config: str = "config.followup.json", experiment_id: str = ""):
    from embedding_rl.config import Config
    from embedding_rl.followup import conditions, select_variant

    base = Config.load(config)
    experiment_id = experiment_id or f"knobs-{uuid.uuid4().hex[:10]}"
    print(f"EXPERIMENT {experiment_id}", flush=True)
    probe = experiment_remote.remote(base.to_dict(), experiment_id, "probe")
    print(f"PROBE {json.dumps(probe['comparison'])}", flush=True)
    records = []
    grid = dict(conditions(base))
    for label, settings in grid.items():
        records.append(experiment_remote.remote(settings.to_dict(), experiment_id, "train", label))
    selected = select_variant(records)
    print(f"SELECTED {selected} for seed repeats against control", flush=True)
    initial_seed = base.seed if base.training_seed is None else base.training_seed
    for seed in (initial_seed + 1, initial_seed + 2):
        for label in ("control", selected):
            settings = replace(grid[label], training_seed=seed)
            records.append(experiment_remote.remote(settings.to_dict(), experiment_id, "train", label))
    result = experiment_remote.remote(base.to_dict(), experiment_id, "summary",
                                      records=records, selected=selected)
    print(f"COMPARISON {json.dumps(result)}", flush=True)
