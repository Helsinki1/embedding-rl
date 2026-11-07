"""Bounded LLM grading and RL on the existing full-corpus BrowseComp cache."""
import json
from pathlib import Path

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-graded")
SECRETS = [modal.Secret.from_name("embedding-rl-openai", required_keys=["OPENAI_API_KEY"]),
           modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])]


def experiment_dir(experiment_id):
    if Path(experiment_id).name != experiment_id:
        raise ValueError("Invalid experiment ID")
    return ROOT / "experiments" / experiment_id


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume},
              timeout=7200, secrets=SECRETS, max_containers=1)
def gpu_stage(experiment_id, stage, label="", seed=42):
    from embedding_rl.config import write_json
    from embedding_rl.followup import run_record
    from embedding_rl.graded_judge import JudgedReward
    from embedding_rl.judge_experiment import base_config, load_judge, prepare_plan, train_settings
    from embedding_rl.train import train

    directory = experiment_dir(experiment_id)
    base = base_config()
    try:
        if stage == "prepare":
            result = prepare_plan(ROOT, base, directory)
            return {k: v for k, v in result.items() if k not in {"train_pools", "calibration", "config"}}
        if stage == "rankings":
            from embedding_rl.graded_evaluate import collect_rankings
            return collect_rankings(ROOT, base, directory)
        if stage != "train":
            raise ValueError("Unknown GPU stage")
        marker = directory / f"{label}-seed{seed}.json"
        if marker.exists():
            return json.loads(marker.read_text())
        config = train_settings(base, label, seed)
        reward = None
        if label.startswith("llm"):
            calibration = json.loads((directory / "calibration-gpt-6.1-terra.json").read_text())
            if not calibration["passed"]:
                raise ValueError("Judge calibration did not pass")
            judge, bundle = load_judge(ROOT, config, directory, commit=volume.commit)
            reward = JudgedReward(judge, bundle)
        result = train(ROOT, config, "rl", "cuda", on_checkpoint=volume.commit, reward_provider=reward)
        record = run_record(ROOT, label, config, result)
        write_json(marker, record)
        return record
    finally:
        volume.commit()


@app.function(image=image, cpu=4, memory=16384, volumes={ROOT: volume}, timeout=14400,
              secrets=SECRETS, max_containers=1)
def judge_stage(experiment_id, stage):
    from embedding_rl.config import write_json
    from embedding_rl.judge_experiment import base_config, calibrate, load_judge

    directory = experiment_dir(experiment_id)
    try:
        judge, _ = load_judge(ROOT, base_config(), directory, commit=volume.commit)
        plan = json.loads((directory / "plan-top25.json").read_text())
        if stage == "calibrate":
            result = calibrate(judge, plan, directory)
            return {k: v for k, v in result.items() if k != "rows"}
        if stage == "grade":
            if not json.loads((directory / "calibration-gpt-6.1-terra.json").read_text())["passed"]:
                raise ValueError("Judge calibration did not pass")
            count = judge.ensure(plan["train_pools"])
            result = {"new_pairs": count, "total_pairs": sum(len(v) for v in judge.grades.values()),
                      "api_usd_upper_estimate": judge.budget.total()}
            write_json(directory / "grading-summary.json", result)
            return result
        if stage == "evaluate":
            from embedding_rl.graded_evaluate import evaluate_judge
            from embedding_rl.tracking import Tracker
            result = evaluate_judge(judge, directory)
            tracker = Tracker(directory, base_config(), "graded-comparison", {"experiment_id": experiment_id})
            if tracker.run:
                import wandb
                tracker.run.log({"comparison": wandb.Table(
                    columns=["condition", "seed", "benchmark_ndcg", "llm_ndcg", "strict_llm_ndcg",
                             "training_signal_fraction", "wandb_url"],
                    data=[[r["label"], r["seed"], r["benchmark"]["ndcg@10"],
                           r["llm_metrics"]["llm_ndcg@10"], r["llm_metrics"]["llm_strict_ndcg@10"],
                           r.get("training_signal_fraction"), r.get("wandb_url")] for r in result["runs"]],
                )})
            result["wandb_url"] = tracker.run.url if tracker.run else None
            write_json(directory / "comparison.json", result)
            tracker.finish({"api_usd_upper_estimate": result["api_usd_upper_estimate"],
                            "validation_queries": result["validation_queries"], "test_evaluated": False})
            return {k: v for k, v in result.items() if k not in {"runs", "stronger_audit"}}
        raise ValueError("Unknown judge stage")
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "prepare", experiment_id: str = "graded-20261106", label: str = "llm_raw",
         seed: int = 42):
    if stage in {"binary-controls", "llm-training"}:
        labels = ["binary_normalized"] if stage == "binary-controls" else ["llm_raw", "llm_normalized"]
        for training_seed in (42, 43, 44):
            for condition in labels:
                print(f"START {condition} seed={training_seed}", flush=True)
                record = gpu_stage.remote(experiment_id, "train", condition, training_seed)
                print(f"FINISHED {condition} seed={training_seed} "
                      + json.dumps(record["summary"]), flush=True)
        return
    if stage in {"prepare", "train", "rankings"}:
        result = gpu_stage.remote(experiment_id, stage, label, seed)
    else:
        result = judge_stage.remote(experiment_id, stage)
    print(json.dumps(result, indent=2))
