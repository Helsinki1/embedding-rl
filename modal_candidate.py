"""Matched no-API Gaussian RLOO/CMP experiment on frozen judged candidate pools."""
from dataclasses import replace
import json
from pathlib import Path
import random
import statistics

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-candidate-cmp")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume},
              timeout=7200, secrets=[modal.Secret.from_name("wandb-secret")], max_containers=1)
def stage_remote(stage, label="unprojected", seed=42, records=None,
                 experiment_id="candidate-cmp-20261106", kl_weight=0.0, pool_every=0,
                 grade_source="llm", repeat_all=False, steps=500):
    import torch
    from embedding_rl.candidate_policy import AlternatingPoolReward, CachedPoolReward, project_gradient_actions
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.followup import run_record
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.objectives import ndcg, sample_actions
    from embedding_rl.tracking import Tracker
    from embedding_rl.train import make_encoder, train

    if Path(experiment_id).name != experiment_id:
        raise ValueError("Invalid experiment ID")
    directory = ROOT / "experiments" / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    if steps not in {500, 2000}:
        raise ValueError("This experiment supports only the declared500/2000-update budgets")
    config = replace(base_config(), training_seed=seed, samples_per_query=32,
                     exploration_radius=.45, kl_weight=kl_weight, learning_rate=5e-6, steps=steps,
                     eval_steps=(10, 50, 100, 250, 500, 1000, 1500, 2000)
                     if steps == 2000 else base_config().eval_steps)
    if label not in {"unprojected", "projected"}:
        raise ValueError("Unknown candidate-pool estimator")
    matched = None
    if steps == 2000:
        if (experiment_id, label, grade_source, pool_every, kl_weight) != (
                "alternating-longer-20261106", "projected", "benchmark", 10, 0):
            raise ValueError("Longer runs must match the predeclared selected mixed recipe")
        matched = json.loads((ROOT / "experiments/alternating-binary-pool-20261106"
                              / f"projected-seed{seed}.json").read_text())
        previous_settings = {k: v for k, v in matched["settings"].items() if k not in {"steps", "eval_steps"}}
        current_settings = {k: v for k, v in config.to_dict().items() if k not in {"steps", "eval_steps"}}
        if previous_settings != current_settings:
            raise ValueError("Longer mixed recipe changed settings beyond the update budget")
        control = json.loads((ROOT / "experiments/longer-20261106" / f"both-2000-seed{seed}.json").read_text())
        if digest(control["settings"]) != digest(config.to_dict()):
            raise ValueError("Full-corpus control settings differ")
    try:
        if stage == "summary":
            groups = {name: [r for r in records if r["label"] == name]
                      for name in ("unprojected", "projected")}
            rows = []
            per_query = {}
            for name, runs in groups.items():
                values = [r["summary"]["final_validation_ndcg"] for r in runs]
                rows.append({"condition": name, "seeds": [r["settings"]["training_seed"] for r in runs],
                             "mean_ndcg": statistics.mean(values),
                             "seed_std": statistics.stdev(values) if len(values) > 1 else None,
                             "final_values": values,
                             "informative_groups": 1 - statistics.mean(r["mean_zero_advantage_fraction"] for r in runs),
                             "fixed_training_pool_before": statistics.mean(r["fixed_pool"]["before"] for r in runs),
                             "fixed_training_pool_after": statistics.mean(r["fixed_pool"]["after"] for r in runs)})
                measurements = []
                for run in runs:
                    val = json.loads((ROOT / "runs" / run["summary"]["run_id"] / "validation.json").read_text())
                    measurements.append({r["query_id"]: r["ndcg@10"] for r in val["final"]["per_query"]})
                per_query[name] = {q: statistics.mean(r[q] for r in measurements) for q in measurements[0]}
            base, projected = per_query["unprojected"], per_query["projected"]
            result = {"comparison": rows, "runs": records, "test_evaluated": False,
                      "paired_query_bootstrap": paired_bootstrap(list(base.values()), [projected[q] for q in base]),
                      "api_calls": 0, "validation_queries": len(base),
                      "pool_every_updates": pool_every,
                      "grade_source": grade_source,
                      "caveat": "Judged pools contain known training evidence; validation searches the full corpus. Alternating runs retain full-corpus rewards on other steps. Seed42 screens the recipe; validation tuning is exploratory."}
            tracker = Tracker(directory, config, "candidate-comparison", {"experiment_id": directory.name})
            if tracker.run:
                import wandb
                tracker.run.log({"comparison": wandb.Table(
                    columns=["condition", "seed", "final_ndcg", "run_url"],
                    data=[[r["label"], r["settings"]["training_seed"],
                           r["summary"]["final_validation_ndcg"], r["summary"]["wandb_url"]] for r in records])})
            result["wandb_url"] = tracker.run.url if tracker.run else None
            tracker.finish({"paired_mean_delta": result["paired_query_bootstrap"]["mean_delta"], "api_calls": 0})
            write_json(directory / "comparison.json", result)
            return {k: v for k, v in result.items() if k != "runs"}

        _, manifest, bundle, tensors = load_prepared(ROOT, config)
        source = ROOT / "experiments/graded-20261106"
        plan = json.loads((source / "plan-top25.json").read_text())
        if plan["prepared_digest"] != digest(manifest):
            raise ValueError("Candidate plan belongs to different prepared data")
        maker = AlternatingPoolReward if pool_every else CachedPoolReward
        extra = {"pool_every": pool_every} if pool_every else {}
        reward = maker(plan, source / "judge-terra-v1", bundle, tensors["documents"],
                       projected=label == "projected" or stage == "probe", grade_source=grade_source, **extra)
        if matched is not None and digest(matched["reward_metadata"]) != digest(reward.metadata):
            raise ValueError("Longer mixed recipe changed the reward environment")
        qids = random.Random(config.seed).sample(sorted(bundle["splits"]["train"]), 64)
        references = {q: tensors["queries"][i] for i, q in enumerate(bundle["query_vector_ids"])}
        write_json(directory / "protocol.json", {
            "prepared_digest": digest(manifest),
            "common_settings": {k: v for k, v in config.to_dict().items() if k != "training_seed"},
            "effective_cosine_anchor": config.kl_weight * tensors["documents"].shape[1]
            / config.exploration_radius**2,
            "candidate_supervision": {k: v for k, v in reward.metadata.items() if k != "gradient_estimator"},
            "initial_seed": 42, "repeat_seeds": [43, 44],
            "repeat_rule": ("Run selected projected binary-pool recipe for all three preselected seeds regardless of results" if steps == 2000 else
                            "Run both estimators for all three preselected seeds regardless of results" if repeat_all else
                            "Repeat both estimators if either alternating seed42 run beats the existing matched full-corpus500-step seed42 run by more than0.001" if pool_every else
                            "Repeat both estimators only if projected beats unprojected seed42 final validation by more than0.001"),
            "training_environment": "Frozen original top25 plus known training evidence; all candidates actually judged. Alternating mode uses full-corpus binary reward on other steps.",
            "pool_every_updates": pool_every,
            "evaluation_environment": "All100195 frozen documents;83 validation queries;166 test queries untouched",
            "reward": f"Candidate-pool {grade_source} nDCG@10; IDCG over the whole fixed pool",
            "grade_source": grade_source,
            "steps": steps,
            "initialization": "Fresh pretrained encoder per seed; previous weights lack optimizer state",
            "matched_controls": "longer-20261106, seeds42/43/44 at2000updates" if steps == 2000 else None,
            "projection": "Conditional averaging of isotropic Gaussian noise onto the fixed candidate span, only in the score-function gradient",
            "limitations": "Changed training environment, not full-index CMP or a complete RELER reproduction",
        })
        if stage == "probe":
            marker = directory / "probe.json"
            if marker.exists():
                return json.loads(marker.read_text())["summary"]
            generator = torch.Generator(device="cuda").manual_seed(1309)
            rows = []
            with torch.no_grad():
                for qid in qids:
                    docs, grades, _, basis = reward.candidates(qid, "cuda")
                    mean = references[qid].to("cuda").expand(64, -1)
                    actions, sigma = sample_actions(mean, .45, 32, generator)
                    projected = project_gradient_actions(mean, actions, [basis] * 64)
                    scores, projected_scores = actions @ docs.T, projected @ docs.T
                    rewards = ndcg(scores, grades, 10)
                    projected_rewards = ndcg(projected_scores, grades, 10)
                    advantage = (rewards - rewards.mean(1, keepdim=True)) * 32 / 31
                    plain = (advantage[..., None] * (actions - mean[:, None]) / sigma**2).mean(1)
                    reduced = (advantage[..., None] * (projected - mean[:, None]) / sigma**2).mean(1)
                    plain -= (plain * mean).sum(-1, keepdim=True) * mean
                    reduced -= (reduced * mean).sum(-1, keepdim=True) * mean
                    rows.append({"query_id": qid, "basis_rank": basis.shape[1],
                                 "plain_variance": plain.var(0).sum().item(),
                                 "projected_variance": reduced.var(0).sum().item(),
                                 "max_score_error": (scores - projected_scores).abs().max().item(),
                                 "changed_reward_fraction": (rewards != projected_rewards).float().mean().item(),
                                 "informative_groups": (rewards.max(1).values > rewards.min(1).values).float().mean().item()})
            summary = {"queries": len(rows), "draw_groups_per_query": 64,
                       "variance_reduction_factor": sum(r["plain_variance"] for r in rows)
                       / max(sum(r["projected_variance"] for r in rows), 1e-15),
                       "mean_basis_rank": statistics.mean(r["basis_rank"] for r in rows),
                       "max_score_error": max(r["max_score_error"] for r in rows),
                       "mean_changed_reward_fraction": statistics.mean(r["changed_reward_fraction"] for r in rows),
                       "informative_groups": statistics.mean(r["informative_groups"] for r in rows),
                       "scope": "Query-embedding tangent gradients at the original encoder, not full network gradients"}
            if summary["max_score_error"] > 1e-5 or summary["mean_changed_reward_fraction"] > .001:
                raise ValueError("Projection did not numerically preserve candidate rewards")
            write_json(marker, {"summary": summary, "rows": rows})
            return summary
        if stage != "train":
            raise ValueError("Unknown stage")
        marker = directory / f"{label}-seed{seed}.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            if digest(record["settings"]) != digest(config.to_dict()):
                raise ValueError("Completed run settings differ")
            saved_metadata = json.loads((ROOT / "runs" / record["summary"]["run_id"] / "run.json").read_text())
            if digest(saved_metadata["supervision"]) != digest(reward.metadata):
                raise ValueError("Completed run reward environment differs")
            return record
        with torch.no_grad():
            before_vectors = torch.stack([references[q] for q in qids]).to("cuda")
            before = CachedPoolReward.__call__(reward, qids, before_vectors[:, None], None, 10)[0].mean().item()
        print(f"START {label} seed={seed}", flush=True)
        result = train(ROOT, config, "rl", "cuda", on_checkpoint=volume.commit, reward_provider=reward,
                       policy_action_transform=reward.gradient_actions if label == "projected" else None)
        record = run_record(ROOT, label, config, result)
        record["reward_metadata"] = reward.metadata
        encoder = make_encoder(config, manifest, "cuda")
        saved = torch.load(ROOT / "runs" / result["run_id"] / "last.pt", map_location="cpu", weights_only=True)
        encoder.model.load_state_dict(saved["state_dict"])
        vectors = encoder.encode_batches([bundle["queries"][q] for q in qids], 32, query=True).to("cuda")
        with torch.no_grad():
            after = CachedPoolReward.__call__(reward, qids, vectors[:, None], None, 10)[0].mean().item()
        record["fixed_pool"] = {"before": before, "after": after, "training_query_ids": qids}
        if pool_every:
            record["update_counts"] = reward.counts
        if matched is not None:
            at500 = next(r for r in record["evaluations"] if r["step"] == 500)
            record["matched_500_run"] = matched["summary"]["run_id"]
            record["step500_difference"] = at500["validation/ndcg@10"] - matched["summary"]["final_validation_ndcg"]
            if reward.counts != {"full_corpus": 1800, "candidate_pool": 200}:
                raise ValueError("Unexpected mixed update counts")
        write_json(marker, record)
        print("FINISHED " + json.dumps({"label": label, "seed": seed, **result,
                                       "fixed_pool_before": before, "fixed_pool_after": after}), flush=True)
        return record
    finally:
        volume.commit()


@app.local_entrypoint()
def main(experiment_id: str = "candidate-cmp-20261106", kl_weight: float = 0.0, pool_every: int = 0,
         grade_source: str = "llm", repeat_all: bool = False, steps: int = 500):
    options = {"experiment_id": experiment_id, "kl_weight": kl_weight, "pool_every": pool_every,
               "grade_source": grade_source, "repeat_all": repeat_all, "steps": steps}
    if steps == 2000:
        records = [stage_remote.remote("train", "projected", seed, **options) for seed in (42, 43, 44)]
        print("LONGER_DONE " + json.dumps([r["summary"] for r in records]), flush=True)
        return
    print("PROBE " + json.dumps(stage_remote.remote("probe", **options)), flush=True)
    records = [stage_remote.remote("train", label, 42, **options) for label in ("unprojected", "projected")]
    gain = records[1]["summary"]["final_validation_ndcg"] - records[0]["summary"]["final_validation_ndcg"]
    repeat = gain > .001
    if pool_every:
        reference = json.loads(b"".join(volume.read_file("experiments/unanchored-knobs-20261106/both-seed42.json")))
        repeat = max(r["summary"]["final_validation_ndcg"] for r in records) > reference["summary"]["final_validation_ndcg"] + .001
    if repeat or repeat_all:
        for seed in (43, 44):
            for label in ("unprojected", "projected"):
                records.append(stage_remote.remote("train", label, seed, **options))
    print("COMPARISON " + json.dumps(stage_remote.remote("summary", records=records, **options)), flush=True)
