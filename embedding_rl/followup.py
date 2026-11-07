"""Controlled sampling probe and summaries for the three-knob experiment."""
from dataclasses import replace
import json
import math
import random
import statistics

import torch

from .config import digest, write_json
from .data import load_prepared
from .objectives import retrieval_rewards
from .tracking import Tracker


def conditions(base):
    # KL = ||mean-reference||^2 / (2 sigma^2). Scale its coefficient so
    # changing sampling radius does not also loosen the displacement penalty.
    return [(label, replace(base, samples_per_query=count, exploration_radius=radius,
                            kl_weight=base.kl_weight * (radius / base.exploration_radius) ** 2))
            for label, count, radius in [
                ("control", 8, 0.15), ("more_samples", 32, 0.15),
                ("wider", 8, 0.45), ("both", 32, 0.45),
            ]]


@torch.no_grad()
def sampling_probe(root, config, directory, experiment_id, device="cpu"):
    _, manifest, bundle, tensors = load_prepared(root, config)
    if config.dataset != "browsecomp-plus" or config.monitor_train_queries < 1:
        raise ValueError("Sampling probe requires BrowseComp and a positive training monitor count")
    qids = random.Random(config.seed).sample(
        sorted(bundle["splits"]["train"]),
        min(config.monitor_train_queries, len(bundle["splits"]["train"])),
    )
    references = {qid: tensors["queries"][i]
                  for i, qid in enumerate(bundle["query_vector_ids"])}
    docs = tensors["documents"].to(device)
    doc_index = {doc["id"]: i for i, doc in enumerate(bundle["docs"])}
    generator = torch.Generator(device=device).manual_seed(config.seed)
    records = {label: [] for label, _ in conditions(config)}
    originals = []
    for qid in qids:
        mean = references[qid].to(device)[None, :]
        grades = torch.zeros(1, len(docs), device=device)
        for docid, grade in bundle["qrels"][qid].items():
            grades[0, doc_index[docid]] = grade
        original, _ = retrieval_rewards(mean[:, None, :], docs, grades, config.ndcg_k)
        original = original.item()
        originals.append(original)
        # Identical draws across radii; the 8-sample condition is a prefix of 32.
        noise = torch.randn(1, 32, mean.shape[-1], generator=generator, device=device)
        for radius in (0.15, 0.45):
            actions = mean[:, None, :] + noise * radius / math.sqrt(mean.shape[-1])
            rewards, winners = retrieval_rewards(actions, docs, grades, config.ndcg_k)
            for label, settings in conditions(config):
                if settings.exploration_radius != radius:
                    continue
                values = rewards[0, :settings.samples_per_query]
                records[label].append({
                    "query_id": qid, "original_ndcg": original,
                    "reward_mean": values.mean().item(), "best_reward": values.max().item(),
                    "positive_sample_fraction": (values > 0).float().mean().item(),
                    "has_signal": bool(values.max() > values.min()),
                    "best_improves_original": bool(values.max() > original),
                    "unique_top1": len(winners[0, :settings.samples_per_query].unique()),
                })
    comparison = []
    for label, settings in conditions(config):
        rows = records[label]
        comparison.append({
            "condition": label, "samples": settings.samples_per_query,
            "radius": settings.exploration_radius,
            **{key: statistics.mean(row[key] for row in rows)
               for key in ("reward_mean", "best_reward", "positive_sample_fraction",
                           "has_signal", "best_improves_original", "unique_top1")},
        })
    directory.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(directory, config, "sampling-probe", {"experiment_id": experiment_id})
    try:
        result = {"config_digest": digest(config.to_dict()), "prepared_manifest": manifest,
                  "query_ids": qids, "original_mean_ndcg": statistics.mean(originals),
                  "comparison": comparison, "per_query": records,
                  "wandb_url": tracker.run.url if tracker.run and config.wandb_mode == "online"
                  else None}
        if tracker.run:
            import wandb
            tracker.run.log({"sampling_comparison": wandb.Table(
                columns=list(comparison[0]), data=[list(row.values()) for row in comparison])})
        write_json(directory / "probe.json", result)
        tracker.finish({"queries": len(qids), "original_mean_ndcg": result["original_mean_ndcg"]})
        return result
    except BaseException:
        tracker.finish(success=False)
        raise


def run_record(root, label, config, summary):
    directory = root / "runs" / summary["run_id"]
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    training = [row for row in records if "train/zero_advantage_fraction" in row]
    evaluations = [row for row in records if f"validation/ndcg@{config.ndcg_k}" in row]
    return {"label": label, "settings": config.to_dict(), "summary": summary,
            "evaluations": evaluations,
            "mean_zero_advantage_fraction": statistics.mean(
                row["train/zero_advantage_fraction"] for row in training)}


def select_variant(records):
    # Select using the fixed final step, not the highest point on a noisy curve.
    return max((row for row in records if row["label"] != "control"),
               key=lambda row: row["summary"]["final_validation_ndcg"])["label"]


def summarize(records, selected):
    grouped = {}
    for row in records:
        grouped.setdefault(row["label"], []).append(row)
    rows = []
    for label, runs in grouped.items():
        values = [r["summary"]["final_validation_ndcg"] for r in runs]
        rows.append({"condition": label, "seeds": [r["settings"]["training_seed"] for r in runs],
                     "final_ndcg_mean": statistics.mean(values),
                     "final_ndcg_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                     "final_ndcg_values": values,
                     "initial_ndcg": runs[0]["summary"]["initial_validation_ndcg"],
                     "mean_zero_advantage_fraction": statistics.mean(
                         r["mean_zero_advantage_fraction"] for r in runs)})
    controls = {r["settings"]["training_seed"]: r["summary"]["final_validation_ndcg"]
                for r in grouped["control"]}
    paired = [{"seed": r["settings"]["training_seed"],
               "delta_vs_control": r["summary"]["final_validation_ndcg"]
               - controls[r["settings"]["training_seed"]]} for r in grouped[selected]]
    return {"selected_variant": selected, "comparison": rows, "paired_deltas": paired,
            "mean_paired_delta": statistics.mean(r["delta_vs_control"] for r in paired),
            "selection_rule": "Highest final-step validation nDCG among initial-seed non-control runs",
            "test_evaluated": False, "runs": records}
