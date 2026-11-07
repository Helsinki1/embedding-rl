"""One-step query-encoder policies; frozen document vectors; clean validation/test."""
import hashlib
import json
import random
import uuid
from pathlib import Path

import torch

from .config import Config, digest, write_json
from .data import load_prepared
from .evaluate import evaluate
from .objectives import gaussian_kl, infonce_loss, retrieval_rewards, rloo_loss, sample_actions
from .tracking import Tracker


def code_fingerprint():
    files = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


def make_encoder(config, manifest, device):
    from .model import Encoder
    return Encoder(config.model_name, manifest["revisions"]["model"], device,
                   config.max_length, config.query_prefix)


def checkpoint(encoder, directory, step, score, filename="best.pt"):
    temporary = directory / (filename + ".tmp")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in encoder.model.state_dict().items()},
                "step": step, "validation_ndcg": score}, temporary)
    temporary.replace(directory / filename)


def train(root, config, method, device="cpu", encoder_factory=make_encoder, on_checkpoint=None,
          reward_provider=None, policy_action_transform=None):
    if method not in {"baseline", "infonce", "rl"}:
        raise ValueError("method must be baseline, infonce, or rl")
    if reward_provider is not None and (method != "rl" or config.dataset != "browsecomp-plus"):
        raise ValueError("Custom rewards require BrowseComp RL")
    if policy_action_transform is not None and method != "rl":
        raise ValueError("Policy-gradient action transforms require RL")
    training_seed = config.seed if config.training_seed is None else config.training_seed
    torch.manual_seed(training_seed)
    rng = random.Random(training_seed)
    generator = torch.Generator(device=device).manual_seed(training_seed)
    path, manifest, bundle, tensors = load_prepared(root, config)
    benchmark_labels = config.dataset == "browsecomp-plus"
    if benchmark_labels:
        ids = bundle["splits"]["train"]
        labels = {qid: bundle["qrels"][qid] for qid in ids} if method != "baseline" else {}
        supervision = {"source": "browsecomp-plus/evidence", "prepared": digest(manifest)}
    else:
        from .judge import judge_settings, training_labels
        labels = training_labels(path, config, manifest, bundle) if method != "baseline" else {}
        supervision = judge_settings(config, manifest)
    if reward_provider is not None:
        supervision = reward_provider.metadata
    directory = root / "runs" / f"{method}-{uuid.uuid4().hex[:10]}"
    directory.mkdir(parents=True)
    metadata = {
        "method": method, "prepared_manifest": manifest, "code_sha256": code_fingerprint(),
        "supervision": supervision,
        "experiment_id": digest({"prepared": manifest, "supervision": supervision}),
        "eligible_training_queries": len(labels), "run_id": directory.name,
    }
    write_json(directory / "run.json", {"config": config.to_dict(), **metadata})
    tracker = Tracker(directory, config, method, metadata)
    try:
        encoder = encoder_factory(config, manifest, device)
        documents = tensors["documents"]
        training_documents = documents.to(device) if benchmark_labels else documents
        doc_index = {doc["id"]: i for i, doc in enumerate(bundle["docs"])}
        reference = {qid: tensors["queries"][i] for i, qid in enumerate(bundle["query_vector_ids"])}
        before = evaluate(encoder, bundle, documents, bundle["splits"]["validation"], config)
        probe_ids = random.Random(config.seed).sample(
            sorted(bundle["splits"]["train"]),
            min(config.monitor_train_queries, len(bundle["splits"]["train"])),
        )
        probe_before = evaluate(encoder, bundle, documents, probe_ids, config) if probe_ids else None
        probe_current = probe_before
        current = before
        best = before
        key = f"ndcg@{config.ndcg_k}"
        best_score, best_step = before["metrics"][key], 0
        checkpoint(encoder, directory, best_step, best_score)
        if on_checkpoint:
            on_checkpoint()
        initial_metrics = {f"validation/{k}": v for k, v in before["metrics"].items()}
        if probe_before:
            initial_metrics.update({f"fixed_train/{k}": v for k, v in probe_before["metrics"].items()})
        tracker.log(initial_metrics, 0)
        if method != "baseline":
            optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=config.learning_rate,
                                          weight_decay=0)
            ids = sorted(labels)
            for step in range(1, config.steps + 1):
                batch = rng.sample(ids, min(config.batch_size, len(ids)))
                if benchmark_labels:
                    candidates = training_documents
                    grades = torch.zeros(len(batch), len(documents), device=device)
                    for row, qid in enumerate(batch):
                        for docid, grade in labels[qid].items():
                            grades[row, doc_index[docid]] = grade
                else:
                    indices = torch.tensor([bundle["pools"][qid] for qid in batch])
                    candidates = documents[indices].to(device)
                    grades = torch.stack([labels[qid] for qid in batch]).to(device)
                means = encoder.encode([bundle["queries"][qid] for qid in batch], query=True)
                ref = torch.stack([reference[qid] for qid in batch]).to(device)
                actions, sigma = sample_actions(means, config.exploration_radius,
                                                config.samples_per_query, generator)
                if reward_provider is None:
                    rewards, winners = retrieval_rewards(actions, candidates, grades, config.ndcg_k)
                else:
                    rewards, winners = reward_provider(batch, actions, candidates, config.ndcg_k)
                kl = gaussian_kl(means, ref, sigma)
                if method == "rl":
                    gradient_actions = (policy_action_transform(batch, means, actions)
                                        if policy_action_transform is not None else actions)
                    policy_loss, advantages = rloo_loss(
                        means, gradient_actions, rewards, sigma, config.advantage_std_floor)
                    loss = policy_loss + config.kl_weight * kl
                else:
                    loss = infonce_loss(means, candidates, grades, config.temperature,
                                        positive_grade=1 if benchmark_labels else 2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0,
                                                     error_if_nonfinite=True)
                optimizer.step()
                metrics = {
                    "train/loss": loss.item(), "train/reward_mean": rewards.mean().item(),
                    "train/reward_std": rewards.std(dim=1).mean().item(), "train/kl": kl.item(),
                    "train/zero_advantage_fraction":
                        (rewards.max(1).values == rewards.min(1).values).float().mean().item(),
                    "train/grad_norm": norm.item(),
                    "train/query_drift": (1 - (means * ref).sum(-1)).mean().item(),
                    "train/unique_top1": sum(len(row.unique()) for row in winners) / len(batch),
                }
                if method == "rl":
                    metrics["train/advantage_abs_mean"] = advantages.abs().mean().item()
                if step % config.eval_every == 0 or step in config.eval_steps or step == config.steps:
                    current = evaluate(encoder, bundle, documents,
                                       bundle["splits"]["validation"], config)
                    metrics.update({f"validation/{k}": v for k, v in current["metrics"].items()})
                    if probe_ids:
                        probe_current = evaluate(encoder, bundle, documents, probe_ids, config)
                        metrics.update({f"fixed_train/{k}": v
                                        for k, v in probe_current["metrics"].items()})
                    if current["metrics"][key] > best_score:
                        best, best_score, best_step = current, current["metrics"][key], step
                        checkpoint(encoder, directory, best_step, best_score)
                        if on_checkpoint:
                            on_checkpoint()
                tracker.log(metrics, step)
                print(json.dumps({"step": step, **metrics}), flush=True)
        final_step = config.steps if method != "baseline" else 0
        checkpoint(encoder, directory, final_step, current["metrics"][key], "last.pt")
        summary = {"run_id": directory.name, "method": method, "best_step": best_step,
                   "final_step": final_step, "final_validation_ndcg": current["metrics"][key],
                   "initial_validation_ndcg": before["metrics"][key],
                   "best_validation_ndcg": best_score,
                   "validation_delta": best_score - before["metrics"][key],
                   "wandb_url": tracker.run.url if tracker.run and config.wandb_mode == "online"
                   else None}
        write_json(directory / "validation.json", {"before": before, "best": best, "final": current})
        if probe_before:
            write_json(directory / "training_probe.json", {
                "query_ids": probe_ids, "before": probe_before, "final": probe_current,
            })
        write_json(directory / "summary.json", summary)
        tracker.examples(before, best)
        tracker.finish(summary)
        return summary
    except BaseException:
        tracker.finish(success=False)
        raise


def final_test(root, run_id, device="cpu", encoder_factory=make_encoder):
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be a run directory name")
    directory = root / "runs" / run_id
    result_path = directory / "test.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    metadata = json.loads((directory / "run.json").read_text())
    config = Config(**metadata["config"])
    _, manifest, bundle, tensors = load_prepared(root, config)
    if manifest != metadata["prepared_manifest"]:
        raise ValueError("Run and prepared dataset provenance differ")
    encoder = encoder_factory(config, manifest, device)
    before = evaluate(encoder, bundle, tensors["documents"], bundle["splits"]["test"], config)
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    encoder.model.load_state_dict(saved["state_dict"])
    result = evaluate(encoder, bundle, tensors["documents"], bundle["splits"]["test"], config)
    result.update({"run_id": run_id, "checkpoint_step": saved["step"]})
    result["before_metrics"] = before["metrics"]
    result["delta"] = {key: value - before["metrics"][key]
                       for key, value in result["metrics"].items()}
    test_dir = directory / "final-test"
    test_dir.mkdir(exist_ok=True)
    tracker = Tracker(test_dir, config, "final-test", metadata)
    tracker.log({f"test/{k}": v for k, v in result["metrics"].items()}, 0)
    tracker.finish(result["metrics"])
    write_json(result_path, result)
    return result
