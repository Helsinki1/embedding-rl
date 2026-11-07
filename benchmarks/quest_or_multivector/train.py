"""Train either three-vector architecture against the same frozen augmented corpus."""
import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch

from benchmarks.quest_or.scripts.evaluate import evaluate
from benchmarks.quest_or_synthetic.data import load_augmented_corpus
from benchmarks.quest_or_balanced.data import load_splits
from benchmarks.quest_or_training.data import gold_indices, sha256, write_json
from benchmarks.quest_or_training.train import restore_weights, save_weights, training_batches
from .model import ThreeVectorEncoder
from .objectives import multi_vector_infonce, multi_vector_reinforce

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def retrieve(encoder, rows, documents, batch_size, max_k):
    was_training = encoder.model.training
    encoder.model.eval()
    rankings, similarities = [], []
    try:
        for start in range(0, len(rows), batch_size):
            vectors = encoder.encode([r["query"] for r in rows[start:start + batch_size]])
            rankings.extend((vectors @ documents.T).topk(max_k, dim=-1).indices.cpu().tolist())
            cosine = vectors @ vectors.transpose(-1, -2)
            similarities.extend(cosine[:, [0, 0, 1], [1, 2, 2]].cpu().tolist())
    finally:
        encoder.model.train(was_training)
    return rankings, similarities


def score_rankings(rows, rankings, titles, k):
    predictions = []
    for row, heads in zip(rows, rankings):
        ids = dict.fromkeys(index for head in heads for index in head[:k])
        predictions.append({"id": row["id"], "query": row["query"],
                            "docs": [titles[index] for index in ids]})
    summary, details = evaluate(rows, predictions)
    return summary, predictions, details


def choose_cutoff(rows, rankings, titles, max_k):
    curve = []
    for k in range(1, max_k + 1):
        result, _, _ = score_rankings(rows, rankings, titles, k)
        curve.append({"k": k, **result["overall"]})
    return max(curve, key=lambda r: r["f1"]), curve


def diagnostics(rows, rankings, titles, k, similarities):
    unique_relevant = [0, 0, 0]
    duplicate_count = 0
    for row, ranked in zip(rows, rankings):
        heads = [{titles[i] for i in head[:k]} for head in ranked]
        duplicate_count += sum(map(len, heads)) - len(set.union(*heads))
        for i in range(3):
            other = set.union(*(heads[j] for j in range(3) if j != i))
            unique_relevant[i] += len((heads[i] - other) & set(row["docs"]))
    count = len(rows)
    return {
        "mean_duplicate_retrievals": duplicate_count / count,
        "mean_unique_relevant_per_slot": [value / count for value in unique_relevant],
        "mean_pairwise_slot_cosine": [sum(row[i] for row in similarities) / count for i in range(3)],
    }


@torch.no_grad()
def freeze_reference_batches(encoder, rows, config):
    """Match each training microbatch exactly: BF16 padding changes amplify in KL."""
    was_training = encoder.model.training
    cached = getattr(encoder, "evaluation_kv_cache", False)
    encoder.model.eval()
    encoder.evaluation_kv_cache = False
    references = []
    try:
        for step, batch in enumerate(training_batches(len(rows), config), start=1):
            chunks = []
            for start in range(0, len(batch), config["micro_batch_size"]):
                ids = batch[start:start + config["micro_batch_size"]]
                chunks.append(encoder.encode([rows[i]["query"] for i in ids]))
            references.append(torch.cat(chunks))
            if step % 40 == 0 or step == config["steps"]:
                print(json.dumps({"reference_batches": step, "total_batches": config["steps"]}), flush=True)
            if step == config["steps"]:
                break
    finally:
        encoder.evaluation_kv_cache = cached
        encoder.model.train(was_training)
    return references


def train_one(config, model_path, documents, titles, train_rows, dev_rows, test_rows,
              directory, device, provenance, encoder_factory=ThreeVectorEncoder,
              started_at_monotonic=None):
    config = dict(config)
    method = config.setdefault("method", "infonce")
    if method not in ("infonce", "rl"):
        raise ValueError("Unknown training method")
    epoch_steps = config["epochs"] * math.ceil(len(train_rows) / config["batch_size"])
    config["steps"] = min(epoch_steps, config.get("max_steps", epoch_steps))
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    started = time.monotonic() if started_at_monotonic is None else started_at_monotonic
    update_deadline = (started + config.get("max_run_seconds", float("inf"))
                       - config.get("finalization_reserve_seconds", 0))
    stop_reason = "max_steps" if config["steps"] < epoch_steps else "epochs_complete"
    encoder = encoder_factory(model_path, config, device)
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    lookup = {title: i for i, title in enumerate(titles)}
    gold = gold_indices(train_rows, lookup, device)
    gold_indices(dev_rows, lookup, device)
    write_json(directory / "run.json", {
        "method": "multi_vector_" + method, "variant": config["variant"], "config": config,
        "provenance": provenance, "trainable_parameters": sum(p.numel() for p in parameters),
        "eligible_training_queries": len(train_rows), "validation_queries": len(dev_rows),
        "maximum_training_examples": min(config["epochs"] * len(train_rows),
                                           config["steps"] * config["batch_size"]),
        "document_count": len(documents), "output_shape": [3, documents.shape[-1]],
        "document_encoder": "frozen", "query_training": "LoRA q/k/v/o plus three FP32 slots",
        "feedback_norm": (encoder.model.feedback_norm.item()
                          if hasattr(encoder.model, "feedback_norm") else None),
        "feedback_gradient": "end-to-end, no detach", "cache_during_training": False,
        "loss": ("Union answer-set F1 REINFORCE plus joint Gaussian KL to initialization"
                 if method == "rl" else "logsumexp over slots, then full-corpus multi-positive InfoNCE"),
    })
    initial_rankings, similarities = retrieve(
        encoder, dev_rows, documents, config["eval_batch_size"], config["max_k"])
    best, curve = choose_cutoff(dev_rows, initial_rankings, titles, config["max_k"])
    initial = dict(best)
    best_step, best_trained, best_trained_step = 0, None, None
    save_weights(encoder, directory / "best.pt")
    write_json(directory / "initial_validation.json", {
        "best": initial, "curve": curve,
        "diagnostics": diagnostics(dev_rows, initial_rankings, titles, best["k"], similarities),
    })
    reference, reward_k = None, None
    if method == "rl":
        reward_k = initial["k"]
        reference = freeze_reference_batches(encoder, train_rows, config)
        write_json(directory / "rl_reference.json", {
            "initial_checkpoint_sha256": sha256(directory / "best.pt"),
            "shape": [sum(len(batch) for batch in reference), *reference[0].shape[1:]],
            "batching": "Exact seeded training microbatches and padding, without KV cache",
            "reward_k": reward_k,
            "reward_cutoff_selection": "Initial validation F1; fixed before updates",
            "sigma_per_coordinate": config["exploration_radius"] / math.sqrt(documents.shape[-1]),
            "kl_reduction": "Sum over all three slots and dimensions, mean over queries",
            "exploration": "Independent output noise after all three means are computed",
        })
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    encoder.model.train()
    examples_seen = 0
    with (directory / "metrics.jsonl").open("w") as log:
        for step, batch in enumerate(training_batches(len(train_rows), config), start=1):
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            objective_metrics = {}
            for start in range(0, len(batch), config["micro_batch_size"]):
                ids = batch[start:start + config["micro_batch_size"]]
                embeddings = encoder.encode([train_rows[i]["query"] for i in ids])
                if method == "rl":
                    loss, extra = multi_vector_reinforce(
                        embeddings, reference[step - 1][start:start + len(ids)],
                        documents, [gold[i] for i in ids], reward_k,
                        config["samples_per_query"], config["exploration_radius"],
                        config["kl_weight"], generator)
                    if step == 1 and extra["kl"] > 1e-3:
                        raise RuntimeError("Initial KL is nonzero; reference batching or computation differs")
                else:
                    loss = multi_vector_infonce(
                        embeddings, documents, [gold[i] for i in ids], config["temperature"])
                    extra = {}
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                weight = len(ids) / len(batch)
                for key, value in extra.items():
                    objective_metrics[key] = objective_metrics.get(key, 0.0) + weight * value
                (loss * weight).backward()
                loss_value += loss.item() * weight
                del embeddings, loss
            slot_grads = (encoder.model.slots.grad.float().norm(dim=-1).tolist()
                          if hasattr(encoder.model, "slots") else [])
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            examples_seen += len(batch)
            metrics = {**objective_metrics, "step": step, "examples_seen": examples_seen, "loss": loss_value,
                       "grad_norm": norm.item(), "slot_grad_norms": slot_grads,
                       "update_seconds": time.monotonic() - step_started}
            time_limit_reached = time.monotonic() >= update_deadline
            if time_limit_reached:
                stop_reason = "wall_time_budget"
            if step % config["eval_every"] == 0 or step == config["steps"] or time_limit_reached:
                rankings, similarities = retrieve(
                    encoder, dev_rows, documents, config["eval_batch_size"], config["max_k"])
                current, curve = choose_cutoff(dev_rows, rankings, titles, config["max_k"])
                metrics["validation"] = current
                metrics["diagnostics"] = diagnostics(
                    dev_rows, rankings, titles, current["k"], similarities)
                write_json(directory / f"validation-{step:04d}.json", curve)
                save_weights(encoder, directory / "last.pt")
                if best_trained is None or current["f1"] > best_trained["f1"]:
                    best_trained, best_trained_step = current, step
                    save_weights(encoder, directory / "best_trained.pt")
                if current["f1"] > best["f1"]:
                    best, best_step = current, step
                    save_weights(encoder, directory / "best.pt")
            metrics["elapsed_seconds"] = time.monotonic() - started
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            write_json(directory / "progress.json", {"stage": "training", **metrics,
                                                      "total_steps": config["steps"]})
            print(json.dumps({"variant": config["variant"], **metrics}), flush=True)
            if step == config["steps"] or time_limit_reached:
                break
    selection = {"checkpoint_step": best_step, "k": best["k"], "validation": best,
                 "initial_validation": initial, "rl_reward_k": reward_k,
                 "checkpoint_sha256": sha256(directory / "best.pt")}
    write_json(directory / "selection.json", selection)
    write_json(directory / "best_trained_selection.json", {
        "checkpoint_step": best_trained_step, "validation": best_trained,
        "checkpoint_sha256": sha256(directory / "best_trained.pt"),
    })
    restore_weights(encoder, directory / "best.pt")
    # Checkpoint and cutoff are saved before test labels enter evaluation.
    gold_indices(test_rows, lookup, device)
    rankings, similarities = retrieve(
        encoder, test_rows, documents, config["eval_batch_size"], best["k"])
    summary, predictions, details = score_rankings(test_rows, rankings, titles, best["k"])
    (directory / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    write_json(directory / "test_details.json", details)
    result = {"method": method, "variant": config["variant"], **selection, "test": summary,
              "test_diagnostics": diagnostics(test_rows, rankings, titles, best["k"], similarities),
              "elapsed_seconds": time.monotonic() - started, "stop_reason": stop_reason,
              "completed_steps": step, "examples_seen": examples_seen}
    if str(device).startswith("cuda"):
        result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    write_json(directory / "summary.json", result)
    write_json(directory / "progress.json", {"stage": "complete", "step": step,
                                              "examples_seen": examples_seen,
                                              "stop_reason": stop_reason})
    return result


def validate_config(config):
    if config.get("method", "infonce") not in ("infonce", "rl"):
        raise ValueError("Unknown training method")
    for key in ("epochs", "batch_size", "micro_batch_size", "eval_batch_size", "eval_every",
                "max_length", "lora_rank", "lora_alpha", "max_k"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["micro_batch_size"] > config["batch_size"]:
        raise ValueError("micro_batch_size exceeds effective batch_size")
    if config["learning_rate"] <= 0 or config["temperature"] <= 0 or config["slot_init_noise"] < 0:
        raise ValueError("Invalid training parameter")
    if "max_steps" in config and (not isinstance(config["max_steps"], int) or config["max_steps"] < 1):
        raise ValueError("max_steps must be a positive integer")
    if "max_run_seconds" in config:
        if not 0 < config.get("finalization_reserve_seconds", 0) < config["max_run_seconds"]:
            raise ValueError("Reserve positive finalization time within the run budget")
    if config.get("method") == "rl":
        if (not isinstance(config["samples_per_query"], int) or config["samples_per_query"] < 2
                or config["exploration_radius"] <= 0 or config["kl_weight"] < 0):
            raise ValueError("Invalid RL sampling or KL configuration")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("infonce", "rl"), default="infonce")
    parser.add_argument("--variant", required=True, choices=("one_pass", "three_pass"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=HERE.parent / "quest_or_balanced")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh run directory")
    config_path = args.config or HERE / ("config.rl.json" if args.method == "rl" else "config.json")
    config = json.loads(config_path.read_text())
    config["method"] = args.method
    config["variant"] = args.variant
    validate_config(config)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps({"stage": "verifying_data", "variant": args.variant}), flush=True)
    train, dev, test, manifest = load_splits(args.split, args.dataset)
    source_manifest = json.loads((args.dataset / "manifest.json").read_text())
    print(json.dumps({"stage": "loading_corpus", "training_queries": len(train)}), flush=True)
    documents, titles, corpus = load_augmented_corpus(
        args.work_dir, args.dataset, args.embeddings, args.device)
    if config["max_k"] > len(documents):
        raise ValueError("Retrieval cutoff exceeds corpus size")
    provenance = {
        "dataset": manifest, "source_dataset": source_manifest, "corpus": corpus,
        "split_manifest_sha256": sha256(args.split / "manifest.json"),
        "model": "Qwen/Qwen3-Embedding-8B", "revision": source_manifest["corpus"]["model_revision"],
        "code_sha256": {str(path.relative_to(HERE.parent)): sha256(path)
                        for name in ("quest_or_multivector", "quest_or_balanced", "quest_or_training", "quest_or_synthetic")
                        for path in sorted((HERE.parent / name).glob("*.py"))},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "peft", "numpy")},
    }
    result = train_one(config, args.work_dir / "models/Qwen3-Embedding-8B", documents,
                       titles, train, dev, test, args.output, args.device, provenance,
                       started_at_monotonic=started)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
