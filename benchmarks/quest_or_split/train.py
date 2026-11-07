"""Evaluate frozen thirds and train separate rank-32 RL/InfoNCE query encoders."""
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
from benchmarks.quest_or_training.train import (
    QwenEncoder, encode_rows, restore_weights, save_weights, training_batches,
)
from .objectives import infonce, piece_rankings, prepare_documents, reinforce

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def retrieve(encoder, rows, documents, batch_size, max_k):
    was_training = encoder.model.training
    encoder.model.eval()
    rankings = []
    try:
        for start in range(0, len(rows), batch_size):
            means = encoder.encode([r["query"] for r in rows[start:start + batch_size]])
            rankings.extend(piece_rankings(means, documents, max_k).cpu().tolist())
    finally:
        encoder.model.train(was_training)
    return rankings


def score_rankings(rows, rankings, titles, k):
    if len(rows) != len(rankings):
        raise ValueError("Expected rankings for every query")
    predictions = []
    for row, pieces in zip(rows, rankings):
        ids = dict.fromkeys(index for piece in pieces for index in piece[:k])
        predictions.append({"id": row["id"], "query": row["query"],
                            "docs": [titles[index] for index in ids]})
    summary, details = evaluate(rows, predictions)
    return summary, predictions, details


def choose_cutoff(rows, rankings, titles, max_k):
    curve = []
    for k in range(1, max_k + 1):
        result, _, _ = score_rankings(rows, rankings, titles, k)
        curve.append({"k": k, **result["overall"]})
    return max(curve, key=lambda row: row["f1"]), curve


def retrieval_diagnostics(rows, rankings, titles, k):
    duplicates, sizes, relevant = [], [], [0, 0, 0]
    for row, ranked in zip(rows, rankings):
        pieces = [{titles[i] for i in piece[:k]} for piece in ranked]
        merged = set.union(*pieces)
        sizes.append(len(merged))
        duplicates.append(sum(map(len, pieces)) - len(merged))
        for i in range(3):
            other = set.union(*(pieces[j] for j in range(3) if j != i))
            relevant[i] += len((pieces[i] - other) & set(row["docs"]))
    return {"mean_union_size": sum(sizes) / len(rows),
            "mean_duplicate_retrievals": sum(duplicates) / len(rows),
            "mean_unique_relevant_per_piece": [value / len(rows) for value in relevant]}


def evaluate_selection(encoder, rows, documents, titles, config, directory, selection):
    rankings = retrieve(encoder, rows, documents, config["eval_batch_size"], selection["k"])
    summary, predictions, details = score_rankings(rows, rankings, titles, selection["k"])
    (directory / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    write_json(directory / "test_details.json", details)
    result = {**selection, "test": summary,
              "diagnostics": retrieval_diagnostics(rows, rankings, titles, selection["k"])}
    write_json(directory / "summary.json", result)
    return result


def train_one(method, config, model_path, documents, titles, train_rows, dev_rows, test_rows,
              directory, device, provenance, encoder_factory=QwenEncoder):
    if method not in ("frozen", "rl", "infonce"):
        raise ValueError("Unknown method")
    config = dict(config)
    config["steps"] = config["epochs"] * math.ceil(len(train_rows) / config["batch_size"])
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    started = time.monotonic()
    encoder = encoder_factory(model_path, config, device)
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    lookup = {title: i for i, title in enumerate(titles)}
    gold = gold_indices(train_rows, lookup, device) if method != "frozen" else []
    gold_indices(dev_rows, lookup, device)
    write_json(directory / "run.json", {
        "method": method, "config": config, "provenance": provenance,
        "trainable_parameters": sum(p.numel() for p in parameters) if method != "frozen" else 0,
        "training_queries": len(train_rows), "validation_queries": len(dev_rows),
        "document_count": len(titles), "slice_dimensions": [part.shape[-1] for part in documents],
        "document_encoder": "frozen, matching coordinate slices normalized separately",
        "query_training": "LoRA q/k/v/o; original last-token pooling; no new head or slots",
        "loss": "union-F1 Gaussian REINFORCE" if method == "rl" else
                "full-corpus multi-positive InfoNCE over logsumexp of three slice scores",
    })
    initial_rankings = retrieve(encoder, dev_rows, documents, config["eval_batch_size"], config["max_k"])
    best, curve = choose_cutoff(dev_rows, initial_rankings, titles, config["max_k"])
    initial = dict(best)
    write_json(directory / "initial_validation.json", {"best": initial, "curve": curve})
    if method == "frozen":
        selection = {"method": method, "checkpoint_step": 0, "k": best["k"], "validation": best}
        write_json(directory / "selection.json", selection)
        result = evaluate_selection(encoder, test_rows, documents, titles, config, directory, selection)
        write_json(directory / "progress.json", {"stage": "complete", "step": 0})
        return result
    if not parameters:
        raise ValueError("No trainable parameters")
    save_weights(encoder, directory / "best.pt")
    reference = None
    if method == "rl":
        write_json(directory / "progress.json", {"stage": "encoding_initial_policy"})
        reference = encode_rows(encoder, train_rows, config["eval_batch_size"]).detach()
    reward_k = initial["k"]
    best_step, best_trained, best_trained_step = 0, None, None
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    encoder.model.train()
    examples_seen = 0
    with (directory / "metrics.jsonl").open("w") as log:
        for step, batch in enumerate(training_batches(len(train_rows), config), start=1):
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            metrics = {"loss": 0.0}
            for start in range(0, len(batch), config["micro_batch_size"]):
                ids = batch[start:start + config["micro_batch_size"]]
                means = encoder.encode([train_rows[i]["query"] for i in ids])
                if method == "rl":
                    loss, extra = reinforce(
                        means, reference[ids], documents, [gold[i] for i in ids], reward_k,
                        config["samples_per_query"], config["exploration_radius"],
                        config["kl_weight"], generator)
                else:
                    loss = infonce(means, documents, [gold[i] for i in ids], config["temperature"])
                    extra = {}
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                weight = len(ids) / len(batch)
                (loss * weight).backward()
                metrics["loss"] += loss.item() * weight
                for key, value in extra.items():
                    metrics[key] = metrics.get(key, 0.0) + weight * value
                del means, loss
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            examples_seen += len(batch)
            metrics.update(step=step, examples_seen=examples_seen, grad_norm=norm.item(),
                           update_seconds=time.monotonic() - step_started)
            if step % config["eval_every"] == 0 or step == config["steps"]:
                rankings = retrieve(encoder, dev_rows, documents, config["eval_batch_size"], config["max_k"])
                current, curve = choose_cutoff(dev_rows, rankings, titles, config["max_k"])
                metrics["validation"] = current
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
            print(json.dumps({"method": method, **metrics}), flush=True)
    selections = {
        "selected": {"checkpoint_step": best_step, "k": best["k"], "validation": best,
                     "checkpoint": "best.pt"},
        "best_trained": {"checkpoint_step": best_trained_step, "k": best_trained["k"],
                         "validation": best_trained, "checkpoint": "best_trained.pt"},
        "final": {"checkpoint_step": config["steps"], "k": current["k"],
                  "validation": current, "checkpoint": "last.pt"},
    }
    for selection in selections.values():
        selection.update(method=method, rl_reward_k=reward_k if method == "rl" else None,
                         checkpoint_sha256=sha256(directory / selection["checkpoint"]))
    # All checkpoint/cutoff decisions are fixed before any held-out test score.
    write_json(directory / "selection.json", selections)
    gold_indices(test_rows, lookup, device)
    results = {}
    for label, selection in selections.items():
        restore_weights(encoder, directory / selection["checkpoint"])
        destination = directory / label
        destination.mkdir()
        results[label] = evaluate_selection(
            encoder, test_rows, documents, titles, config, destination, selection)
    results["elapsed_seconds"] = time.monotonic() - started
    if str(device).startswith("cuda"):
        results["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    write_json(directory / "summary.json", results)
    write_json(directory / "progress.json", {"stage": "complete", "step": config["steps"],
                                              "examples_seen": examples_seen})
    return results


def validate_config(config):
    for key in ("epochs", "batch_size", "micro_batch_size", "eval_batch_size", "eval_every",
                "max_length", "lora_rank", "lora_alpha", "max_k", "samples_per_query"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["micro_batch_size"] > config["batch_size"] or config["samples_per_query"] < 2:
        raise ValueError("Invalid batch/sample settings")
    if min(config["learning_rate"], config["temperature"], config["exploration_radius"]) <= 0:
        raise ValueError("Invalid learning/exploration settings")
    if config["kl_weight"] < 0:
        raise ValueError("KL weight must be nonnegative")
    if not 0 < config.get("gpu_memory_fraction", 1.0) <= 1:
        raise ValueError("GPU memory fraction must be in (0, 1]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("frozen", "rl", "infonce"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=HERE.parent / "quest_or_balanced")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    config = json.loads(args.config.read_text())
    validate_config(config)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    if str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(config.get("gpu_memory_fraction", 1.0), args.device)
    print(json.dumps({"stage": "verifying_data", "method": args.method}), flush=True)
    train, dev, test, manifest = load_splits(args.split, args.dataset)
    corpus_manifest = json.loads((args.dataset / "manifest.json").read_text())
    raw, titles, corpus = load_augmented_corpus(args.work_dir, args.dataset, args.embeddings, args.device)
    documents = prepare_documents(raw)
    del raw
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    if config["max_k"] > len(titles):
        raise ValueError("Retrieval cutoff exceeds corpus size")
    provenance = {
        "dataset": manifest, "corpus": corpus,
        "model": "Qwen/Qwen3-Embedding-8B", "revision": corpus_manifest["corpus"]["model_revision"],
        "split_manifest_sha256": sha256(args.split / "manifest.json"),
        "code_sha256": {str(path.relative_to(HERE.parent)): sha256(path)
                        for name in ("quest_or_split", "quest_or_balanced", "quest_or_training", "quest_or_synthetic")
                        for path in sorted((HERE.parent / name).glob("*.py"))},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "peft", "numpy")},
    }
    result = train_one(args.method, config, args.work_dir / "models/Qwen3-Embedding-8B",
                       documents, titles, train, dev, test, args.output, args.device, provenance)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
