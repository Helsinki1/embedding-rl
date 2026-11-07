"""One shared training/evaluation path for all eleven specified approaches."""
import argparse
import collections
import json
import math
import os
import time
from pathlib import Path

import torch

from benchmarks.quest_or_training.train import QwenEncoder, restore_weights, save_weights, training_batches
from benchmarks.quest_or_training.data import gold_indices
from benchmarks.quest_or_training import objectives as single
from benchmarks.quest_or_split import objectives as split
from benchmarks.quest_or_multivector import objectives as multi
from benchmarks.quest_or_multivector.model import ThreeVectorEncoder
from benchmarks.quest_or_multivector.train import freeze_reference_batches
from benchmarks.quest_or_llm_union.union import components_for
from .build import HERE, read_rows, sha256, write_json, write_rows
from .data import load_corpus, load_splits
from .metrics import evaluate


def round_robin(heads, budget):
    result, seen = [], set()
    for rank in range(max(map(len, heads), default=0)):
        for head in heads:
            if rank < len(head) and head[rank] not in seen:
                seen.add(head[rank])
                result.append(head[rank])
                if len(result) == budget:
                    return result
    return result


def predictions(rows, rankings, titles, k=None, budget=None):
    result = []
    for row, heads in zip(rows, rankings):
        if budget is None:
            heads = [h[:k] for h in heads]
        ids = round_robin(heads, budget or sum(map(len, heads)))
        result.append({"id": row["id"], "docs": [titles[i] for i in ids]})
    return result


def select_cutoff(rows, rankings, titles, max_k):
    # Union F1 alone selects the checkpoint/cutoff. Other labels cannot influence it.
    curve = []
    for k in range(1, max_k + 1):
        pred = predictions(rows, rankings, titles, k=k)
        scores = []
        for row, p in zip(rows, pred):
            gold, got = set(row["docs"]), set(p["docs"])
            scores.append(2 * len(gold & got) / (len(gold) + len(got)))
        curve.append({"k": k, "f1": sum(scores) / len(scores),
                      "mean_results": sum(len(p["docs"]) for p in pred) / len(pred)})
    return max(curve, key=lambda r: r["f1"]), curve


@torch.no_grad()
def retrieve(encoder, rows, documents, architecture, batch_size, decompositions=None, max_k=100):
    was_training = encoder.model.training
    encoder.model.eval()
    texts, counts = [], []
    for row in rows:
        queries = components_for(row["query"], decompositions[row["query"]])[0] if architecture == "llm" else [row["query"]]
        texts.extend(queries)
        counts.append(len(queries))
    rankings, cosines = [], []
    torch.cuda.synchronize()
    began = time.monotonic()
    try:
        for start in range(0, len(texts), batch_size):
            vectors = encoder.encode(texts[start:start + batch_size])
            if architecture == "split":
                ranked = split.piece_rankings(vectors, documents, max_k)
            else:
                if vectors.ndim == 2:
                    vectors = vectors[:, None]
                ranked = (vectors @ documents.T).topk(max_k, dim=-1).indices
                if vectors.shape[1] == 3:
                    cosine = vectors @ vectors.transpose(-1, -2)
                    cosines.extend(cosine[:, [0, 0, 1], [1, 2, 2]].cpu().tolist())
            rankings.extend(ranked.cpu().tolist())
    finally:
        encoder.model.train(was_training)
    if architecture == "llm":
        grouped, index = [], 0
        for count in counts:
            grouped.append([r[0] for r in rankings[index:index + count]])
            index += count
        rankings = grouped
    torch.cuda.synchronize()
    return rankings, {"query_encoding_and_retrieval_seconds": time.monotonic() - began,
                      "encoder_inputs": len(texts), "queries": len(rows),
                      "mean_pairwise_slot_cosine": [sum(r[i] for r in cosines) / len(cosines) for i in range(3)] if cosines else None}


def objective(architecture, method, embeddings, reference, documents, gold, reward_k, config, generator):
    module = split if architecture == "split" else multi if architecture in ("one_pass", "three_pass") else single
    if method == "infonce":
        fn = module.multi_vector_infonce if module is multi else module.infonce
        return fn(embeddings, documents, gold, config["temperature"]), {}
    fn = module.multi_vector_reinforce if module is multi else module.reinforce
    return fn(embeddings, reference, documents, gold, reward_k, config["samples_per_query"],
              config["exploration_radius"], config["kl_weight"], generator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("single", "split", "one_pass", "three_pass", "llm"), required=True)
    parser.add_argument("--method", choices=("frozen", "infonce", "rl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ubuntu/quest-spec-v2-20261106"))
    args = parser.parse_args()
    if args.architecture == "llm" and args.method != "frozen":
        raise ValueError("LLM control is frozen")
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(.58)
    config = json.loads((HERE / "protocol.json").read_text())
    config["variant"] = args.architecture
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    train, validation, test, provenance = load_splits()
    config["steps"] = config["epochs"] * math.ceil(len(train) / config["batch_size"])
    base = Path("/home/ubuntu/quest-or-baseline-20261106")
    documents, titles, corpus_provenance = load_corpus(
        base, Path("/home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic"),
        Path("/home/ubuntu/quest-or-synthetic-20261106/embeddings"), args.root / "embeddings", "cuda")
    if args.architecture == "split":
        documents = split.prepare_documents(documents)
    encoder_class = ThreeVectorEncoder if args.architecture in ("one_pass", "three_pass") else QwenEncoder
    encoder = encoder_class(str(base / "models/Qwen3-Embedding-8B"), config, "cuda")
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    lookup = {title: i for i, title in enumerate(titles)}
    gold = gold_indices(train, lookup, "cuda")
    gold_indices(validation, lookup, "cuda")
    decompositions = None
    if args.architecture == "llm":
        decomposition_manifest = json.loads((HERE / "decomposition/manifest.json").read_text())
        if decomposition_manifest["manifest_sha256"] != provenance["manifest_sha256"]:
            raise ValueError("Decomposition uses a different split")
        if sha256(HERE / "decomposition/decompositions.jsonl") != decomposition_manifest["cache_sha256"]:
            raise ValueError("Decomposition cache changed")
        decompositions = {r["query"]: r for r in read_rows(HERE / "decomposition/decompositions.jsonl")}
        provenance["decomposition"] = decomposition_manifest
    write_json(args.output / "run.json", {"architecture": args.architecture, "method": args.method,
        "config": config, "provenance": provenance, "corpus": corpus_provenance, "documents": len(titles),
        "trainable_parameters": sum(p.numel() for p in parameters) if args.method != "frozen" else 0,
        "code_sha256": {str(p.relative_to(HERE.parent)): sha256(p) for package in
                        (HERE, HERE.parent / "quest_or_training", HERE.parent / "quest_or_split", HERE.parent / "quest_or_multivector")
                        for p in sorted(package.glob("*.py"))},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "rl_kl_reduction": "Sum over independent policy coordinates/slots, mean over queries; same coefficient across methods"})
    max_k = 60 if args.architecture == "single" else 20

    def validation_result(step):
        ranks, timing = retrieve(encoder, validation, documents, args.architecture, config["eval_batch_size"], decompositions)
        selected, curve = select_cutoff(validation, ranks, titles, max_k)
        summary, _ = evaluate(validation, predictions(validation, ranks, titles, k=selected["k"]))
        write_json(args.output / f"validation-{step:04d}.json", {"selected": selected, "curve": curve,
                                                               "metrics": summary, "timing": timing})
        return selected, summary

    initial, _ = validation_result(0)
    best, best_step = initial, 0
    save_weights(encoder, args.output / "initial.pt")
    examples_seen, seen_sources = 0, collections.Counter()
    stop_reason = "frozen" if args.method == "frozen" else "epochs_complete"
    if args.method != "frozen":
        # Strictly trained checkpoint is the primary result; initialization is a diagnostic.
        best = None
        reference = freeze_reference_batches(encoder, train, config) if args.method == "rl" else None
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
        encoder.model.train()
        with (args.output / "metrics.jsonl").open("w") as log:
            for step, batch in enumerate(training_batches(len(train), config), 1):
                step_start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                metrics = collections.defaultdict(float)
                for start in range(0, len(batch), config["micro_batch_size"]):
                    ids = batch[start:start + config["micro_batch_size"]]
                    embeddings = encoder.encode([train[i]["query"] for i in ids])
                    anchor = reference[step - 1][start:start + len(ids)] if reference is not None else None
                    loss, extra = objective(args.architecture, args.method, embeddings, anchor, documents,
                                            [gold[i] for i in ids], initial["k"], config, generator)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite objective")
                    if step == 1 and extra.get("kl", 0) > 1e-3:
                        raise RuntimeError("Initial KL nonzero: reference computation differs")
                    weight = len(ids) / len(batch)
                    (loss * weight).backward()
                    for key, value in {"loss": loss.item(), **extra}.items():
                        metrics[key] += weight * value
                if hasattr(encoder.model, "slots"):
                    metrics["slot_grad_norms"] = encoder.model.slots.grad.float().norm(dim=-1).tolist()
                metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True).item()
                optimizer.step()
                examples_seen += len(batch)
                seen_sources.update(train[i]["source"] for i in batch)
                metrics.update(step=step, examples_seen=examples_seen, seen_sources=dict(seen_sources),
                               update_seconds=time.monotonic() - step_start)
                stop = time.monotonic() - started >= config["max_run_seconds"] - config["finalization_reserve_seconds"]
                if step % config["eval_every"] == 0 or step == config["steps"] or stop:
                    current, summary = validation_result(step)
                    metrics["validation"] = summary
                    save_weights(encoder, args.output / "last.pt")
                    if best is None or current["f1"] > best["f1"]:
                        best, best_step = current, step
                        save_weights(encoder, args.output / "best.pt")
                metrics["elapsed_seconds"] = time.monotonic() - started
                log.write(json.dumps(dict(metrics)) + "\n")
                log.flush()
                write_json(args.output / "progress.json", dict(metrics))
                if step % 10 == 0 or step == config["steps"] or stop:
                    print(json.dumps(dict(metrics)), flush=True)
                if stop:
                    stop_reason = "wall_time_budget"
                    break
        restore_weights(encoder, args.output / "best.pt")
    selection = {"step": best_step, "k": best["k"], "validation": best, "initial_validation": initial,
                 "rl_reward_k": initial["k"] if args.method == "rl" else None,
                 "checkpoint_sha256": sha256(args.output / ("initial.pt" if args.method == "frozen" else "best.pt")),
                 "selection_rule": "Best validation union F1 among updated checkpoints; smaller k breaks ties"}
    # Freeze choices before reading any test labels.
    write_json(args.output / "selection.json", selection)
    gold_indices(test, lookup, "cuda")
    ranks, timing = retrieve(encoder, test, documents, args.architecture, config["eval_batch_size"], decompositions)
    predicted = predictions(test, ranks, titles, k=best["k"])
    summary, details = evaluate(test, predicted)
    write_rows(args.output / "test_predictions.jsonl", predicted)
    write_rows(args.output / "test_details.jsonl", details)
    write_json(args.output / "test_rankings.json", {"ids": [r["id"] for r in test], "corpus_indices": ranks})
    fixed = {}
    for budget in (10, 20, 60, 100):
        fixed[str(budget)], _ = evaluate(test, predictions(test, ranks, titles, budget=budget))
    summary.update(architecture=args.architecture, method=args.method, selection=selection,
                   fixed_budget=fixed, examples_seen=examples_seen, seen_sources=dict(seen_sources),
                   stop_reason=stop_reason, timing=timing, elapsed_seconds=time.monotonic() - started,
                   peak_gpu_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                   dataset_manifest_sha256=provenance["manifest_sha256"])
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"completed": args.output.name, "overall": summary["overall"], "elapsed": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
