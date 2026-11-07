"""Train and evaluate ordered joint objectives without changing historical runners."""
import argparse
import collections
import json
import math
import time
from pathlib import Path

import torch

from benchmarks.quest_or_spec_v2.build import HERE as DATASET, sha256, write_json, write_rows
from benchmarks.quest_or_spec_v2.data import load_corpus, load_splits
from benchmarks.quest_or_spec_v2.metrics import average, evaluate, set_metrics
from benchmarks.quest_or_spec_v2.run import predictions, retrieve, select_cutoff
from benchmarks.quest_or_training.data import gold_indices
from benchmarks.quest_or_training.train import restore_weights, save_weights, training_batches
from benchmarks.quest_or_multivector.objectives import multi_vector_infonce
from .model import make_encoder
from .objectives import ordered_clause_infonce

HERE = Path(__file__).resolve().parent


def slot_diagnostics(rows, ranks, titles, k):
    slots = []
    for slot in range(3):
        items = [set_metrics(r["component_docs"][slot], [titles[i] for i in heads[slot][:k]])
                 for r, heads in zip(rows, ranks)
                 if slot < len(r["component_docs"]) and r["component_docs"][slot]]
        slots.append({"slot": slot+1, "scorable_queries": len(items), **average(items)})
    prefix = {}
    for count in (1, 2, 3):
        summary, _ = evaluate(rows, predictions(rows, [r[:count] for r in ranks], titles, budget=20))
        prefix[str(count)] = summary["overall"]
    return {"slot_against_same_position_component": slots,
            "prefix_fixed_budget_20": prefix,
            "note": "Diagnostics only; primary metrics still use the final merged result set. No matching."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", required=True,
                   choices=("pooled_mlp", "attention_mlp", "one_pass", "three_pass"))
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/quest-ordered-joint-20261106"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--corpus-root", type=Path, default=Path("/home/ubuntu/quest-spec-v2-20261106"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--micro-batch-size", type=int)
    p.add_argument("--clause-weight", type=float)
    args = p.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(.58)
    config = {**json.loads((DATASET / "protocol.json").read_text()),
              **json.loads((HERE / "config.json").read_text()), "variant": args.architecture}
    for name in ("epochs", "micro_batch_size"):
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                raise ValueError(f"{name} must be positive")
            config[name] = value
    if config["micro_batch_size"] > config["batch_size"]:
        raise ValueError("Microbatch cannot exceed the effective batch size")
    if args.clause_weight is not None:
        if not math.isfinite(args.clause_weight) or args.clause_weight < 0:
            raise ValueError("Clause weight must be finite and nonnegative")
        config["clause_weight"] = args.clause_weight
    joint = config["clause_weight"] > 0
    method = "ordered_joint_infonce" if joint else "document_infonce"
    config["schedule"] = f"{method} from the first update, {config['epochs']} epoch(s)"
    config["decomposition_supervision"] = joint
    if not joint:
        config["assignment"] = "none; no decomposition targets"
        config["priority"] = "no explicit priority supervision"
    torch.manual_seed(config["seed"])
    train, validation, test, provenance = load_splits()
    steps_per_epoch = math.ceil(len(train) / config["batch_size"])
    config["steps"] = config["epochs"] * steps_per_epoch
    config["steps_per_epoch"] = steps_per_epoch
    teacher_dir = args.root / "teachers"
    teacher_manifest = None
    if joint:
        teacher_manifest = json.loads((teacher_dir / "manifest.json").read_text())
        if (teacher_manifest["provenance"]["manifest_sha256"] != provenance["manifest_sha256"]
                or sha256(teacher_dir / "teachers.pt") != teacher_manifest["teachers_sha256"]):
            raise ValueError("Teacher cache provenance mismatch")
        cache = torch.load(teacher_dir / "teachers.pt", map_location="cpu", weights_only=True)
        if cache["query_ids"] != [r["id"] for r in train]:
            raise ValueError("Teacher rows do not match the exact training order")
        teachers, target_ids = cache["vectors"].cuda(), cache["target_ids"].cuda()
        teachers.requires_grad_(False)
        del cache
    base = Path("/home/ubuntu/quest-or-baseline-20261106")
    documents, titles, corpus_provenance = load_corpus(
        base, Path("/home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic"),
        Path("/home/ubuntu/quest-or-synthetic-20261106/embeddings"), args.corpus_root / "embeddings", "cuda")
    encoder = make_encoder(str(base / "models/Qwen3-Embedding-8B"), config, "cuda")
    parameters = [v for v in encoder.model.parameters() if v.requires_grad]
    lookup = {t: i for i, t in enumerate(titles)}
    gold = gold_indices(train, lookup, "cuda")
    gold_indices(validation, lookup, "cuda")
    write_json(args.output / "run.json", {
        "architecture": args.architecture, "method": method, "config": config,
        "provenance": provenance, "teacher_manifest": teacher_manifest,
        "corpus": corpus_provenance, "documents": len(titles),
        "trainable_parameters": sum(v.numel() for v in parameters),
        "code_sha256": {str(f.relative_to(HERE.parent)): sha256(f)
                         for package in (HERE, DATASET, HERE.parent / "quest_or_training",
                                         HERE.parent / "quest_or_multivector")
                         for f in sorted(package.glob("*.py"))},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "document_encoder": "frozen", "teacher_encoder": "original frozen Qwen; training clauses only" if joint else "unused",
        "query_encoder": "fresh rank-32/alpha-64 q/k/v/o LoRA, plus architecture-specific head/slots",
        "loss": "collective full-corpus multi-positive document InfoNCE" + (" + ordered clause InfoNCE" if joint else ""),
        "reward": "none; F1 only selects validation checkpoint/cutoff and evaluates test",
        "order_note": "Canonical source component order; not a learned salience ranking. QUEST paraphrases may reorder mentions." if joint else "No decomposition supervision or slot-to-clause assignment."})

    def validation_result(step):
        ranks, timing = retrieve(encoder, validation, documents, args.architecture, config["eval_batch_size"])
        selected, curve = select_cutoff(validation, ranks, titles, 20)
        summary, _ = evaluate(validation, predictions(validation, ranks, titles, k=selected["k"]))
        write_json(args.output / f"validation-{step:04d}.json", {
            "selected": selected, "curve": curve, "metrics": summary, "timing": timing,
            "slot_diagnostics": slot_diagnostics(validation, ranks, titles, selected["k"])})
        return selected, summary

    initial, _ = validation_result(0)
    save_weights(encoder, args.output / "initial.pt")
    best, best_step = None, None
    examples_seen, seen_sources = 0, collections.Counter()
    stop_reason = "epochs_complete"
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    encoder.model.train()
    with (args.output / "metrics.jsonl").open("w") as log:
        for step, batch in enumerate(training_batches(len(train), config), 1):
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            metrics = collections.defaultdict(float)
            for start in range(0, len(batch), config["micro_batch_size"]):
                ids = batch[start:start+config["micro_batch_size"]]
                embeddings = encoder.encode([train[i]["query"] for i in ids])
                doc_loss = multi_vector_infonce(embeddings, documents, [gold[i] for i in ids], config["temperature"])
                if joint:
                    clause_loss, extra = ordered_clause_infonce(embeddings, teachers, target_ids[ids], config["clause_temperature"])
                else:
                    clause_loss, extra = embeddings.new_zeros(()), {}
                loss = doc_loss + config["clause_weight"] * clause_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite joint loss")
                weight = len(ids) / len(batch)
                (loss * weight).backward()
                for key, value in {"loss": loss.item(), "document_loss": doc_loss.item(),
                                   "clause_loss": clause_loss.item(), **extra}.items():
                    metrics[key] += weight * value
            if hasattr(encoder.model, "slots"):
                metrics["slot_grad_norms"] = encoder.model.slots.grad.float().norm(dim=-1).tolist()
            if hasattr(encoder.model, "head") and hasattr(encoder.model.head, "readers"):
                metrics["reader_grad_norms"] = encoder.model.head.readers.grad.float().norm(dim=-1).tolist()
            metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True).item()
            optimizer.step()
            examples_seen += len(batch)
            seen_sources.update(train[i]["source"] for i in batch)
            metrics.update(step=step, examples_seen=examples_seen, seen_sources=dict(seen_sources),
                           update_seconds=time.monotonic()-before,
                           epoch=(step-1)//steps_per_epoch+1,
                           epoch_progress=step/steps_per_epoch)
            stop = time.monotonic()-started >= config["max_run_seconds"]-config["finalization_reserve_seconds"]
            within_epoch = (step-1) % steps_per_epoch + 1
            if within_epoch % config["eval_every"] == 0 or within_epoch == steps_per_epoch or stop:
                current, summary = validation_result(step)
                metrics["validation"] = summary
                save_weights(encoder, args.output / "last.pt")
                if within_epoch == steps_per_epoch:
                    save_weights(encoder, args.output / f"epoch-{step//steps_per_epoch}.pt")
                if best is None or current["f1"] > best["f1"]:
                    best, best_step = current, step
                    save_weights(encoder, args.output / "best.pt")
            metrics["elapsed_seconds"] = time.monotonic()-started
            log.write(json.dumps(dict(metrics))+"\n")
            log.flush()
            write_json(args.output / "progress.json", {"stage": "training", **dict(metrics)})
            if step % 10 == 0 or step == config["steps"] or stop:
                print(json.dumps(dict(metrics)), flush=True)
            if stop:
                stop_reason = "wall_time_budget"
                break
    restore_weights(encoder, args.output / "best.pt")
    selection = {"step": best_step, "k": best["k"], "validation": best, "initial_validation": initial,
                 "checkpoint_sha256": sha256(args.output / "best.pt"),
                 "selection_rule": "Best updated checkpoint on validation union F1; no test tuning"}
    write_json(args.output / "selection.json", selection)
    gold_indices(test, lookup, "cuda")
    ranks, timing = retrieve(encoder, test, documents, args.architecture, config["eval_batch_size"])
    predicted = predictions(test, ranks, titles, k=best["k"])
    summary, details = evaluate(test, predicted)
    write_rows(args.output / "test_predictions.jsonl", predicted)
    write_rows(args.output / "test_details.jsonl", details)
    write_json(args.output / "test_rankings.json", {"ids": [r["id"] for r in test], "corpus_indices": ranks})
    fixed = {str(b): evaluate(test, predictions(test, ranks, titles, budget=b))[0] for b in (10, 20, 60, 100)}
    summary.update(architecture=args.architecture, method=method, selection=selection,
                   fixed_budget=fixed, slot_diagnostics=slot_diagnostics(test, ranks, titles, best["k"]),
                   completed_steps=step, examples_seen=examples_seen, seen_sources=dict(seen_sources),
                   stop_reason=stop_reason, timing=timing, elapsed_seconds=time.monotonic()-started,
                   peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                   dataset_manifest_sha256=provenance["manifest_sha256"])
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "progress.json", {"stage": "complete", "step": step, "examples_seen": examples_seen})
    print(json.dumps({"completed": args.architecture, "overall": summary["overall"]}), flush=True)


if __name__ == "__main__":
    main()
