"""Train only against frozen interpretation embeddings; no retrieval imports."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from .model import InterpretationModel, MODEL_ID, REVISION, diagnostics, matching_loss, tokenize


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_weights(model, path):
    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    started = time.time()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    backbone = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                        attn_implementation="sdpa").to("cuda")
    backbone.config.use_cache = False
    backbone = get_peft_model(backbone, LoraConfig(r=32, lora_alpha=64, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    model = InterpretationModel(backbone, tokenizer.eos_token_id, args.architecture)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
    rows, gold = {}, {}
    for name in ("train", "validation", "test"):
        rows[name] = [json.loads(line) for line in (args.data / f"{name}.jsonl").read_text().splitlines()]
        gold[name] = torch.load(args.teacher / f"{name}.pt", map_location="cpu", weights_only=True)
        assert gold[name]["ids"] == [r["id"] for r in rows[name]]
    teacher_manifest = json.loads((args.teacher / "manifest.json").read_text())
    for name in rows:
        assert teacher_manifest["split_sha256"][name] == hashlib.sha256((args.data / f"{name}.jsonl").read_bytes()).hexdigest()
    config = {"architecture": args.architecture, "model": MODEL_ID, "revision": REVISION,
              "seed": 42, "epochs": 3, "batch_size": 32, "microbatch": 8,
              "learning_rate": 1e-4, "lora_rank": 32, "lora_alpha": 64,
              "loss": "minimum one-to-one cosine distance; unmatched third slot masked for 2 labels",
              "counts": {k: len(v) for k, v in rows.items()}, "teacher": teacher_manifest,
              "corpus_access": False, "gpu": torch.cuda.get_device_name(),
              "deadline": args.deadline,
              "trainable_parameters": sum(p.numel() for p in parameters)}
    write_json(args.output / "run.json", config)

    @torch.no_grad()
    def evaluate(name, save=False):
        model.eval()
        vectors = []
        for i in range(0, len(rows[name]), 16):
            tokens = tokenize(tokenizer, [r["query"] for r in rows[name][i:i+16]], "cuda")
            vectors.append(model(**tokens).cpu())
        predictions = torch.cat(vectors)
        result = diagnostics(predictions, gold[name]["targets"], gold[name]["counts"])
        result["by_arity"] = {}
        for n in (2, 3):
            mask = gold[name]["counts"] == n
            if mask.any():
                result["by_arity"][str(n)] = diagnostics(predictions[mask], gold[name]["targets"][mask], gold[name]["counts"][mask])
        if save:
            torch.save({"ids": gold[name]["ids"], "embeddings": predictions,
                        "per_query_loss": matching_loss(predictions, gold[name]["targets"], gold[name]["counts"])},
                       args.output / f"{name}_predictions.pt")
        return result

    original_baselines = {name: diagnostics(gold[name]["original_query_vectors"][:, None].expand(-1, 3, -1),
                          gold[name]["targets"], gold[name]["counts"]) for name in ("validation",)}
    initial = evaluate("validation")
    write_json(args.output / "initial_validation.json", {"slots": initial, "original_query_repeated": original_baselines["validation"]})
    save_weights(model, args.output / "initial.pt")
    best, best_epoch = float("inf"), None
    step, examples = 0, 0
    stopped_for_time = False
    rng = random.Random(42)
    with (args.output / "metrics.jsonl").open("w") as log:
        for epoch in range(1, 4):
            order = list(range(len(rows["train"])))
            rng.shuffle(order)
            model.train()
            for start in range(0, len(order), 32):
                if time.time() > args.deadline - 100:
                    stopped_for_time = True
                    break
                indices = order[start:start+32]
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.
                for offset in range(0, len(indices), 8):
                    ix = indices[offset:offset+8]
                    tokens = tokenize(tokenizer, [rows["train"][i]["query"] for i in ix], "cuda")
                    predictions = model(**tokens)
                    loss = matching_loss(predictions, gold["train"]["targets"][ix].to("cuda"), gold["train"]["counts"][ix]).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss")
                    (loss * len(ix) / len(indices)).backward()
                    loss_value += loss.item() * len(ix) / len(indices)
                grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                examples += len(indices)
                item = {"step": step, "epoch": epoch, "examples_seen": examples,
                        "loss": loss_value, "grad_norm": grad.item(), "seconds": time.time()-started}
                log.write(json.dumps(item) + "\n"); log.flush()
                if step % 10 == 0:
                    print(json.dumps({"architecture": args.architecture, **item}), flush=True)
            current = evaluate("validation")
            write_json(args.output / f"validation_epoch_{epoch}.json", current)
            print(json.dumps({"architecture": args.architecture, "epoch": epoch, "validation": current}), flush=True)
            if current["matching_cosine_loss"] < best:
                best, best_epoch = current["matching_cosine_loss"], epoch
                save_weights(model, args.output / "best.pt")
            if stopped_for_time:
                break
    save_weights(model, args.output / "last.pt")
    # Freeze selection before the first test-model forward pass or test metric.
    write_json(args.output / "selection.json", {"best_epoch": best_epoch, "validation_loss": best,
                                               "step": step, "selection_time": time.time()})
    model.load_state_dict(torch.load(args.output / "best.pt", weights_only=True), strict=False)
    test = evaluate("test", save=True)
    test_baseline = diagnostics(gold["test"]["original_query_vectors"][:, None].expand(-1, 3, -1),
                                gold["test"]["targets"], gold["test"]["counts"])
    result = {"architecture": args.architecture, "best_epoch": best_epoch,
              "steps": step, "examples_seen": examples, "completed_epochs": examples / len(rows["train"]),
              "stopped_for_time": stopped_for_time, "initial_validation": initial,
              "selected_validation_loss": best, "test": test, "test_original_query_repeated": test_baseline,
              "test_loss_improvement_over_original": test_baseline["matching_cosine_loss"]-test["matching_cosine_loss"],
              "wall_seconds": time.time()-started, "peak_gpu_gb": torch.cuda.max_memory_allocated()/1024**3,
              "corpus_access": False}
    write_json(args.output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("mtp", "autoregressive"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    train(parser.parse_args())
