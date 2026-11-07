"""Train two independent whole-query Qwen3 baselines against frozen QUEST vectors.

Run from the repository root: python -m benchmarks.quest_or_training.train --help
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.quest_or.scripts.evaluate import evaluate

from .data import DIMENSION, INSTRUCTION, gold_indices, sha256, write_json
from .objectives import infonce, reinforce

HERE = Path(__file__).resolve().parent


class QwenEncoder:
    """LoRA changes Qwen's query encoder; pooling and output dimension stay fixed."""

    def __init__(self, model_path, config, device):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        self.device, self.max_length = device, config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        base = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        if base.config.hidden_size != DIMENSION:
            raise ValueError("Expected the original 4096-dimensional Qwen3-Embedding-8B")
        base.config.use_cache = False
        self.model = get_peft_model(base, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
        ))
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.enable_input_require_grads()

    def encode(self, texts):
        # Match the frozen corpus pipeline's original-query prompt byte for byte.
        prompts = [f"Instruct: {INSTRUCTION}\nQuery:{text}" for text in texts]
        tokens = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if tokens.input_ids.shape[-1] > self.max_length:
            raise ValueError("Query exceeds max_length; raise the limit instead of silently truncating")
        tokens = tokens.to(self.device)
        # Left padding makes the last position the last real token in every row.
        hidden = self.model(**tokens, use_cache=False).last_hidden_state[:, -1]
        return F.normalize(hidden.float(), dim=-1)


@torch.no_grad()
def encode_rows(encoder, rows, batch_size):
    was_training = encoder.model.training
    encoder.model.eval()
    try:
        batches = []
        for start in range(0, len(rows), batch_size):
            batches.append(encoder.encode([r["query"] for r in rows[start:start + batch_size]]))
            if len(rows) > 1000 and (len(batches) % 200 == 0 or start + batch_size >= len(rows)):
                print(json.dumps({"reference_encoded": min(start + batch_size, len(rows)),
                                  "reference_total": len(rows)}), flush=True)
        return torch.cat(batches)
    finally:
        encoder.model.train(was_training)


@torch.no_grad()
def retrieve(encoder, rows, documents, batch_size, k):
    embeddings = encode_rows(encoder, rows, batch_size)
    return (embeddings @ documents.T).topk(k, dim=-1).indices.cpu().tolist()


def score_rankings(rows, rankings, titles, k):
    predictions = [{"id": row["id"], "query": row["query"],
                    "docs": [titles[index] for index in ranking[:k]]}
                   for row, ranking in zip(rows, rankings)]
    summary, details = evaluate(rows, predictions)
    return summary, predictions, details


def choose_cutoff(rows, rankings, titles, max_k):
    curve = []
    for k in range(1, max_k + 1):
        summary, _, _ = score_rankings(rows, rankings, titles, k)
        curve.append({"k": k, **summary["overall"]})
    # Iteration is ascending: a tie retains the smaller cutoff.
    best = max(curve, key=lambda result: result["f1"])
    return best, curve


def save_weights(encoder, path):
    # Only adapter/trainable weights; no duplicated 8B base checkpoint.
    state = {name: value.detach().cpu().clone()
             for name, value in encoder.model.named_parameters() if value.requires_grad}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_weights(encoder, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    trainable = {name for name, value in encoder.model.named_parameters() if value.requires_grad}
    if set(saved) != trainable:
        raise ValueError("Checkpoint trainable parameter names do not match the encoder")
    encoder.model.load_state_dict(saved, strict=False)


def training_batches(count, config):
    rng = random.Random(config["seed"])
    if "epochs" in config:
        for _ in range(config["epochs"]):
            order = list(range(count))
            rng.shuffle(order)
            for start in range(0, count, config["batch_size"]):
                yield order[start:start + config["batch_size"]]
    else:
        for _ in range(config["steps"]):
            yield rng.sample(range(count), min(config["batch_size"], count))


def train_one(method, config, model_path, documents, titles, train_rows, dev_rows, test_rows,
              directory, device, provenance, encoder_factory=QwenEncoder):
    """One fresh initialization per method. Test evaluation occurs after selection is saved."""
    if method not in ("rl", "infonce"):
        raise ValueError("Unknown training method")
    config = dict(config)
    if "epochs" in config:
        config["steps"] = config["epochs"] * math.ceil(len(train_rows) / config["batch_size"])
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    encoder = encoder_factory(model_path, config, device)
    encoder.model.eval()
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    write_json(directory / "run.json", {
        "method": method, "config": config, "provenance": provenance,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "training_queries": len(train_rows), "validation_queries": len(dev_rows),
        "output_dimension": documents.shape[-1], "document_count": len(documents),
        "document_encoder": "frozen", "query_training": "LoRA, q/k/v/o projections",
    })
    title_to_index = {title: i for i, title in enumerate(titles)}
    gold = gold_indices(train_rows, title_to_index, device)
    gold_indices(dev_rows, title_to_index, device)  # Fail early on missing validation labels.
    reference = (encode_rows(encoder, train_rows, config["batch_size"]).detach()
                 if method == "rl" else None)
    rankings = retrieve(encoder, dev_rows, documents, config["batch_size"], config["max_k"])
    best, curve = choose_cutoff(dev_rows, rankings, titles, config["max_k"])
    initial = dict(best)
    # Reward cutoff is chosen once on development, before any gradient update.
    reward_k = best["k"]
    best_step = 0
    best_trained, best_trained_step = None, None
    best_path = directory / "best.pt"
    save_weights(encoder, best_path)
    write_json(directory / "initial_validation.json", {"best": initial, "curve": curve})
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    encoder.model.train()
    examples_seen = 0
    with (directory / "metrics.jsonl").open("w") as log:
        for step, batch in enumerate(training_batches(len(train_rows), config), start=1):
            examples_seen += len(batch)
            optimizer.zero_grad(set_to_none=True)
            means = encoder.encode([train_rows[i]["query"] for i in batch])
            if method == "rl":
                loss, metrics = reinforce(
                    means, reference[batch], documents, [gold[i] for i in batch], reward_k,
                    config["samples_per_query"], config["exploration_radius"],
                    config["kl_weight"], generator,
                )
            else:
                loss = infonce(means, documents, [gold[i] for i in batch], config["temperature"])
                metrics = {}
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            metrics.update({"step": step, "examples_seen": examples_seen,
                            "loss": loss.item(), "grad_norm": norm.item()})
            if step % config["eval_every"] == 0 or step == config["steps"]:
                rankings = retrieve(encoder, dev_rows, documents, config["batch_size"],
                                    config["max_k"])
                current, curve = choose_cutoff(dev_rows, rankings, titles, config["max_k"])
                metrics["validation"] = current
                write_json(directory / f"validation-{step:04d}.json", curve)
                if best_trained is None or current["f1"] > best_trained["f1"]:
                    best_trained, best_trained_step = current, step
                    save_weights(encoder, directory / "best_trained.pt")
                if current["f1"] > best["f1"]:
                    best, best_step = current, step
                    save_weights(encoder, best_path)
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            print(json.dumps({"method": method, **metrics}), flush=True)
    save_weights(encoder, directory / "last.pt")
    write_json(directory / "best_trained_selection.json", {
        "checkpoint_step": best_trained_step, "k": best_trained["k"],
        "validation": best_trained, "selection_rule": "Best validation checkpoint after an update",
        "checkpoint_sha256": sha256(directory / "best_trained.pt"),
    })
    selection = {"checkpoint_step": best_step, "k": best["k"], "validation": best,
                 "initial_validation": initial, "rl_reward_k": reward_k if method == "rl" else None,
                 "checkpoint_sha256": sha256(best_path)}
    write_json(directory / "selection.json", selection)
    restore_weights(encoder, best_path)
    encoder.model.eval()
    if hasattr(encoder.model, "save_pretrained"):
        encoder.model.save_pretrained(directory / "adapter")
        encoder.tokenizer.save_pretrained(directory / "adapter")
    # The held-out set is introduced only after checkpoint and cutoff are frozen.
    gold_indices(test_rows, title_to_index, device)
    rankings = retrieve(encoder, test_rows, documents, config["batch_size"], best["k"])
    summary, predictions, details = score_rankings(test_rows, rankings, titles, best["k"])
    (directory / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    write_json(directory / "test_details.json", details)
    result = {"method": method, **selection, "test": summary}
    write_json(directory / "summary.json", result)
    return result


def validate_config(config):
    for key in ("steps", "batch_size", "eval_every", "max_length", "lora_rank", "lora_alpha", "max_k"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if "epochs" in config and (not isinstance(config["epochs"], int) or config["epochs"] < 1):
        raise ValueError("epochs must be a positive integer")
    if (config["samples_per_query"] < 2 or config["exploration_radius"] <= 0
            or config["kl_weight"] < 0 or config["temperature"] <= 0
            or config["learning_rate"] <= 0):
        raise ValueError("Invalid loss or optimizer configuration")


def main():
    # All new QUEST/NQ runs share the canonical 3,600/400/1,000 query files.
    from benchmarks.quest_or_synthetic.train import main as run_shared
    run_shared()


if __name__ == "__main__":
    main()
