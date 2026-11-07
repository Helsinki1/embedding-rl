"""Cache frozen, ordered clause targets using only the canonical training split."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from benchmarks.quest_or_spec_v2.data import load_splits
from benchmarks.quest_or_spec_v2.build import HERE as DATASET, sha256, write_json, write_rows
from benchmarks.quest_or_training.data import INSTRUCTION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/home/ubuntu/quest-or-baseline-20261106/models/Qwen3-Embedding-8B")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.58)
    train, _, _, provenance = load_splits()
    texts = list(dict.fromkeys(c for r in train for c in r["components"]))
    lookup = {t: i for i, t in enumerate(texts)}
    target_ids = torch.full((len(train), 3), -1, dtype=torch.long)
    for i, row in enumerate(train):
        target_ids[i, :len(row["components"])] = torch.tensor([lookup[c] for c in row["components"]])
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa").cuda().eval()
    model.requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            prompts = [f"Instruct: {INSTRUCTION}\nQuery:{t}" for t in texts[start:start+args.batch_size]]
            batch = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to("cuda")
            if batch.input_ids.shape[-1] > 512:
                raise ValueError("Clause exceeds 512 tokens; no silent truncation")
            hidden = model(**batch, use_cache=False).last_hidden_state[:, -1]
            chunks.append(F.normalize(hidden.float(), dim=-1).cpu())
            if start % (args.batch_size * 20) == 0 or start + args.batch_size >= len(texts):
                progress = {"done": min(start+args.batch_size, len(texts)), "total": len(texts),
                            "elapsed_seconds": time.monotonic()-started}
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    vectors = torch.cat(chunks)
    if not torch.isfinite(vectors).all() or not torch.allclose(vectors.norm(dim=-1), torch.ones(len(vectors)), atol=1e-5):
        raise ValueError("Invalid teacher vectors")
    torch.save({"vectors": vectors, "target_ids": target_ids,
                "query_ids": [r["id"] for r in train], "texts": texts}, args.output / "teachers.pt")
    write_rows(args.output / "ordered_targets.jsonl", [
        {"id": r["id"], "source": r["source"], "query": r["query"], "clauses": r["components"],
         "target_ids": target_ids[i].tolist()} for i, r in enumerate(train)])
    write_json(args.output / "manifest.json", {
        "training_queries": len(train), "unique_clauses": len(texts), "provenance": provenance,
        "input_split": "train only", "order": "canonical component order; identity assignment",
        "target_source": "existing explicit source-derived OR clauses; no validation/test teacher targets",
        "model_path": args.model, "instruction": INSTRUCTION,
        "model_config_sha256": sha256(Path(args.model) / "config.json"),
        "teachers_sha256": sha256(args.output / "teachers.pt"),
        "ordered_targets_sha256": sha256(args.output / "ordered_targets.jsonl"),
        "training_file_sha256": sha256(DATASET / "data/train.jsonl"),
        "elapsed_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
