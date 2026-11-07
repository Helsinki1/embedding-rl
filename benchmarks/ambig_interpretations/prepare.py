"""Encode human interpretation text with the unchanged base Qwen teacher."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import MODEL_ID, REVISION, tokenize


def prepare(data_dir, output, model_path, deadline):
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    started = time.time()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                     attn_implementation="sdpa").to("cuda").eval()
    model.requires_grad_(False)
    partitions = {name: [json.loads(line) for line in (data_dir / f"{name}.jsonl").read_text().splitlines()]
                  for name in ("train", "validation", "test")}
    texts = sorted({text for rows in partitions.values() for r in rows
                    for text in [r["query"], *r["interpretations"]]})
    vectors = []
    max_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            if time.time() > deadline - 120:
                raise TimeoutError("Teacher preparation exhausted experiment deadline")
            tokens = tokenize(tokenizer, texts[start:start + 32], "cuda")
            max_tokens = max(max_tokens, tokens.input_ids.shape[1])
            result = model(**tokens, use_cache=False).last_hidden_state[:, -1]
            vectors.append(F.normalize(result.float(), dim=-1).cpu())
            if start % 1024 == 0:
                print(json.dumps({"stage": "teacher", "done": min(start + 32, len(texts)),
                                  "total": len(texts), "seconds": time.time() - started}), flush=True)
    vectors = torch.cat(vectors)
    lookup = {s: i for i, s in enumerate(texts)}
    for name, rows in partitions.items():
        targets = torch.zeros(len(rows), 3, vectors.shape[-1])
        for i, row in enumerate(rows):
            for j, text in enumerate(row["interpretations"]):
                targets[i, j] = vectors[lookup[text]]
        torch.save({"targets": targets,
                    "counts": torch.tensor([len(r["interpretations"]) for r in rows]),
                    "original_query_vectors": vectors[[lookup[r["query"]] for r in rows]],
                    "ids": [r["id"] for r in rows]}, output / f"{name}.pt")
    info = {"model": MODEL_ID, "revision": REVISION, "dimension": vectors.shape[-1],
            "unique_texts": len(texts), "max_token_length": max_tokens,
            "frozen_teacher": True, "corpus_access": False,
            "teacher_seconds": time.time() - started,
            "split_sha256": {n: hashlib.sha256((data_dir / f"{n}.jsonl").read_bytes()).hexdigest() for n in partitions}}
    (output / "manifest.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--deadline", type=float, required=True)
    a = p.parse_args()
    prepare(a.data, a.output, a.model, a.deadline)
