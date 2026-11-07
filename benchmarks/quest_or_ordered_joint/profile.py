"""Measure a full microbatch before the three-epoch follow-up; use train data only."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from benchmarks.quest_or_spec_v2.build import HERE as DATASET, write_json
from benchmarks.quest_or_spec_v2.data import load_corpus, load_splits
from benchmarks.quest_or_training.data import INSTRUCTION, gold_indices
from benchmarks.quest_or_training.train import training_batches
from benchmarks.quest_or_multivector.objectives import multi_vector_infonce
from .model import make_encoder
from .objectives import ordered_clause_infonce


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(.58)
    config = {**json.loads((DATASET / "protocol.json").read_text()),
              **json.loads((Path(__file__).parent / "config.json").read_text()),
              "variant": "three_pass", "epochs": 3, "micro_batch_size": 32}
    torch.manual_seed(config["seed"])
    train, _, _, provenance = load_splits()
    cache = torch.load(args.root / "teachers/teachers.pt", weights_only=True, map_location="cpu")
    teachers, target_ids = cache["vectors"].cuda(), cache["target_ids"].cuda()
    del cache
    base = Path("/home/ubuntu/quest-or-baseline-20261106")
    documents, titles, _ = load_corpus(
        base, Path("/home/ubuntu/quest-or-synthetic-20261106/project/benchmarks/quest_or_synthetic"),
        Path("/home/ubuntu/quest-or-synthetic-20261106/embeddings"),
        Path("/home/ubuntu/quest-spec-v2-20261106/embeddings"), "cuda")
    gold = gold_indices(train, {t: i for i, t in enumerate(titles)}, "cuda")
    encoder = make_encoder(str(base / "models/Qwen3-Embedding-8B"), config, "cuda")
    params = [p for p in encoder.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=0)
    encoder.model.train()
    batches = []
    for batch in training_batches(len(train), config):
        batches.append(batch)
        if len(batches) == 8:
            break
    lengths = [len(x) for x in encoder.tokenizer(
        [f"Instruct: {INSTRUCTION}\nQuery:{r['query']}" for r in train],
        padding=False, truncation=False)["input_ids"]]
    batches.append(sorted(range(len(train)), key=lambda i: lengths[i], reverse=True)[:32])
    times, losses = [], []
    for step, ids in enumerate(batches):
        torch.cuda.synchronize()
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        vectors = encoder.encode([train[i]["query"] for i in ids])
        doc = multi_vector_infonce(vectors, documents, [gold[i] for i in ids], config["temperature"])
        clause, _ = ordered_clause_infonce(vectors, teachers, target_ids[ids], config["clause_temperature"])
        loss = doc + config["clause_weight"] * clause
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.monotonic()-began)
        losses.append(loss.item())
        print(json.dumps({"step": step+1, "longest_training_batch": step == 8,
                          "seconds": times[-1], "loss": losses[-1]}), flush=True)
    result = {"config": config, "provenance": provenance, "update_seconds": times,
              "median_regular_update_seconds": statistics.median(times[1:8]),
              "estimated_798_update_seconds": 798*statistics.mean(times[1:8]),
              "max_training_query_tokens": max(lengths),
              "peak_allocated_gb": torch.cuda.max_memory_allocated()/1024**3,
              "elapsed_seconds": time.monotonic()-started,
              "training_data_only": True, "disposable_weights": True}
    write_json(args.root / "profile-three_pass.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
