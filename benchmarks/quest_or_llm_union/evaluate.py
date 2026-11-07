"""Frozen Qwen embeddings per LLM component, full-corpus search, deduplicated union."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from benchmarks.quest_or.scripts.evaluate import evaluate
from benchmarks.quest_or_balanced.data import load_splits
from benchmarks.quest_or_synthetic.build import read_rows, sha256, write_json, write_rows
from benchmarks.quest_or_synthetic.data import load_augmented_corpus
from benchmarks.quest_or_training.train import QwenEncoder
from .union import flatten_queries, union_predictions

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def retrieve_components(encoder, rows, documents, batch_size, k):
    encoder.model.eval()
    rankings = []
    for start in range(0, len(rows), batch_size):
        vectors = encoder.encode([r["query"] for r in rows[start:start + batch_size]])
        rankings.extend((vectors @ documents.T).topk(k, dim=-1).indices.cpu().tolist())
    return rankings


def main():
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=HERE.parent / "quest_or_balanced")
    parser.add_argument("--corpus-dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--decompositions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    _, dev, test, manifest = load_splits(args.split, args.corpus_dataset)
    decomposition_manifest = json.loads((args.decompositions / "decomposition_manifest.json").read_text())
    if decomposition_manifest["split_manifest_sha256"] != sha256(args.split / "manifest.json"):
        raise ValueError("Decompositions use a different query split")
    cache_path = args.decompositions / "decompositions.jsonl"
    if sha256(cache_path) != decomposition_manifest["decompositions_sha256"]:
        raise ValueError("Decomposition cache changed")
    records = read_rows(cache_path)
    cache = {r["query"]: r for r in records}
    expected = {r["query"] for r in dev + test}
    if len(cache) != len(records) or set(cache) != expected:
        raise ValueError("Decompositions must exactly cover validation and test")
    if any(r["model"] != decomposition_manifest["model"] or r["prompt_sha256"] != decomposition_manifest["prompt_sha256"] for r in records):
        raise ValueError("Decomposition model/prompt mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    documents, titles, corpus = load_augmented_corpus(args.work_dir, args.corpus_dataset, args.embeddings, args.device)
    config = json.loads((HERE.parent / "quest_or_balanced/config.json").read_text())
    encoder = QwenEncoder(args.work_dir / "models/Qwen3-Embedding-8B", config, args.device)
    # Zero-initialized adapters leave the base model unchanged; no optimizer or updates.
    encoder.model.requires_grad_(False)
    write_json(args.output / "run.json", {
        "method": "llm_decomposer_multi_query_deduped_untrained", "training_updates": 0,
        "encoder": "Qwen/Qwen3-Embedding-8B", "dimension_per_component": 4096,
        "split_manifest_sha256": sha256(args.split / "manifest.json"), "split": manifest,
        "decomposition": decomposition_manifest, "corpus": corpus, "config": config,
        "cutoff_candidates": list(range(1, 21)), "retrieval": "Full corpus cosine top-k per component, then set union by document ID; no union-size truncation",
        "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy")},
        "code_sha256": {str(p.relative_to(HERE.parent)): sha256(p) for d in (HERE, HERE.parent / "quest_or_training", HERE.parent / "quest_or_balanced", HERE.parent / "quest_or_synthetic") for p in sorted(d.glob("*.py"))},
    })
    flat, offsets, fallbacks = flatten_queries(dev, cache)
    ranks = retrieve_components(encoder, flat, documents, config["batch_size"], 20)
    curve = []
    for k in range(1, 21):
        summary, _ = evaluate(dev, union_predictions(dev, offsets, ranks, titles, k))
        curve.append({"k": k, **summary["overall"]})
    selected = max(curve, key=lambda row: (row["f1"], -row["k"]))
    write_json(args.output / "validation_curve.json", curve)
    write_json(args.output / "selection.json", {"k": selected["k"], "validation": selected,
                                                "rule": "Maximum validation macro F1, smaller k on ties"})
    print(json.dumps({"stage": "validation_selected", **selected}), flush=True)
    # Only evaluate test after committing the validation-only cutoff choice.
    flat, offsets, test_fallbacks = flatten_queries(test, cache)
    ranks = retrieve_components(encoder, flat, documents, config["batch_size"], selected["k"])
    predictions = union_predictions(test, offsets, ranks, titles, selected["k"])
    summary, details = evaluate(test, predictions)
    summary["by_source"] = {}
    for name, synthetic in (("quest", False), ("nq", True)):
        chosen = [i for i, r in enumerate(test) if bool(r["metadata"].get("synthetic")) == synthetic]
        result, _ = evaluate([test[i] for i in chosen], [predictions[i] for i in chosen])
        summary["by_source"][name] = result["overall"]
    result = {"method": "llm_decomposer_multi_query_deduped_untrained", "k_per_component": selected["k"],
              "validation": selected, "test": summary, "validation_fallbacks": len(fallbacks),
              "test_fallbacks": len(test_fallbacks), "test_component_counts": dict(Counter(end-start for start,end in offsets)),
              "elapsed_evaluation_seconds": time.time() - started}
    write_rows(args.output / "test_predictions.jsonl", predictions)
    write_json(args.output / "test_details.json", details)
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
