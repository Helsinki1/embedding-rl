"""Run one compact baseline, then score frozen and trained encoders on the new test set."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from benchmarks.quest_or_synthetic.build import sha256, write_json, write_rows
from benchmarks.quest_or_synthetic.data import load_augmented_corpus
from benchmarks.quest_or_training.train import (
    QwenEncoder, restore_weights, retrieve, score_rankings, train_one, validate_config,
)
from .data import load_splits

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("rl", "infonce"), required=True)
    parser.add_argument("--dataset", type=Path, default=HERE)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--corpus-dataset", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    config = json.loads(args.config.read_text())
    validate_config(config)
    train, dev, test, manifest = load_splits(args.dataset, args.corpus_dataset)
    corpus_manifest = json.loads((args.corpus_dataset / "manifest.json").read_text())
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda:0"
    documents, titles, corpus = load_augmented_corpus(args.work_dir, args.corpus_dataset, args.embeddings, device)
    provenance = {"dataset": manifest, "corpus": corpus, "model": "Qwen/Qwen3-Embedding-8B",
                  "revision": corpus_manifest["corpus"]["model_revision"], "time_limit_seconds": 1795,
                  "split_manifest_sha256": sha256(args.dataset / "manifest.json"),
                  "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy")},
                  "code_sha256": {str(p.relative_to(HERE.parent)): sha256(p)
                                  for directory in (HERE, HERE.parent / "quest_or_training", HERE.parent / "quest_or_synthetic")
                                  for p in sorted(directory.glob("*.py"))}}
    model = args.work_dir / "models/Qwen3-Embedding-8B"
    train_one(args.method, config, model, documents, titles, train, dev, test, args.output, device, provenance)
    gc.collect()
    torch.cuda.empty_cache()
    trained = json.loads((args.output / "best_trained_selection.json").read_text())
    initial = json.loads((args.output / "initial_validation.json").read_text())["best"]
    choices = {"frozen": {"checkpoint_step": 0, "k": initial["k"], "validation": initial},
               "trained": trained}
    root = args.output / "evaluations"
    root.mkdir()
    write_json(root / "selection.json", choices)
    torch.manual_seed(config["seed"])
    encoder = QwenEncoder(model, config, device)
    for label, choice in choices.items():
        if label == "trained":
            restore_weights(encoder, args.output / "best_trained.pt")
        ranking = retrieve(encoder, test, documents, config["batch_size"], choice["k"])
        summary, predictions, details = score_rankings(test, ranking, titles, choice["k"])
        summary["by_source"] = {}
        for source, synthetic in (("quest", False), ("nq", True)):
            indices = [i for i, r in enumerate(test) if bool(r["metadata"].get("synthetic")) == synthetic]
            grouped, _, _ = score_rankings([test[i] for i in indices], [ranking[i] for i in indices], titles, choice["k"])
            summary["by_source"][source] = grouped["overall"]
        target = root / label
        target.mkdir()
        write_json(target / "summary.json", {"method": args.method, "evaluation": label, **choice, "test": summary})
        write_json(target / "test_details.json", details)
        write_rows(target / "test_predictions.jsonl", predictions)
        print(json.dumps({"evaluation": label, "test": summary["overall"]}), flush=True)
    write_json(args.output / "completed.json", {"started": started, "finished": time.time(),
                                                "elapsed_seconds": time.time() - started})


if __name__ == "__main__":
    main()
