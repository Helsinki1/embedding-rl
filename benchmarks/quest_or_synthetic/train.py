"""Explicit training entry point for QUEST plus the frozen synthetic augmentation."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

from .build import HERE, sha256, write_json
from .data import load_augmented_corpus
from benchmarks.quest_or_balanced.data import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE)
    parser.add_argument("--split", type=Path, default=HERE.parent / "quest_or_balanced")
    parser.add_argument("--work-dir", type=Path, default=Path("/home/ubuntu/quest-or-baseline-20261106"))
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE.parent / "quest_or_balanced/config.json")
    parser.add_argument("--method", choices=("rl", "infonce", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from benchmarks.quest_or_training.train import train_one, validate_config

    config = json.loads(args.config.read_text())
    validate_config(config)
    train, validation, test, manifest = load_splits(args.split, args.dataset)
    corpus_manifest = json.loads((args.dataset / "manifest.json").read_text())
    methods = ("rl", "infonce") if args.method == "both" else (args.method,)
    for method in methods:
        if (args.output / method).exists():
            raise FileExistsError("Choose a fresh training output directory")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    vectors, titles, corpus = load_augmented_corpus(args.work_dir, args.dataset, args.embeddings, args.device)
    if config["max_k"] > len(titles):
        raise ValueError("max_k exceeds corpus size")
    provenance = {"dataset": manifest, "corpus": corpus,
                  "model": "Qwen/Qwen3-Embedding-8B", "revision": corpus_manifest["corpus"]["model_revision"],
                  "split_manifest_sha256": sha256(args.split / "manifest.json"),
                  "sampling": ("shuffled complete passes without replacement" if "epochs" in config
                               else "uniform over original and synthetic training rows"),
                  "code_sha256": {str(path.relative_to(HERE.parent)): sha256(path)
                                  for directory in (HERE, HERE.parent / "quest_or_training")
                                  for path in sorted(directory.glob("*.py"))},
                  "packages": {name: importlib.metadata.version(name)
                               for name in ("torch", "transformers", "peft", "numpy")}}
    results = []
    for method in methods:
        results.append(train_one(method, config, args.work_dir / "models/Qwen3-Embedding-8B",
                                 vectors, titles, train, validation, test,
                                 args.output / method, args.device, provenance))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output / f"{args.method}-results.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
