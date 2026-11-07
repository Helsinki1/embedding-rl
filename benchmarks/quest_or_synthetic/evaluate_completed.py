"""Score frozen, best-trained, and final encoders against the expanded corpus."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.quest_or_training.data import sha256, write_json
from benchmarks.quest_or_training.train import QwenEncoder, restore_weights, retrieve, score_rankings

from .data import load_augmented_corpus, load_bundle
from benchmarks.quest_or_balanced.data import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path(__file__).resolve().parent.parent / "quest_or_balanced")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (args.run / "summary.json").exists():
        raise ValueError("Training must complete before evaluation")
    metadata = json.loads((args.run / "run.json").read_text())
    config = metadata["config"]
    if metadata["provenance"]["dataset"].get("version") == "5000-v1":
        _, _, test, manifest = load_splits(args.split, args.dataset)
    else:
        _, _, test, manifest = load_bundle(args.dataset)
    if manifest != metadata["provenance"]["dataset"]:
        raise ValueError("Dataset differs from the completed run")
    root = args.run / "extra_evaluation"
    root.mkdir(exist_ok=False)
    initial = json.loads((args.run / "initial_validation.json").read_text())["best"]
    trained = json.loads((args.run / "best_trained_selection.json").read_text())
    final_curve = json.loads((args.run / f"validation-{config['steps']:04d}.json").read_text())
    final = max(final_curve, key=lambda row: (row["f1"], -row["k"]))
    selections = {
        "frozen": {"checkpoint_step": 0, "k": initial["k"], "validation": initial},
        "best_trained": trained,
        "final": {"checkpoint_step": config["steps"], "k": final["k"], "validation": final,
                  "checkpoint_sha256": sha256(args.run / "last.pt")},
    }
    # Freeze every checkpoint/cutoff choice before computing any of these test scores.
    write_json(root / "selection.json", selections)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    documents, titles, _ = load_augmented_corpus(
        args.work_dir, args.dataset, args.embeddings, args.device
    )
    encoder = QwenEncoder(args.work_dir / "models/Qwen3-Embedding-8B", config, args.device)
    for label, checkpoint in (("frozen", None), ("best_trained", "best_trained.pt"),
                              ("final", "last.pt")):
        if checkpoint:
            restore_weights(encoder, args.run / checkpoint)
        chosen = selections[label]
        rankings = retrieve(encoder, test, documents, config["batch_size"], chosen["k"])
        summary, predictions, details = score_rankings(test, rankings, titles, chosen["k"])
        destination = root / label
        destination.mkdir()
        result = {"method": metadata["method"], "evaluation": label, **chosen, "test": summary}
        write_json(destination / "summary.json", result)
        write_json(destination / "test_details.json", details)
        (destination / "test_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
