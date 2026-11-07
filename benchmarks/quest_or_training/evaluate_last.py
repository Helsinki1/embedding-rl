"""Secondary evaluation of the pre-specified final update, without test-based selection."""
import argparse
import json
from pathlib import Path

import torch

from .data import load_corpus, load_splits, sha256, write_json
from .train import QwenEncoder, restore_weights, retrieve, score_rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    metadata = json.loads((args.run / "run.json").read_text())
    config = metadata["config"]
    here = Path(__file__).resolve().parent
    _, _, test, spec = load_splits(here.parent / "quest_or", here / "split.json")
    if spec != metadata["provenance"]["split"]:
        raise ValueError("Split differs from the completed training run")
    destination = args.run / "last_evaluation"
    destination.mkdir(exist_ok=False)
    curve = json.loads((args.run / f"validation-{config['steps']:04d}.json").read_text())
    best = max(curve, key=lambda item: (item["f1"], -item["k"]))
    selection = {"checkpoint_step": config["steps"], "k": best["k"], "validation": best,
                 "checkpoint_sha256": sha256(args.run / "last.pt"),
                 "selection_rule": "Fixed final training update; cutoff from its validation curve"}
    write_json(destination / "selection.json", selection)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    documents, titles, _ = load_corpus(args.work_dir, args.device)
    encoder = QwenEncoder(args.work_dir / "models/Qwen3-Embedding-8B", config, args.device)
    restore_weights(encoder, args.run / "last.pt")
    rankings = retrieve(encoder, test, documents, config["batch_size"], best["k"])
    summary, predictions, details = score_rankings(test, rankings, titles, best["k"])
    result = {"method": metadata["method"] + "_last", **selection, "test": summary}
    write_json(destination / "summary.json", result)
    write_json(destination / "test_details.json", details)
    (destination / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
