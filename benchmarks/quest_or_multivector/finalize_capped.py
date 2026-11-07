"""Score a preserved development-selected checkpoint after a requested early stop."""
import argparse
import json
import time
from pathlib import Path

import torch

from benchmarks.quest_or_synthetic.data import load_augmented_corpus, load_bundle
from benchmarks.quest_or_balanced.data import load_splits
from benchmarks.quest_or_training.data import gold_indices, sha256, write_json
from benchmarks.quest_or_training.train import restore_weights
from .model import ThreeVectorEncoder
from .train import choose_cutoff, diagnostics, retrieve, score_rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path(__file__).resolve().parent.parent / "quest_or_balanced")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    directory = args.run
    if (directory / "summary.json").exists():
        raise FileExistsError("Run is already finalized")
    request = json.loads((directory / "budget_cap_request.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    config = run["config"]
    initial = json.loads((directory / "initial_validation.json").read_text())["best"]
    best, selected_step = initial, 0
    for line in (directory / "metrics.jsonl").read_text().splitlines():
        metrics = json.loads(line)
        if "validation" in metrics and metrics["validation"]["f1"] > best["f1"]:
            best, selected_step = metrics["validation"], metrics["step"]
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(config["seed"])
    if run["provenance"]["dataset"].get("version") == "5000-v1":
        _, dev, test, manifest = load_splits(args.split, args.dataset)
    else:
        _, dev, test, manifest = load_bundle(args.dataset)
    if manifest != run["provenance"]["dataset"]:
        raise ValueError("Evaluation split differs from the training run")
    documents, titles, _ = load_augmented_corpus(
        args.work_dir, args.dataset, args.embeddings, args.device)
    encoder = ThreeVectorEncoder(args.work_dir / "models/Qwen3-Embedding-8B", config, args.device)
    restore_weights(encoder, directory / "best.pt")
    rankings, similarities = retrieve(encoder, dev, documents, config["eval_batch_size"], config["max_k"])
    verified, curve = choose_cutoff(dev, rankings, titles, config["max_k"])
    if abs(verified["f1"] - best["f1"]) > 1e-6 or verified["k"] != best["k"]:
        raise ValueError("Preserved checkpoint does not reproduce its recorded development score")
    selection = {
        "checkpoint_step": selected_step, "k": best["k"], "validation": verified,
        "initial_validation": initial, "checkpoint_sha256": sha256(directory / "best.pt"),
        "termination": "user_requested_time_budget", "budget_cap": request,
        "finalizer_source_sha256": sha256(Path(__file__)),
    }
    write_json(directory / "selection.json", selection)
    write_json(directory / "capped_validation.json", {"curve": curve, "diagnostics": diagnostics(
        dev, rankings, titles, best["k"], similarities)})
    gold_indices(test, {title: i for i, title in enumerate(titles)}, args.device)
    rankings, similarities = retrieve(encoder, test, documents, config["eval_batch_size"], best["k"])
    summary, predictions, details = score_rankings(test, rankings, titles, best["k"])
    (directory / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    write_json(directory / "test_details.json", details)
    elapsed = time.time() - request["original_launcher"]["started_at"]
    result = {"variant": config["variant"], **selection, "test": summary,
              "test_diagnostics": diagnostics(test, rankings, titles, best["k"], similarities),
              "elapsed_seconds": elapsed,
              "comparison_note": "Time-capped pilots; different amounts of training data were processed"}
    write_json(directory / "summary.json", result)
    write_json(directory / "progress.json", {
        "stage": "complete", "termination": "user_requested_time_budget",
        "last_observed_update": request["last_observed_update"],
        "latest_saved_checkpoint_step": request["latest_completed_checkpoint_step"],
        "selected_checkpoint_step": selected_step, "elapsed_seconds": elapsed,
    })
    status_path = directory.parent.parent / "status" / (config["variant"] + "-launch.json")
    status = json.loads(status_path.read_text())
    write_json(status_path, {**status, "stage": "complete", "termination": "user_requested_time_budget",
                             "training_exit_code": status.get("returncode"), "returncode": 0,
                             "finalized_at": time.time(), "elapsed_seconds": elapsed})
    print(json.dumps({"variant": config["variant"], "selected_step": selected_step,
                      "test": summary["overall"], "elapsed_minutes": elapsed / 60}), flush=True)


if __name__ == "__main__":
    main()
