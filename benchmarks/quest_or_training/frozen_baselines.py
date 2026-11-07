"""Rescore the frozen single-query and OR-union methods on the shared new split."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.quest_or.scripts.evaluate import evaluate

from .data import load_splits, sha256, write_json

HERE = Path(__file__).resolve().parent


def cached_predictions(rows, method, k, lookup, titles, indices):
    predictions = []
    for row in rows:
        components = (["original"] if method == "frozen_single" else
                      [str(i) for i in range(len(row["components"]))])
        docs = dict.fromkeys(titles[int(index)] for component in components
                             for index in indices[lookup[(row["id"], component)], :k])
        predictions.append({"id": row["id"], "query": row["query"], "docs": list(docs)})
    return predictions


def rescore(work_dir, benchmark, split_path, output):
    train, validation, test, spec = load_splits(benchmark, split_path)
    results_dir = work_dir / "results"
    query_ids = json.loads((results_dir / "queries_ids.json").read_text())
    progress = json.loads((results_dir / "queries_progress.json").read_text())
    if progress["done"] != progress["total"] or progress["done"] != len(query_ids):
        raise ValueError("Frozen query embeddings are incomplete")
    lookup = {(r["id"], r["component"]): i for i, r in enumerate(query_ids)}
    if len(lookup) != len(query_ids):
        raise ValueError("Duplicate query/component cache identities")
    for row in train + validation + test:
        expected = [("original", row["query"])] + [(str(i), c) for i, c in enumerate(row["components"])]
        for component, text in expected:
            if query_ids[lookup[(row["id"], component)]]["text"] != text:
                raise ValueError("Frozen retrieval cache query text mismatch")
    titles = json.loads((results_dir / "retrieval_document_titles.json").read_text())
    shard_titles = [r["title"] for rank in (0, 1)
                    for r in json.loads((results_dir / f"corpus_{rank}_ids.json").read_text())]
    if titles != shard_titles or len(titles) != 325427 or len(set(titles)) != len(titles):
        raise ValueError("Retrieval cache/corpus document ordering mismatch")
    with np.load(results_dir / "top1000.npz") as cache:
        indices = cache["indices"]
    if indices.shape[0] != len(query_ids) or indices.shape[1] < 60:
        raise ValueError("Retrieval cache has an unexpected shape")
    if indices.min() < 0 or indices.max() >= len(titles):
        raise ValueError("Invalid cached document indices")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {
        "split": spec, "source": "Frozen Qwen3 vLLM query vectors and full-corpus cosine search",
        "cache_sha256": {name: sha256(results_dir / name)
                         for name in ("top1000.npz", "queries_ids.json", "queries_progress.json")},
        "encoder_revision": progress["revision"],
    })
    selected = {}
    for method, max_k in (("frozen_single", 60), ("frozen_or_union", 20)):
        trials = []
        for k in range(1, max_k + 1):
            predictions = cached_predictions(validation, method, k, lookup, titles, indices)
            metrics, _ = evaluate(validation, predictions)
            trials.append({"k": k, **metrics["overall"]})
        selected[method] = max(trials, key=lambda r: (r["f1"], -r["k"]))
        write_json(output / f"{method}_validation_curve.json", trials)
    write_json(output / "selection.json", selected)
    summaries = []
    for method, selection in selected.items():
        predictions = cached_predictions(test, method, selection["k"], lookup, titles, indices)
        metrics, details = evaluate(test, predictions)
        (output / f"{method}_test_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
        )
        write_json(output / f"{method}_test_details.json", details)
        summaries.append({"method": method, "k": selection["k"],
                          "validation": selection, "test": metrics})
    write_json(output / "summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=HERE.parent / "quest_or")
    parser.add_argument("--splits", type=Path, default=HERE / "split.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rescore(args.work_dir, args.benchmark, args.splits, args.output), indent=2))


if __name__ == "__main__":
    main()
