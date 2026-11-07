"""Freeze 5,000 queries as 3,600 training, 400 validation, and 1,000 test."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def group(row):
    return row["metadata"]["domain"] + ":" + row["metadata"]["template"]


def normalized_query(row):
    return " ".join(row["query"].casefold().split())


def stratified_sample(candidates, population, size, seed):
    groups = defaultdict(list)
    for row in candidates:
        groups[group(row)].append(row)
    if sum(map(len, groups.values())) < size:
        raise ValueError("Not enough unexposed queries for the requested split")
    ideal = {key: size * count / sum(population.values()) for key, count in population.items()}
    allocation = {key: min(math.floor(ideal[key]), len(groups[key])) for key in population}
    while sum(allocation.values()) < size:
        eligible = [key for key in sorted(population) if allocation[key] < len(groups[key])]
        key = max(eligible, key=lambda k: ideal[k] - allocation[k])
        allocation[key] += 1
    if any(abs(allocation[key] - ideal[key]) > 1.01 for key in population):
        raise ValueError("Exposure exclusions prevent a representative split")
    rng, chosen = random.Random(seed), []
    for key in sorted(population):
        rows = sorted(groups[key], key=lambda r: r["id"])
        rng.shuffle(rows)
        chosen.extend(rows[:allocation[key]])
    rng.shuffle(chosen)
    return chosen


def partition(rows, exposed_ids, prior_evaluation_ids, seed=20261106,
              training_size=3600, validation_size=400, test_size=1000):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    if len({normalized_query(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate normalized queries; group them before splitting")
    population = Counter(map(group, rows))
    excluded_test = exposed_ids | prior_evaluation_ids
    test = stratified_sample([r for r in rows if r["id"] not in excluded_test], population,
                             test_size, seed)
    test_ids = {r["id"] for r in test}
    development_side = [r for r in rows if r["id"] not in test_ids]
    validation = stratified_sample([r for r in development_side if r["id"] not in exposed_ids],
                                   population, validation_size, seed + 1)
    validation_ids = {r["id"] for r in validation}
    train = stratified_sample([r for r in development_side if r["id"] not in validation_ids],
                              population, training_size, seed + 2)
    return train, validation, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Frozen split already exists")
    source_paths = [args.source / "data" / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl")]
    original_train, original_dev, original_test = map(read_rows, source_paths)
    exposure = json.loads(args.exposure.read_text())
    exposed = set()
    for record in exposure:
        if record["training_data_sha256"] != sha256(source_paths[0]):
            raise ValueError("Checkpoint used a different training pool")
        config = record["config"]
        if config["epochs"] != 1:
            raise ValueError("Exposure reconstruction expects one shuffled epoch")
        order = list(range(len(original_train)))
        random.Random(config["seed"]).shuffle(order)
        for index in order[:record["exclude_through_step"] * config["batch_size"]]:
            exposed.add(original_train[index]["id"])
    prior_evaluation = {r["id"] for r in original_dev + original_test}
    all_rows = original_train + original_dev + original_test
    train, dev, test = partition(all_rows, exposed, prior_evaluation)
    data = args.output / "data"
    data.mkdir(parents=True, exist_ok=True)
    counts, strata = {}, {}
    for name, rows in (("train", train), ("validation", dev), ("test", test)):
        (data / (name + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
        counts[name] = len(rows)
        strata[name] = dict(sorted(Counter(map(group, rows)).items()))
    provenance = args.output / "provenance"
    provenance.mkdir(exist_ok=True)
    (provenance / "checkpoint_exposure.json").write_text(json.dumps(exposure, indent=2) + "\n")
    (provenance / "excluded_from_new_holdouts.json").write_text(json.dumps({
        "training_exposure_ids": sorted(exposed), "previous_evaluation_ids": sorted(prior_evaluation)
    }, indent=2) + "\n")
    manifest = {
        "version": "5000-v1", "seed": 20261106, "source_queries": len(all_rows),
        "total_queries": len(train) + len(dev) + len(test), "counts": counts,
        "training_development_side": len(train) + len(dev), "test_fraction": len(test) / len(all_rows),
        "stratification": "domain/source and two-versus-three-part OR template",
        "strata": strata, "exposed_training_queries_excluded": len(exposed),
        "prior_evaluation_queries_excluded_from_test": len(prior_evaluation),
        "exposure_safety_margin": "Two additional batches beyond the last logged/requested update",
        "test_unit": "Unseen complete query combinations; component questions/documents may recur",
        "initialization": "Fresh pinned base model for each variant; prior trained weights are not reused",
        "source_sha256": {str(p.relative_to(args.source)): sha256(p) for p in source_paths},
        "files": {str(p.relative_to(args.output)): sha256(p)
                  for p in sorted(data.glob("*.jsonl")) + sorted(provenance.glob("*.json"))},
    }
    manifest["test_fraction"] = len(test) / manifest["total_queries"]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("total_queries", "counts", "training_development_side",
                                             "test_fraction", "exposed_training_queries_excluded")}))


if __name__ == "__main__":
    main()
