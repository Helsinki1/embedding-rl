"""Join the actual AMER release to original human-written AmbigNQ rewrites."""
import argparse
import collections
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return " ".join(text.lower().split()).rstrip("?.")


def build(mode):
    source = HERE / "source"
    destination = HERE / "data"
    if destination.exists():
        raise FileExistsError("Refusing to replace frozen experiment data")
    original = {}
    with zipfile.ZipFile(io.BytesIO((source / "ambignq_light.zip").read_bytes())) as z:
        for name in z.namelist():
            if name.endswith(".json"):
                for row in json.loads(z.read(name)):
                    original[row["id"]] = {**row, "original_split": name}
    accepted = {3} if mode == "exactly_three" else {2, 3}
    audit = {}

    def convert(filename):
        rows, exclusions = [], collections.Counter()
        raw = [json.loads(line) for line in (source / filename).read_text().splitlines()]
        seen = set()
        for r in raw:
            item = original.get(str(r.get("id")))
            if item is None:
                exclusions["missing_original_rewrites"] += 1
                continue
            candidates = []
            for annotation in item["annotations"]:
                if annotation["type"] != "multipleQAs":
                    continue
                pairs = sorted({p["question"].strip() for p in annotation["qaPairs"]})
                if len(pairs) in accepted and len({normalized(p) for p in pairs}) == len(pairs):
                    candidates.append(pairs)
            if not candidates:
                exclusions["no_eligible_annotation"] += 1
                continue
            key = normalized(item["question"])
            if key in seen:
                exclusions["duplicate_query"] += 1
                continue
            seen.add(key)
            # Prefer three meanings when both arities are allowed; otherwise a
            # deterministic lexical choice among acceptable human annotations.
            chosen = sorted(candidates, key=lambda x: (-len(x), x))[0]
            rows.append({"id": str(r["id"]), "query": item["question"],
                         "interpretations": chosen, "original_split": item["original_split"]})
        audit[filename] = {"raw_rows": len(raw), "accepted_before_overlap": len(rows),
                           "exclusions": dict(exclusions)}
        return rows

    training = convert("amer_data__ambigqa_train.jsonl")
    test = convert("amer_data__eval_data__ambigqa.jsonl")
    test_ids = {r["id"] for r in test}
    test_texts = {normalized(r["query"]) for r in test}
    before = len(training)
    training = [r for r in training if r["id"] not in test_ids and normalized(r["query"]) not in test_texts]
    audit["train_test_overlaps_removed"] = before - len(training)
    # Keep exact shared interpretation texts together within training/validation.
    parent = list(range(len(training)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, r in enumerate(training):
        for text in [r["query"], *r["interpretations"]]:
            key = normalized(text)
            if key in owners:
                parent[root(i)] = root(owners[key])
            owners[key] = i
    groups = collections.defaultdict(list)
    for i, row in enumerate(training):
        groups[root(i)].append(row)
    groups = list(groups.values())
    random.Random(42).shuffle(groups)
    validation, train = [], []
    target = max(1, round(len(training) * .1))
    for group in groups:
        (validation if len(validation) < target else train).extend(group)
    partitions = {"train": train, "validation": validation, "test": test}
    destination.mkdir()
    for name, rows in partitions.items():
        (destination / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {"mode": mode, "seed": 42, "audit": audit,
                "counts": {n: len(r) for n, r in partitions.items()},
                "arity_counts": {n: dict(collections.Counter(len(r['interpretations']) for r in rows)) for n, rows in partitions.items()},
                "sources": {p.name: sha(p) for p in source.iterdir() if p.is_file()},
                "splits_sha256": {n: sha(destination / f"{n}.jsonl") for n in partitions},
                "teacher_targets": "Original human-written disambiguated questions, never answer strings or document text",
                "corpus_access": False,
                "note": "Custom corpus-free subset of downloaded AMER release; not the paper's 5,044-example retrieval split."}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exactly_three", "two_or_three"), default="exactly_three")
    build(parser.parse_args().mode)
