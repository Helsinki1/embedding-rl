"""Build frozen, reproducible OR training examples from NQ's training split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

HERE = Path(__file__).resolve().parent
DATASET = "sentence-transformers/natural-questions"
REVISION = "f9e894e1081e206e577b4eaa9ee6de2b06ae6f17"
SOURCE_FILE = "pair/train-00000-of-00001.parquet"
SOURCE_SHA256 = "8cfb5e5a7cb1cd09dbb035a581aa42ce5db5852acb3e70f070d89d7035887de9"
OPERATORS = re.compile(r"\b(?:and|or|not|without|except|excluding|neither|nor|both)\b|n['’]t\b|[&|]", re.I)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def normalize_query(query):
    return " ".join(query.split()).casefold()


def prepare_source(rows):
    """Group repeated queries, union ALL their supplied positives, dedup exact passages."""
    grouped, passages, excluded = {}, {}, Counter()
    source_count = 0
    for index, row in enumerate(rows):
        source_count += 1
        query, passage = row.get("query"), row.get("answer")
        if not isinstance(query, str) or not isinstance(passage, str) or not query.strip() or not passage.strip():
            excluded["empty_query_or_passage"] += 1
            continue
        if OPERATORS.search(query):
            excluded["explicit_operator_in_source_query"] += 1
            continue
        if len(query) > 256:
            excluded["source_query_over_256_characters"] += 1
            continue
        key = normalize_query(query)
        doc_id = "nq:" + digest(passage)
        passages[doc_id] = passage
        atom = grouped.setdefault(key, {
            "id": "nq-query:" + digest(key), "query": query,
            "docs": set(), "source_rows": [],
        })
        atom["docs"].add(doc_id)
        atom["source_rows"].append(index)
    atoms = sorted(grouped.values(), key=lambda row: row["id"])
    for atom in atoms:
        atom["docs"] = sorted(atom["docs"])
    corpus = [{"title": title, "text": passages[title], "source": DATASET}
              for title in sorted(passages)]
    return atoms, corpus, {"source_rows": source_count, "excluded_rows": dict(excluded),
                           "retained_rows": source_count - sum(excluded.values()),
                           "unique_queries": len(atoms), "unique_passages": len(corpus)}


def compose(atoms):
    return {
        "id": "nq-or:" + digest("\n".join(sorted(atom["id"] for atom in atoms))),
        "query": " OR ".join("(" + atom["query"] + ")" for atom in atoms),
        "docs": sorted({doc for atom in atoms for doc in atom["docs"]}),
        "components": [atom["query"] for atom in atoms],
        "component_ids": [atom["id"] for atom in atoms],
        "component_docs": [atom["docs"] for atom in atoms],
        "split": "train",
        "metadata": {"domain": "natural_questions", "template": " or ".join(["_"] * len(atoms)),
                     "synthetic": True, "source_dataset": DATASET, "source_split": "train",
                     "positive_labels": "union_of_supplied_positives_not_exhaustive"},
    }


def generate(atoms, pairs, triples, seed):
    if len(atoms) < 3 or pairs < 0 or triples < 0 or pairs + triples == 0:
        raise ValueError("Need at least three source queries and a positive example count")
    rng = random.Random(seed)
    anchors = list(range(len(atoms)))
    rng.shuffle(anchors)
    seen, examples = set(), []
    for index, arity in enumerate([2] * pairs + [3] * triples):
        anchor = anchors[index % len(anchors)]
        for _ in range(10000):
            others = rng.sample(range(len(atoms)), arity - 1)
            if anchor in others:
                continue
            chosen = [atoms[i] for i in [anchor, *others]]
            row = compose(chosen)
            if row["id"] in seen or len(row["docs"]) < arity:
                continue
            seen.add(row["id"])
            examples.append(row)
            break
        else:
            raise ValueError("Cannot generate enough distinct OR combinations")
    return examples


def frozen_base(benchmark, split_path):
    spec = json.loads(split_path.read_text())
    for name, expected in spec["source_sha256"].items():
        if sha256(benchmark / name) != expected:
            raise ValueError("Frozen QUEST source changed: " + name)
    rows = read_rows(benchmark / "data/validation.jsonl") + read_rows(benchmark / "data/test.jsonl")
    lookup = {row["id"]: row for row in rows}
    ids = spec["train_ids"] + spec["validation_ids"] + spec["test_ids"]
    if len(set(ids)) != len(ids) or set(ids) != set(lookup) or len(lookup) != len(rows):
        raise ValueError("Invalid QUEST split partition")
    return [[lookup[qid] for qid in spec[key]]
            for key in ("train_ids", "validation_ids", "test_ids")], spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--benchmark", type=Path, default=HERE.parent / "quest_or")
    parser.add_argument("--splits", type=Path, default=HERE.parent / "quest_or_training/split.json")
    parser.add_argument("--pairs", type=int, default=50000)
    parser.add_argument("--triples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists() or (args.output / "data").exists():
        raise FileExistsError("Use a fresh output directory; frozen data is never overwritten")
    cache = args.output / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    source_path = cache / "natural-questions-train.parquet"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_FILE}"
    if not source_path.exists():
        temporary = source_path.with_suffix(".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(source_path)
    if sha256(source_path) != SOURCE_SHA256:
        raise ValueError("NQ source checksum mismatch")
    import pyarrow.parquet as pq
    atoms, corpus, counts = prepare_source(pq.read_table(source_path, columns=["query", "answer"]).to_pylist())
    examples = generate(atoms, args.pairs, args.triples, args.seed)
    (train, validation, test), base_spec = frozen_base(args.benchmark, args.splits)
    data = args.output / "data"
    data.mkdir()
    write_rows(data / "corpus_added.jsonl", corpus)
    write_rows(data / "source_queries.jsonl", atoms)
    write_rows(data / "train_or2.jsonl", examples[:args.pairs])
    write_rows(data / "train_or3.jsonl", examples[args.pairs:])
    write_rows(data / "train_synthetic.jsonl", examples)
    write_rows(data / "train.jsonl", [*train, *examples])
    write_rows(data / "validation.jsonl", validation)
    write_rows(data / "test.jsonl", test)
    provenance = args.output / "provenance"
    provenance.mkdir(exist_ok=True)
    write_json(provenance / "base_split.json", base_spec)
    urllib.request.urlretrieve(f"https://huggingface.co/datasets/{DATASET}/raw/{REVISION}/README.md",
                               provenance / "source_README.md")
    counts.update({"or2": args.pairs, "or3": args.triples, "synthetic_train": len(examples),
                   "original_train": len(train), "combined_train": len(train) + len(examples),
                   "validation": len(validation), "test": len(test),
                   "source_queries_used": len({qid for row in examples for qid in row["component_ids"]}),
                   "base_corpus": 325427, "combined_corpus": 325427 + len(corpus)})
    manifest = {
        "version": 1, "name": "QUEST + deterministic NQ OR augmentation", "seed": args.seed,
        "source": {"dataset": DATASET, "revision": REVISION, "split": "train", "config": "pair",
                   "file": SOURCE_FILE, "url": url, "sha256": SOURCE_SHA256},
        "counts": counts,
        "generation": {"pairs": args.pairs, "triples": args.triples,
                       "query_join": "(q1) OR (q2) [OR (q3)]", "positive_merge": "set union",
                       "query_dedup": "casefold + whitespace normalization; union all source positives",
                       "document_dedup": "exact UTF-8 passage SHA256; nq: namespaced ID",
                       "source_query_filter": OPERATORS.pattern, "maximum_source_query_characters": 256,
                       "pairing": "seeded shuffled anchors with sampled partners; no repeated unordered combinations",
                       "source_query_coverage": "every eligible query anchors once when example count permits"},
        "base_split_sha256": sha256(args.splits),
        "corpus": {"base": "quest_or full corpus, unchanged", "append": "data/corpus_added.jsonl",
                   "added_document_encoder_input": "text only; source passage already contains title text",
                   "model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"},
        "assumptions": ["Existing QUEST and new source are treated as disjoint, as requested; no overlap audit.",
                        "Source positives are known positives, not exhaustive relevance judgments.",
                        "Evaluation query/label rows are unchanged; an expanded corpus changes retrieval difficulty."],
        "files": {str(path.relative_to(args.output)): sha256(path)
                  for path in sorted(data.glob("*.jsonl"))},
        "builder_sha256": sha256(Path(__file__)),
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
