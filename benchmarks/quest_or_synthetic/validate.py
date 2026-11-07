"""Audit the complete generated artifact without API calls or GPU work."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .build import HERE, OPERATORS, compose, digest, normalize_query, read_rows
from .data import load_bundle


def validate(dataset=HERE):
    dataset = Path(dataset)
    train, validation, test, manifest = load_bundle(dataset)
    corpus = read_rows(dataset / "data/corpus_added.jsonl")
    docs = {row["title"]: row["text"] for row in corpus}
    assert len(docs) == len(corpus) == manifest["counts"]["unique_passages"]
    assert all(title == "nq:" + digest(text) for title, text in docs.items())
    atoms = read_rows(dataset / "data/source_queries.jsonl")
    by_id = {row["id"]: row for row in atoms}
    assert len(by_id) == len(atoms) == manifest["counts"]["unique_queries"]
    assert len({normalize_query(row["query"]) for row in atoms}) == len(atoms)
    for atom in atoms:
        assert atom["id"] == "nq-query:" + digest(normalize_query(atom["query"]))
        assert atom["docs"] == sorted(set(atom["docs"])) and atom["docs"]
        assert set(atom["docs"]) <= docs.keys()
        assert not OPERATORS.search(atom["query"])
    synthetic = read_rows(dataset / "data/train_synthetic.jsonl")
    pairs = read_rows(dataset / "data/train_or2.jsonl")
    triples = read_rows(dataset / "data/train_or3.jsonl")
    assert synthetic == pairs + triples
    assert len(pairs) == manifest["counts"]["or2"]
    assert len(triples) == manifest["counts"]["or3"]
    assert train[manifest["counts"]["original_train"]:] == synthetic
    assert len({row["id"] for row in train}) == len(train)
    assert len({row["query"] for row in synthetic}) == len(synthetic)
    assert all(len(row["components"]) == 2 for row in pairs)
    assert all(len(row["components"]) == 3 for row in triples)
    used, positive_counts = set(), Counter()
    duplicates_removed = 0
    for row in synthetic:
        components = [by_id[qid] for qid in row["component_ids"]]
        assert len(set(row["component_ids"])) == len(components)
        assert row == compose(components)
        assert set(row["docs"]) <= docs.keys()
        assert len(row["docs"]) >= len(components)
        used.update(row["component_ids"])
        positive_counts[len(row["docs"])] += 1
        duplicates_removed += sum(map(len, row["component_docs"])) - len(row["docs"])
    assert len(used) == manifest["counts"]["source_queries_used"]
    return {"status": "passed", "synthetic_examples": len(synthetic),
            "combined_training_examples": len(train), "added_documents": len(corpus),
            "validation_queries": len(validation), "test_queries": len(test),
            "source_queries_used": len(used), "positives_per_synthetic_query": dict(positive_counts),
            "duplicate_positive_occurrences_removed": duplicates_removed,
            "checks": ["frozen file hashes", "source and document identities", "exact OR positive unions",
                       "all positive documents present", "unique synthetic combinations",
                       "original training rows preserved", "evaluation rows unchanged"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset), indent=2))


if __name__ == "__main__":
    main()
