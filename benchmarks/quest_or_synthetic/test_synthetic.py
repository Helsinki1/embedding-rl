"""Focused checks of union labels, deterministic generation, and vector alignment."""
import json

import numpy as np
import pytest

from .build import compose, generate, prepare_source, sha256, write_json, write_rows
from .data import load_added_vectors


def test_repeated_query_aggregates_all_positives_and_filters_operators():
    atoms, corpus, counts = prepare_source([
        {"query": "What colour is the sky?", "answer": "a"},
        {"query": "what  COLOUR is the sky?", "answer": "b"},
        {"query": "What colour is grass?", "answer": "a"},
        {"query": "red and blue", "answer": "bad"},
        {"query": "red OR blue", "answer": "bad"},
        {"query": "why isn't grass blue", "answer": "bad"},
        {"query": "", "answer": "bad"},
    ])
    assert len(atoms) == 2
    assert sorted(map(lambda atom: len(atom["docs"]), atoms)) == [1, 2]
    assert len(corpus) == 2
    assert counts["excluded_rows"] == {"explicit_operator_in_source_query": 3, "empty_query_or_passage": 1}


def test_union_deduplicates_shared_positives_without_dropping_other_positives():
    row = compose([{"id": "q1", "query": "first question", "docs": ["a", "b"]},
                   {"id": "q2", "query": "second question", "docs": ["b", "c"]}])
    assert row["query"] == "(first question) OR (second question)"
    assert row["docs"] == ["a", "b", "c"]
    assert row["component_docs"] == [["a", "b"], ["b", "c"]]


def test_generation_is_deterministic_unique_and_covers_sources():
    atoms, _, _ = prepare_source([{"query": f"question {i}", "answer": f"passage {i}"} for i in range(12)])
    first = generate(atoms, 10, 10, 42)
    assert first == generate(atoms, 10, 10, 42)
    assert first != generate(atoms, 10, 10, 43)
    assert len({row["id"] for row in first}) == 20
    assert len({qid for row in first for qid in row["component_ids"]}) == 12
    assert [len(row["components"]) for row in first] == [2] * 10 + [3] * 10


def test_added_vectors_validate_identity_and_normalize(tmp_path):
    dataset, embeddings = tmp_path / "dataset", tmp_path / "embeddings"
    (dataset / "data").mkdir(parents=True)
    embeddings.mkdir()
    write_rows(dataset / "data/corpus_added.jsonl", [{"title": "nq:a", "text": "first"},
                                                    {"title": "nq:b", "text": "second"}])
    corpus_hash = sha256(dataset / "data/corpus_added.jsonl")
    write_json(dataset / "manifest.json", {"counts": {"unique_passages": 2},
                                           "files": {"data/corpus_added.jsonl": corpus_hash},
                                           "corpus": {"model_revision": "test"}})
    write_json(embeddings / "document_ids.json", ["nq:a", "nq:b"])
    np.save(embeddings / "documents.npy", np.array([[3, 4], [0, 2]], dtype=np.float16))
    progress = {"done": 2, "total": 2, "revision": "test", "document_input": "text", "max_length": 32768,
                "dimension": 2, "corpus_sha256": corpus_hash,
                "ids_sha256": sha256(embeddings / "document_ids.json"),
                "vectors_sha256": sha256(embeddings / "documents.npy")}
    write_json(embeddings / "progress.json", progress)
    vectors, titles, _ = load_added_vectors(dataset, embeddings, "cpu", dimension=2)
    assert titles == ["nq:a", "nq:b"]
    np.testing.assert_allclose(vectors.numpy(), [[0.6, 0.8], [0, 1]])
    write_json(embeddings / "document_ids.json", ["nq:b", "nq:a"])
    with pytest.raises(ValueError, match="checksum"):
        load_added_vectors(dataset, embeddings, "cpu", dimension=2)
    progress["ids_sha256"] = sha256(embeddings / "document_ids.json")
    write_json(embeddings / "progress.json", progress)
    with pytest.raises(ValueError, match="misaligned"):
        load_added_vectors(dataset, embeddings, "cpu", dimension=2)


def test_frozen_package_integrity_and_exact_unions():
    from .validate import validate
    report = validate()
    assert report["synthetic_examples"] == 100000
    assert report["combined_training_examples"] == 100363
