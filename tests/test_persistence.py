import copy
from dataclasses import replace
import json

import pytest
import torch

from conftest import TinyEncoder
from embedding_rl.data import cached_documents, load_prepared
from embedding_rl.export import export_reranker
from embedding_rl.train import train


def test_corpus_cache_survives_query_config_changes_but_rejects_incompatible_data(prepared):
    root, config, bundle = prepared
    path, _, _, tensors = load_prepared(root, config)
    changed = replace(config, seed=100, max_train_queries=1, query_prefix="new query prefix")
    vectors, source = cached_documents(root, changed, bundle)
    assert torch.equal(vectors, tensors["documents"])
    assert source == str(path.relative_to(root))
    assert cached_documents(root, replace(config, max_length=256), bundle) == (None, None)
    for key in ("model", "corpus"):
        incompatible = copy.deepcopy(bundle)
        incompatible["revisions"][key] = "different revision"
        assert cached_documents(root, config, incompatible) == (None, None)
    reordered = copy.deepcopy(bundle)
    reordered["docs"].reverse()
    assert cached_documents(root, config, reordered) == (None, None)


def test_export_persists_candidates_page_text_and_split_boundaries(prepared):
    root, config, bundle = prepared
    run = train(root, config, "baseline", encoder_factory=TinyEncoder)
    result = export_reranker(root, run["run_id"], k=2, encoder_factory=TinyEncoder)
    path = root / "exports" / "reranker" / run["run_id"] / "top-2"
    assert result["test_exported"] is False
    assert not (path / "test.jsonl").exists()
    assert result["query_counts"] == {"train": 2, "validation": 1}
    rows = {}
    for split in ("train", "validation"):
        rows[split] = [json.loads(line) for line in (path / f"{split}.jsonl").read_text().splitlines()]
        assert [row["query_id"] for row in rows[split]] == bundle["splits"][split]
        assert all(row["split"] == split for row in rows[split])
    candidates = rows["train"][0]["candidates"]
    assert [c["document_id"] for c in candidates] == ["d0", "d1"]
    assert [c["relevance"] for c in candidates] == [0, 1]
    docs = [json.loads(line) for line in (path / "documents.jsonl").read_text().splitlines()]
    assert {doc["document_id"] for doc in docs} == {"d0", "d1"}
    assert all(doc["text"] == "Synthetic fixture" for doc in docs)

    def forbidden(*args):
        raise AssertionError("A complete export should be reusable without model inference")

    assert export_reranker(root, run["run_id"], k=2, encoder_factory=forbidden) == result
    (path / "train.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="Export file changed"):
        export_reranker(root, run["run_id"], k=2, encoder_factory=forbidden)


def test_export_loads_checkpoint_and_does_not_inject_missing_positives(prepared):
    root, config, _ = prepared
    run = train(root, config, "baseline", encoder_factory=TinyEncoder)
    original = export_reranker(root, run["run_id"], k=1, encoder_factory=TinyEncoder)
    assert original["queries_with_retrieved_evidence"]["train"] == 0
    checkpoint_path = root / "runs" / run["run_id"] / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["state_dict"]["weight"] = torch.tensor([[0., 1.], [1., 0.]])
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="Export source changed"):
        export_reranker(root, run["run_id"], k=1, encoder_factory=TinyEncoder)
    export_reranker(root, run["run_id"], k=2, encoder_factory=TinyEncoder)
    path = root / "exports" / "reranker" / run["run_id"] / "top-2" / "train.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    assert first["candidates"][0]["document_id"] == "d1"


def test_export_browsecomp_includes_full_text_and_checks_corpus_alignment(monkeypatch):
    from datasets import Dataset
    from embedding_rl.config import Config
    from embedding_rl.export import export_documents

    bundle = {"docs": [{"id": "a"}, {"id": "b"}], "revisions": {"corpus": "fixed"}}
    corpus = Dataset.from_list([
        {"docid": "a", "url": "https://example.com/a", "text": "Full page A"},
        {"docid": "b", "url": "https://example.com/b", "text": "Full page B"},
    ])

    def load(name, split, revision):
        assert name == "Tevatron/browsecomp-plus-corpus" and revision == "fixed"
        return corpus

    monkeypatch.setattr("datasets.load_dataset", load)
    rows = list(export_documents(bundle, Config(), {"b"}))
    assert rows == [{"document_id": "b", "url": "https://example.com/b", "text": "Full page B"}]
    bundle["docs"].reverse()
    with pytest.raises(ValueError, match="row order"):
        list(export_documents(bundle, Config(), {"b"}))
