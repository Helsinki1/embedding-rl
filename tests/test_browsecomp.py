import base64
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from conftest import TinyEncoder
from embedding_rl.browsecomp import CANARY, load_browsecomp, split_queries
from embedding_rl.config import Config, digest, prepared_dir, write_json
from embedding_rl.data import load_prepared
from embedding_rl.objectives import infonce_loss, retrieval_rewards, rloo_loss
from embedding_rl.train import final_test, train


def test_three_way_split_keeps_shared_evidence_and_duplicate_queries_together():
    qrels = {str(i): {f"d{i // 2}": 1} for i in range(30)}
    queries = {str(i): f"query {i}" for i in range(30)}
    queries["2"] = queries["0"].upper() + "  "
    splits = split_queries(qrels, queries, 21, 3, 42)
    assert splits == split_queries(qrels, queries, 21, 3, 42)
    assert sorted(q for ids in splits.values() for q in ids) == sorted(qrels)
    owners = {qid: name for name, ids in splits.items() for qid in ids}
    assert owners["0"] == owners["1"] == owners["2"] == owners["3"]
    docsets = [{d for q in ids for d in qrels[q]} for ids in splits.values()]
    for i, docs in enumerate(docsets):
        assert all(not docs & others for others in docsets[i + 1:])


def test_loader_decodes_only_queries_and_labels_and_pilot_preserves_holdout(monkeypatch):
    import datasets
    import huggingface_hub

    def encode(text):
        key = hashlib.sha256(CANARY.encode()).digest()
        return base64.b64encode(bytes(b ^ key[i % len(key)]
                                     for i, b in enumerate(text.encode()))).decode()

    rows = [{"query_id": str(i), "query": encode(f"question {i}"),
             "evidence_docs": [{"docid": encode(f"d{i}")}],
             "gold_docs": [{"docid": encode(f"d{i}")}],
             "answer": "must not be decoded or stored"} for i in range(10)]
    corpus = datasets.Dataset.from_list([
        {"docid": f"d{i}", "text": "A webpage", "url": f"https://example.com/{i}"}
        for i in range(10)
    ])
    calls = []

    def fake_load(name, **kwargs):
        calls.append((name, kwargs))
        return corpus if name.endswith("corpus") else rows

    class API:
        def model_info(self, *args, **kwargs):
            return type("Info", (), {"sha": "resolved-model"})()

        def dataset_info(self, name, **kwargs):
            return type("Info", (), {"sha": f"resolved-{name}"})()

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    config = Config(train_queries=7, validation_queries=1)
    full = load_browsecomp(config)
    pilot = load_browsecomp(replace(config, max_train_queries=2))
    assert full["queries"]["0"] == "question 0"
    assert full["qrels"]["0"] == full["gold_qrels"]["0"] == {"d0": 1}
    assert "answer" not in full and "text" not in full["docs"][0]
    assert len(full["docs"]) == 10
    assert pilot["full_splits"] == full["full_splits"]
    assert pilot["splits"]["test"] == full["splits"]["test"]
    assert len(pilot["splits"]["train"]) == 2
    assert all(kwargs["revision"].startswith("resolved-") for _, kwargs in calls)


def test_full_corpus_reward_uses_all_evidence_and_drives_correct_gradient():
    means = torch.tensor([[0.8, 0.6]], requires_grad=True)
    actions = torch.tensor([[[1., 0.], [0., 1.]]])
    docs = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    grades = torch.tensor([[0., 1., 1.]])
    rewards, winners = retrieval_rewards(actions, docs, grades, k=1)
    assert rewards.tolist() == [[0., 1.]]
    assert winners.tolist() == [[0, 1]]
    loss, _ = rloo_loss(means, actions, rewards, sigma=0.5)
    loss.backward()
    # Gradient descent must rotate toward the rewarded y direction.
    assert means.grad[0, 0] > 0 and means.grad[0, 1] < 0
    assert actions.grad is None
    assert torch.isfinite(infonce_loss(means, docs, grades, 0.1, positive_grade=1))


@pytest.mark.parametrize("method", ["baseline", "infonce", "rl"])
def test_browsecomp_training_without_judge_or_candidate_pools(prepared, monkeypatch, method):
    root, old_config, _ = prepared
    _, _, bundle, tensors = load_prepared(root, old_config)
    config = replace(old_config, dataset="browsecomp-plus")
    bundle.pop("pools")
    bundle["gold_qrels"] = bundle["qrels"]
    path = prepared_dir(root, config)
    manifest = {"schema": 2, "preparation": config.preparation_settings(),
                "revisions": bundle["revisions"], "data_fingerprint": digest(bundle)}
    write_json(path / "data.json", bundle)
    write_json(path / "manifest.json", manifest)
    torch.save(tensors, path / "vectors.pt")

    def forbidden(*args, **kwargs):
        raise AssertionError("BrowseComp training must not call the judge")

    monkeypatch.setattr("embedding_rl.judge.training_labels", forbidden)
    import embedding_rl.train as training
    original = training.evaluate
    evaluated = []

    def recording(*args):
        evaluated.append(args[3])
        return original(*args)

    monkeypatch.setattr(training, "evaluate", recording)
    result = train(root, config, method, encoder_factory=TinyEncoder)
    assert all(ids == bundle["splits"]["validation"] for ids in evaluated)
    run_dir = root / "runs" / result["run_id"]
    assert not (run_dir / "test.json").exists()
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["supervision"]["source"] == "browsecomp-plus/evidence"
    if method != "baseline":
        records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        assert any(row.get("train/grad_norm", 0) > 0 for row in records)
    result = final_test(root, result["run_id"], encoder_factory=TinyEncoder)
    assert evaluated[-2:] == [bundle["splits"]["test"]] * 2
    assert "gold_ndcg@2" in result["metrics"]
    assert result["delta"]["ndcg@2"] == pytest.approx(
        result["metrics"]["ndcg@2"] - result["before_metrics"]["ndcg@2"]
    )
    assert torch.equal(tensors["documents"], torch.load(
        path / "vectors.pt", weights_only=True)["documents"])
    assert final_test(root, result["run_id"], encoder_factory=TinyEncoder) == result


def test_browsecomp_rejects_judge_stage_before_loading_data(tmp_path):
    from embedding_rl.judge import judge
    with pytest.raises(ValueError, match="skip the judge stage"):
        judge(tmp_path, Config())


def test_preparation_encodes_full_corpus_without_test_queries_or_bm25(tmp_path, monkeypatch):
    from datasets import Dataset
    from embedding_rl.data import prepare

    config = Config(train_queries=1, validation_queries=1, encode_batch_size=2)
    bundle = {
        "revisions": {"model": "fixture", "corpus": "fixture", "qrels": "fixture"},
        "docs": [{"id": f"d{i}", "title": f"url{i}"} for i in range(3)],
        "queries": {"train": "1,0", "val": "0,1", "test": "NEVER ENCODE"},
        "splits": {"train": ["train"], "validation": ["val"], "test": ["test"]},
        "qrels": {"train": {"d0": 1}, "val": {"d1": 1}, "test": {"d2": 1}},
        "_corpus": Dataset.from_dict({"text": ["1,0", "0,1", "-1,0"]}),
    }
    monkeypatch.setattr("embedding_rl.browsecomp.load_browsecomp", lambda _: bundle)
    calls = []

    class Encoder(TinyEncoder):
        def __init__(self, name, revision, device, max_length, prefix):
            super().__init__(config, {}, device)

        def encode_batches(self, texts, batch_size, query=False):
            calls.extend(texts)
            return super().encode_batches(texts, batch_size, query)

    monkeypatch.setattr("embedding_rl.model.Encoder", Encoder)
    prepare(tmp_path, config)
    _, _, saved, tensors = load_prepared(tmp_path, config)
    assert "NEVER ENCODE" not in calls
    assert saved["query_vector_ids"] == ["train", "val"]
    assert "_corpus" not in saved
    assert saved["pools"] == {}
    assert tensors["documents"].tolist() == [[1., 0.], [0., 1.], [-1., 0.]]
    count = len(calls)
    prepare(tmp_path, config)
    assert len(calls) == count
