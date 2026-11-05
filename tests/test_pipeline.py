from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from embedding_rl.data import grouped_split, load_prepared, mixed_pool
from embedding_rl.judge import Grade, Grades, cache_dir, judge, validate_grades
from embedding_rl.train import final_test, train
from conftest import TinyEncoder


class FakeJudge:
    def __init__(self, fail_after=None):
        self.responses = self
        self.calls = []
        self.fail_after = fail_after

    def parse(self, **kwargs):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("Simulated connection failure")
        payload = json.loads(kwargs["input"][1]["content"])
        self.calls.append(payload)
        grades = Grades(grades=[Grade(document_id=d["document_id"],
                                      grade=3 if d["document_id"] == "d1" else 0,
                                      reason="Synthetic test label") for d in payload["documents"]])
        return SimpleNamespace(status="completed", id="fake-response", model="fake",
                               output_parsed=grades, usage=None)


def test_grouped_split_is_deterministic_and_keeps_evidence_families_together():
    qrels = {str(i): {str(i // 2): 1} for i in range(20)}
    train_ids, val_ids = grouped_split(qrels, 8, 5, 42)
    assert (train_ids, val_ids) == grouped_split(qrels, 8, 5, 42)
    assert not set(train_ids) & set(val_ids)
    assert not {d for q in train_ids for d in qrels[q]} & {d for q in val_ids for d in qrels[q]}


def test_mixed_pool_is_unique_and_contains_both_retrievers():
    assert mixed_pool([4, 3, 2, 1], [1, 2, 3, 4], 3) == [0, 3, 1]


def test_judge_resumes_after_failure_and_never_sees_validation_or_test(prepared):
    root, config, bundle = prepared
    partial = FakeJudge(fail_after=1)
    with pytest.raises(RuntimeError, match="Simulated"):
        judge(root, config, partial)
    rest = FakeJudge()
    assert judge(root, config, rest)["new_pairs"] == 6
    assert judge(root, config, FakeJudge())["new_pairs"] == 0
    allowed = {bundle["queries"][qid] for qid in bundle["splits"]["train"]}
    assert {p["claim"] for p in partial.calls + rest.calls} <= allowed
    path, manifest, _, _ = load_prepared(root, config)
    assert cache_dir(path, config, manifest) != cache_dir(
        path, replace(config, judge_model="different"), manifest
    )


def test_judge_budget_checked_before_any_paid_request(prepared):
    root, config, _ = prepared
    client = FakeJudge()
    with pytest.raises(ValueError, match="exceed"):
        judge(root, replace(config, judge_max_new_pairs=1), client)
    assert client.calls == []


def test_judge_rejects_duplicate_or_missing_ids():
    g = Grade(document_id="a", grade=0, reason="none")
    for parsed in (None, Grades(grades=[g, g]), Grades(grades=[g])):
        with pytest.raises(ValueError):
            validate_grades(parsed, ["a", "b"])


@pytest.mark.parametrize("method", ["baseline", "infonce", "rl"])
def test_end_to_end_training_and_explicit_final_test(prepared, monkeypatch, method):
    root, config, bundle = prepared
    judge(root, config, FakeJudge())
    import embedding_rl.train as training
    original = training.evaluate
    evaluated = []

    def recording_evaluate(encoder, data, docs, ids, cfg):
        evaluated.append(ids)
        return original(encoder, data, docs, ids, cfg)

    monkeypatch.setattr(training, "evaluate", recording_evaluate)
    result = train(root, config, method, encoder_factory=TinyEncoder)
    assert all(ids == bundle["splits"]["validation"] for ids in evaluated)
    assert result["best_validation_ndcg"] >= result["initial_validation_ndcg"]
    directory = root / "runs" / result["run_id"]
    assert (directory / "best.pt").exists()
    assert not (directory / "test.json").exists()
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    assert checkpoint["step"] == result["best_step"]
    if method != "baseline":
        records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        assert any(row.get("train/grad_norm", 0) > 0 for row in records)
    test = final_test(root, result["run_id"], encoder_factory=TinyEncoder)
    assert evaluated[-1] == bundle["splits"]["test"]
    assert 0 <= test["metrics"]["ndcg@2"] <= 1
    assert final_test(root, result["run_id"], encoder_factory=TinyEncoder) == test


def test_real_wandb_offline_logging(prepared, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    root, config, _ = prepared
    result = train(root, replace(config, wandb_mode="offline"), "baseline",
                   encoder_factory=TinyEncoder)
    assert list((root / "runs" / result["run_id"]).glob("wandb/offline-run-*/run-*.wandb"))
