from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest
import torch

from embedding_rl.graded_judge import (
    Assessments, DollarBudget, JudgedReward, WebJudge, agreement, excerpt, fixed_max_dcg, validate,
)
from embedding_rl.objectives import rloo_loss, sample_actions


def assessment(grade=2, index=0, uncertain=False):
    return {"document_id": "d1", "grade": grade, "evidence_indices": [index],
            "query_clue_index": 0, "reason": "An explicit date matches a distinctive clue.",
            "uncertain": uncertain}


def test_positive_grades_require_real_quotes_and_query_clues():
    docs = [{"document_id": "d1", "text": "The bridge opened in 1902."}]
    query = "Which bridge opened in 1902?"
    parsed = Assessments(assessments=[assessment(uncertain=True)])
    assert validate(parsed, query, docs)["d1"]["effective_grade"] == 1
    with pytest.raises(ValueError, match="verifiable"):
        validate(Assessments(assessments=[assessment(index=100)]), query, docs)
    with pytest.raises(ValueError, match="exactly once"):
        validate(Assessments(assessments=[assessment(), assessment()]), query, docs)


def test_excerpt_can_find_evidence_beyond_prefix_without_gold_labels():
    text = "Unrelated text. " * 2000 + "A violet viaduct opened in 1902. " * 20
    result = excerpt("Which violet viaduct opened in 1902?", text)
    assert len(result) <= 9000 and "violet viaduct" in result


def test_budget_reserves_concurrent_and_uncertain_requests(tmp_path):
    budget = DollarBudget(tmp_path, ceiling=0.005)
    def reserve(i):
        try:
            budget.reserve(str(i), "gpt-6-luna", "small", 3200)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve, range(8)))
    assert sum(accepted) == 1
    assert budget.total() <= 0.005
    restarted = DollarBudget(tmp_path, ceiling=0.005)
    assert restarted.total() == budget.total()
    first = next(iter(budget.entries))
    budget.settle(first, {"input_tokens": 100, "output_tokens": 20}, "response", "gpt-6-luna")
    assert budget.entries[first]["actual_usd"] == pytest.approx(0.0000225)


def test_paid_invalid_response_is_cached_and_accounted(tmp_path):
    response = SimpleNamespace(
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 200, "output_tokens": 100}),
        id="response", model="gpt-6-luna", status="completed",
        output_text=json.dumps({"assessments": [assessment(index=100)]}),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    judge = WebJudge(tmp_path, {"queries": {"q": "Which bridge opened in 1902?"}},
                     {"d1": "The bridge opened in 1902."}, client=client)
    with pytest.raises(ValueError):
        judge.assess("q", ["d1"])
    assert judge.budget.total() > 0
    assert len(list((tmp_path / "calls").glob("*.json"))) == 1
    assert not judge.grades


def test_positive_labels_are_confirmed_conservatively_and_cached(tmp_path):
    judge = WebJudge(tmp_path, {"queries": {"q": "A question?"}}, {"d1": "A page."},
                     client=SimpleNamespace())
    calls = []
    def assess(qid, docids, tag="primary", model=None):
        calls.append(tag)
        return {"d1": {"effective_grade": 2 if tag == "primary" else 1}}
    judge.assess = assess
    assert judge.ensure({"q": ["d1"]}, workers=1) == 1
    assert judge.grades["q"]["d1"]["effective_grade"] == 1
    assert judge.ensure({"q": ["d1"]}, workers=1) == 0
    assert calls == ["primary", "positive-confirmation"]


def test_fixed_max_reward_and_group_normalization_do_not_invent_signal():
    assert fixed_max_dcg(torch.tensor([[3., 3.]])).item() == pytest.approx(1)
    assert fixed_max_dcg(torch.tensor([[0., 0.]])).item() == 0
    means = torch.tensor([[.8, .6]], requires_grad=True)
    actions, sigma = sample_actions(means, .15, 8)
    rewards = torch.ones(1, 8) * .17
    loss, advantages = rloo_loss(means, actions, rewards, sigma, std_floor=.05)
    loss.backward()
    assert advantages.abs().sum() == 0
    assert means.grad.abs().sum() == 0
    rewards[0, 0] += .00001
    _, advantages = rloo_loss(means, actions, rewards, sigma, std_floor=.05)
    assert advantages.abs().max() < .001


def test_reward_judges_every_full_corpus_top_result_and_rejects_holdout():
    class Judge:
        settings = {}
        directory = "cache"
        grades = {}
        def ensure(self, requests):
            assert requests == {"train": {"d0", "d1"}}
            self.grades = {"train": {"d0": {"effective_grade": 1}, "d1": {"effective_grade": 3}}}
    reward = JudgedReward(Judge(), {"docs": [{"id": "d0"}, {"id": "d1"}],
                                   "splits": {"train": ["train"]}})
    actions = torch.tensor([[[1., 0.], [0., 1.]]])
    docs = torch.eye(2)
    values, winners = reward(["train"], actions, docs, 1)
    assert values.tolist()[0] == pytest.approx([1 / 7, 1.0])
    assert winners.tolist() == [[0, 1]]
    with pytest.raises(ValueError, match="non-training"):
        reward(["validation"], actions, docs, 1)


def test_agreement_does_not_hide_large_disagreements():
    result = agreement([0, 1, 2, 3], [0, 1, 0, 3])
    assert result["exact"] == .75
    assert result["within_one"] == .75
    assert result["useful_boundary_agreement"] == .75


def test_evaluation_uses_shared_pool_ideal_and_strict_metric():
    from embedding_rl.graded_evaluate import score_runs
    ids = [f"d{i}" for i in range(12)]
    judgments = {"q": {docid: {"effective_grade": 1 if i == 0 else 3 if i == 11 else 0}
                        for i, docid in enumerate(ids)}}
    runs = [{"label": "topical", "rankings": {"q": ids[:10]}},
            {"label": "useful", "rankings": {"q": [ids[11], *ids[1:10]]}}]
    result = score_runs(runs, {"q": ids}, judgments, ["q"])
    assert result[0]["llm_metrics"]["llm_ndcg@10"] < .15
    assert result[0]["llm_metrics"]["llm_strict_ndcg@10"] == 0
    assert result[1]["llm_metrics"]["llm_strict_ndcg@10"] == pytest.approx(1)


def test_llm_reward_integration_preserves_benchmark_eval_and_test_separation(prepared):
    from conftest import TinyEncoder
    from test_followup import browsecomp_fixture
    from embedding_rl.train import train
    root, config, bundle = browsecomp_fixture(prepared)
    calls = []
    class Reward:
        metadata = {"source": "synthetic graded reward"}
        def __call__(self, batch, actions, documents, k):
            assert set(batch) <= set(bundle["splits"]["train"])
            calls.append(batch)
            return actions[:, :, 0], torch.zeros(actions.shape[:2], dtype=torch.int64)
    result = train(root, config, "rl", encoder_factory=TinyEncoder, reward_provider=Reward())
    path = root / "runs" / result["run_id"]
    assert len(calls) == config.steps
    assert json.loads((path / "run.json").read_text())["supervision"]["source"] == "synthetic graded reward"
    assert json.loads((path / "validation.json").read_text())["before"]["per_query"][0]["query_id"] == "q2"
    assert not (path / "test.json").exists()
