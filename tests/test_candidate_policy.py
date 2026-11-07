import json

import pytest
import torch
import torch.nn.functional as F

from embedding_rl.candidate_policy import (
    AlternatingPoolReward, CachedPoolReward, candidate_basis, project_gradient_actions,
)
from embedding_rl.config import write_json
from embedding_rl.objectives import ndcg, rloo_loss, sample_actions


def test_projection_preserves_candidate_rankings_with_redundant_documents():
    torch.manual_seed(19)
    docs = F.normalize(torch.randn(5, 16, dtype=torch.float64), dim=-1)
    docs[-1] = docs[0]
    basis = candidate_basis(docs)
    assert basis.shape == (16, 4)
    mean = F.normalize(torch.randn(1, 16, dtype=torch.float64), dim=-1)
    actions, _ = sample_actions(mean, .45, 32)
    projected = project_gradient_actions(mean, actions, [basis])
    assert torch.allclose(actions @ docs.T, projected @ docs.T, atol=1e-10)
    grades = torch.tensor([3., 0., 2., 1., 3.])
    before = ndcg(F.normalize(actions, dim=-1) @ docs.T, grades, 3)
    after = ndcg(F.normalize(projected, dim=-1) @ docs.T, grades, 3)
    assert torch.equal(before, after)


def test_projected_surrogate_has_the_conditional_score_gradient():
    torch.manual_seed(27)
    docs = F.normalize(torch.randn(3, 12), dim=-1)
    basis = candidate_basis(docs)
    mean = F.normalize(torch.randn(1, 12), dim=-1).requires_grad_()
    actions, sigma = sample_actions(mean, .45, 32)
    reward = ndcg(actions @ docs.T, torch.tensor([3., 1., 0.]), 2)
    gradient_actions = project_gradient_actions(mean, actions, [basis])
    assert not gradient_actions.requires_grad
    loss, advantage = rloo_loss(mean, gradient_actions, reward, sigma)
    loss.backward()
    expected = -(advantage[..., None] * (gradient_actions - mean.detach()[:, None]) / sigma**2).mean(1)
    assert torch.allclose(mean.grad, expected, atol=1e-5)
    assert mean.grad.norm() > 0


def test_conditional_projection_reduces_variance_without_resolved_mean_shift():
    torch.manual_seed(4821)
    groups, count, dim = 1000, 32, 16
    docs = F.normalize(torch.randn(3, dim, dtype=torch.float64), dim=-1)
    basis = candidate_basis(docs)
    mean = F.normalize(torch.randn(1, dim, dtype=torch.float64), dim=-1).expand(groups, -1)
    actions, sigma = sample_actions(mean, .45, count)
    projected = project_gradient_actions(mean, actions, [basis] * groups)
    reward = ndcg(actions @ docs.T, torch.tensor([3., 1., 0.]), 2)
    advantage = (reward - reward.mean(1, keepdim=True)) * count / (count - 1)
    plain = (advantage[..., None] * (actions - mean[:, None]) / sigma**2).mean(1)
    reduced = (advantage[..., None] * (projected - mean[:, None]) / sigma**2).mean(1)
    difference = reduced - plain
    z = difference.mean(0).abs() / (difference.std(0) / groups**.5).clamp_min(1e-12)
    assert z.max() < 5
    assert reduced.var(0).sum() < plain.var(0).sum() / 2


def cache_fixture(tmp_path):
    bundle = {"docs": [{"id": "a"}, {"id": "b"}, {"id": "outside"}],
              "splits": {"train": ["q0", "q1"], "validation": ["v"], "test": ["t"]}}
    plan = {"train_pools": {qid: ["a", "b"] for qid in bundle["splits"]["train"]}}
    for qid in plan["train_pools"]:
        write_json(tmp_path / "pairs" / f"{qid}.json", {
            "query_id": qid, "settings_digest": "frozen-rubric",
            "grades": {"a": {"effective_grade": 0}, "b": {"effective_grade": 3}},
        })
    return plan, bundle, torch.eye(3)


def test_cached_pool_requires_complete_labels_and_isolates_training_queries(tmp_path):
    plan, bundle, docs = cache_fixture(tmp_path)
    reward = CachedPoolReward(plan, tmp_path, bundle, docs, projected=True)
    actions = torch.tensor([[[.1, .9, 5.], [.9, .1, 5.]]])
    scores, winners = reward(["q0"], actions, docs, 1)
    assert scores.tolist() == [[1., 0.]]
    assert winners.tolist() == [[1, 0]]  # The high-scoring outside document is not in this environment.
    for qid in ("v", "t"):
        with pytest.raises(ValueError, match="non-training"):
            reward([qid], actions, docs, 1)
    bad = json.loads((tmp_path / "pairs/q0.json").read_text())
    del bad["grades"]["b"]
    write_json(tmp_path / "pairs/q0.json", bad)
    with pytest.raises(ValueError, match="unjudged"):
        CachedPoolReward(plan, tmp_path, bundle, docs)


def test_pool_provenance_rejects_holdout_inclusion_and_mixed_rubrics(tmp_path):
    plan, bundle, docs = cache_fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly match"):
        CachedPoolReward({"train_pools": {**plan["train_pools"], "v": ["a"]}}, tmp_path, bundle, docs)
    path = tmp_path / "pairs/q1.json"
    cached = json.loads(path.read_text())
    cached["settings_digest"] = "changed-rubric"
    write_json(path, cached)
    with pytest.raises(ValueError, match="Mixed"):
        CachedPoolReward(plan, tmp_path, bundle, docs)


def test_alternation_keeps_full_corpus_reward_and_gradient_unprojected(tmp_path):
    plan, bundle, docs = cache_fixture(tmp_path)
    bundle["qrels"] = {q: {"outside": 1} for q in bundle["splits"]["train"]}
    reward = AlternatingPoolReward(plan, tmp_path, bundle, docs, projected=True, pool_every=2)
    means = torch.tensor([[.1, .1, .1]])
    actions = torch.tensor([[[.1, .9, 5.], [.9, .1, 5.]]])
    full, winners = reward(["q0"], actions, docs, 1)
    assert full.tolist() == [[1., 1.]] and winners.tolist() == [[2, 2]]
    assert torch.equal(reward.gradient_actions(["q0"], means, actions), actions)
    pool, winners = reward(["q0"], actions, docs, 1)
    assert pool.tolist() == [[1., 0.]] and winners.tolist() == [[1, 0]]
    projected = reward.gradient_actions(["q0"], means, actions)
    assert projected[0, :, 2].tolist() == pytest.approx([.1, .1])
    assert reward.counts == {"full_corpus": 1, "candidate_pool": 1}
    with pytest.raises(ValueError, match="non-training"):
        reward(["v"], actions, docs, 1)


def test_binary_pool_control_keeps_candidates_but_changes_only_label_source(tmp_path):
    plan, bundle, docs = cache_fixture(tmp_path)
    bundle["qrels"] = {q: {"a": 1} for q in bundle["splits"]["train"]}
    judged = AlternatingPoolReward(plan, tmp_path, bundle, docs, projected=True, pool_every=2)
    binary = AlternatingPoolReward(plan, tmp_path / "no-cache", bundle, docs,
                                   projected=True, pool_every=2, grade_source="benchmark")
    actions = torch.tensor([[[.9, .1, 0.], [.1, .9, 0.]]])
    assert torch.equal(judged(["q0"], actions, docs, 1)[0], binary(["q0"], actions, docs, 1)[0])
    judged_scores, judged_winners = judged(["q0"], actions, docs, 1)
    binary_scores, binary_winners = binary(["q0"], actions, docs, 1)
    assert torch.equal(judged_winners, binary_winners)
    assert judged_scores.tolist() == [[0., 1.]]
    assert binary_scores.tolist() == [[1., 0.]]
    assert judged.metadata["pool_digest"] == binary.metadata["pool_digest"]
    assert judged.metadata["grades_digest"] != binary.metadata["grades_digest"]
    assert binary.metadata["judge_settings_digest"] is None
    assert binary.metadata["grade_source"] == "benchmark"
    with pytest.raises(ValueError, match="non-training"):
        binary(["t"], actions, docs, 1)


def test_training_transforms_only_the_policy_gradient_not_reward_actions(prepared):
    from conftest import TinyEncoder
    from test_followup import browsecomp_fixture
    from embedding_rl.train import train
    root, config, bundle = browsecomp_fixture(prepared)
    calls = []
    class Reward:
        metadata = {"source": "synthetic candidate reward"}
        def __call__(self, batch, actions, documents, k):
            assert set(batch) <= set(bundle["splits"]["train"])
            calls.append(actions.clone())
            return actions[:, :, 0].clone(), torch.zeros(actions.shape[:2], dtype=torch.int64)
    def transform(batch, means, actions):
        assert torch.equal(actions, calls[-1])
        # Zero all gradient noise: reward still varies, but the score gradient must vanish.
        return means.detach()[:, None].expand_as(actions)
    result = train(root, config, "rl", encoder_factory=TinyEncoder,
                   reward_provider=Reward(), policy_action_transform=transform)
    run = root / "runs" / result["run_id"]
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(calls) == config.steps
    assert all(row["train/grad_norm"] < 1e-6 for row in metrics if "train/grad_norm" in row)
    assert not (run / "test.json").exists()
    assert json.loads((run / "validation.json").read_text())["final"]["per_query"][0]["query_id"] == "q2"
