import math

import pytest
import torch

from embedding_rl.objectives import gaussian_kl, infonce_loss, ndcg, rloo_loss, sample_actions


def test_ndcg_uses_full_pool_ideal_even_when_relevant_doc_is_outside_top_k():
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    grades = torch.tensor([[1.0, 0.0, 3.0]])
    assert ndcg(scores, grades, k=1).item() == pytest.approx(1 / 7)
    assert ndcg(grades, grades, k=2).item() == pytest.approx(1)
    assert ndcg(scores, torch.zeros_like(grades)).item() == 0


def test_reinforce_matches_known_expected_reward_gradient():
    # For reward z_0, E[reward] = mean_0, so its gradient must be [1, 0].
    # This catches accidentally differentiating through the sampled action.
    means = torch.tensor([[0.3, -0.2]], requires_grad=True)
    actions, sigma = sample_actions(means, radius=0.4, count=50000,
                                    generator=torch.Generator().manual_seed(7))
    assert not actions.requires_grad
    loss, advantages = rloo_loss(means, actions, actions[:, :, 0], sigma)
    loss.backward()
    assert means.grad[0, 0].item() == pytest.approx(-1, abs=0.03)
    assert means.grad[0, 1].item() == pytest.approx(0, abs=0.03)
    assert advantages.sum().item() == pytest.approx(0, abs=0.003)


def test_constant_reward_has_no_policy_gradient():
    means = torch.tensor([[0.5, 0.5]], requires_grad=True)
    actions, sigma = sample_actions(means, 0.2, 8)
    loss, _ = rloo_loss(means, actions, torch.ones(1, 8), sigma)
    loss.backward()
    assert torch.equal(means.grad, torch.zeros_like(means))


def test_exact_gaussian_kl_and_no_reference_gradient():
    means = torch.tensor([[1.0, 0.0]], requires_grad=True)
    ref = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = gaussian_kl(means, ref, sigma=0.5)
    assert loss.item() == pytest.approx(4)
    loss.backward()
    assert ref.grad is None
    assert means.grad.tolist() == [[4, -4]]


def test_infonce_uses_all_positives_and_rejects_unlabeled_query():
    means = torch.tensor([[0.0, 1.0]], requires_grad=True)
    docs = torch.zeros(1, 4, 2)
    grades = torch.tensor([[3, 2, 1, 0]])
    assert infonce_loss(means, docs, grades, 0.1).item() == pytest.approx(math.log(2))
    with pytest.raises(ValueError):
        infonce_loss(means, docs, torch.zeros_like(grades), 0.1)
