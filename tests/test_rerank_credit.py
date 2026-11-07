import itertools

import pytest
import torch

from rerank_credit import credit_loss
from rerank_features import ranking_log_probability, ranking_reward, sample_rankings


@pytest.mark.parametrize('estimator', ['placement', 'marginalize_all'])
def test_credit_estimator_matches_exact_expected_gradient(estimator):
    logits = torch.tensor([[.3, -.5, .8]], dtype=torch.float64, requires_grad=True)
    orders = torch.tensor(list(itertools.permutations(range(3))))[None]
    probabilities = ranking_log_probability(logits, orders, 2).exp()
    gains = torch.tensor([[0., 1., 2.]], dtype=torch.float64)
    ideal = torch.tensor([3.], dtype=torch.float64)
    rewards = ranking_reward(orders, gains, ideal, 2)
    expected = torch.autograd.grad((probabilities * rewards).sum(), logits, retain_graph=True)[0]
    observed = torch.zeros_like(expected)
    for a, b in itertools.product(range(6), repeat=2):
        gradient = torch.autograd.grad(-credit_loss(logits, orders[:, [a, b]], gains, ideal, estimator, 2), logits, retain_graph=True)[0]
        observed += probabilities[0, a].detach() * probabilities[0, b].detach() * gradient
    assert torch.allclose(expected, observed, atol=1e-12)


def test_marginalized_first_position_is_exact_and_has_no_sampling_variance():
    logits = torch.tensor([[.3, -.5, .8]], dtype=torch.float64, requires_grad=True)
    gains = torch.tensor([[0., 1., 2.]], dtype=torch.float64)
    ideal = torch.tensor([3.], dtype=torch.float64)
    exact = torch.autograd.grad((logits.softmax(-1) * gains).sum() / ideal[0], logits, retain_graph=True)[0]
    for seed in (42, 43, 44):
        orders = sample_rankings(logits, 2, torch.Generator().manual_seed(seed))
        gradient = torch.autograd.grad(-credit_loss(logits, orders, gains, ideal, 'marginalize_all', 1), logits, retain_graph=True)[0]
        assert torch.allclose(gradient, exact, atol=1e-12)
