import torch

from rerank_features import policy_loss, ranking_log_probability, ranking_reward, sample_rankings, train_policy
from rerank_neural import neural_logits, train_neural_policy


def fixture():
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(5, 25, 4, generator=generator)
    neural = torch.randn(5, 25, 8, generator=generator)
    gains = torch.randint(0, 2, (5, 25), generator=generator).float()
    ideal = torch.full((5,), 10.)
    return features, neural, gains, ideal


def test_zero_residual_preserves_logits_probabilities_and_scalar_gradient():
    features, neural, gains, ideal = fixture()
    theta = torch.tensor([1., 0., 0., 0.], requires_grad=True)
    delta = torch.zeros(8, requires_grad=True)
    original = features @ theta
    expanded = neural_logits(theta, delta, features, neural)
    assert torch.equal(original, expanded)
    orders = sample_rankings(original, 32, torch.Generator().manual_seed(4))
    assert torch.equal(ranking_log_probability(original, orders, 10), ranking_log_probability(expanded, orders, 10))
    rewards = ranking_reward(orders, gains, ideal)
    first = torch.autograd.grad(policy_loss(original, orders, rewards), theta)[0]
    second, residual = torch.autograd.grad(policy_loss(expanded, orders, rewards), (theta, delta))
    assert torch.equal(first, second)
    assert torch.isfinite(residual).all() and residual.norm() > 0


def test_frozen_residual_reproduces_old_training_updates():
    features, neural, gains, ideal = fixture()
    original, _ = train_policy(features, gains, ideal, 42, steps=15)
    theta, delta, _ = train_neural_policy(features, neural, gains, ideal, 42, steps=15, delta_learning_rate=0)
    assert torch.equal(theta, original)
    assert torch.equal(delta, torch.zeros_like(delta))
