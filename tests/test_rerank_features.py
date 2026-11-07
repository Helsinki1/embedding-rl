import itertools

import torch

from rerank_features import normalize_features, policy_loss, ranking_log_probability, ranking_reward, sample_rankings


def test_pl_probability_and_rloo_gradient_match_exact_enumeration():
    logits = torch.tensor([[.3, -.5, .8]], dtype=torch.float64, requires_grad=True)
    orders = torch.tensor(list(itertools.permutations(range(3))))[None]
    logp = ranking_log_probability(logits, orders, 2)
    probability = logp.exp()
    assert torch.allclose(probability.sum(), torch.tensor(1., dtype=torch.float64))
    manual = []
    for a, b, c in orders[0].tolist():
        scores = logits.exp()[0]
        manual.append(scores[a] / scores.sum() * scores[b] / (scores[b] + scores[c]))
    assert torch.allclose(probability[0], torch.stack(manual))
    gains = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
    reward = ranking_reward(orders, gains, torch.tensor([2.], dtype=torch.float64), k=2)
    exact = torch.autograd.grad((probability * reward).sum(), logits, retain_graph=True)[0]
    estimate = torch.zeros_like(logits)
    for a, b in itertools.product(range(6), repeat=2):
        pair_orders, pair_rewards = orders[:, [a, b]], reward[:, [a, b]]
        gradient = torch.autograd.grad(-policy_loss(logits, pair_orders, pair_rewards, 2), logits, retain_graph=True)[0]
        estimate += probability[0, a].detach() * probability[0, b].detach() * gradient
    assert torch.allclose(exact, estimate, atol=1e-12)
    assert reward.max() == .5  # Whole-corpus ideal may exceed the candidate-set ideal.


def test_gumbel_sampling_matches_top_one_distribution_and_has_no_duplicates():
    logits = torch.tensor([[0., .5, 1.]])
    orders = sample_rankings(logits, 30000, torch.Generator().manual_seed(13))
    observed = torch.bincount(orders[0, :, 0], minlength=3).float() / 30000
    assert torch.allclose(observed, logits.softmax(-1)[0], atol=.01)
    assert torch.equal(orders.sort(-1).values, torch.arange(3).expand_as(orders))


def test_feature_normalization_is_per_query_and_handles_constants():
    raw = torch.tensor([[[1., 4.], [3., 4.]], [[10., 1.], [30., 2.]]])
    result = normalize_features(raw)
    assert torch.allclose(result[..., 0], torch.tensor([[-1., 1.], [-1., 1.]]))
    assert torch.equal(result[0, :, 1], torch.zeros(2))
    changed = raw.clone()
    changed[1] += 100
    assert torch.equal(normalize_features(changed)[0], result[0])
