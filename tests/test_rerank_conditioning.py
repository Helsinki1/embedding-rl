import torch

from rerank_conditioning import fit_residualization, original_coordinates, residualize


def test_residualization_preserves_the_full_policy_family():
    generator = torch.Generator().manual_seed(72)
    train_features = torch.randn(9, 7, 4, generator=generator, dtype=torch.float64)
    train_features -= train_features.mean(1, keepdim=True)
    train_neural = torch.randn(9, 7, 6, generator=generator, dtype=torch.float64)
    fitted = fit_residualization(train_features, train_neural)
    heldout_features = torch.randn(3, 7, 4, generator=generator, dtype=torch.float64)
    heldout_neural = torch.randn(3, 7, 6, generator=generator, dtype=torch.float64)
    theta = torch.randn(4, generator=generator, dtype=torch.float64)
    delta = torch.randn(6, generator=generator, dtype=torch.float64)
    transformed = heldout_features @ theta + residualize(heldout_features, heldout_neural, fitted) @ delta
    old_theta, old_delta = original_coordinates(theta, delta, fitted)
    original = heldout_features @ old_theta + heldout_neural @ old_delta
    torch.testing.assert_close(transformed.log_softmax(-1), original.log_softmax(-1), atol=1e-12, rtol=1e-12)
    assert torch.equal(transformed.argsort(-1), original.argsort(-1))


def test_training_projection_and_fixed_transform_ignore_new_query_values():
    generator = torch.Generator().manual_seed(73)
    features = torch.randn(12, 7, 4, generator=generator, dtype=torch.float64)
    features -= features.mean(1, keepdim=True)
    neural = torch.randn(12, 7, 6, generator=generator, dtype=torch.float64)
    fitted = fit_residualization(features, neural)
    assert fitted['projection'].untyped_storage().nbytes() == fitted['projection'].numel() * fitted['projection'].element_size()
    residual = residualize(features, neural, fitted) * fitted['scale']
    cross_product = features.reshape(-1, 4).T @ residual.reshape(-1, 6)
    torch.testing.assert_close(cross_product, torch.zeros_like(cross_product), atol=1e-12, rtol=0)
    before = {k: v.clone() for k, v in fitted.items()}
    residualize(features[:2] * 1e4, neural[:2] * -1e5, fitted)
    assert all(torch.equal(fitted[k], before[k]) for k in fitted)
    assert (fitted['scale'] > 0).all()
