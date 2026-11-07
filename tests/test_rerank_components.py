import pytest
import torch

from rerank_components import original_component_weights, transform_component
from rerank_conditioning import fit_residualization


@pytest.mark.parametrize('mode', ['remove_only', 'scale_only'])
def test_component_intervention_preserves_policy_family(mode):
    generator = torch.Generator().manual_seed(91)
    features = torch.randn(8, 7, 4, generator=generator, dtype=torch.float64)
    neural = torch.randn(8, 7, 6, generator=generator, dtype=torch.float64)
    fitted = fit_residualization(features[:5], neural[:5])
    theta = torch.randn(4, generator=generator, dtype=torch.float64)
    delta = torch.randn(6, generator=generator, dtype=torch.float64)
    new = features[5:] @ theta + transform_component(features[5:], neural[5:], fitted, mode) @ delta
    old_theta, old_delta = original_component_weights(theta, delta, fitted, mode)
    old = features[5:] @ old_theta + neural[5:] @ old_delta
    torch.testing.assert_close(new.log_softmax(-1), old.log_softmax(-1), atol=1e-12, rtol=1e-12)
    assert torch.equal(new.argsort(-1), old.argsort(-1))
