import torch

from rerank_anchor import first_choice_kl, train_anchored_policy
from rerank_neural import train_neural_policy


def test_kl_reference_nonnegative_shift_invariance_and_detached_reference():
    reference = torch.tensor([[1., -2., .4], [0., .7, -1.]], dtype=torch.float64, requires_grad=True)
    current = torch.tensor([[.2, -.5, 2.], [-3., 1.2, .3]], dtype=torch.float64, requires_grad=True)
    assert first_choice_kl(reference, reference).item() == 0
    loss = first_choice_kl(current, reference)
    assert loss.item() > 0
    torch.testing.assert_close(loss, first_choice_kl(current + 123., reference - 77.))
    expected = (current.softmax(-1) * (current.softmax(-1) / reference.detach().softmax(-1)).log()).sum(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert reference.grad is None


def test_zero_anchor_reproduces_unregularized_updates_exactly():
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(5, 25, 4, generator=generator)
    neural = torch.randn(5, 25, 8, generator=generator)
    gains = torch.randint(0, 2, (5, 25), generator=generator).float()
    ideal = torch.full((5,), 10.)
    first = train_neural_policy(features, neural, gains, ideal, 42, steps=20)
    second = train_anchored_policy(features, neural, gains, ideal, 42, beta=0, steps=20)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
