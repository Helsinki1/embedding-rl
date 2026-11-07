"""Placement and marginalized estimators for additive ranking rewards."""
import torch

from rerank_features import ranking_reward, sample_rankings


def credit_loss(logits, orders, gains, ideal, estimator, k=10):
    """Same expected nDCG objective; only the gradient estimator differs."""
    samples = orders.shape[1]
    if samples < 2 or estimator not in {"placement", "marginalize_all"}:
        raise ValueError("Unknown estimator or too few samples")
    ordered = logits[:, None, :].expand(-1, samples, -1).gather(-1, orders)
    selected_gains = gains[:, None, :].expand_as(ordered).gather(-1, orders)
    denominator = ordered.flip(-1).logcumsumexp(-1).flip(-1)
    log_terms = (ordered - denominator)[..., :k]
    prefix_including = log_terms.cumsum(-1)
    discounts = torch.log2(torch.arange(k, dtype=logits.dtype) + 2)
    divisor = ideal[:, None, None].clamp_min(1e-12) * discounts
    if estimator == "placement":
        utility = selected_gains[..., :k] / divisor
        baseline = (utility.sum(1, keepdim=True) - utility) / (samples - 1)
        return -((utility - baseline).detach() * prefix_including).sum(-1).mean()
    positions = torch.arange(logits.shape[-1])[None, :] < torch.arange(k)[:, None]
    conditional = ordered[..., None, :].expand(-1, -1, k, -1).masked_fill(positions, -torch.inf).softmax(-1)
    utility = (conditional * selected_gains[..., None, :]).sum(-1) / divisor
    baseline = (utility.sum(1, keepdim=True) - utility) / (samples - 1)
    prefix_before = torch.cat([torch.zeros_like(prefix_including[..., :1]), prefix_including[..., :-1]], dim=-1)
    # Differentiate the conditional expectation and the sampled prefix probability.
    return -(utility + (utility - baseline).detach() * prefix_before).sum(-1).mean()


def train_credit_policy(features, gains, ideal, seed, estimator, steps=500):
    generator = torch.Generator().manual_seed(seed)
    weights = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]))
    optimizer = torch.optim.Adam([weights], lr=.01)
    history = []
    for step in range(1, steps + 1):
        indices = torch.randint(len(features), (16,), generator=generator)
        logits = features[indices] @ weights
        orders = sample_rankings(logits, 32, generator)
        loss = credit_loss(logits, orders, gains[indices], ideal[indices], estimator)
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(weights.grad).all():
            raise ValueError("Nonfinite credit-assignment gradient")
        optimizer.step()
        if step == 1 or step % 50 == 0:
            rewards = ranking_reward(orders, gains[indices], ideal[indices])
            history.append({"step": step, "train/loss": loss.item(), "train/reward": rewards.mean().item(),
                **{f"weight/{name}": value for name, value in zip(("rrf", "dense", "bm25", "bge"), weights.detach().tolist(), strict=True)}})
    return weights.detach(), history
