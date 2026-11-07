"""One-step Plackett–Luce policy gradients for a small frozen-feature reranker."""
import torch


def normalize_features(raw):
    """Normalize each signal within its query's fixed candidate set."""
    return (raw - raw.mean(-2, keepdim=True)) / raw.std(-2, unbiased=False, keepdim=True).clamp_min(1e-6)


def ranking_log_probability(logits, orders, k):
    """Probability of the first k placements; remaining items stay in denominators."""
    expanded = logits[:, None, :].expand(-1, orders.shape[1], -1)
    ordered = expanded.gather(-1, orders)
    denominator = ordered.flip(-1).logcumsumexp(-1).flip(-1)
    return (ordered - denominator)[..., :k].sum(-1)


def sample_rankings(logits, samples, generator):
    uniform = torch.rand(logits.shape[0], samples, logits.shape[1], generator=generator, dtype=logits.dtype)
    noise = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(logits.dtype).tiny)))
    return (logits.detach()[:, None, :] + noise).argsort(dim=-1, descending=True, stable=True)


def ranking_reward(orders, gains, ideal, k=10):
    expanded = gains[:, None, :].expand(-1, orders.shape[1], -1)
    discounts = torch.log2(torch.arange(k, dtype=gains.dtype) + 2)
    dcg = (expanded.gather(-1, orders[..., :k]) / discounts).sum(-1)
    return torch.where(ideal[:, None] > 0, dcg / ideal[:, None].clamp_min(1e-12), 0)


def policy_loss(logits, orders, rewards, k=10):
    if rewards.shape[1] < 2:
        raise ValueError("RLOO needs at least two samples")
    rewards = rewards.detach()
    baseline = (rewards.sum(1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
    advantage = rewards - baseline
    return -(advantage * ranking_log_probability(logits, orders, k)).mean()


def evaluate_weights(weights, features, gains, ideal):
    orders = (features @ weights).argsort(dim=-1, descending=True, stable=True)[:, None, :]
    return ranking_reward(orders, gains, ideal)[:, 0]


def train_policy(features, gains, ideal, seed, steps=500, batch_size=16, samples=32, learning_rate=.01):
    """Receives training tensors only; all evaluation happens after final update."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]))
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    history = []
    for step in range(1, steps + 1):
        indices = torch.randint(len(features), (batch_size,), generator=generator)
        logits = features[indices] @ weights
        orders = sample_rankings(logits, samples, generator)
        rewards = ranking_reward(orders, gains[indices], ideal[indices])
        loss = policy_loss(logits, orders, rewards)
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(weights.grad).all():
            raise ValueError("Nonfinite policy gradient")
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append({"step": step, "train/loss": loss.item(), "train/reward": rewards.mean().item(),
                "train/informative_fraction": (rewards.max(1).values != rewards.min(1).values).float().mean().item(),
                **{f"weight/{name}": value for name, value in zip(("rrf", "dense", "bm25", "bge"), weights.detach().tolist(), strict=True)}})
    return weights.detach(), history
