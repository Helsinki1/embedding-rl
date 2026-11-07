"""A nested ranking policy with frozen neural features and a learned residual."""
import torch

from rerank_features import policy_loss, ranking_reward, sample_rankings


def neural_logits(theta, delta, scalar_features, neural_features):
    return scalar_features @ theta + neural_features @ delta


def evaluate_neural(theta, delta, scalar_features, neural_features, gains, ideal):
    logits = neural_logits(theta, delta, scalar_features, neural_features)
    orders = logits.argsort(dim=-1, descending=True, stable=True)[:, None, :]
    return ranking_reward(orders, gains, ideal)[:, 0]


def train_neural_policy(features, neural, gains, ideal, seed, steps=500, delta_learning_rate=.0001):
    """Only training tensors enter optimization; the pretrained model is absent."""
    generator = torch.Generator().manual_seed(seed)
    theta = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]))
    delta = torch.nn.Parameter(torch.zeros(neural.shape[-1]))
    optimizer = torch.optim.Adam([{"params": [theta], "lr": .01}, {"params": [delta], "lr": delta_learning_rate}])
    history = []
    for step in range(1, steps + 1):
        indices = torch.randint(len(features), (16,), generator=generator)
        logits = neural_logits(theta, delta, features[indices], neural[indices])
        orders = sample_rankings(logits, 32, generator)
        rewards = ranking_reward(orders, gains[indices], ideal[indices])
        loss = policy_loss(logits, orders, rewards)
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(theta.grad).all() or not torch.isfinite(delta.grad).all():
            raise ValueError("Nonfinite neural-policy gradient")
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append({"step": step, "train/loss": loss.item(), "train/reward": rewards.mean().item(),
                "train/informative_fraction": (rewards.max(1).values != rewards.min(1).values).float().mean().item(),
                "head/delta_norm": delta.detach().norm().item(), "head/delta_max_abs": delta.detach().abs().max().item(),
                **{f"weight/{name}": value for name, value in zip(("rrf", "dense", "bm25", "bge"), theta.detach().tolist(), strict=True)}})
    return theta.detach(), delta.detach(), history
