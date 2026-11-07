"""First-choice KL regularization for the cached neural ranking policy."""
import torch

from rerank_features import policy_loss, ranking_reward, sample_rankings
from rerank_neural import neural_logits


def first_choice_kl(logits, reference_logits):
    """Mean-query KL(current || fixed reference), not full-ranking KL."""
    log_prob = logits.log_softmax(-1)
    reference = reference_logits.detach().log_softmax(-1)
    return (log_prob.exp() * (log_prob - reference)).sum(-1).mean()


def train_anchored_policy(features, neural, gains, ideal, seed, beta, steps=500):
    """Only training tensors enter optimization; the initial hybrid is fixed."""
    if beta < 0:
        raise ValueError("KL strength must be nonnegative")
    generator = torch.Generator().manual_seed(seed)
    theta = torch.nn.Parameter(torch.tensor([1., 0., 0., 0.]))
    delta = torch.nn.Parameter(torch.zeros(neural.shape[-1]))
    reference = (features @ theta.detach()).detach()
    optimizer = torch.optim.Adam([{"params": [theta], "lr": .01}, {"params": [delta], "lr": .0001}])
    history = []
    for step in range(1, steps + 1):
        indices = torch.randint(len(features), (16,), generator=generator)
        logits = neural_logits(theta, delta, features[indices], neural[indices])
        orders = sample_rankings(logits, 32, generator)
        rewards = ranking_reward(orders, gains[indices], ideal[indices])
        pg_loss = policy_loss(logits, orders, rewards)
        kl = first_choice_kl(logits, reference[indices])
        loss = pg_loss + beta * kl
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(theta.grad).all() or not torch.isfinite(delta.grad).all():
            raise ValueError("Nonfinite anchored-policy gradient")
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append({"step": step, "train/loss": loss.item(), "train/policy_loss": pg_loss.item(),
                "train/first_choice_kl": kl.item(), "train/reward": rewards.mean().item(),
                "train/informative_fraction": (rewards.max(1).values != rewards.min(1).values).float().mean().item(),
                "head/delta_norm": delta.detach().norm().item(),
                **{f"weight/{name}": value for name, value in zip(("rrf", "dense", "bm25", "bge"), theta.detach().tolist(), strict=True)}})
    return theta.detach(), delta.detach(), history
