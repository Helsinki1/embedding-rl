"""Single-vector Gaussian REINFORCE and full-corpus multi-positive InfoNCE."""
import math

import torch
import torch.nn.functional as F


def answer_f1(indices, gold_indices):
    """indices: [batch, samples, k]; each retrieved index occurs at most once."""
    return torch.stack([
        2 * torch.isin(predictions, gold).sum(-1).float() / (predictions.shape[-1] + len(gold))
        for predictions, gold in zip(indices, gold_indices)
    ])


def reinforce(means, reference, documents, gold_indices, k, samples, radius, kl_weight,
              generator=None):
    if samples < 2 or radius <= 0:
        raise ValueError("REINFORCE needs at least two samples and positive exploration radius")
    sigma = radius / math.sqrt(means.shape[-1])
    # Detached LATENT samples: normalization and retrieval belong to the environment.
    # Differentiating through a reparameterized sample would cancel the score gradient.
    with torch.no_grad():
        actions = means.detach()[:, None] + sigma * torch.randn(
            len(means), samples, means.shape[-1], device=means.device, generator=generator
        )
        indices = (F.normalize(actions, dim=-1) @ documents.T).topk(k, dim=-1).indices
        rewards = answer_f1(indices, gold_indices)
        baseline = (rewards.sum(-1, keepdim=True) - rewards) / (samples - 1)
        advantages = rewards - baseline
        # Avoid a spurious gradient from FP32 subtraction when every reward is tied.
        tied = rewards.max(-1).values == rewards.min(-1).values
        advantages.masked_fill_(tied[:, None], 0)
    log_prob = torch.distributions.Normal(means[:, None], sigma).log_prob(actions).sum(-1)
    policy_loss = -(advantages * log_prob).mean()
    # Exact KL(N(mu, sigma² I) || N(mu_initial, sigma² I)), before normalization.
    kl = (means - reference.detach()).square().sum(-1).mean() / (2 * sigma**2)
    return policy_loss + kl_weight * kl, {
        "policy_loss": policy_loss.detach().item(), "kl": kl.detach().item(),
        "reward_f1": rewards.mean().item(),
        "reward_std": rewards.std(-1, unbiased=False).mean().item(),
        "zero_advantage_fraction": (advantages.abs().sum(-1) == 0).float().mean().item(),
    }


def infonce(means, documents, gold_indices, temperature):
    """Average -log p(positive) over ALL gold answers, with a full-corpus denominator.

    This is the supervised multi-positive InfoNCE extension: each answer receives
    equal target mass. It cannot satisfy an OR query by learning only one answer.
    """
    if temperature <= 0 or any(len(gold) == 0 for gold in gold_indices):
        raise ValueError("Positive temperature and nonempty gold sets are required")
    logits = (means @ documents.T) / temperature
    positive_logits = torch.stack([
        logits[i, gold].mean() for i, gold in enumerate(gold_indices)
    ])
    return (torch.logsumexp(logits, dim=-1) - positive_logits).mean()
