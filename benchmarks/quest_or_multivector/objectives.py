"""Collective retrieval objectives shared by both three-vector architectures."""
import math

import torch
import torch.nn.functional as F


def multi_vector_infonce(embeddings, documents, positives, temperature):
    if embeddings.ndim != 3 or embeddings.shape[1] != 3 or temperature <= 0:
        raise ValueError("Expected [batch, 3, dimension] and a positive temperature")
    if len(positives) != len(embeddings) or any(len(p) == 0 for p in positives):
        raise ValueError("Every query needs a nonempty positive set")
    # Each document can match ANY slot. The denominator includes the entire corpus.
    logits = (embeddings @ documents.T) / temperature
    union_logits = torch.logsumexp(logits, dim=1)
    # Average over every positive, not logsumexp over positives (which can ignore some).
    positive_scores = torch.stack([union_logits[i, p].mean() for i, p in enumerate(positives)])
    return (torch.logsumexp(union_logits, dim=-1) - positive_scores).mean()


def union_answer_f1(indices, positives):
    """Set F1 of [batch, samples, heads, k] retrievals; count duplicates once."""
    ordered = indices.flatten(-2).sort(dim=-1).values
    unique = torch.ones_like(ordered, dtype=torch.bool)
    unique[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    return torch.stack([
        2 * (torch.isin(predictions, gold) & first).sum(-1).float()
        / (first.sum(-1) + len(gold))
        for predictions, first, gold in zip(ordered, unique, positives)
    ])


def multi_vector_reinforce(means, reference, documents, positives, k, samples,
                           radius, kl_weight, generator=None):
    """Joint Gaussian policy over three output vectors, rewarded by union F1.

    Noise is sampled around completed outputs; it does not enter the architecture's
    internal feedback. KL sums all three slots, as the joint policy log-probability does.
    """
    if means.ndim != 3 or means.shape[1] != 3 or reference.shape != means.shape:
        raise ValueError("Expected matching [batch, 3, dimension] policy/reference means")
    if samples < 2 or radius <= 0 or kl_weight < 0 or not 1 <= k <= len(documents):
        raise ValueError("Invalid exploration, KL weight, or retrieval cutoff")
    if len(positives) != len(means) or any(len(p) == 0 for p in positives):
        raise ValueError("Every query needs a nonempty positive set")
    sigma = radius / math.sqrt(means.shape[-1])
    with torch.no_grad():
        # Detached samples are required for the score-function policy gradient.
        actions = means.detach()[:, None] + sigma * torch.randn(
            len(means), samples, *means.shape[1:], device=means.device, generator=generator)
        indices = (F.normalize(actions, dim=-1) @ documents.T).topk(k, dim=-1).indices
        rewards = union_answer_f1(indices, positives)
        baseline = (rewards.sum(-1, keepdim=True) - rewards) / (samples - 1)
        advantages = rewards - baseline
        tied = rewards.max(-1).values == rewards.min(-1).values
        advantages.masked_fill_(tied[:, None], 0)
    log_prob = torch.distributions.Normal(means[:, None], sigma).log_prob(actions).sum((-1, -2))
    policy_loss = -(advantages * log_prob).mean()
    kl = (means - reference.detach()).square().sum((-1, -2)).mean() / (2 * sigma**2)
    return policy_loss + kl_weight * kl, {
        "policy_loss": policy_loss.detach().item(), "kl": kl.detach().item(),
        "reward_f1": rewards.mean().item(),
        "reward_std": rewards.std(-1, unbiased=False).mean().item(),
        "zero_advantage_fraction": (advantages.abs().sum(-1) == 0).float().mean().item(),
    }
