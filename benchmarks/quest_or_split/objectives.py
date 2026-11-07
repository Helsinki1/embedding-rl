"""Matching coordinate slices, collective InfoNCE, and union-F1 REINFORCE."""
import math

import torch
import torch.nn.functional as F


def split_vectors(vectors):
    """Balanced contiguous thirds; retain every coordinate, normalize each third."""
    if vectors.shape[-1] < 3:
        raise ValueError("Need at least three embedding coordinates")
    return tuple(F.normalize(piece.float(), dim=-1)
                 for piece in torch.tensor_split(vectors, 3, dim=-1))


def prepare_documents(documents):
    pieces = split_vectors(documents)
    if any(not torch.isfinite(piece).all() or (piece.norm(dim=-1) == 0).any()
           for piece in pieces):
        raise ValueError("Document slices must be finite and nonzero")
    return tuple(piece.contiguous() for piece in pieces)


def piece_rankings(means, documents, k):
    return torch.stack([(query @ corpus.T).topk(k, dim=-1).indices
                        for query, corpus in zip(split_vectors(means), documents)], dim=-2)


def union_f1(rankings, positives):
    """[batch, ..., 3, k] -> [batch, ...], deduplicating before counting."""
    ids = rankings.flatten(-2).sort(dim=-1).values
    unique = torch.ones_like(ids, dtype=torch.bool)
    unique[..., 1:] = ids[..., 1:] != ids[..., :-1]
    return torch.stack([
        2 * (torch.isin(row, gold) & keep).sum(-1).float()
        / (keep.sum(-1) + len(gold))
        for row, keep, gold in zip(ids, unique, positives)
    ])


def infonce(means, documents, positives, temperature):
    if temperature <= 0 or len(positives) != len(means) or any(len(p) == 0 for p in positives):
        raise ValueError("Need positive temperature and nonempty labels for each query")
    logits = torch.stack([(query @ corpus.T) / temperature
                          for query, corpus in zip(split_vectors(means), documents)], dim=1)
    union = torch.logsumexp(logits, dim=1)
    positive = torch.stack([union[i, ids].mean() for i, ids in enumerate(positives)])
    return (torch.logsumexp(union, dim=-1) - positive).mean()


def reinforce(means, reference, documents, positives, k, samples, radius, kl_weight,
              generator=None):
    if samples < 2 or radius <= 0 or kl_weight < 0:
        raise ValueError("Invalid Gaussian policy settings")
    parts, anchors = split_vectors(means), split_vectors(reference.detach())
    actions, rankings, sigmas = [], [], []
    with torch.no_grad():
        for part, corpus in zip(parts, documents):
            sigma = radius / math.sqrt(part.shape[-1])
            action = part.detach()[:, None] + sigma * torch.randn(
                len(part), samples, part.shape[-1], device=part.device, generator=generator)
            actions.append(action)
            sigmas.append(sigma)
            rankings.append((F.normalize(action, dim=-1) @ corpus.T).topk(k, dim=-1).indices)
        rewards = union_f1(torch.stack(rankings, dim=-2), positives)
        baseline = (rewards.sum(-1, keepdim=True) - rewards) / (samples - 1)
        advantages = rewards - baseline
        tied = rewards.max(-1).values == rewards.min(-1).values
        advantages.masked_fill_(tied[:, None], 0)
    # Score-function gradient for the joint action: all three pieces share union F1.
    log_prob = sum(torch.distributions.Normal(part[:, None], sigma).log_prob(action).sum(-1)
                   for part, sigma, action in zip(parts, sigmas, actions))
    policy_loss = -(advantages * log_prob).mean()
    kl = sum((part - anchor).square().sum(-1).mean() / (2 * sigma ** 2)
             for part, anchor, sigma in zip(parts, anchors, sigmas))
    return policy_loss + kl_weight * kl, {
        "policy_loss": policy_loss.detach().item(), "kl": kl.detach().item(),
        "reward_f1": rewards.mean().item(),
        "reward_std": rewards.std(-1, unbiased=False).mean().item(),
        "zero_advantage_fraction": (advantages.abs().sum(-1) == 0).float().mean().item(),
    }
