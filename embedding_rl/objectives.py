"""Pure tensor math, independent of APIs, models, and Modal."""
import math
import torch
import torch.nn.functional as F


def ndcg(scores: torch.Tensor, grades: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Last axis is the complete judged pool. Leading axes may include samples.

    Exponential gain (2**grade - 1); IDCG always uses the WHOLE fixed pool.
    All-zero relevance is defined as zero, never a division by zero.
    """
    k = min(k, scores.shape[-1])
    while grades.ndim < scores.ndim:
        grades = grades.unsqueeze(-2)
    grades = grades.expand_as(scores)
    order = scores.argsort(dim=-1, descending=True, stable=True)[..., :k]
    gain = torch.exp2(grades.float()) - 1
    discount = torch.log2(torch.arange(k, device=scores.device, dtype=torch.float32) + 2)
    dcg = (gain.gather(-1, order) / discount).sum(-1)
    ideal = (gain.sort(dim=-1, descending=True).values[..., :k] / discount).sum(-1)
    return torch.where(ideal > 0, dcg / ideal.clamp_min(1e-12), torch.zeros_like(dcg))


def sample_actions(means, radius, count, generator=None):
    """Gaussian LATENT actions; normalization is a deterministic environment step.

    radius is expected noise magnitude, approximately, independent of dimension.
    Detach actions: REINFORCE differentiates log p(z|q), not a sampled path.
    """
    sigma = radius / math.sqrt(means.shape[-1])
    noise = torch.randn(
        means.shape[0], count, means.shape[-1], device=means.device, generator=generator
    )
    actions = means.detach().unsqueeze(1) + sigma * noise
    return actions, sigma


def rloo_loss(means, actions, rewards, sigma, std_floor=0.0):
    count = rewards.shape[1]
    if count < 2:
        raise ValueError("At least two samples are required")
    rewards = rewards.detach()
    baseline = (rewards.sum(1, keepdim=True) - rewards) / (count - 1)
    advantages = rewards - baseline
    if std_floor > 0:
        # A fixed floor prevents tiny reward differences from becoming enormous updates.
        scale = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(std_floor)
        advantages = (advantages / scale).clamp(-5, 5)
    tied = rewards.max(1).values == rewards.min(1).values
    advantages = advantages.masked_fill(tied[:, None], 0)
    distribution = torch.distributions.Normal(means.unsqueeze(1), sigma)
    log_prob = distribution.log_prob(actions.detach()).sum(-1)
    return -(advantages * log_prob).mean(), advantages


def gaussian_kl(means, reference, sigma):
    """Exact KL for equal-variance Gaussian latent policies (sum over dimensions)."""
    return ((means - reference.detach()).square().sum(-1) / (2 * sigma**2)).mean()


def retrieval_rewards(actions, document_vectors, grades, k):
    if document_vectors.ndim == 2:
        scores = F.normalize(actions, dim=-1) @ document_vectors.T
    else:
        scores = torch.einsum("bgd,bcd->bgc", F.normalize(actions, dim=-1), document_vectors)
    return ndcg(scores, grades, k), scores.argmax(-1)


def infonce_loss(means, document_vectors, grades, temperature, positive_grade=2):
    """Multi-positive InfoNCE; binary benchmark labels use positive_grade=1."""
    if document_vectors.ndim == 2:
        logits = (means @ document_vectors.T) / temperature
    else:
        logits = torch.einsum("bd,bcd->bc", means, document_vectors) / temperature
    positives = grades >= positive_grade
    if not positives.any(-1).all():
        raise ValueError(f"Every training query needs at least one grade >= {positive_grade}")
    return (
        torch.logsumexp(logits, -1)
        - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), -1)
    ).mean()
