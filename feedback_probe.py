"""Label-free dense pseudo-relevance feedback for a bounded inference experiment."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def refine_queries(queries, documents, k=3, alpha=.5, temperature=.05, rounds=1):
    if not 1 <= k <= len(documents) or not 0 <= alpha <= 1 or temperature <= 0 or rounds < 1:
        raise ValueError("Invalid feedback settings")
    current = queries.clone()
    for _ in range(rounds):
        scores = current @ documents.T
        indices = scores.argsort(dim=-1, descending=True, stable=True)[:, :k]
        weights = (scores.gather(1, indices) / temperature).softmax(-1)
        centroid = F.normalize((weights[..., None] * documents[indices]).sum(1), dim=-1)
        current = F.normalize((1 - alpha) * current + alpha * centroid, dim=-1)
    return current
