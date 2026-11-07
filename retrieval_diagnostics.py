"""Benchmark-only candidate coverage and a labeled upper bound for reranking."""
import torch

from embedding_rl.objectives import ndcg


@torch.no_grad()
def candidate_metrics(scores, grades, pool_size=25, k=10):
    if not 1 <= k <= pool_size <= scores.shape[1] or scores.shape != grades.shape:
        raise ValueError("Invalid candidate diagnostic shapes/depths")
    order = scores.argsort(dim=-1, descending=True, stable=True)
    pool_grades = grades.gather(1, order[:, :pool_size])
    relevance = grades > 0
    count = (pool_grades > 0).sum(1)
    discounts = torch.log2(torch.arange(k, device=scores.device) + 2)
    ideal = ((2**grades.sort(dim=-1, descending=True).values[:, :k] - 1) / discounts).sum(1)
    oracle_dcg = ((2**pool_grades.sort(dim=-1, descending=True).values[:, :k] - 1) / discounts).sum(1)
    return {
        "ndcg10": ndcg(scores, grades, k),
        "recall25": count / relevance.sum(1).clamp_min(1),
        "hit25": (count > 0).float(),
        "oracle_rerank25_ndcg10": torch.where(ideal > 0, oracle_dcg / ideal.clamp_min(1e-12), 0),
    }
