"""Evaluate a fixed candidate permutation against the whole-corpus ideal ranking."""
import torch


@torch.no_grad()
def reranked_ndcg(candidate_ids, logits, corpus_grades, k=10):
    if candidate_ids.shape != logits.shape or not 1 <= k <= candidate_ids.shape[1]:
        raise ValueError("Invalid reranking shapes/depth")
    if candidate_ids.shape[0] != corpus_grades.shape[0]:
        raise ValueError("Mismatched queries")
    if any(len(set(row)) != len(row) for row in candidate_ids.tolist()):
        raise ValueError("Repeated candidate document")
    labels = corpus_grades.gather(1, candidate_ids)
    order = logits.argsort(dim=-1, descending=True, stable=True)
    discounts = torch.log2(torch.arange(k, device=logits.device) + 2)
    dcg = ((2**labels.gather(1, order[:, :k]) - 1) / discounts).sum(1)
    ideal = ((2**corpus_grades.sort(dim=-1, descending=True).values[:, :k] - 1) / discounts).sum(1)
    return torch.where(ideal > 0, dcg / ideal.clamp_min(1e-12), 0)
