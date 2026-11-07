"""Deterministic, full-corpus evaluation against benchmark qrels."""
import torch

from .objectives import ndcg


@torch.no_grad()
def evaluate(encoder, bundle, vectors, query_ids, config):
    queries = encoder.encode_batches(
        [bundle["queries"][qid] for qid in query_ids], config.encode_batch_size, query=True
    )
    scores = queries @ vectors.T
    doc_index = {doc["id"]: i for i, doc in enumerate(bundle["docs"])}
    grades = torch.zeros_like(scores)
    for row, qid in enumerate(query_ids):
        for doc_id, grade in bundle["qrels"][qid].items():
            grades[row, doc_index[doc_id]] = grade
    order = scores.argsort(dim=-1, descending=True, stable=True)
    relevant = grades > 0
    recall = relevant.gather(1, order[:, :config.recall_k]).sum(1) / relevant.sum(1).clamp_min(1)
    ndcgs = ndcg(scores, grades, config.ndcg_k)
    metrics = {f"ndcg@{config.ndcg_k}": ndcgs.mean().item(),
               f"recall@{config.recall_k}": recall.float().mean().item()}
    gold_ndcgs = None
    if "gold_qrels" in bundle:
        gold_grades = torch.zeros_like(scores)
        for row, qid in enumerate(query_ids):
            for doc_id, grade in bundle["gold_qrels"][qid].items():
                gold_grades[row, doc_index[doc_id]] = grade
        gold_ndcgs = ndcg(scores, gold_grades, config.ndcg_k)
        metrics[f"gold_ndcg@{config.ndcg_k}"] = gold_ndcgs.mean().item()
    examples = []
    for row, qid in enumerate(query_ids[:5]):
        for rank, index in enumerate(order[row, :5].tolist(), 1):
            examples.append({
                "query_id": qid,
                "query": "[benchmark query omitted]" if config.dataset == "browsecomp-plus"
                else bundle["queries"][qid], "rank": rank,
                "document_id": bundle["docs"][index]["id"],
                "title": bundle["docs"][index]["title"],
                "gold_relevance": grades[row, index].item(),
            })
    per_query = [{"query_id": qid, f"ndcg@{config.ndcg_k}": ndcgs[i].item(),
                  f"recall@{config.recall_k}": recall[i].item()}
                 for i, qid in enumerate(query_ids)]
    if gold_ndcgs is not None:
        for row, value in zip(per_query, gold_ndcgs):
            row[f"gold_ndcg@{config.ndcg_k}"] = value.item()
    return {"metrics": metrics, "examples": examples, "per_query": per_query}
