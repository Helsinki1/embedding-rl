"""Fixed-position InfoNCE with no output-to-target matching."""
import torch


def ordered_clause_infonce(embeddings, teachers, target_ids, temperature=.05):
    if embeddings.ndim != 3 or embeddings.shape[1] != 3 or temperature <= 0:
        raise ValueError("Expected three output vectors and positive temperature")
    if target_ids.shape != embeddings.shape[:2]:
        raise ValueError("Expected [batch, 3] ordered target IDs, padded with -1")
    valid = target_ids >= 0
    if not torch.all(valid.sum(-1) >= 2):
        raise ValueError("Every training query must have at least two clauses")
    if torch.any(target_ids[valid] >= len(teachers)):
        raise ValueError("Teacher index out of range")
    flat = embeddings[valid]
    positive_ids = target_ids[valid]
    logits = (flat @ teachers.T) / temperature
    owner = torch.arange(len(embeddings), device=embeddings.device)[:, None].expand_as(target_ids)[valid]
    # Sibling clauses are valid query interpretations, not mandatory negatives.
    siblings = target_ids[owner]
    blocked = (siblings >= 0) & (siblings != positive_ids[:, None])
    r = torch.arange(len(flat), device=embeddings.device)[:, None].expand_as(siblings)
    logits[r[blocked], siblings[blocked]] = -torch.inf
    rows = torch.arange(len(flat), device=embeddings.device)
    loss = torch.logsumexp(logits, -1) - logits[rows, positive_ids]
    # Equal query weight regardless of arity, matching the retrieval objective.
    per_query = torch.zeros(len(embeddings), device=embeddings.device).scatter_add(0, owner, loss)
    mean = (per_query / valid.sum(-1)).mean()
    cosine = (flat.detach() * teachers[positive_ids]).sum(-1)
    slots = torch.arange(3, device=embeddings.device)[None].expand_as(target_ids)[valid]
    extra = {"clause_cosine": cosine.mean().item(),
             "clause_bank_accuracy": (logits.detach().argmax(-1) == positive_ids).float().mean().item()}
    for i in range(3):
        if (slots == i).any():
            extra[f"clause_cosine_slot_{i+1}"] = cosine[slots == i].mean().item()
    return mean, extra
