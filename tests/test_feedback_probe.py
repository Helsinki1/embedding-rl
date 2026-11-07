import pytest
import torch
import torch.nn.functional as F

from feedback_probe import refine_queries


def test_feedback_identity_and_unit_norm():
    torch.manual_seed(7)
    queries = F.normalize(torch.randn(3, 8), dim=-1)
    docs = F.normalize(torch.randn(10, 8), dim=-1)
    assert torch.allclose(refine_queries(queries, docs, alpha=0), queries)
    result = refine_queries(queries, docs, rounds=2)
    assert torch.allclose(result.norm(dim=-1), torch.ones(3))
    assert torch.isfinite(result).all()


def test_feedback_uses_retrieved_vectors_with_fresh_search_each_round():
    query = F.normalize(torch.tensor([[1., .4]]), dim=-1)
    docs = torch.eye(2)
    result = refine_queries(query, docs, k=1, alpha=1)
    assert torch.equal(result, docs[:1])
    assert torch.equal(refine_queries(query, docs, k=1, alpha=1, rounds=2), result)
    # Reordering a non-tied corpus must not change the refined query.
    assert torch.equal(refine_queries(query, docs.flip(0), k=1, alpha=1), result)


def test_feedback_rejects_invalid_protocol():
    vectors = torch.eye(2)
    for settings in ({"k": 3}, {"temperature": 0}, {"alpha": 2}, {"rounds": 0}):
        with pytest.raises(ValueError, match="Invalid"):
            refine_queries(vectors, vectors, **settings)
