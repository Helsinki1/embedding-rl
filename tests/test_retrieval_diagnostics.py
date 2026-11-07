import pytest
import torch

from retrieval_diagnostics import candidate_metrics


def test_reranking_ceiling_cannot_recover_evidence_outside_candidates():
    scores = torch.tensor([[4., 3., 2., 1.], [4., 3., 2., 1.]])
    grades = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 1.]])
    result = candidate_metrics(scores, grades, pool_size=2, k=1)
    assert result["ndcg10"].tolist() == [0., 0.]
    assert result["oracle_rerank25_ndcg10"].tolist() == [1., 0.]
    assert result["recall25"].tolist() == [1., 0.]
    assert result["hit25"].tolist() == [1., 0.]


def test_candidate_ceiling_preserves_global_ideal_and_bounds_actual_ranking():
    scores = torch.tensor([[4., 3., 2., 1.], [4., 3., 2., 1.]])
    grades = torch.tensor([[0., 1., 1., 1.], [0., 0., 0., 0.]])
    result = candidate_metrics(scores, grades, pool_size=2, k=2)
    assert result["oracle_rerank25_ndcg10"][0] < 1
    assert torch.all(result["oracle_rerank25_ndcg10"] >= result["ndcg10"])
    assert result["oracle_rerank25_ndcg10"][1] == 0
    with pytest.raises(ValueError, match="depths"):
        candidate_metrics(scores, grades, pool_size=1, k=2)
