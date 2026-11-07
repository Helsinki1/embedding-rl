import pytest
import torch

from reranking_control import reranked_ndcg


def test_reranking_cannot_recover_outside_candidates_and_preserves_ties():
    grades = torch.tensor([[0., 1., 1., 1.], [0., 0., 1., 1.]])
    candidates = torch.tensor([[0, 1], [0, 1]])
    tied = reranked_ndcg(candidates, torch.zeros(2, 2), grades, k=2)
    better = reranked_ndcg(candidates, torch.tensor([[0., 1.], [0., 1.]]), grades, k=2)
    ideal = 1 + 1 / torch.log2(torch.tensor(3.))
    assert tied[0] == pytest.approx(float(1 / torch.log2(torch.tensor(3.)) / ideal))
    assert better[0] == pytest.approx(float(1 / ideal))
    assert better[1] == tied[1] == 0
    assert (better <= 1).all()


def test_reranking_rejects_duplicates_and_handles_no_labels():
    with pytest.raises(ValueError, match="Repeated"):
        reranked_ndcg(torch.tensor([[0, 0]]), torch.zeros(1, 2), torch.zeros(1, 3), k=2)
    assert reranked_ndcg(torch.tensor([[0, 1]]), torch.zeros(1, 2), torch.zeros(1, 3), k=2).item() == 0
