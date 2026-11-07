import numpy as np
import pytest

from lexical_passages import PassageTokenRows, max_page_scores


def test_passages_cover_each_page_without_crossing_boundaries():
    tokens = np.arange(13, dtype=np.int32)
    offsets = np.array([0, 0, 2, 6, 13])
    rows = PassageTokenRows(tokens, offsets, size=4, stride=3)
    assert list(rows) == [[], [0, 1], [2, 3, 4, 5], [6, 7, 8, 9], [9, 10, 11, 12]]
    assert rows.documents.tolist() == [0, 1, 2, 3, 3]
    for doc in range(4):
        represented = {t for i, row in enumerate(rows) if rows.documents[i] == doc for t in row}
        assert represented == set(tokens[offsets[doc]:offsets[doc + 1]])
    assert all(type(t) is int for row in rows for t in row)
    with pytest.raises(ValueError, match="layout"):
        PassageTokenRows(tokens, offsets, size=2, stride=3)


def test_max_aggregation_deduplicates_pages_without_summing():
    scores = max_page_scores(np.array([1., 4., 2., 3., 3.]), np.array([0, 0, 1, 2, 2]), 3)
    assert scores.tolist() == [4., 2., 3.]
    assert scores.argsort()[::-1].tolist() == [0, 2, 1]
    with pytest.raises(ValueError, match="Every page"):
        max_page_scores(np.array([1.]), np.array([0]), 2)
