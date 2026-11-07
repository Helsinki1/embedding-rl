"""Prevent gold-component leakage, dropped queries, and incorrect union scoring."""
import asyncio
from types import SimpleNamespace

import pytest

from .decompose import request_decomposition
from .union import components_for, flatten_queries, union_predictions


def test_llm_receives_only_query_text():
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(status="completed", output_text='{"keep":true,"components":["a","b"],"reason":"OR"}',
                               model="test", id="test", usage=SimpleNamespace(model_dump=lambda: {}))
    asyncio.run(request_decomposition("a OR b", SimpleNamespace(responses=SimpleNamespace(create=create))))
    assert requests[0]["input"] == "a OR b"
    assert requests[0]["store"] is False


def test_union_deduplicates_and_does_not_truncate_total_results():
    rows = [{"id": "x", "query": "a OR b", "components": ["SECRET GOLD"]}]
    cache = {"a OR b": {"query": "a OR b", "keep": True, "components": ["a", "b"]}}
    flat, offsets, fallbacks = flatten_queries(rows, cache)
    assert flat == [{"query": "a"}, {"query": "b"}] and not fallbacks
    result = union_predictions(rows, offsets, [[0, 1, 2], [1, 2, 3]], ["A", "B", "C", "D"], 2)
    assert result[0]["docs"] == ["A", "B", "C"]


@pytest.mark.parametrize("record", [
    {"keep": False, "components": []},
    {"keep": True, "components": ["a", " A "]},
    {"keep": True, "components": ["a", ""]},
    {"keep": True, "components": ["a"]},
])
def test_invalid_or_rejected_decomposition_preserves_query(record):
    assert components_for("full query", {"query": "full query", **record}) == (["full query"], True)


def test_cache_query_mismatch_is_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        components_for("one", {"query": "two", "keep": False})
