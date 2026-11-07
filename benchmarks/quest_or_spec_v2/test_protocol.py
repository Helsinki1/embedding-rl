import pytest

from .data import load_splits
from .metrics import evaluate, set_metrics


def test_merged_predictions_score_each_gold_component():
    row = {"id": "q", "source": "nq", "docs": ["a", "b", "c"], "components": ["A", "B"],
           "component_docs": [["a", "b"], ["b", "c"]]}
    summary, details = evaluate([row], [{"id": "q", "docs": ["a", "c", "x", "a"]}])
    assert summary["overall"]["union"]["f1"] == pytest.approx(2 / 3)
    assert summary["overall"]["component_macro"]["precision"] == pytest.approx(1 / 3)
    assert summary["overall"]["component_macro"]["recall"] == .5
    assert summary["overall"]["component_macro"]["f1"] == .4
    assert details[0]["component_coverage"] == 1


def test_empty_component_is_not_invented_negative_label():
    row = {"id": "q", "source": "quest", "docs": ["a"], "components": ["A", "B"],
           "component_docs": [["a"], []]}
    summary, _ = evaluate([row], [{"id": "q", "docs": ["a"]}])
    assert summary["overall"]["unscorable_components"] == 1
    assert summary["overall"]["component_macro"]["f1"] == 1
    assert set_metrics([], ["a"])["recall"] is None


def test_incomplete_prediction_set_rejected():
    with pytest.raises(ValueError):
        evaluate([{"id": "q"}], [])


def test_frozen_mixture_and_no_component_leakage():
    train, val, test, provenance = load_splits()
    assert list(map(len, (train, val, test))) == [8500, 500, 1000]
    assert provenance["audit"]["test"]["sources"] == {"quest": 52, "msmarco": 474, "nq": 474}


def test_budget_deduplicates_before_counting_and_uses_all_heads():
    from .run import round_robin
    assert round_robin([[1, 2, 3], [1, 4, 5], [6, 7, 8]], 5) == [1, 6, 2, 4, 7]


def test_cutoff_selection_does_not_use_component_labels():
    from .run import select_cutoff
    rows = [{"id": "q", "docs": ["a", "b"]}]
    best, _ = select_cutoff(rows, [[[0, 2], [1, 2]]], ["a", "b", "x"], 2)
    assert best["k"] == 1
    assert best["f1"] == 1
