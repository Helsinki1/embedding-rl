from dataclasses import replace
import json

import pytest
import torch

from conftest import TinyEncoder
from embedding_rl.config import Config, digest, prepared_dir, write_json
from embedding_rl.data import load_prepared
from embedding_rl.followup import conditions, sampling_probe, select_variant, summarize
from embedding_rl.objectives import gaussian_kl
from embedding_rl.train import train


def browsecomp_fixture(prepared):
    root, old_config, _ = prepared
    _, _, bundle, tensors = load_prepared(root, old_config)
    config = replace(old_config, dataset="browsecomp-plus", training_seed=42,
                     monitor_train_queries=2, eval_every=10, eval_steps=(1,))
    bundle["gold_qrels"] = bundle["qrels"]
    path = prepared_dir(root, config)
    manifest = {"schema": 2, "preparation": config.preparation_settings(),
                "revisions": bundle["revisions"], "data_fingerprint": digest(bundle)}
    write_json(path / "data.json", bundle)
    write_json(path / "manifest.json", manifest)
    torch.save(tensors, path / "vectors.pt")
    return root, config, bundle


def test_training_seed_and_monitoring_do_not_change_data_split_or_cache(tmp_path):
    config = Config()
    changed = replace(config, training_seed=44, monitor_train_queries=64, eval_steps=(10, 100))
    assert prepared_dir(tmp_path, changed) == prepared_dir(tmp_path, config)
    assert prepared_dir(tmp_path, replace(config, seed=43)) != prepared_dir(tmp_path, config)


def test_radius_grid_preserves_effective_kl_displacement_penalty():
    mean, reference = torch.tensor([[0.8, 0.6]]), torch.tensor([[1., 0.]])
    penalties = [cfg.kl_weight * gaussian_kl(mean, reference, cfg.exploration_radius / 2**0.5)
                 for _, cfg in conditions(Config())]
    assert all(float(value) == pytest.approx(float(penalties[0])) for value in penalties)


def test_sampling_probe_uses_training_only_and_nested_draws(prepared):
    root, config, bundle = browsecomp_fixture(prepared)
    result = sampling_probe(root, config, root / "probe", "test")
    assert set(result["query_ids"]) == set(bundle["splits"]["train"])
    rows = result["per_query"]
    for small, large in [("control", "more_samples"), ("wider", "both")]:
        for a, b in zip(rows[small], rows[large]):
            assert a["query_id"] == b["query_id"]
            assert b["best_reward"] >= a["best_reward"]
            assert b["has_signal"] >= a["has_signal"]


def test_fixed_training_monitor_and_explicit_evaluation_steps(prepared):
    root, config, bundle = browsecomp_fixture(prepared)
    ids = []
    for seed in (42, 43):
        result = train(root, replace(config, training_seed=seed), "rl", encoder_factory=TinyEncoder)
        path = root / "runs" / result["run_id"]
        records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        assert [r["step"] for r in records if "validation/ndcg@2" in r] == [0, 1, 3]
        assert [r["step"] for r in records if "fixed_train/ndcg@2" in r] == [0, 1, 3]
        probe = json.loads((path / "training_probe.json").read_text())
        ids.append(probe["query_ids"])
        assert set(ids[-1]) == set(bundle["splits"]["train"])
        validation = json.loads((path / "validation.json").read_text())
        assert validation["final"]["metrics"]["ndcg@2"] == result["final_validation_ndcg"]
        assert not (path / "test.json").exists()
    assert ids[0] == ids[1]


def test_selection_and_summary_use_final_scores_and_match_seed_controls():
    def record(label, seed, score, peak):
        return {"label": label, "settings": {"training_seed": seed},
                "summary": {"final_validation_ndcg": score, "best_validation_ndcg": peak,
                            "initial_validation_ndcg": 0.1},
                "mean_zero_advantage_fraction": 0.5}
    initial = [record("control", 42, .11, .2), record("more_samples", 42, .12, .5),
               record("both", 42, .13, .13)]
    assert select_variant(initial) == "both"
    result = summarize(initial + [record("control", 43, .12, .2),
                                  record("both", 43, .11, .2)], "both")
    assert result["mean_paired_delta"] == pytest.approx(.005)
    assert result["test_evaluated"] is False
