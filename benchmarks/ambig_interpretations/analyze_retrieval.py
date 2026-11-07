"""Verify downloaded retrieval results and summarize paired changes from baseline."""
import argparse
import json
from pathlib import Path
import random
import statistics

from benchmarks.quest_or_spec_v2.build import read_rows, sha256, write_json
from benchmarks.quest_or_spec_v2.data import load_splits
from benchmarks.quest_or_spec_v2.metrics import evaluate


def paired_interval(differences, repeats=2000):
    rng = random.Random(42)
    n = len(differences)
    samples = sorted(statistics.mean(rng.choices(differences, k=n)) for _ in range(repeats))
    return {"mean_delta": statistics.mean(differences),
            "paired_bootstrap_95_percent_interval": [samples[int(.025*repeats)], samples[int(.975*repeats)]],
            "resamples": repeats, "seed": 42}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    _, _, rows, _ = load_splits()
    expected_ids = [r["id"] for r in rows]
    details, summaries = {}, {}
    for architecture in ("single_frozen", "mtp", "autoregressive"):
        root = args.run / architecture
        summary = json.loads((root / "summary.json").read_text())
        predictions = read_rows(root / "test_predictions.jsonl")
        assert [r["id"] for r in predictions] == expected_ids
        recomputed, per_query = evaluate(rows, predictions)
        for key in ("overall", "by_source", "by_arity"):
            assert recomputed[key] == summary[key], (architecture, key)
        rankings = json.loads((root / "test_rankings.json").read_text())
        assert rankings["ids"] == expected_ids and len(rankings["corpus_indices"]) == 1000
        heads = 1 if architecture == "single_frozen" else 3
        for query in rankings["corpus_indices"]:
            assert len(query) == heads
            assert all(len(h) == len(set(h)) == 100 and min(h) >= 0 and max(h) < 489188 for h in query)
        if architecture != "single_frozen":
            checkpoint = args.run.parent / "ambig-20261106-v1" / architecture / "best.pt"
            assert sha256(checkpoint) == summary["selection"]["checkpoint_sha256"]
        details[architecture], summaries[architecture] = per_query, summary
    comparisons = {a: paired_interval([new["f1"]-old["f1"]
                   for new, old in zip(details[a], details["single_frozen"])])
                   for a in ("mtp", "autoregressive")}
    report = {"verified": True, "queries_per_model": 1000, "documents": 489188,
              "paired_union_f1_delta": comparisons,
              "metrics": {a: {"union": s["overall"]["union"],
                              "component_macro": s["overall"]["component_macro"],
                              "component_coverage_at_100": s["fixed_budget"]["100"]["overall"]["component_coverage"],
                              "union_recall_at_20": s["fixed_budget"]["20"]["overall"]["union"]["recall"],
                              "source_union_f1": {name: v["union"]["f1"] for name, v in s["by_source"].items()},
                              "nonoverlap_union_f1": s["no_exact_training_or_checkpoint_validation_overlap"]["overall"]["union"]["f1"],
                              "cutoff_per_vector": s["selection"]["retrieval"]["k"],
                              "mean_pairwise_slot_cosine": s["timing"]["mean_pairwise_slot_cosine"]}
                          for a, s in summaries.items()}}
    write_json(args.run / "analysis.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
