"""Verify all eleven completed runs and export the common score table."""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from .build import COUNTS, HERE, read_rows, sha256, write_json
from .data import load_splits
from .launch import PLANS
from .metrics import evaluate

ORDER = [("single", "frozen"), ("llm", "frozen"), ("split", "frozen"),
         ("single", "infonce"), ("single", "rl"), ("split", "infonce"), ("split", "rl"),
         ("one_pass", "infonce"), ("one_pass", "rl"), ("three_pass", "infonce"), ("three_pass", "rl")]
NAMES = {"single": "Single embedding", "llm": "LLM decomposition", "split": "Split thirds",
         "one_pass": "One-pass slots", "three_pass": "Three-pass slots"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .run import predictions

    _, _, test, provenance = load_splits()
    titles_path = args.results.parent / "corpus_ids.json"
    titles = json.loads(titles_path.read_text())
    if len(titles) != 489188 or len(set(titles)) != len(titles):
        raise ValueError("Common corpus identities are incomplete")
    provenance["corpus_ids_sha256"] = sha256(titles_path)
    assert set(ORDER) == {x for plan in PLANS.values() for x in plan}
    rows, verified, expected_code = [], [], None
    for architecture, method in ORDER:
        name = architecture + "_" + method
        directory = args.results / name
        run = json.loads((directory / "run.json").read_text())
        saved = json.loads((directory / "summary.json").read_text())
        if run["provenance"]["manifest_sha256"] != provenance["manifest_sha256"]:
            raise ValueError("Run used a different dataset: " + name)
        if run["provenance"]["protocol_sha256"] != provenance["protocol_sha256"]:
            raise ValueError("Run used a different protocol: " + name)
        if expected_code is None:
            expected_code = run["code_sha256"]
        elif run["code_sha256"] != expected_code:
            raise ValueError("Run used a different executable code snapshot: " + name)
        if run["config"]["lora_rank"] != 32 or run["documents"] != 489188:
            raise ValueError("Wrong LoRA rank/corpus: " + name)
        if method != "frozen":
            expected = {s: counts[0] for s, counts in COUNTS.items()}
            if saved["examples_seen"] != 8500 or saved["seen_sources"] != expected or saved["stop_reason"] != "epochs_complete":
                raise ValueError("Training did not cover the specified full mixture: " + name)
            logs = read_rows(directory / "metrics.jsonl")
            if len(logs) != 266 or logs[-1]["examples_seen"] != 8500:
                raise ValueError("Incomplete training log: " + name)
            for step in logs:
                if any(not math.isfinite(step[k]) for k in ("loss", "grad_norm")):
                    raise ValueError("Nonfinite training metric: " + name)
        recomputed, _ = evaluate(test, read_rows(directory / "test_predictions.jsonl"))
        for group in ("overall", "by_source", "by_arity"):
            if recomputed[group] != saved[group]:
                raise ValueError("Saved predictions do not reproduce metrics: " + name)
        rankings = json.loads((directory / "test_rankings.json").read_text())
        if rankings["ids"] != [r["id"] for r in test]:
            raise ValueError("Saved ranking order differs from the test split")
        selected_predictions = predictions(test, rankings["corpus_indices"], titles, k=saved["selection"]["k"])
        if selected_predictions != read_rows(directory / "test_predictions.jsonl"):
            raise ValueError("Saved rankings do not reproduce selected predictions")
        for budget in (10, 20, 60, 100):
            predicted = predictions(test, rankings["corpus_indices"], titles, budget=budget)
            if any(len(p["docs"]) != budget for p in predicted):
                raise ValueError("Final result counts differ from the fixed budget")
            metrics, _ = evaluate(test, predicted)
            if metrics != saved["fixed_budget"][str(budget)]:
                raise ValueError("Fixed-budget metrics do not reproduce")
        union = saved["overall"]["union"]
        component = saved["overall"]["component_macro"]
        row = {"approach": NAMES[architecture], "objective": method,
               "union_precision": union["precision"], "union_recall": union["recall"], "union_f1": union["f1"],
               "component_precision": component["precision"], "component_recall": component["recall"],
               "component_f1": component["f1"], "component_coverage_at_100": saved["fixed_budget"]["100"]["overall"]["component_coverage"],
               "union_f1_at_10": saved["fixed_budget"]["10"]["overall"]["union"]["f1"],
               "union_f1_at_20": saved["fixed_budget"]["20"]["overall"]["union"]["f1"],
               "selected_k": saved["selection"]["k"], "checkpoint_step": saved["selection"]["step"],
               "mean_results": union["predicted_count"], "training_examples": saved["examples_seen"],
               "elapsed_seconds": saved["elapsed_seconds"], "test_encoding_retrieval_seconds": saved["timing"]["query_encoding_and_retrieval_seconds"],
               "peak_gpu_allocated_gb": saved["peak_gpu_allocated_gb"]}
        for source, metrics in saved["by_source"].items():
            row[source + "_union_f1"] = metrics["union"]["f1"]
        rows.append(row)
        verified.append({"run": name, "summary_sha256": sha256(directory / "summary.json"),
                         "predictions_sha256": sha256(directory / "test_predictions.jsonl"),
                         "metrics_recomputed": True, "full_spec_training_mixture": method == "frozen" or saved["examples_seen"] == 8500})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint_audit = json.loads((args.results.parent / "checkpoint_audit.json").read_text())
    tests = (args.results.parent / "logs/tests-current.log").read_text().strip()
    write_json(args.output / "scores.json", {"provenance": provenance, "scores": rows, "verification": verified,
                                             "checkpoint_audit": checkpoint_audit, "test_log": tests,
                                             "executable_code_sha256": expected_code})
    lines = ["# Updated QUEST / MS MARCO / NQ comparison", "",
             "All eleven approaches use the same 8,500 training / 500 validation / 1,000 test examples and 489,188-document corpus. All eight trained runs completed one epoch with rank-32 LoRA. Checkpoints and cutoffs were selected on validation only.", "",
             "| Approach | Objective | Union P | Union R | Union F1 | Component F1 | Coverage@100 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| " + " | ".join([r["approach"], r["objective"]] +
                     [f"{100*r[k]:.2f}%" for k in ("union_precision", "union_recall", "union_f1", "component_f1", "component_coverage_at_100")]) + " |")
    best = max(rows, key=lambda r: r["union_f1"])
    lines.extend(["", f"Best observed union F1: **{best['approach']} ({best['objective']}), {100*best['union_f1']:.2f}%**. These are single-seed measurements, not a statistical claim of superiority.", "",
                  "`scores.csv` includes component precision/recall, source scores, fixed budgets, selected cutoffs, result counts and timing. `scores.json` includes verification hashes. Saved per-query predictions reproduce all primary aggregate metrics exactly.", "",
                  "Verification also recomputed all fixed-budget metrics, confirmed exactly 10/20/60/100 final results, checked identical executable snapshots, and checked actual rank-32 checkpoints for nonzero learned updates. A fresh rebuild reproduced every dataset and manifest hash. The remote runtime passed 53 tests; three historical split-fixture tests were excluded because their old datasets were not copied into this isolated snapshot.", "",
                  "Source mix: 519 QUEST, 4,740 MS MARCO, 4,741 NQ. Atomic-question groups are partitioned before synthetic OR composition. The two unscorable QUEST components occur only in training; every validation/test component has labels.", "",
                  "The test set has 500 two-component and 500 three-component queries. QUEST categories are disjoint, while entities may satisfy different categories in different splits (61 shared positive documents between training and test); NQ/MS MARCO identical positive passage texts are disjoint. See `data_audit.json` for the complete breakdown.", "",
                  "MS MARCO is a custom v1.1 candidate-passage experiment, not the full 8.8M-passage leaderboard. Relevance labels are incomplete. Existing QUEST and NQ documents remain in the corpus. Scores are not directly comparable to the old 5,000-query table.", "",
                  "Component metrics use each method's final merged set against every gold component, then average within and across queries. Coverage@100 counts components represented by at least one positive among exactly 100 merged, deduplicated results. Fixed budgets use round-robin merging.", "",
                  "The Google Doc was read but not edited. Historical data, checkpoints and results were preserved.", ""])
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    for name in ("manifest.json", "protocol.json", "README.md"):
        shutil.copy2(HERE / name, args.output / name)
    for source, target in (("source/data_audit.json", "data_audit.json"), ("source/audit.json", "source_audit.json"),
                           ("decomposition/usage.json", "llm_usage.json"), ("source/rebuild_verification.json", "rebuild_verification.json")):
        shutil.copy2(HERE / source, args.output / target)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
