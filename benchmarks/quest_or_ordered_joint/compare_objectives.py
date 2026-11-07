"""Verify the matched three-epoch joint versus document-only comparison."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from benchmarks.quest_or_spec_v2.build import read_rows, write_json
from benchmarks.quest_or_spec_v2.data import load_splits
from benchmarks.quest_or_spec_v2.metrics import evaluate
from .collect import ordered_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    train, validation, test, provenance = load_splits()
    gold = json.loads((root / "gold_corpus_indices.json").read_text())
    models, table, epoch_rows = {}, [], []
    rng = np.random.default_rng(42)
    for variant in ("one_pass", "three_pass"):
        records, details = {}, {}
        for method, group in (("joint", "joint"), ("document_only", "results")):
            directory = root / group / variant
            run = json.loads((directory / "run.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            metrics = read_rows(directory / "metrics.jsonl")
            assert run["config"]["epochs"] == 3 and run["config"]["steps"] == 798
            assert run["config"]["batch_size"] == run["config"]["micro_batch_size"] == 32
            assert run["config"]["clause_weight"] == (1 if method == "joint" else 0)
            assert run["documents"] == 489188
            assert summary["dataset_manifest_sha256"] == provenance["manifest_sha256"]
            assert summary["completed_steps"] == len(metrics) == 798
            assert summary["examples_seen"] == 3*len(train) == 25500
            assert summary["seen_sources"] == {s: 3*n for s, n in provenance["audit"]["train"]["sources"].items()}
            assert summary["stop_reason"] == "epochs_complete" and summary["elapsed_seconds"] < 1800
            assert all(np.isfinite(x["loss"]) and abs(x["loss"]-x["document_loss"]-run["config"]["clause_weight"]*x["clause_loss"]) < 1e-4 for x in metrics)
            if method == "document_only":
                assert run["teacher_manifest"] is None and run["teacher_encoder"] == "unused"
                assert run["method"] == summary["method"] == "document_infonce"
                assert all(x["clause_loss"] == 0 and x["loss"] == x["document_loss"] for x in metrics)
            actual, per_query = evaluate(test, read_rows(directory / "test_predictions.jsonl"))
            for field in ("overall", "by_source", "by_arity"):
                assert actual[field] == summary[field], (variant, method, field)
            details[method] = {r["id"]: r["f1"] for r in per_query}
            curve = []
            for file in sorted(directory.glob("validation-*.json")):
                step = int(file.stem.split("-")[1])
                curve.append({"step": step, "epochs": step/266, **json.loads(file.read_text())["selected"]})
            assert [x["step"] for x in curve] == [0, 89, 178, 266, 355, 444, 532, 621, 710, 798]
            selected = next(x for x in curve if x["step"] == summary["selection"]["step"])
            assert selected["f1"] == max(x["f1"] for x in curve if x["step"] > 0)
            assert selected["k"] == summary["selection"]["k"]
            records[method] = {"run": run, "summary": summary, "validation_curve": curve,
                               "ordered_diagnostics": ordered_diagnostics(directory, gold)}
            union = summary["overall"]["union"]
            table.append({"architecture": variant, "objective": method,
                          "precision": union["precision"], "recall": union["recall"], "f1": union["f1"],
                          "component_f1": summary["overall"]["component_macro"]["f1"],
                          "f1_at_20": summary["fixed_budget"]["20"]["overall"]["union"]["f1"],
                          "selected_update": summary["selection"]["step"],
                          "seconds": summary["elapsed_seconds"]})
            for epoch in (1, 2, 3):
                endpoint = next(x for x in curve if x["step"] == epoch*266)
                epoch_rows.append({"architecture": variant, "objective": method,
                                   "epoch": epoch, "validation_f1": endpoint["f1"]})
        joint, only = records["joint"]["run"], records["document_only"]["run"]
        for field in ("seed", "epochs", "batch_size", "micro_batch_size", "eval_batch_size", "eval_every",
                      "learning_rate", "lora_rank", "lora_alpha", "temperature", "slot_init_noise", "feedback"):
            assert joint["config"][field] == only["config"][field], (variant, field)
        assert records["joint"]["summary"]["selection"]["initial_validation"] == records["document_only"]["summary"]["selection"]["initial_validation"]
        for source in ("quest_or_multivector/model.py", "quest_or_multivector/objectives.py", "quest_or_training/train.py"):
            assert joint["code_sha256"][source] == only["code_sha256"][source]
        delta = np.array([details["document_only"][r["id"]]-details["joint"][r["id"]] for r in test])
        samples = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)]
        records["comparison"] = {"document_only_minus_joint_f1_points": 100*delta.mean(),
                                  "paired_query_bootstrap_95pct_points": (100*np.quantile(samples, [.025, .975])).tolist(),
                                  "replicates": 5000}
        models[variant] = records
    result = {"models": models, "rows": table, "epoch_rows": epoch_rows, "provenance": provenance,
              "all_primary_metrics_independently_recomputed": True,
              "limitations": ["One seed; intervals measure query-sampling uncertainty, not training-seed variation.",
                              "The frozen test set was observed in preceding experiments. No test-based tuning was used.",
                              "Checkpoints and cutoffs are selected on validation only, so the selected checkpoint may precede epoch 3.",
                              "Joint clause supervision excludes sibling clauses from negatives; this comparison applies to that specific objective."]}
    write_json(root / "comparison.json", result)
    for filename, rows in (("comparison.csv", table), ("validation-by-epoch.csv", epoch_rows)):
        with (root / filename).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    labels = {"joint": "Joint InfoNCE", "document_only": "Document InfoNCE only"}
    for ax, variant in zip(axes, models):
        for method, color in (("joint", "#8795a5"), ("document_only", "#125db0")):
            curve = models[variant][method]["validation_curve"][1:]
            ax.plot([x["epochs"] for x in curve], [100*x["f1"] for x in curve], "o-", color=color, label=labels[method])
            selected = models[variant][method]["summary"]["selection"]
            ax.scatter([selected["step"]/266], [100*selected["validation"]["f1"]], marker="*", s=130, color=color, zorder=5)
        ax.set(title=variant.replace("_", " ").capitalize(), xlabel="Epochs", ylabel="Validation union F1 (%)", xticks=[0, 1, 2, 3], xlim=(0, 3.1))
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.savefig(root / "validation-curves.svg"); fig.savefig(root / "validation-curves.png", dpi=160); plt.close(fig)
    rows_html = "".join(f'<tr><td>{r["architecture"].replace("_", " ")}</td><td>{labels[r["objective"]]}</td>'
                        f'<td>{100*r["precision"]:.2f}</td><td>{100*r["recall"]:.2f}</td><td><b>{100*r["f1"]:.2f}</b></td>'
                        f'<td>{r["selected_update"]}</td><td>{r["seconds"]/60:.1f}</td></tr>' for r in table)
    differences_html = "".join(f'<li>{variant.replace("_", " ")}: document-only minus joint = '
                               f'{model["comparison"]["document_only_minus_joint_f1_points"]:+.2f} F1 points; '
                               f'paired-query 95% interval [{model["comparison"]["paired_query_bootstrap_95pct_points"][0]:+.2f}, '
                               f'{model["comparison"]["paired_query_bootstrap_95pct_points"][1]:+.2f}].</li>' for variant, model in models.items())
    (root / "report.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>Three-epoch InfoNCE comparison</title>
<style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:44px auto;padding:0 24px;color:#172432}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;text-align:right;border-bottom:1px solid #dce3e9}}td:first-child,th:first-child{{text-align:left}}th{{background:#eef3f7}}img{{max-width:100%}}</style>
<h1>Three epochs with and without decomposition supervision</h1>
<p>Both architectures were trained from fresh Qwen3-Embedding-8B weights with rank-32 LoRA for three epochs. Each run completed 798 updates and 25,500 query exposures. Dataset, corpus, seed, learning rate, effective batch size 32, microbatch 32, architecture and validation schedule are matched. The document-only runs use only collective full-corpus document InfoNCE: no clause targets are loaded and no clause loss is computed.</p>
<table><thead><tr><th>Architecture</th><th>Objective</th><th>Precision %</th><th>Recall %</th><th>Test F1 %</th><th>Selected update</th><th>Minutes</th></tr></thead><tbody>{rows_html}</tbody></table>
<ul>{differences_html}</ul>
<h2>Validation progress</h2><img src="validation-curves.svg" alt="Validation F1 for matched joint and document-only three-epoch runs">
<p>Stars mark validation-selected checkpoints. All runs nevertheless finished three epochs; test was evaluated only after training and selection.</p>
<h2>Checks and limitations</h2><p>Test scores were independently recomputed from saved predictions. The collector checks exact data identity, all 798 updates, source counts, finite losses, matched settings and initialization scores, and zero clause loss in every document-only update. Every run completed within the 30-minute cap.</p>
<ul>{''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])}</ul>
<p><a href="comparison.csv">Results CSV</a> · <a href="validation-by-epoch.csv">Epoch CSV</a> · <a href="comparison.json">Full results and provenance</a></p>''')
    print(json.dumps({variant: {"joint_f1": model["joint"]["summary"]["overall"]["union"]["f1"],
                               "document_only_f1": model["document_only"]["summary"]["overall"]["union"]["f1"],
                               **model["comparison"]} for variant, model in models.items()}, indent=2))


if __name__ == "__main__":
    main()
