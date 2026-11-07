"""Verify and report the two three-epoch autoregressive follow-up runs."""
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
    args = parser.parse_args()
    root = args.root
    train, validation, test, provenance = load_splits()
    gold_indices = json.loads((root / "gold_corpus_indices.json").read_text())
    output, curves = {}, {}
    table = []
    rng = np.random.default_rng(42)
    for variant in ("one_pass", "three_pass"):
        directory = root / "results" / variant
        run = json.loads((directory / "run.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        logs = read_rows(directory / "metrics.jsonl")
        assert run["config"]["epochs"] == 3 and run["config"]["steps"] == 798
        assert run["config"]["micro_batch_size"] == run["config"]["batch_size"] == 32
        assert run["documents"] == 489188
        assert summary["dataset_manifest_sha256"] == provenance["manifest_sha256"]
        assert summary["completed_steps"] == len(logs) == 798
        assert summary["examples_seen"] == 3*len(train) == 25500
        assert summary["seen_sources"] == {s: 3*n for s, n in provenance["audit"]["train"]["sources"].items()}
        assert summary["stop_reason"] == "epochs_complete" and summary["elapsed_seconds"] < 1800
        assert all(np.isfinite(x["loss"]) and abs(x["loss"]-x["document_loss"]-x["clause_loss"]) < 1e-4 for x in logs)
        recomputed, new_details = evaluate(test, read_rows(directory / "test_predictions.jsonl"))
        for field in ("overall", "by_source", "by_arity"):
            assert summary[field] == recomputed[field], (variant, field)
        old_dir = root / "references" / variant
        old = json.loads((old_dir / "summary.json").read_text())
        old_run = json.loads((old_dir / "run.json").read_text())
        for field in ("seed", "batch_size", "learning_rate", "lora_rank", "lora_alpha", "temperature",
                      "clause_weight", "clause_temperature", "slot_init_noise", "feedback", "assignment"):
            assert run["config"][field] == old_run["config"][field], (variant, field)
        assert run["teacher_manifest"]["teachers_sha256"] == old_run["teacher_manifest"]["teachers_sha256"]
        assert summary["selection"]["initial_validation"] == old["selection"]["initial_validation"]
        old_scores, old_details = evaluate(test, read_rows(old_dir / "test_predictions.jsonl"))
        assert old_scores["overall"] == old["overall"]
        assert old["dataset_manifest_sha256"] == provenance["manifest_sha256"]
        by_id = {x["id"]: x["f1"] for x in old_details}
        differences = np.array([x["f1"]-by_id[x["id"]] for x in new_details])
        samples = [rng.choice(differences, len(differences), replace=True).mean() for _ in range(5000)]
        validations = []
        for file in sorted(directory.glob("validation-*.json")):
            step = int(file.stem.split("-")[1])
            selected = json.loads(file.read_text())["selected"]
            validations.append({"step": step, "epochs": step/266, **selected})
        assert [x["step"] for x in validations] == [0, 89, 178, 266, 355, 444, 532, 621, 710, 798]
        curves[variant] = validations
        epochs = [{"epoch": epoch, **next(x for x in validations if x["step"] == epoch*266)}
                  for epoch in (1, 2, 3)]
        selected = next(x for x in validations if x["step"] == summary["selection"]["step"])
        assert selected["f1"] == max(x["f1"] for x in validations if x["step"] > 0)
        assert selected["k"] == summary["selection"]["k"]
        output[variant] = {
            "primary_metrics_recomputed": True, "summary": summary,
            "prior_one_epoch_summary": old, "validation_curve": validations,
            "epoch_end_validation": epochs, "delta_test_f1_points": 100*differences.mean(),
            "paired_query_bootstrap_95pct_points": (100*np.quantile(samples, [.025, .975])).tolist(),
            "ordered_retrieval_diagnostics": ordered_diagnostics(directory, gold_indices),
            "bootstrap_note": "Query-sampling uncertainty for these checkpoints; not training-seed uncertainty."}
        for label, metrics in (("Previous 1 epoch", old), ("New 3 epochs", summary)):
            union = metrics["overall"]["union"]
            table.append({"architecture": variant, "run": label, "precision": union["precision"],
                          "recall": union["recall"], "f1": union["f1"],
                          "component_f1": metrics["overall"]["component_macro"]["f1"],
                          "f1_at_20": metrics["fixed_budget"]["20"]["overall"]["union"]["f1"],
                          "selected_update": metrics["selection"]["step"],
                          "k_per_output": metrics["selection"]["k"], "seconds": metrics["elapsed_seconds"]})
    result = {"models": output, "rows": table, "provenance": provenance,
              "limitations": [
                  "One seed. The same frozen test set was observed in prior experiments.",
                  "Microbatch grouping changed from 8 to 32 to meet the 30-minute cap; effective batch size remains 32. Mathematical gradient equivalence was tested, but floating-point rounding can differ.",
                  "Checkpoint and cutoff selection use validation only. Epoch curves are validation scores, not test scores.",
                  "The selected checkpoint may precede the final epoch. All three epochs nevertheless completed.",
                  "Priority remains stored clause order; sibling clauses are excluded from clause negatives."]}
    write_json(root / "comparison.json", result)
    with (root / "comparison.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (root / "validation-by-epoch.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=["architecture", "epoch", "step", "f1", "k"])
        writer.writeheader()
        writer.writerows({"architecture": variant, **{k: x[k] for k in ("epoch", "step", "f1", "k")}}
                         for variant in output for x in output[variant]["epoch_end_validation"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    for ax, variant in zip(axes, ("one_pass", "three_pass")):
        curve = curves[variant]
        ax.plot([x["epochs"] for x in curve[1:]], [100*x["f1"] for x in curve[1:]],
                "o-", color="#125db0", label="New three-epoch run, microbatch 32")
        prior = []
        for file in sorted((root / "references" / variant).glob("validation-*.json")):
            step = int(file.stem.split("-")[1])
            if step:
                prior.append((step/266, 100*json.loads(file.read_text())["selected"]["f1"]))
        ax.plot(*zip(*prior), "s--", color="#8795a5", label="Prior one-epoch run, microbatch 8")
        selected = output[variant]["summary"]["selection"]
        ax.scatter([selected["step"]/266], [100*selected["validation"]["f1"]],
                   marker="*", s=150, color="#d47912", zorder=5, label="Selected checkpoint")
        ax.set(title=variant.replace("_", " ").capitalize(), xlabel="Epochs", ylabel="Validation union F1 (%)",
               xticks=[0, 1, 2, 3], xlim=(0, 3.1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    fig.savefig(root / "validation-curves.svg")
    fig.savefig(root / "validation-curves.png", dpi=160)
    plt.close(fig)
    body = "".join(f'<tr><td>{html.escape(r["architecture"].replace("_", " "))}</td><td>{r["run"]}</td>'
                   f'<td>{100*r["precision"]:.2f}</td><td>{100*r["recall"]:.2f}</td><td><b>{100*r["f1"]:.2f}</b></td>'
                   f'<td>{r["selected_update"]}</td><td>{r["seconds"]/60:.1f}</td></tr>' for r in table)
    epoch_body = "".join(f'<tr><td>{variant.replace("_", " ")}</td>'
                         + "".join(f'<td>{100*x["f1"]:.2f}</td>' for x in output[variant]["epoch_end_validation"])
                         + "</tr>" for variant in output)
    deltas = "".join(f'<li>{variant.replace("_", " ")}: {value["delta_test_f1_points"]:+.2f} test F1 points; '
                      f'paired-query 95% interval [{value["paired_query_bootstrap_95pct_points"][0]:+.2f}, '
                      f'{value["paired_query_bootstrap_95pct_points"][1]:+.2f}].</li>' for variant, value in output.items())
    (root / "report.html").write_text(f'''<!doctype html><meta charset="utf-8">
<title>Three-epoch autoregressive results</title>
<style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:44px auto;padding:0 24px;color:#172432}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;text-align:right;border-bottom:1px solid #dce3e9}}td:first-child,th:first-child{{text-align:left}}th{{background:#eef3f7}}img{{max-width:100%}}</style>
<h1>Three-epoch autoregressive results</h1>
<p>Both fresh joint-loss models completed 798 updates and 25,500 training-query exposures: three passes through the same 8,500-query training set. Validation contains 500 queries; test contains 1,000; the frozen corpus contains 489,188 documents. The loss, rank-32 LoRA, learning rate, effective batch size, and fixed clause assignment are unchanged.</p>
<h2>Test results after validation-only selection</h2>
<table><thead><tr><th>Architecture</th><th>Run</th><th>Precision %</th><th>Recall %</th><th>F1 %</th><th>Selected update</th><th>Minutes</th></tr></thead><tbody>{body}</tbody></table>
<ul>{deltas}</ul><p>Intervals measure query-sampling uncertainty, not variation across training seeds. Prior results used microbatch 8; new runs use microbatch 32 with the same effective batch of 32. This preserves the mathematical objective and reduces overhead, while floating-point rounding can differ.</p>
<h2>Validation progress at each completed epoch</h2>
<table><thead><tr><th>Architecture</th><th>Epoch 1 F1 %</th><th>Epoch 2 F1 %</th><th>Epoch 3 F1 %</th></tr></thead><tbody>{epoch_body}</tbody></table>
<img src="validation-curves.svg" alt="Validation F1 over three epochs for both autoregressive models">
<p>The curve also includes checks within each epoch. Selection uses the best updated validation checkpoint across all three epochs; the endpoint table is diagnostic. Test results were computed after training and selection were complete.</p>
<h2>Verification and limits</h2><p>Primary metrics were independently recomputed from saved predictions, and dataset identity, all 798 updates, source counts, finite losses, and the 30-minute cap were checked. Eleven CPU tests passed, including accumulated-versus-full-batch gradients for both architectures. The disposable GPU profile used training data only.</p>
<ul>{''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])}</ul>
<p><a href="comparison.csv">Results CSV</a> · <a href="validation-by-epoch.csv">Epoch CSV</a> · <a href="comparison.json">Full metrics and provenance</a></p>''')
    print(json.dumps({variant: {"test_f1": value["summary"]["overall"]["union"]["f1"],
                               "delta_points": value["delta_test_f1_points"],
                               "epoch_end_validation": value["epoch_end_validation"]}
                      for variant, value in output.items()}, indent=2))


if __name__ == "__main__":
    main()
