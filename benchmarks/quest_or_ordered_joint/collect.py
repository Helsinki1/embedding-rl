"""Independently recompute predictions and report completed ordered experiments."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from benchmarks.quest_or_spec_v2.build import read_rows, write_json
from benchmarks.quest_or_spec_v2.data import load_splits
from benchmarks.quest_or_spec_v2.metrics import evaluate

NAMES = {"pooled_mlp": "Pooled MLP + ordered joint loss",
         "attention_mlp": "Attention + MLP + ordered joint loss",
         "one_pass": "One-pass slots + ordered joint loss",
         "three_pass": "Three-pass feedback + ordered joint loss",
         "single_frozen": "Frozen single vector",
         "single_infonce": "Single vector, document InfoNCE",
         "llm_frozen": "Frozen LLM decomposition + union",
         "one_pass_infonce": "One-pass slots, document InfoNCE",
         "three_pass_infonce": "Three-pass feedback, document InfoNCE"}


def ordered_diagnostics(directory, gold):
    """Read fixed positions; never optimize a slot-to-component assignment."""
    rankings = json.loads((directory / "test_rankings.json").read_text())
    recall = [[[] for _ in range(3)] for _ in range(3)]
    jaccard = []
    for qid, heads in zip(rankings["ids"], rankings["corpus_indices"]):
        components = gold[qid]
        if len(components) != 3:
            continue
        for slot in range(3):
            for component in range(3):
                target = set(components[component])
                if target:
                    recall[slot][component].append(len(set(heads[slot][:5]) & target) / len(target))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            first, second = set(heads[a][:10]), set(heads[b][:10])
            jaccard.append(len(first & second) / len(first | second))
    matrix = [[float(np.mean(x)) for x in row] for row in recall]
    summary = json.loads((directory / "summary.json").read_text())
    return {"population": "500 held-out three-clause queries; fixed output and component order",
            "recall_at_5_matrix_rows_slots_columns_components": matrix,
            "diagonal_mean_recall_at_5": float(np.trace(matrix) / 3),
            "off_diagonal_mean_recall_at_5": float((np.sum(matrix) - np.trace(matrix)) / 6),
            "mean_pairwise_top10_jaccard_three_clause_queries": float(np.mean(jaccard)),
            "mean_pairwise_slot_cosine_all_test_queries": summary["timing"]["mean_pairwise_slot_cosine"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    train, validation, test, provenance = load_splits()
    gold = json.loads((args.root / "gold_corpus_indices.json").read_text())
    assert set(gold) == {r["id"] for r in test}
    index_to_title, title_to_index = {}, {}
    for row in test:
        assert len(gold[row["id"]]) == len(row["component_docs"])
        for indices, titles in zip(gold[row["id"]], row["component_docs"]):
            assert len(indices) == len(titles)
            for index, title in zip(indices, titles):
                assert 0 <= index < 489188
                assert index_to_title.setdefault(index, title) == title
                assert title_to_index.setdefault(title, index) == index
    rows, details, verified = [], {}, {}
    for key, label in NAMES.items():
        group = "results" if key in ("pooled_mlp", "attention_mlp", "one_pass", "three_pass") else "references"
        directory = args.root / group / key
        summary = json.loads((directory / "summary.json").read_text())
        run = json.loads((directory / "run.json").read_text())
        predictions = read_rows(directory / "test_predictions.jsonl")
        recomputed, per_query = evaluate(test, predictions)
        assert summary["dataset_manifest_sha256"] == provenance["manifest_sha256"], key
        assert run["documents"] == 489188, key
        assert summary["overall"] == recomputed["overall"], key
        assert summary["by_source"] == recomputed["by_source"], key
        assert summary["by_arity"] == recomputed["by_arity"], key
        if group == "results":
            assert summary["examples_seen"] == len(train) and summary["completed_steps"] == 266, key
            assert summary["seen_sources"] == provenance["audit"]["train"]["sources"], key
            assert summary["stop_reason"] == "epochs_complete", key
            assert summary["elapsed_seconds"] < 1800, key
            assert run["config"]["assignment"].startswith("identity"), key
            logs = read_rows(directory / "metrics.jsonl")
            assert len(logs) == 266, key
            assert all(np.isfinite(x["loss"]) and abs(x["loss"]-x["document_loss"]-x["clause_loss"]) < 1e-4 for x in logs)
        details[key] = {d["id"]: d for d in per_query}
        overall = summary["overall"]
        fixed = summary["fixed_budget"]["20"]["overall"]
        rows.append({"key": key, "method": label, "group": group,
                     "precision": overall["union"]["precision"], "recall": overall["union"]["recall"],
                     "f1": overall["union"]["f1"], "component_f1": overall["component_macro"]["f1"],
                     "f1_at_20": fixed["union"]["f1"], "coverage_at_100": summary["fixed_budget"]["100"]["overall"]["component_coverage"],
                     "mean_results": overall["union"]["predicted_count"], "k_per_vector": summary["selection"]["k"],
                     "checkpoint_step": summary["selection"]["step"], "examples_seen": summary["examples_seen"],
                     "seconds": summary["elapsed_seconds"], "trainable_parameters": run["trainable_parameters"]})
        verified[key] = {"primary_metrics_recomputed": True, "heldout_queries": len(test),
                         "sources": summary["by_source"], "arity": summary["by_arity"],
                         "slot_diagnostics": summary.get("slot_diagnostics")}
        if group == "results" or key in ("one_pass_infonce", "three_pass_infonce"):
            ranks = json.loads((directory / "test_rankings.json").read_text())
            assert ranks["ids"] == [r["id"] for r in test]
            pred_by_id = {r["id"]: r["docs"] for r in predictions}
            for row, heads in zip(test, ranks["corpus_indices"]):
                selected = set(i for head in heads for i in head[:summary["selection"]["k"]])
                expected = set(row["docs"]) & set(pred_by_id[row["id"]])
                observed = {index_to_title[i] for i in selected if i in index_to_title} & set(row["docs"])
                assert observed == expected and len(selected) == len(pred_by_id[row["id"]]), key
            verified[key]["ordered_retrieval_diagnostics"] = ordered_diagnostics(directory, gold)
    comparisons = []
    rng = np.random.default_rng(42)
    for a, b in (("one_pass", "one_pass_infonce"), ("three_pass", "three_pass_infonce"),
                 ("attention_mlp", "pooled_mlp")):
        differences = np.array([details[a][r["id"]]["f1"]-details[b][r["id"]]["f1"] for r in test])
        estimates = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(5000)])
        comparisons.append({"a": a, "b": b, "delta_f1_percentage_points": 100*differences.mean(),
                            "paired_query_bootstrap_95pct": (100*np.quantile(estimates,[.025,.975])).tolist(),
                            "replicates": 5000, "note": "Query-sampling uncertainty for these checkpoints; not variation across training seeds."})
    with (args.root / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"rows": rows, "comparisons": comparisons, "verified": verified, "provenance": provenance,
              "limitations": ["One seed and one epoch; lambda=1 was fixed before results, not tuned.",
                              "Priority is canonical component order, not a learned importance ranking.",
                              "QUEST source order can differ from paraphrase mention order.",
                              "Two-clause examples leave slot 3 without clause supervision.",
                              "Sibling clauses were excluded from clause negatives, so the objective does not directly contrast interpretations of the same query.",
                              "Original relevance judgments are incomplete; unjudged documents score as nonrelevant.",
                              "MLP variants lack document-only matched controls; only slot architectures isolate the new loss."]}
    write_json(args.root / "comparison.json", result)
    (args.root / "README.md").write_text('''# Ordered joint training results

Open `report.html` for the comparison, training curves, and fixed-position slot diagnostics.
`comparison.csv` contains the main numeric table; `comparison.json` contains source/arity
breakdowns, paired query intervals, and verification results.

## Experiment

Four fresh Qwen3-Embedding-8B encoders: pooled-output MLP, final-token attention readers
plus MLP, one-pass causal slots, and three-pass completed-state feedback. All use the
canonical 8,500/500/1,000 split, 489,188 frozen document vectors, rank-32 LoRA, one epoch,
266 optimizer updates, seed 42, and validation-only checkpoint/cutoff selection.

The loss is document InfoNCE plus ordered clause InfoNCE, weight 1, jointly from the
first update. Output i always targets canonical clause i. There is no permutation
search, warm-up, RL reward, or test tuning. Clause negatives exclude sibling clauses;
two-clause examples leave the third output with document supervision only.
"Priority" here is canonical component order, not an importance ranking or unequal
retrieval budgets. See `code/benchmarks/quest_or_ordered_joint/README.md` for details.

## Files and reproduction

- `results/`: four new runs, with predictions, rankings, validation curves, training
  logs, configuration, provenance, selected checkpoints' checksums, and test metrics.
- `references/`: existing controls on the same frozen benchmark. They were not retrained.
- `teachers/`: ordered training targets and teacher cache provenance; no held-out targets.
- `code/`: executed experiment source and shared dependencies. Post-run collection code
  is also included. Dataset/model/vector files are reused from the existing repository
  and compute server; this is not a standalone model distribution.
- `export_checksums.json`: checksums for the server-exported evidence.
- `verification.json`: local metric, source, count, and export checks plus the CPU test result.

Remote run root: `/home/ubuntu/quest-ordered-joint-20261106` on the configured compute host.
The selected trainable adapters/heads remain at `results/<architecture>/best.pt` under
that root. Initial and final checkpoints are retained alongside them. Full base model
weights and teacher vector tensors are not duplicated into this lightweight report.

From the original repository, rerun the report with:

```bash
python -m benchmarks.quest_or_ordered_joint.collect --root /path/to/ordered-joint-training
```

Training and teacher-preparation commands are in the experiment README. Each run had
a launcher-enforced 1,800-second budget including evaluation; shared teacher preparation
was recorded separately. This is a single-seed comparison, not a multi-seed conclusion.
''')
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for key in ("pooled_mlp", "attention_mlp", "one_pass", "three_pass"):
        logs = read_rows(args.root / "results" / key / "metrics.jsonl")
        for ax, signal in zip(axes, ("document_loss", "clause_loss")):
            values = np.array([r[signal] for r in logs])
            smooth = np.convolve(values, np.ones(15)/15, mode="valid")
            ax.plot(np.arange(15, len(logs)+1), smooth, label=key.replace("_", " "))
            ax.set(xlabel="Optimizer update", ylabel="InfoNCE loss",
                   title=signal.replace("_", " ").capitalize()+" (15-update mean)")
            ax.grid(alpha=.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(args.root / "training-curves.svg")
    fig.savefig(args.root / "training-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), constrained_layout=True)
    for ax, key in zip(axes, ("pooled_mlp", "attention_mlp", "one_pass", "three_pass")):
        matrix = np.array(verified[key]["ordered_retrieval_diagnostics"]["recall_at_5_matrix_rows_slots_columns_components"])
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=[0, 1, 2], xticklabels=[1, 2, 3], yticks=[0, 1, 2],
               yticklabels=[1, 2, 3], xlabel="Target component", ylabel="Output slot",
               title=key.replace("_", " "))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100*matrix[i,j]:.1f}%", ha="center", va="center",
                        color="white" if matrix[i,j] > .55 else "#172432")
    fig.savefig(args.root / "ordered-slot-recall.svg")
    fig.savefig(args.root / "ordered-slot-recall.png", dpi=160)
    plt.close(fig)
    collapse_rows = "".join(
        f'<tr><td>{html.escape(NAMES[key])}</td>'
        f'<td>{100*verified[key]["ordered_retrieval_diagnostics"]["diagonal_mean_recall_at_5"]:.1f}%</td>'
        f'<td>{100*verified[key]["ordered_retrieval_diagnostics"]["off_diagonal_mean_recall_at_5"]:.1f}%</td>'
        f'<td>{np.mean(verified[key]["ordered_retrieval_diagnostics"]["mean_pairwise_slot_cosine_all_test_queries"]):.4f}</td>'
        f'<td>{100*verified[key]["ordered_retrieval_diagnostics"]["mean_pairwise_top10_jaccard_three_clause_queries"]:.1f}%</td></tr>'
        for key in ("pooled_mlp", "attention_mlp", "one_pass", "three_pass", "one_pass_infonce", "three_pass_infonce"))
    table = "".join(f'<tr><td>{html.escape(r["method"])}</td><td>{100*r["precision"]:.2f}</td>'
                    f'<td>{100*r["recall"]:.2f}</td><td><b>{100*r["f1"]:.2f}</b></td>'
                    f'<td>{100*r["f1_at_20"]:.2f}</td><td>{100*r["coverage_at_100"]:.2f}</td>'
                    f'<td>{r["mean_results"]:.1f}</td><td>{r["seconds"]/60:.1f}</td></tr>' for r in rows)
    comparisons_html = "".join(f'<li>{html.escape(NAMES[c["a"]])} minus {html.escape(NAMES[c["b"]])}: '
                               f'<b>{c["delta_f1_percentage_points"]:+.2f} F1 points</b>; '
                               f'paired-query 95% interval [{c["paired_query_bootstrap_95pct"][0]:+.2f}, '
                               f'{c["paired_query_bootstrap_95pct"][1]:+.2f}].</li>' for c in comparisons)
    best = max(rows[:4], key=lambda r:r["f1"])
    (args.root / "report.html").write_text(f'''<!doctype html><meta charset="utf-8">
<title>Ordered joint InfoNCE results</title><style>body{{font:16px/1.55 system-ui;max-width:1180px;margin:48px auto;padding:0 24px;color:#172432}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:right;border-bottom:1px solid #dce3e9}}td:first-child,th:first-child{{text-align:left}}th{{background:#eef3f7}}p{{max-width:950px}}.lead{{font-size:20px}}img{{max-width:100%}}</style>
<h1>Ordered joint InfoNCE: four completed experiments</h1>
<p class="lead">Best new variant: <b>{html.escape(best["method"])}</b>, test F1 <b>{100*best["f1"]:.2f}%</b>.</p>
<p>Each new run used 8,500 training queries, 500 validation queries, and 1,000 test queries against 489,188 frozen document vectors. All completed 266 updates in one epoch with fresh rank-32 LoRA. The loss was document InfoNCE plus ordered-clause InfoNCE (weight 1, temperature .05). No reinforcement learning, output matching, or test-based tuning was used.</p>
<p>Priority means fixed canonical clause order. Synthetic questions follow mention order; QUEST uses original template order. This experiment does not label the most important interpretation. Frozen Qwen encoded 20,956 unique training clauses. Both MLP heads use residual corrections to the original embedding; the attention variant learns three readers over final token states.</p>
<table><thead><tr><th>Method</th><th>Precision %</th><th>Recall %</th><th>F1 %</th><th>F1 @ 20 %</th><th>Component coverage @ 100 %</th><th>Mean results</th><th>Minutes</th></tr></thead><tbody>{table}</tbody></table>
<p>The first four rows are new joint-loss runs. The remaining rows are existing controls verified against the same dataset and corpus. The primary F1 uses each model's validation-selected checkpoint and cutoff. F1 @ 20 uses exactly 20 deduplicated documents for every method.</p>
<h2>Matched comparisons</h2><ul>{comparisons_html}</ul><p>Intervals describe query-sampling uncertainty for one trained checkpoint per method; they do not establish robustness across training seeds.</p>
<h2>Training signals</h2><img src="training-curves.svg" alt="Document and ordered clause InfoNCE losses over updates">
<h2>Did the slots specialize?</h2><p>Each cell shows recall@5 for one output slot against one component on the 500 three-clause test queries. Strong ordering should favor the diagonal. No assignment search is used. This is an additional diagnostic, not a selection criterion.</p>
<img src="ordered-slot-recall.svg" alt="Fixed-position slot versus component recall matrices">
<table><thead><tr><th>Method</th><th>Assigned-component recall@5</th><th>Other-component recall@5</th><th>Mean slot cosine</th><th>Mean top-10 Jaccard</th></tr></thead><tbody>{collapse_rows}</tbody></table>
<p>The document-only autoregressive controls already exhibited strong fixed-position alignment. Consequently, ordering in a joint-loss model alone is not evidence that the added loss caused that ordering. Recall and Jaccard above use three-clause queries; cosine uses all test queries.</p>
<p>Higher cosine and Jaccard indicate more similar vectors and retrieved sets. Fixed target positions alone do not guarantee specialization. This implementation excludes sibling clauses from the negative bank; direct competition between a query's interpretations is therefore absent. Any follow-up using sibling negatives or an explicit separation term would require a new, clearly labeled experiment.</p>
<h2>Limits</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])}</ul>
<p>Primary metrics were independently recomputed from saved per-query predictions. Detailed source/arity results, ordered-slot diagnostics, provenance and confidence intervals are in <a href="comparison.json">comparison.json</a>; the table is also available as <a href="comparison.csv">CSV</a>.</p>''')
    print(json.dumps({"verified_runs":len(rows),"best_new":best,"comparisons":comparisons},indent=2))


if __name__ == "__main__":
    main()
