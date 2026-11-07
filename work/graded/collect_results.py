"""Collect immutable completed results, compute paired comparisons, and persist a report."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

import modal

EXPERIMENT = "graded-20261106"
OUT = Path("/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06")
VOLUME = modal.Volume.from_name("embedding-rl-data")


def download(remote, local):
    payload = b"".join(VOLUME.read_file(remote))
    if local.endswith(".json"):
        json.loads(payload)
    path = OUT / local
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


comparison = json.loads(download(f"experiments/{EXPERIMENT}/comparison.json", "comparison.json"))
calibration = json.loads(download(f"experiments/{EXPERIMENT}/calibration-gpt-6.1-terra.json", "calibration.json"))
budget = json.loads(download(f"experiments/{EXPERIMENT}/judge/budget.json", "budget.json"))
download(f"experiments/{EXPERIMENT}/grading-summary.json", "grading-summary.json")
download(f"experiments/{EXPERIMENT}/rankings.json", "rankings.json")
new_runs = [r for r in comparison["runs"] if r["label"] in {"binary_normalized", "llm_raw", "llm_normalized"}]
jobs = [(f"runs/{run['run_id']}/{name}", f"{run['run_id']}/{name}") for run in new_runs
        for name in ("summary.json", "run.json", "metrics.jsonl", "validation.json", "training_probe.json")]
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lambda args: download(*args), jobs))

audit = {(r["label"], r["seed"]): r for r in comparison["stronger_audit"]}
rows = []
for run in comparison["runs"]:
    rows.append({"condition": run["label"], "seed": run["seed"], "run_id": run["run_id"],
                 "evidence_ndcg_83": run["benchmark"]["ndcg@10"], "recall100_83": run["benchmark"]["recall@100"],
                 "llm_ndcg_83": run["llm_metrics"]["llm_ndcg@10"],
                 "strict_llm_ndcg_83": run["llm_metrics"]["llm_strict_ndcg@10"],
                 "independent_llm_ndcg_12": audit[(run["label"], run["seed"])]["llm_metrics"]["llm_ndcg@10"],
                 "pilot_10_ndcg": run["pilot_10_ndcg"],
                 "training_signal_fraction": run.get("training_signal_fraction"),
                 "wandb_url": run.get("wandb_url")})
with (OUT / "comparison.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def bootstrap(values):
    rng = random.Random(918)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return {"mean": statistics.mean(values), "lower95": means[250], "upper95": means[9749]}


def average_query_scores(runs, field, metric):
    qids = [row["query_id"] for row in runs[0][field]]
    indices = [{row["query_id"]: row[metric] for row in run[field]} for run in runs]
    return {qid: statistics.mean(index[qid] for index in indices) for qid in qids}


groups = {}
for run in comparison["runs"]:
    groups.setdefault(run["label"], []).append(run)
comparisons = []
for label in ("binary_normalized", "llm_raw", "llm_normalized"):
    for control in ("original_encoder", "previous_control"):
        for field, metric in (("benchmark_per_query", "ndcg@10"), ("llm_per_query", "llm_ndcg@10"),
                              ("llm_per_query", "llm_strict_ndcg@10")):
            a = average_query_scores(groups[control], field, metric)
            b = average_query_scores(groups[label], field, metric)
            comparisons.append({"condition": label, "control": control, "metric": metric,
                                **bootstrap([b[q] - a[q] for q in a])})
(OUT / "paired-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")

group_rows = []
for label, runs in groups.items():
    selected = [r for r in rows if r["condition"] == label]
    group_rows.append({"condition": label, "runs": len(runs),
                      **{key: statistics.mean(r[key] for r in selected) for key in (
                          "evidence_ndcg_83", "llm_ndcg_83", "strict_llm_ndcg_83", "independent_llm_ndcg_12")},
                      "signal": statistics.mean(r["training_signal_fraction"] for r in selected)
                      if selected[0]["training_signal_fraction"] is not None else None})
(OUT / "group-means.json").write_text(json.dumps(group_rows, indent=2) + "\n")

lines = ["# LLM-graded embedding RL — 2026-11-06", "",
         f"[W&B comparison]({comparison['wandb_url']})", "",
         "## Comparable results", "",
         "All scores below evaluate final checkpoints on the same 83 validation queries. "
         "The original 10-step pilot has been re-evaluated on that full split. "
         "Original benchmark metrics remain separate from the new LLM metrics.", "",
         "| Condition | Seeds/runs | Evidence nDCG@10 (83 queries) | Graded nDCG@10 (83) | Strict graded nDCG@10 (83) | Independent judge nDCG@10 (12) | Informative training groups |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for row in group_rows:
    signal = f"{row['signal']:.1%}" if row["signal"] is not None else "—"
    lines.append(f"| {row['condition']} | {row['runs']} | {row['evidence_ndcg_83']:.8f} | "
                 f"{row['llm_ndcg_83']:.6f} | {row['strict_llm_ndcg_83']:.6f} | "
                 f"{row['independent_llm_ndcg_12']:.6f} | {signal} |")
lines += ["", "The independent judge column covers only 12 fixed validation queries and is not "
          "directly comparable to the 83-query columns. Strict graded nDCG gives topical-only "
          "pages zero gain. Informative training groups have at least two different sampled rewards.", "",
          "## Paired differences versus previous binary control", "",
          "Means are first averaged across three training seeds for each query, then bootstrapped "
          "over the 83 queries. These are approximate query-bootstrap intervals, not a final-test result.", "",
          "| New condition | Metric | Mean delta | 95% interval |",
          "|---|---|---:|---|"]
for row in comparisons:
    if row["control"] == "previous_control":
        lines.append(f"| {row['condition']} | {row['metric']} | {row['mean']:+.8f} | "
                     f"[{row['lower95']:+.8f}, {row['upper95']:+.8f}] |")
lines += ["", "## Judge design and consistency", "",
          "The rubric adapts the [TREC relevance scale](https://trec.nist.gov/data/deep2021.html) "
          "to multi-hop web evidence: 0 irrelevant, 1 topical, 2 useful partial evidence, 3 decisive evidence. "
          "It requires actual facts supporting specific query clues, including useful intermediate evidence; "
          "generic keyword overlap is insufficient.", "",
          "GPT-6.1 Terra grades numbered page passages and cites their IDs; the program extracts the cited "
          "text. Positive grades require valid page and question citations. Uncertain matches are capped at 1. "
          "Every positive is independently graded again and the lower effective grade is cached. "
          "Rank, model identity and benchmark labels are hidden from the judge.", "",
          "Pages longer than 9,000 characters use the opening and query-matching windows from the full "
          "scraped page. This may miss evidence; these are judgments of the supplied excerpts, not "
          "complete human assessments of every page. The encoder still reads its original first 512 tokens.", "",
          f"Training-only audit: 32 queries, 128 page pairs (32 known evidence, 64 retrieved-unjudged, "
          f"32 random). Consensus versus a fresh reordered pass: **{calibration['repeat']['exact']:.1%} exact**, "
          f"weighted kappa **{calibration['repeat']['quadratic_weighted_kappa']:.3f}**. Against GPT-6.1 Sol: "
          f"**{calibration['stronger']['exact']:.1%} exact**, "
          f"**{calibration['stronger']['useful_boundary_agreement']:.1%}** agreement on useful evidence (grade 2+). "
          "Agreement is not proof of human-level correctness; negative cases contribute to these rates.", "",
          "A cheaper preliminary grader was less consistent and overly generous to topical matches. "
          "It was not used for training. Preliminary costs remain included in the ledger. "
          "The [relevance-labeling research](https://www.microsoft.com/en-us/research/publication/on-the-use-of-llms-for-relevance-labelling/) "
          "also motivates checking keyword bias and retaining an independent benchmark evaluation.", "",
          "## Training and normalization", "",
          "581 training queries; all 100,195 documents searched. Each new condition uses 8 Gaussian "
          "samples per query, radius 0.15, batch size 4, learning rate 1e-6, KL coefficient 0.01, "
          "500 updates and seeds 42/43/44. Evaluations occur at 0, 10, 50, 100, 250 and 500. "
          "No InfoNCE training was used. The 166 final-test queries remain untouched.", "",
          "LLM reward uses exponential gains (2^grade−1), rank discounts, and a fixed denominator "
          "representing grade 3 at all ten positions. This fixed-max DCG is bounded 0–1 and its scale "
          "does not change with newly judged pages. Every sampled top-ten page is actually judged; "
          "missing judgments trigger grading, never an automatic zero.", "",
          "All RL runs subtract the other samples’ average reward. Normalized variants additionally "
          "divide by the within-query reward standard deviation with a 0.05 floor, then clip advantages "
          "to ±5. Identical rewards produce zero advantages. This avoids amplifying arbitrarily small "
          "differences, but changes query weighting and reward strength relative to KL.", "",
          "The old binary reward retains its per-query nDCG normalization. Consequently, comparisons "
          "between reward sources change both labels and scale; they do not isolate label granularity "
          "alone. Normalization comparisons within each source are controlled.", "",
          "LLM evaluation uses one common pool per query: every model’s top ten plus known evidence. "
          "Each query/page grade and ideal DCG is shared across systems. These pooled LLM nDCGs "
          "are not numerically comparable to binary benchmark nDCG.", "",
          "## Original pilot, on its original ten validation queries", "",
          "| Condition / seed | Evidence nDCG@10 |", "|---|---:|"]
for row in rows:
    if row["condition"] in {"original_encoder", "first_pilot", "llm_raw", "llm_normalized"}:
        lines.append(f"| {row['condition']} / {row['seed']} | {row['pilot_10_ndcg']:.8f} |")
if (OUT / "unanchored-summary.json").exists():
    anchor = json.loads((OUT / "anchor-comparison.json").read_text())
    sampling = json.loads((OUT / "unanchored-summary.json").read_text())
    both = next(r for r in sampling["comparison"] if r["condition"] == "both")
    lines += ["", "## Subsequent binary-reward optimization", "",
              "These experiments retain the original binary evidence reward and require no new LLM calls. "
              "They use the same 83 validation queries and 581 training queries. The improvements below "
              "must not be attributed to the new LLM rubric.", "",
              "| Setup | Steps | Seeds | Mean validation evidence nDCG@10 |",
              "|---|---:|---:|---:|",
              f"| Previous strong anchor, 8 samples | 500 | 3 | {anchor['previous_control_mean']:.6f} |",
              f"| No anchor, learning rate 5e-6, 8 samples | 500 | 3 | {anchor['final_ndcg_mean']:.6f} |",
              f"| No anchor, learning rate 5e-6, 32 samples, radius 0.45 | 500 | 3 | {both['final_ndcg_mean']:.6f} |"]
    if (OUT / "longer-summary.json").exists():
        longer = json.loads((OUT / "longer-summary.json").read_text())
        lines.append(f"| Same 32-sample setup, longer training | 2000 | 3 | "
                     f"{longer['mean_validation_by_step']['2000']:.6f} |")
    lines += ["", "The original narrow Gaussian KL acted like a cosine penalty with coefficient 170.67. "
              "The anchor ablation was motivated by RELER's much weaker fixed-index cosine anchor. "
              "With the stronger constraint removed, increasing samples plus exploration helped in all "
              "three matched seeds. Wider exploration alone did not improve its single screening seed. "
              "Changing the anchor and learning rate together is a combined intervention; the four-way "
              "screen is retained in anchor-comparison.json.", "",
              "These are exploratory validation findings, with repeated tuning on the same queries. "
              "Seed replication reduces dependence on training randomness but does not replace an "
              "untouched final test. Longer runs start afresh and retain their actual step-500 readings "
              "because CUDA results need not be bitwise identical to earlier runs.", "",
              f"[Sampling comparison]({sampling['wandb_url']}). "
              "Detailed seed trajectories and bootstrap results are in the accompanying JSON artifacts."]
lines += ["", "## Cost, verification and persistence", "",
          f"Conservative API upper estimate: **${budget['charged_or_reserved_usd']:.2f}**, "
          "including preliminary audits, repeated positives, validation and the stronger judge. "
          "The ledger reserves a worst-case amount before calls, disables SDK retries, and charges "
          "input at up to 125% of standard prices without subtracting cache discounts. "
          "This is not an invoice; actual billed cost may be lower. Internal ceiling $95; authorized ceiling $100.", "",
          "Verified prices: [Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra), "
          "[Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol). Modal compute is separate.", "",
          "41 automated tests passed, including concurrent budget reservations, retained invalid paid "
          "responses, grounded citations, conservative confirmation, group normalization, full-corpus "
          "reward grading, shared-pool evaluation, and train/validation/test separation. Ruff and "
          "git diff whitespace checks passed.", "",
          f"Modal volume `embedding-rl-data`, `/experiments/{EXPERIMENT}`: judge cache, "
          "rubric/protocol, API ledger, comparison and source snapshot. `/runs/<run_id>` retains "
          "best and final weights, provenance, metrics and validation results. "
          "Judgments are reusable for reranker training, with train and validation IDs kept separate.", "",
          "Local files include CSV/JSON comparisons, individual run artifacts, calibration, budget "
          "and source snapshot. Calibration files contain benchmark excerpts; keep them private. "
          "W&B receives metrics and run links, not benchmark query text or judge excerpts.", ""]
(OUT / "results.md").write_text("\n".join(lines))
manifest = {"experiment_id": EXPERIMENT, "wandb_url": comparison["wandb_url"],
            "runs": [{"run_id": r["run_id"], "label": r["label"], "seed": r["seed"]} for r in new_runs],
            "files": {str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "artifact-index.json"}}
(OUT / "artifact-index.json").write_text(json.dumps(manifest, indent=2) + "\n")
with VOLUME.batch_upload(force=True) as batch:
    for name in ("results.md", "comparison.csv", "paired-comparisons.json", "group-means.json", "artifact-index.json"):
        batch.put_file(str(OUT / name), f"/experiments/{EXPERIMENT}/{name}")
print(json.dumps({"groups": group_rows, "api_cost_upper": budget['charged_or_reserved_usd'],
                  "wandb_url": comparison['wandb_url'], "report": str(OUT/'results.md')}, indent=2))
