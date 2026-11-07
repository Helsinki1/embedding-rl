"""Download this experiment's small artifacts and write a local comparison."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import modal


EXPERIMENT = "knobs-a5f18dcb93"
OUTPUT = Path("/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-followup-2026-11-06")
VOLUME = modal.Volume.from_name("embedding-rl-data")


def download(remote, local):
    payload = b"".join(VOLUME.read_file(remote))
    path = OUTPUT / local
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


comparison = json.loads(download(f"experiments/{EXPERIMENT}/comparison.json", "comparison.json"))
probe = json.loads(download(f"experiments/{EXPERIMENT}/probe/probe.json", "sampling-probe.json"))
jobs = []
for run in comparison["runs"]:
    run_id = run["summary"]["run_id"]
    for filename in ("run.json", "summary.json", "metrics.jsonl", "validation.json", "training_probe.json"):
        jobs.append((f"runs/{run_id}/{filename}", f"{run_id}/{filename}"))
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lambda args: download(*args), jobs))

rows = []
changes = []
for run in comparison["runs"]:
    summary = run["summary"]
    run_id = summary["run_id"]
    validation = json.loads((OUTPUT / run_id / "validation.json").read_text())
    train_probe = json.loads((OUTPUT / run_id / "training_probe.json").read_text())
    before = {r["query_id"]: r["ndcg@10"] for r in validation["before"]["per_query"]}
    differences = [{"query_id": r["query_id"], "before": before[r["query_id"]],
                    "final": r["ndcg@10"], "delta": r["ndcg@10"] - before[r["query_id"]]}
                   for r in validation["final"]["per_query"]
                   if r["ndcg@10"] != before[r["query_id"]]]
    changes.append({"run_id": run_id, "changes": differences})
    row = {
        "condition": run["label"], "seed": run["settings"]["training_seed"],
        "samples": run["settings"]["samples_per_query"],
        "radius": run["settings"]["exploration_radius"],
        "initial_validation_ndcg": summary["initial_validation_ndcg"],
        "final_validation_ndcg": summary["final_validation_ndcg"],
        "validation_queries_improved": sum(r["delta"] > 0 for r in differences),
        "validation_queries_worse": sum(r["delta"] < 0 for r in differences),
        "fixed_train_initial_ndcg": train_probe["before"]["metrics"]["ndcg@10"],
        "fixed_train_final_ndcg": train_probe["final"]["metrics"]["ndcg@10"],
        "training_signal_fraction": 1 - run["mean_zero_advantage_fraction"],
        "run_id": run_id, "wandb_url": summary["wandb_url"],
    }
    rows.append(row)

with (OUTPUT / "run-comparison.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUTPUT / "per-query-changes.json").write_text(json.dumps(changes, indent=2) + "\n")

lines = [
    "# Embedding RL: three-knob follow-up — 2026-11-06", "",
    "More and wider samples increased the availability of informative reward comparisons, "
    "but this experiment found no validation advantage over the control. "
    "All eight runs finished 500 optimizer steps.", "",
    f"[W&B comparison]({comparison['wandb_url']}) · "
    f"[Sampling probe]({probe['wandb_url']})", "",
    "## Protocol", "",
    "- BrowseComp-Plus: 100,195 frozen document vectors; 581 training, 83 validation, "
    "166 untouched test queries. BAAI/bge-small-en-v1.5 query encoder.",
    "- One-step Gaussian-action REINFORCE/RLOO; binary evidence nDCG@10 reward. "
    "No InfoNCE training or LLM judge used.",
    "- 8 versus 32 samples, sampling radius 0.15 versus 0.45; 500 updates each, "
    "with evaluations at 0, 10, 50, 100, 250 and 500. Batch size 4; learning rate 1e-6.",
    "- KL coefficient 0.01 at radius 0.15 and 0.09 at radius 0.45. "
    "This preserves the penalty on encoder-mean displacement while changing sampling spread.",
    "- All four conditions use seed 42. They tied on final validation nDCG, so the "
    "predefined stable tie-break selected more_samples for paired repeats against control "
    "at seeds 43 and 44. Wider-only and combined settings have one training seed each.",
    "- Split seed stays 42. Each condition uses the same minibatch sequence for a given "
    "training seed. Comparisons are matched on optimizer steps, not compute cost.", "",
    "## Paired sampling probe", "",
    "64 fixed training queries, identical noise draws across radii; 8 draws are a prefix "
    "of the 32 draws. A query is informative when sampled actions receive different rewards.", "",
    "| Condition | Samples | Radius | Informative queries | Mean sampled reward | Mean best-of-samples reward |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in probe["comparison"]:
    lines.append(f"| {row['condition']} | {row['samples']} | {row['radius']:.2f} | "
                 f"{round(row['has_signal'] * 64)}/64 ({row['has_signal']:.1%}) | "
                 f"{row['reward_mean']:.6f} | {row['best_reward']:.6f} |")
lines += ["", "Best-of-samples uses known relevance labels to pick an action. It is an "
          "exploration diagnostic, not the quality of a deployable encoder.", "",
          "## Final deterministic encoder evaluation", "",
          "| Condition | Seed | Initial validation nDCG@10 | Final validation nDCG@10 | Improved / worse queries | Fixed-training nDCG@10 (final) | W&B |",
          "|---|---:|---:|---:|---:|---:|---|"]
for row in rows:
    lines.append(f"| {row['condition']} | {row['seed']} | {row['initial_validation_ndcg']:.8f} | "
                 f"{row['final_validation_ndcg']:.8f} | {row['validation_queries_improved']} / "
                 f"{row['validation_queries_worse']} | {row['fixed_train_final_ndcg']:.8f} | "
                 f"[run]({row['wandb_url']}) |")
lines += ["", f"The fixed set of 64 training queries starts at nDCG@10 "
          f"{rows[0]['fixed_train_initial_ndcg']:.8f}. Improvements here are in-sample, "
          "not evidence of generalization.", "",
          "### Validation scores by update", "",
          "| Condition / seed | 0 | 10 | 50 | 100 | 250 | 500 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
for run in comparison["runs"]:
    evaluations = {r["step"]: r["validation/ndcg@10"] for r in run["evaluations"]}
    scores = " | ".join(f"{evaluations[s]:.8f}" for s in (0, 10, 50, 100, 250, 500))
    lines.append(f"| {run['label']} / {run['settings']['training_seed']} | {scores} |")
lines += ["", "## What this supports", "",
          "- More and wider sampling can expose additional reward differences in the current setup.",
          "- Extra steps and the tested sample-count/radius settings did not produce a meaningful "
          "validation improvement. The 32-sample setting tied the control at seeds 42 and 43 "
          "and finished slightly worse at seed 44.",
          "- This does not establish that exploration can never help, or identify whether KL strength, "
          "the learning rate, document representation, or sparse labels are the limiting cause. "
          "Those variables were not independently tested.",
          "- This validation split has only 83 queries and many zero scores. The earlier 10-query "
          "pilot score is not directly comparable. Final test data was not evaluated.", "",
          "## Persistence and reproduction", "",
          f"Modal volume: `embedding-rl-data`, experiment directory `/experiments/{EXPERIMENT}`. "
          "Run checkpoints (`best.pt`, `last.pt`), provenance and metrics remain under `/runs/<run_id>`. "
          "Prepared corpus/query cache: `/prepared/79f6ff30a298db92`.", "",
          "This local folder includes comparison JSON/CSV, per-query metric changes, run artifacts "
          "and a pre-run source snapshot. It contains no API keys or benchmark query text. "
          "Checkpoint files contain model weights, not optimizer state for exact interrupted-step resumption.", "",
          "```bash", "cd /home/davidx/Downloads/embedding-rl",
          "WANDB_SECRET_NAME=wandb-secret modal run modal_followup.py --config config.followup.json",
          "```", "",
          "Use `--experiment-id knobs-a5f18dcb93` with the unchanged source/config to reuse completed "
          "stages; omit it to launch a fresh experiment. The named Modal secret supplies the W&B key.", "",
          "Validation: 31 automated tests, Ruff and git diff whitespace checks passed before launch.", ""]
(OUTPUT / "results.md").write_text("\n".join(lines))
manifest = {
    "experiment_id": EXPERIMENT, "modal_volume": "embedding-rl-data",
    "experiment_path": f"/experiments/{EXPERIMENT}",
    "wandb_url": comparison["wandb_url"], "test_evaluated": False,
    "runs": [{"run_id": r["run_id"], "condition": r["condition"], "seed": r["seed"],
              "checkpoint_directory": f"/runs/{r['run_id']}", "wandb_url": r["wandb_url"]} for r in rows],
    "local_artifacts": {str(p.relative_to(OUTPUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "artifact-index.json"},
}
(OUTPUT / "artifact-index.json").write_text(json.dumps(manifest, indent=2) + "\n")
with VOLUME.batch_upload(force=True) as batch:
    for name in ("results.md", "run-comparison.csv", "per-query-changes.json", "artifact-index.json"):
        batch.put_file(str(OUTPUT / name), f"/experiments/{EXPERIMENT}/{name}")
print(json.dumps({"comparison_url": comparison["wandb_url"], "rows": rows,
                  "paired_deltas": comparison["paired_deltas"], "report": str(OUTPUT / "results.md")}, indent=2))
