"""Shared-pool LLM evaluation plus unchanged benchmark metrics for all checkpoints."""
from concurrent.futures import ThreadPoolExecutor
import json
import random

import torch

from .config import write_json
from .data import load_prepared
from .evaluate import evaluate
from .graded_judge import fixed_max_dcg
from .train import make_encoder


@torch.no_grad()
def collect_rankings(root, config, directory, device="cuda"):
    _, manifest, bundle, tensors = load_prepared(root, config)
    followup = json.loads((root / "experiments/knobs-a5f18dcb93/comparison.json").read_text())
    runs = [{"label": "original_encoder", "run_id": None, "seed": None},
            {"label": "first_pilot", "run_id": "rl-4f5e40ad1b", "seed": 42}]
    runs += [{"label": "previous_" + r["label"], "run_id": r["summary"]["run_id"],
              "seed": r["settings"]["training_seed"]} for r in followup["runs"]]
    for path in sorted(directory.glob("*-seed*.json")):
        record = json.loads(path.read_text())
        runs.append({"label": record["label"], "run_id": record["summary"]["run_id"],
                     "seed": record["settings"]["training_seed"]})
    qids = bundle["splits"]["validation"]
    pilot_metadata = json.loads((root / "runs/rl-4f5e40ad1b/run.json").read_text())
    from .config import Config
    _, _, pilot, _ = load_prepared(root, Config(**pilot_metadata["config"]))
    pilot_ids = set(pilot["splits"]["validation"])
    if not pilot_ids <= set(qids):
        raise ValueError("Pilot validation is not a subset of the full validation split")
    requests = {qid: set(bundle["qrels"][qid]) for qid in qids}
    docids = [doc["id"] for doc in bundle["docs"]]
    for run in runs:
        encoder = make_encoder(config, manifest, device)
        if run["run_id"]:
            run_dir = root / "runs" / run["run_id"]
            summary = json.loads((run_dir / "summary.json").read_text())
            run["wandb_url"] = summary.get("wandb_url")
            logs = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            training = [row for row in logs if "train/zero_advantage_fraction" in row]
            run["training_signal_fraction"] = 1 - sum(row["train/zero_advantage_fraction"]
                                                       for row in training) / len(training)
            checkpoint = torch.load(root / "runs" / run["run_id"] / "last.pt",
                                    map_location="cpu", weights_only=True)
            encoder.model.load_state_dict(checkpoint["state_dict"])
            run["checkpoint_step"] = checkpoint["step"]
        result = evaluate(encoder, bundle, tensors["documents"], qids, config)
        run["benchmark"] = result["metrics"]
        run["benchmark_per_query"] = result["per_query"]
        run["pilot_10_ndcg"] = sum(r["ndcg@10"] for r in result["per_query"]
                                   if r["query_id"] in pilot_ids) / len(pilot_ids)
        vectors = encoder.encode_batches([bundle["queries"][qid] for qid in qids],
                                          config.encode_batch_size, query=True)
        order = (vectors @ tensors["documents"].T).argsort(descending=True, stable=True)[:, :25]
        run["rankings"] = {qid: [docids[i] for i in order[row].tolist()]
                           for row, qid in enumerate(qids)}
        for qid in qids:
            requests[qid].update(run["rankings"][qid][:10])
        print(json.dumps({"evaluated": run["label"], "seed": run["seed"],
                          "ndcg": run["benchmark"]["ndcg@10"]}), flush=True)
        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()
    result = {"runs": runs, "requests": {q: sorted(ids) for q, ids in requests.items()},
              "validation_query_ids": qids, "pilot_validation_ids": sorted(pilot_ids),
              "test_evaluated": False}
    write_json(directory / "rankings.json", result)
    return {"runs": len(runs), "validation_queries": len(qids),
            "judge_pairs": sum(len(v) for v in requests.values())}


def paired_bootstrap(first, second):
    delta = torch.tensor(second, dtype=torch.float64) - torch.tensor(first, dtype=torch.float64)
    generator = torch.Generator().manual_seed(918)
    samples = delta[torch.randint(len(delta), (10000, len(delta)), generator=generator)].mean(1)
    return {"mean_delta": delta.mean().item(), "lower_95": samples.quantile(.025).item(),
            "upper_95": samples.quantile(.975).item(), "queries_improved": (delta > 0).sum().item(),
            "queries_worse": (delta < 0).sum().item()}


def score_runs(runs, pools, judgments, qids):
    output = []
    discounts = torch.log2(torch.arange(10) + 2)
    for run in runs:
        rows = []
        for qid in qids:
            values = judgments[qid]
            grades = torch.tensor([values[docid]["effective_grade"]
                                   for docid in run["rankings"][qid][:10]], dtype=torch.float32)
            ideal = sorted((values[docid]["effective_grade"] for docid in pools[qid]), reverse=True)[:10]
            ideal_grades = torch.tensor(ideal, dtype=torch.float32)
            denominator = ((2**ideal_grades - 1) / discounts[:len(ideal)]).sum().item()
            dcg = ((2**grades - 1) / discounts).sum().item()
            strict = grades.masked_fill(grades < 2, 0)
            strict_ideal = ideal_grades.masked_fill(ideal_grades < 2, 0)
            strict_denom = ((2**strict_ideal - 1) / discounts[:len(ideal)]).sum().item()
            rows.append({"query_id": qid, "llm_ndcg@10": dcg / denominator if denominator else 0,
                         "llm_strict_ndcg@10": ((2**strict - 1) / discounts).sum().item()
                         / strict_denom if strict_denom else 0,
                         "llm_fixed_max_dcg@10": fixed_max_dcg(grades).item(),
                         "useful_precision@10": (grades >= 2).float().mean().item()})
        metrics = {key: sum(r[key] for r in rows) / len(rows) for key in rows[0] if key != "query_id"}
        output.append({k: v for k, v in run.items() if k != "rankings"}
                      | {"llm_metrics": metrics, "llm_per_query": rows})
    return output


def evaluate_judge(judge, directory):
    rankings = json.loads((directory / "rankings.json").read_text())
    requests = rankings["requests"]
    judge.ensure(requests)
    runs = score_runs(rankings["runs"], requests, judge.grades, rankings["validation_query_ids"])
    original = runs[0]
    for run in runs:
        run["paired_vs_original"] = {
            "benchmark": paired_bootstrap([r["ndcg@10"] for r in original["benchmark_per_query"]],
                                           [r["ndcg@10"] for r in run["benchmark_per_query"]]),
            "llm": paired_bootstrap([r["llm_ndcg@10"] for r in original["llm_per_query"]],
                                     [r["llm_ndcg@10"] for r in run["llm_per_query"]]),
        }
    # Preselected random validation subset; this stronger judge is never used in training.
    audit_ids = random.Random(1308).sample(sorted(requests), 12)
    tasks = [(qid, requests[qid][start:start + 6]) for qid in audit_ids
             for start in range(0, len(requests[qid]), 6)]
    def audit(task):
        qid, ids = task
        return qid, judge.assess(qid, ids, tag="heldout-stronger", model="gpt-6.1-sol")
    stronger = {qid: {} for qid in audit_ids}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for qid, values in pool.map(audit, tasks):
            stronger[qid].update(values)
    independent = score_runs(rankings["runs"], requests, stronger, audit_ids)
    result = {"runs": runs, "stronger_audit": independent, "stronger_query_ids": audit_ids,
              "api_usd_upper_estimate": judge.budget.total(), "validation_queries": len(requests),
              "judged_validation_pairs": sum(len(v) for v in requests.values()),
              "test_evaluated": False, "pool": "union of all systems' top 10 plus known evidence",
              "judge_model": judge.model, "independent_model": "gpt-6.1-sol"}
    write_json(directory / "comparison.json", result)
    return result
