"""Small preselected validation audit of the completed binary-reward winners."""
import copy
import json

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-winner-audit")
AUDIT = ROOT / "experiments/winner-audit-20261106"
SOURCE = ROOT / "experiments/graded-20261106"


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume}, timeout=1800)
def rankings():
    import torch
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.evaluate import evaluate
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.train import make_encoder

    if (AUDIT / "rankings.json").exists():
        existing = json.loads((AUDIT / "rankings.json").read_text())
        return {"runs": len(existing["runs"]), "judge_pairs": sum(map(len, existing["requests"].values()))}
    config = base_config()
    _, manifest, bundle, tensors = load_prepared(ROOT, config)
    old = json.loads((SOURCE / "rankings.json").read_text())
    comparison = json.loads((SOURCE / "comparison.json").read_text())
    qids = comparison["stronger_query_ids"]
    if len(qids) != 12 or not set(qids) <= set(bundle["splits"]["validation"]):
        raise ValueError("Expected the preselected twelve validation queries")
    requests = {q: set(old["requests"][q]) for q in qids}
    runs = []
    for previous in old["runs"]:
        if previous["label"] != "original_encoder" and not (
                previous["label"] == "previous_control" and previous["seed"] == 42):
            continue
        row = copy.deepcopy(previous)
        per_query = {r["query_id"]: r for r in row["benchmark_per_query"]}
        row["benchmark_per_query"] = [per_query[q] for q in qids]
        row["benchmark"] = {key: sum(per_query[q][key] for q in qids) / len(qids)
                            for key in row["benchmark"]}
        row["rankings"] = {q: row["rankings"][q] for q in qids}
        row.pop("pilot_10_ndcg", None)
        runs.append(row)
    docids = [doc["id"] for doc in bundle["docs"]]
    for seed in (42, 43, 44):
        marker = json.loads((ROOT / f"experiments/longer-20261106/both-2000-seed{seed}.json").read_text())
        summary = marker["summary"]
        encoder = make_encoder(config, manifest, "cuda")
        saved = torch.load(ROOT / "runs" / summary["run_id"] / "last.pt", map_location="cpu", weights_only=True)
        if saved["step"] != 2000:
            raise ValueError("Expected the fixed final checkpoint, not a selected earlier peak")
        encoder.model.load_state_dict(saved["state_dict"])
        measured = evaluate(encoder, bundle, tensors["documents"], qids, config)
        vectors = encoder.encode_batches([bundle["queries"][q] for q in qids], 12, query=True)
        order = (vectors @ tensors["documents"].T).argsort(descending=True, stable=True)[:, :10]
        ranked = {q: [docids[i] for i in order[row].tolist()] for row, q in enumerate(qids)}
        runs.append({"label": "binary_2000", "seed": seed, "run_id": summary["run_id"],
                     "checkpoint_step": 2000, "wandb_url": summary["wandb_url"],
                     "benchmark": measured["metrics"], "benchmark_per_query": measured["per_query"],
                     "rankings": ranked})
        for q in qids:
            requests[q].update(ranked[q])
        del encoder
        torch.cuda.empty_cache()
    write_json(AUDIT / "rankings.json", {"runs": runs, "requests": {q: sorted(ids) for q, ids in requests.items()},
                                        "validation_query_ids": qids, "test_evaluated": False,
                                        "selection": "Same12 validation queries selected before the binary ablations; all three final2000-step seeds"})
    volume.commit()
    return {"runs": len(runs), "judge_pairs": sum(map(len, requests.values()))}


@app.function(image=image, cpu=4, memory=16384, volumes={ROOT: volume}, timeout=3600,
              secrets=[modal.Secret.from_name("embedding-rl-openai", required_keys=["OPENAI_API_KEY"])],
              max_containers=1)
def judge():
    from embedding_rl.config import write_json
    from embedding_rl.graded_evaluate import evaluate_judge
    from embedding_rl.judge_experiment import base_config, load_judge
    if (AUDIT / "comparison.json").exists():
        result = json.loads((AUDIT / "comparison.json").read_text())
        return {"api_usd_upper_estimate": result["api_usd_upper_estimate"], "reused": True}
    # Always use the original cache AND shared ledger. Never create an audit-specific budget.
    grader, _ = load_judge(ROOT, base_config(), SOURCE, commit=volume.commit)
    before = grader.budget.total()
    try:
        result = evaluate_judge(grader, AUDIT)
        result["audit_incremental_api_usd_upper_estimate"] = grader.budget.total() - before
        result["scope"] = "Only the same12 preselected validation queries; original benchmark and both LLM graders shown separately"
        write_json(AUDIT / "comparison.json", result)
        return {k: v for k, v in result.items() if k not in {"runs", "stronger_audit", "stronger_query_ids"}}
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    print(json.dumps(rankings.remote()), flush=True)
    print(json.dumps(judge.remote()), flush=True)
