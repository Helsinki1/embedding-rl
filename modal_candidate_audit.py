"""Label-blind rankings, then benchmark candidate coverage and oracle-reranking diagnostics."""
import json
import statistics

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-candidate-coverage")


@app.function(image=image.add_local_python_source("feedback_probe", "retrieval_diagnostics"),
              gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume}, timeout=1800)
def audit():
    import torch
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.train import make_encoder
    from feedback_probe import refine_queries
    from retrieval_diagnostics import candidate_metrics

    directory = ROOT / "experiments/candidate-coverage-20261106"
    if (directory / "comparison.json").exists():
        return json.loads((directory / "comparison.json").read_text())["summary"]
    config = base_config()
    _, manifest, bundle, tensors = load_prepared(ROOT, config)
    qids = bundle["splits"]["validation"]
    if len(qids) != 83 or set(qids) & set(bundle["splits"]["test"]):
        raise ValueError("Unexpected validation IDs")
    models = [{"family": "original", "seed": None, "run_id": None, "expected": .037363700568675995}]
    for family, experiment, prefix in (("full2000", "longer-20261106", "both-2000"),
                                       ("mixed2000", "alternating-longer-20261106", "projected")):
        for seed in (42, 43, 44):
            record = json.loads((ROOT / "experiments" / experiment / f"{prefix}-seed{seed}.json").read_text())
            models.append({"family": family, "seed": seed, "run_id": record["summary"]["run_id"],
                           "expected": record["summary"]["final_validation_ndcg"]})
    write_json(directory / "protocol.json", {"models": models, "feedback_settings": {"k": 3, "alpha": .5, "temperature": .05, "rounds": 1},
        "variants_predeclared": ["plain", "feedback"], "candidate_count": 25, "ranking_metric": "nDCG@10",
        "oracle_is_deployable": False, "description": "Known benchmark relevance can optimally reorder only the natural top25; labels are never used to produce deployable rankings.",
        "api_calls": 0, "training_updates": 0, "validation_queries": 83, "test_evaluated": False})
    volume.commit()
    documents = tensors["documents"].to("cuda")
    index = {d["id"]: i for i, d in enumerate(bundle["docs"])}
    grades = torch.zeros(len(qids), len(documents), device="cuda")
    for i, qid in enumerate(qids):
        for docid, grade in bundle["qrels"][qid].items():
            grades[i, index[docid]] = grade
    rows, vectors = [], []
    with torch.no_grad():
        for model in models:
            encoder = make_encoder(config, manifest, "cuda")
            if model["run_id"]:
                saved = torch.load(ROOT / "runs" / model["run_id"] / "last.pt", map_location="cpu", weights_only=True)
                encoder.model.load_state_dict(saved["state_dict"])
            queries = encoder.encode_batches([bundle["queries"][q] for q in qids], 32, query=True).to("cuda")
            for variant, value in (("plain", queries), ("feedback", refine_queries(queries, documents))):
                scores = value @ documents.T
                metrics = candidate_metrics(scores, grades)
                mean = {k: v.mean().item() for k, v in metrics.items()}
                if variant == "plain" and abs(mean["ndcg10"] - model["expected"]) > 1e-5:
                    raise ValueError("Saved baseline does not reproduce")
                rows.append({**model, "variant": variant, "means": mean,
                             "per_query": {k: v.cpu().tolist() for k, v in metrics.items()}})
                vectors.append({**model, "variant": variant, "vectors": value.cpu(),
                                "top100": scores.argsort(dim=-1, descending=True, stable=True)[:, :100].cpu()})
            del encoder
            torch.cuda.empty_cache()
    summary = []
    for family in ("original", "full2000", "mixed2000"):
        for variant in ("plain", "feedback"):
            chosen = [r for r in rows if r["family"] == family and r["variant"] == variant]
            summary.append({"family": family, "variant": variant,
                            **{k: statistics.mean(r["means"][k] for r in chosen) for k in chosen[0]["means"]}})
    torch.save({"models": vectors, "query_ids": qids}, directory / "validation-vectors.pt")
    write_json(directory / "comparison.json", {"summary": summary, "runs": rows, "query_ids": qids,
        "test_evaluated": False, "api_calls": 0, "caveat": "Oracle scores use incomplete benchmark evidence labels and are diagnostic ceilings, not achieved reranker results. Mixed-model feedback is a new fixed-setting composition, not extra RL training."})
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    print("COMPARISON " + json.dumps(audit.remote()), flush=True)
