"""Fixed-setting, no-training, no-API pseudo-relevance-feedback control."""
import json
import statistics

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-feedback-control")
SETTINGS = {
    "paper_default": {"k": 3, "alpha": .5, "temperature": .05, "rounds": 1},
    "gentle": {"k": 3, "alpha": .2, "temperature": .05, "rounds": 1},
    "top10": {"k": 10, "alpha": .5, "temperature": .05, "rounds": 1},
    "two_rounds": {"k": 3, "alpha": .5, "temperature": .05, "rounds": 2},
}


@app.function(image=image.add_local_python_source("feedback_probe"), gpu="T4", cpu=4,
              memory=16384, volumes={ROOT: volume}, timeout=1800, max_containers=1)
def probe():
    import torch
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.objectives import ndcg
    from embedding_rl.train import make_encoder
    from feedback_probe import refine_queries

    directory = ROOT / "experiments/feedback-20261106"
    if (directory / "comparison.json").exists():
        return json.loads((directory / "comparison.json").read_text())["summary"]
    config = base_config()
    _, manifest, bundle, tensors = load_prepared(ROOT, config)
    qids = bundle["splits"]["validation"]
    if len(qids) != 83 or set(qids) & set(bundle["splits"]["test"]):
        raise ValueError("Unexpected validation split")
    documents = tensors["documents"].to("cuda")
    write_json(directory / "protocol.json", {
        "variants": SETTINGS, "selected_before_evaluation": True,
        "source": "https://arxiv.org/html/2605.11374v4",
        "implementation": "Softmax over current topK cosine scores; normalize weighted document centroid; normalize interpolation with current query; repeat fresh retrieval for each round.",
        "scope": "Own vector-only adaptation with fixed paper-inspired settings, not a full program-search reproduction. Full100195-document scoring, not restricted shortlist reranking.",
        "models": "Original encoder and all three existing longer-20261106 final2000-step checkpoints",
        "refinement_uses_labels": False, "extra_encoder_calls": 0,
        "retrieval_passes": {name: value["rounds"] + 1 for name, value in SETTINGS.items()},
        "api_calls": 0, "training_updates": 0, "validation_queries": 83, "test_evaluated": False,
    })
    volume.commit()
    models = [{"name": "original", "seed": None, "run_id": None, "expected": .037363700568675995}]
    for seed in (42, 43, 44):
        record = json.loads((ROOT / "experiments/longer-20261106" / f"both-2000-seed{seed}.json").read_text())
        models.append({"name": "binary2000", "seed": seed, "run_id": record["summary"]["run_id"],
                       "expected": record["summary"]["final_validation_ndcg"]})
    index = {d["id"]: i for i, d in enumerate(bundle["docs"])}
    # These labels are used only after label-free query refinement, for reporting.
    grades = torch.zeros(len(qids), len(documents), device="cuda")
    for i, qid in enumerate(qids):
        for docid, grade in bundle["qrels"][qid].items():
            grades[i, index[docid]] = grade

    def measure(vectors):
        scores = vectors @ documents.T
        order = scores.argsort(dim=-1, descending=True, stable=True)
        values = ndcg(scores, grades, 10)
        relevant = grades > 0
        recall = relevant.gather(1, order[:, :100]).sum(1) / relevant.sum(1).clamp_min(1)
        return {"ndcg": values.cpu().tolist(), "recall100": recall.cpu().tolist(),
                "mean_ndcg": values.mean().item(), "mean_recall100": recall.mean().item()}

    rows = []
    with torch.no_grad():
        for model in models:
            encoder = make_encoder(config, manifest, "cuda")
            if model["run_id"]:
                saved = torch.load(ROOT / "runs" / model["run_id"] / "last.pt", map_location="cpu", weights_only=True)
                encoder.model.load_state_dict(saved["state_dict"])
            vectors = encoder.encode_batches([bundle["queries"][q] for q in qids], 32, query=True).to("cuda")
            baseline = measure(vectors)
            if abs(baseline["mean_ndcg"] - model["expected"]) > 1e-5:
                raise ValueError("Rescored baseline does not match saved final checkpoint")
            for name, settings in {"baseline": None, **SETTINGS}.items():
                result = baseline if settings is None else measure(refine_queries(vectors, documents, **settings))
                rows.append({**model, "variant": name, **result,
                             "paired_delta": paired_bootstrap(baseline["ndcg"], result["ndcg"])})
                print(json.dumps({k: v for k, v in rows[-1].items() if k not in {"ndcg", "recall100"}}), flush=True)
            del encoder
            torch.cuda.empty_cache()
    summary = []
    for model_name in ("original", "binary2000"):
        def average_queries(variant):
            group = [r for r in rows if r["name"] == model_name and r["variant"] == variant]
            return [statistics.mean(r["ndcg"][i] for r in group) for i in range(len(qids))]
        baseline = average_queries("baseline")
        for variant in ("baseline", *SETTINGS):
            values = average_queries(variant)
            summary.append({"model": model_name, "variant": variant, "mean_ndcg": statistics.mean(values),
                            "paired_query_bootstrap": paired_bootstrap(baseline, values)})
    write_json(directory / "comparison.json", {"summary": summary, "runs": rows, "query_ids": qids,
        "test_evaluated": False, "api_calls": 0, "caveat": "Fixed variants evaluated on83 validation queries; any best-variant choice remains exploratory. No per-query oracle routing."})
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    print("COMPARISON " + json.dumps(probe.remote()), flush=True)
