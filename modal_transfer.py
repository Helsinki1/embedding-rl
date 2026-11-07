"""Predeclared frozen-query-encoder regression checks on two external BEIR datasets."""
import csv
import json
from pathlib import Path
import statistics

import modal

from modal_common import ROOT, volume

app = modal.App("embedding-rl-transfer-check")
transfer_image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    .pip_install("pytrec-eval-terrier==0.5.11")
    .env({"HF_HOME": "/data/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("embedding_rl", "modal_common"))
DATASETS = {
    "scifact": {"revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
                "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029", "queries": 300, "documents": 5183},
    "nfcorpus": {"revision": "b5026a0e96e8a7ac4f95f482a596389289d46269",
                 "qrels_revision": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65", "queries": 323, "documents": 3633},
}


@app.function(image=transfer_image, gpu="T4", cpu=4, memory=16384, volumes={ROOT: volume}, timeout=3600)
def run():
    import math
    import pytrec_eval
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.train import make_encoder

    directory = ROOT / "experiments/transfer-20261106"
    if (directory / "comparison.json").exists():
        return json.loads((directory / "comparison.json").read_text())["summary"]
    torch.set_num_threads(4)
    config = base_config()
    _, manifest, _, _ = load_prepared(ROOT, config)
    models = [{"family": "original", "seed": None, "run_id": None}]
    for family, experiment, prefix in (("full2000", "longer-20261106", "both-2000"),
                                       ("mixed2000", "alternating-longer-20261106", "projected")):
        for seed in (42, 43, 44):
            record = json.loads((ROOT / "experiments" / experiment / f"{prefix}-seed{seed}.json").read_text())
            models.append({"family": family, "seed": seed, "run_id": record["summary"]["run_id"]})
    protocol = {"datasets": DATASETS, "models": models, "base_model": config.model_name,
        "model_revision": manifest["revisions"]["model"], "document_text": "title + newline + text, stripped",
        "document_encoder": "Original frozen BGE, first512 tokens, CLS pooling and unit normalization",
        "query_encoder": "Original or final saved query encoder; unchanged query prefix and max_length",
        "retrieval": "Exact cosine search over each complete external corpus; exclude identical query/document IDs as BEIR does",
        "evaluation": "pytrec-eval-terrier0.5.11, ndcg_cut.10 and recall.100; nDCG uses linear graded gains",
        "api_calls": 0, "training_updates": 0, "external_split": "test", "feedback": False,
        "tuning": False, "browsecomp_final_test_evaluated": False,
        "caveat": "Within-dataset transfer diagnostic. These public datasets may occur in base-model pretraining; do not claim contamination-free generalization or compare raw scores across corpora."}
    if (directory / "protocol.json").exists() and json.loads((directory / "protocol.json").read_text()) != protocol:
        raise ValueError("Transfer protocol changed")
    write_json(directory / "protocol.json", protocol)
    volume.commit()
    smoke = pytrec_eval.RelevanceEvaluator({"q": {"a": 2, "b": 1}}, {"ndcg_cut.2"}).evaluate(
        {"q": {"a": 1., "b": 2.}})["q"]["ndcg_cut_2"]
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    if abs(smoke - expected) > 1e-10:
        raise ValueError("Official evaluator does not use expected linear gain convention")
    rows = []
    for name, settings in DATASETS.items():
        folder = directory / name
        if (folder / "runs.json").exists():
            cached = json.loads((folder / "runs.json").read_text())
            if cached["protocol_digest"] != digest(protocol):
                raise ValueError("Transfer result provenance changed")
            rows.extend(cached["runs"])
            continue
        data = {}
        for subset in ("corpus", "queries"):
            path = hf_hub_download(f"BeIR/{name}", f"{subset}/{subset}-00000-of-00001.parquet",
                                   repo_type="dataset", revision=settings["revision"])
            data[subset] = load_dataset("parquet", data_files=path, split="train")
        qrels_path = hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset", revision=settings["qrels_revision"])
        qrels = {}
        with open(qrels_path) as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        query_text = {str(q["_id"]): q["text"] for q in data["queries"]}
        qids = sorted(qrels)
        docids = [str(d["_id"]) for d in data["corpus"]]
        if len(qids) != settings["queries"] or len(docids) != settings["documents"]:
            raise ValueError("Unexpected external dataset size")
        if not set(qids) <= query_text.keys() or any(not set(labels) <= set(docids) for labels in qrels.values()):
            raise ValueError("Qrels reference unknown query or document")
        encoder = make_encoder(config, manifest, "cuda")
        vector_path = folder / "documents.pt"
        if vector_path.exists():
            saved = torch.load(vector_path, weights_only=True)
            if saved["protocol_digest"] != digest(protocol) or saved["document_ids"] != docids:
                raise ValueError("Cached external index differs")
            documents = saved["vectors"]
        else:
            documents = encoder.encode_batches([(d["title"] + "\n" + d["text"]).strip() for d in data["corpus"]], 64)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save({"vectors": documents, "document_ids": docids, "protocol_digest": digest(protocol)}, vector_path)
            volume.commit()
        del encoder
        documents = documents.to("cuda")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
        dataset_rows, vectors = [], []
        with torch.no_grad():
            for model in models:
                encoder = make_encoder(config, manifest, "cuda")
                if model["run_id"]:
                    state = torch.load(ROOT / "runs" / model["run_id"] / "last.pt", weights_only=True, map_location="cpu")
                    encoder.model.load_state_dict(state["state_dict"])
                queries = encoder.encode_batches([query_text[q] for q in qids], 32, query=True)
                scores = (queries.to("cuda") @ documents.T).cpu()
                order = scores.argsort(dim=-1, descending=True, stable=True)[:, :101]
                results = {qid: {docid: score for docid, score in zip(docids, values, strict=True) if docid != qid}
                           for qid, values in zip(qids, scores.tolist(), strict=True)}
                evaluated = evaluator.evaluate(results)
                if evaluated.keys() != qrels.keys():
                    raise ValueError("Evaluator omitted a query")
                values = {metric: [evaluated[q][metric] for q in qids] for metric in ("ndcg_cut_10", "recall_100")}
                row = {"dataset": name, **model, "query_ids": qids, "per_query": values,
                       "means": {k: statistics.mean(v) for k, v in values.items()}}
                dataset_rows.append(row)
                vectors.append({**model, "queries": queries, "top101": order, "scores101": scores.gather(1, order)})
                print("MEASURED " + json.dumps({k: v for k, v in row.items() if k not in {"per_query", "query_ids"}}), flush=True)
                del encoder
                torch.cuda.empty_cache()
        torch.save({"query_ids": qids, "models": vectors}, folder / "query-rankings.pt")
        write_json(folder / "runs.json", {"protocol_digest": digest(protocol), "runs": dataset_rows})
        rows.extend(dataset_rows)
        volume.commit()
    summary = []
    for name in DATASETS:
        original = next(r for r in rows if r["dataset"] == name and r["family"] == "original")
        for family in ("original", "full2000", "mixed2000"):
            chosen = [r for r in rows if r["dataset"] == name and r["family"] == family]
            for metric in ("ndcg_cut_10", "recall_100"):
                after = [statistics.mean(r["per_query"][metric][i] for r in chosen) for i in range(len(original["query_ids"]))]
                summary.append({"dataset": name, "family": family, "metric": metric,
                                "mean": statistics.mean(after), "paired_query_bootstrap": paired_bootstrap(original["per_query"][metric], after)})
    result = {"summary": summary, "runs": rows, "protocol_digest": digest(protocol), "api_calls": 0,
              "training_updates": 0, "browsecomp_final_test_evaluated": False, "external_split": "test",
              "evaluator_linear_gain_smoke_error": abs(smoke - expected)}
    write_json(directory / "comparison.json", result)
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    print("COMPARISON " + json.dumps(run.remote()), flush=True)
