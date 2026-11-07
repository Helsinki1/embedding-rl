"""Train a four-weight contextual-bandit reranker on natural top-25 candidates."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import modal

from modal_common import ROOT, volume
from modal_lexical import lexical_image

app = modal.App("embedding-rl-feature-reranker")
feature_image = lexical_image.add_local_python_source("rerank_features", "modal_lexical")
DIRECTORY = ROOT / "experiments/feature-reranker-20261106"
MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@app.function(image=feature_image, gpu="L4", cpu=4, memory=24576, volumes={ROOT: volume}, timeout=5400, max_containers=1)
def prepare():
    import bm25s
    import numpy as np
    import Stemmer
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import rerank_features
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.export import file_sha256
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.train import make_encoder

    torch.set_num_threads(4)
    config = base_config()
    _, manifest, bundle, tensors = load_prepared(ROOT, config)
    train_ids, val_ids = bundle["splits"]["train"], bundle["splits"]["validation"]
    qids = train_ids + val_ids
    if len(train_ids) != 581 or len(val_ids) != 83 or len(set(qids)) != 664 or set(qids) & set(bundle["splits"]["test"]):
        raise ValueError("Unexpected split boundary")
    run_id = "rl-50597452ad"  # Predeclared seed 42, not the highest validation seed.
    checkpoint = ROOT / "runs" / run_id / "last.pt"
    protocol = {"prepared_digest": digest(manifest), "query_ids_digest": digest(qids),
        "retriever_run_id": run_id, "retriever_checkpoint": "last.pt", "checkpoint_sha256": file_sha256(checkpoint),
        "retriever_seed": 42, "candidate_count": 25, "retrieval": "Equal RRF60, top1000 dense and prefix BM25",
        "features": ["rrf", "dense_cosine", "log1p_bm25", "bge_logit512"],
        "feature_normalization": "Per-query candidate-set mean and population standard deviation, floor 1e-6",
        "reranker_model": MODEL, "reranker_revision": REVISION, "reranker_pair_length": 512, "reranker_dtype": "float16",
        "reranker_text": "Full scraped text with longest_first pair truncation; no URL/title enrichment",
        "policy": "Four linear weights; Plackett-Luce top10 action sampled without replacement by Gumbel sorting",
        "initial_weights": [1, 0, 0, 0], "seeds": [42, 43, 44], "steps": 500, "batch_size": 16, "samples": 32,
        "learning_rate": .01, "optimizer": "Adam", "baseline": "Unscaled leave-one-out reward; no clipping or KL",
        "reward": "Benchmark evidence nDCG@10, whole-corpus ideal denominator; unjudged pages are zero, not verified negatives",
        "train_queries": 581, "validation_queries": 83, "final_checkpoint_only": True,
        "policy_source_sha256": hashlib.sha256(Path(rerank_features.__file__).read_bytes()).hexdigest(),
        "api_calls": 0, "browsecomp_final_test_evaluated": False, "external_datasets_evaluated": False,
        "limitations": "Small feature policy, not full cross-encoder fine-tuning. One fixed retriever seed; three policy-training seeds. Adaptive research on reused validation queries; no hyperparameter grid."}
    if (DIRECTORY / "protocol.json").exists() and json.loads((DIRECTORY / "protocol.json").read_text()) != protocol:
        raise ValueError("Feature experiment protocol changed")
    write_json(DIRECTORY / "protocol.json", protocol)
    volume.commit()
    if (DIRECTORY / "features.pt").exists():
        data = torch.load(DIRECTORY / "features.pt", weights_only=True)
        if data["protocol_digest"] != digest(protocol):
            raise ValueError("Prepared feature provenance differs")
        return {"prepared": True, "queries": len(qids)}
    lexical_dir = ROOT / "experiments/lexical-20261106"
    lexical_protocol = json.loads((lexical_dir / "protocol.json").read_text())
    if lexical_protocol["prepared_digest"] != digest(manifest):
        raise ValueError("Lexical document provenance differs")
    dense_saved = torch.load(lexical_dir / "dense-rankings.pt", weights_only=True)
    prefix_saved = torch.load(lexical_dir / "lexical-scores.pt", weights_only=True)
    if dense_saved["query_ids"] != val_ids or prefix_saved["query_ids"] != val_ids:
        raise ValueError("Cached validation order differs")
    saved_model = next(r for r in dense_saved["models"] if r["run_id"] == run_id)
    candidate_file = DIRECTORY / "candidates.pt"
    if candidate_file.exists():
        candidate_data = torch.load(candidate_file, weights_only=True)
        if candidate_data["protocol_digest"] != digest(protocol):
            raise ValueError("Cached candidate provenance differs")
    else:
        encoder = make_encoder(config, manifest, "cuda")
        encoder.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        with torch.inference_mode():
            train_vectors = encoder.encode_batches([bundle["queries"][q] for q in train_ids], 32, query=True).cpu()
        del encoder
        torch.cuda.empty_cache()
        vectors = torch.cat([train_vectors, saved_model["query_vectors"]])
        dense_scores = vectors @ tensors["documents"].T
        dense_order = dense_scores.argsort(dim=-1, descending=True, stable=True)[:, :1000]
        dense_order[len(train_ids):] = saved_model["top1000"]
        retriever = bm25s.BM25.load(str(lexical_dir / "index-first512"), load_corpus=False, mmap=True)
        tokens = bm25s.tokenize([bundle["queries"][q] for q in train_ids], stopwords="en", stemmer=Stemmer.Stemmer("english"),
                                return_ids=False, show_progress=False)
        train_scores = torch.from_numpy(np.stack([retriever.get_scores(q) if q else np.zeros(len(bundle["docs"]), dtype=np.float32) for q in tokens]))
        lexical_scores = torch.cat([train_scores, prefix_saved["scores"]["first512"]])
        lexical_order = lexical_scores.argsort(dim=-1, descending=True, stable=True)[:, :1000]
        reciprocal = (1 / (60 + torch.arange(1, 1001, dtype=torch.float32))).expand(len(qids), -1)
        fused = torch.zeros_like(dense_scores).scatter(1, dense_order, reciprocal) + torch.zeros_like(dense_scores).scatter(1, lexical_order, reciprocal)
        candidates = fused.argsort(dim=-1, descending=True, stable=True)[:, :25].clone()
        previous = json.loads((ROOT / "experiments/hybrid-reranker-20261106/complete-candidates.json").read_text())
        if previous["query_ids"] != val_ids:
            raise ValueError("Prior candidate query order differs")
        expected = next(r for r in previous["systems"] if r["seed"] == 42 and r["variant"] == "rrf_first512")
        if not torch.equal(candidates[len(train_ids):], torch.tensor(expected["candidates"])):
            raise ValueError("Validation candidate lists differ from their completed control")
        raw = torch.stack([fused.gather(1, candidates), dense_scores.gather(1, candidates), lexical_scores.gather(1, candidates).log1p()], dim=-1)
        candidate_data = {"protocol_digest": digest(protocol), "query_ids": qids, "candidates": candidates, "raw_features": raw}
        torch.save(candidate_data, candidate_file)
        volume.commit()
        print("CANDIDATES_SAVED queries=664 natural_top25; validation candidates exactly match", flush=True)
        del dense_scores, lexical_scores, train_scores, fused, retriever
    candidates = candidate_data["candidates"]
    source_dir = ROOT / "experiments/hybrid-reranker-20261106"
    source_protocol = json.loads((source_dir / "protocol.json").read_text())
    if source_protocol["model"] != MODEL or source_protocol["revision"] != REVISION or source_protocol["prepared_digest"] != digest(manifest):
        raise ValueError("Frozen score source differs")
    source_cache = json.loads((source_dir / "scores-512.json").read_text())
    if source_cache["protocol_digest"] != digest(source_protocol):
        raise ValueError("Frozen score cache provenance differs")
    cache_file = DIRECTORY / "bge-scores.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {"protocol_digest": digest(protocol), "scores": {}}
    if cache["protocol_digest"] != digest(protocol):
        raise ValueError("Feature score cache changed")
    for qi in range(len(train_ids), len(qids)):
        for di in candidates[qi].tolist():
            cache["scores"][f"{qi}:{di}"] = source_cache["scores"][f"{qi - len(train_ids)}:{di}"]
    pending = [(qi, di) for qi, ids in enumerate(candidates.tolist()) for di in ids if f"{qi}:{di}" not in cache["scores"]]
    write_json(cache_file, cache)
    volume.commit()
    print(f"BGE_PENDING {len(pending)}; reused validation pairs=2075", flush=True)
    if pending:
        corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", revision=bundle["revisions"]["corpus"])
        if [str(x) for x in corpus["docid"]] != [d["id"] for d in bundle["docs"]]:
            raise ValueError("Corpus order changed")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16).to("cuda").eval()
        with torch.inference_mode():
            for start in range(0, len(pending), 16):
                pairs = pending[start:start + 16]
                inputs = tokenizer([[bundle["queries"][qids[qi]], corpus[di]["text"]] for qi, di in pairs],
                    padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                values = model(**inputs, return_dict=True).logits.reshape(-1).float().cpu()
                if not torch.isfinite(values).all() or len(values) != len(pairs):
                    raise ValueError("Invalid frozen feature scores")
                for (qi, di), value in zip(pairs, values.tolist(), strict=True):
                    cache["scores"][f"{qi}:{di}"] = value
                if (start + len(pairs)) % 256 == 0 or start + len(pairs) == len(pending):
                    write_json(cache_file, cache)
                    volume.commit()
                    print(f"BGE_CACHED {len(cache['scores'])}/16600", flush=True)
    bge = torch.tensor([[cache["scores"][f"{qi}:{di}"] for di in ids] for qi, ids in enumerate(candidates.tolist())])
    raw = torch.cat([candidate_data["raw_features"], bge[..., None]], dim=-1)
    features = rerank_features.normalize_features(raw)
    gains = torch.tensor([[2 ** bundle["qrels"][qid].get(bundle["docs"][di]["id"], 0) - 1 for di in ids]
                         for qid, ids in zip(qids, candidates.tolist(), strict=True)], dtype=torch.float32)
    ideal = torch.tensor([sum((2**g - 1) / math.log2(rank + 2)
                             for rank, g in enumerate(sorted(bundle["qrels"][qid].values(), reverse=True)[:10])) for qid in qids])
    if not torch.isfinite(features).all() or not torch.isfinite(ideal).all():
        raise ValueError("Nonfinite prepared features or reward denominator")
    output = {"protocol_digest": digest(protocol), "raw_features": raw, "candidate_indices": candidates}
    for split, lo, hi in (("train", 0, len(train_ids)), ("validation", len(train_ids), len(qids))):
        output[split] = {"query_ids": qids[lo:hi], "features": features[lo:hi], "gains": gains[lo:hi], "ideal": ideal[lo:hi]}
    torch.save(output, DIRECTORY / "features.pt")
    write_json(DIRECTORY / "data-summary.json", {"train_queries": 581, "validation_queries": 83,
        "train_with_evidence": int((gains[:581].sum(1) > 0).sum()), "validation_with_evidence": int((gains[581:].sum(1) > 0).sum()),
        "pairs": len(cache["scores"]), "validation_pairs_reused": 2075, "protocol_digest": digest(protocol),
        "all_features_finite": bool(torch.isfinite(features).all()), "test_exported": False})
    volume.commit()
    return {"prepared": True, "queries": len(qids), "pairs": len(cache["scores"])}


@app.function(image=feature_image, cpu=4, memory=4096, volumes={ROOT: volume}, timeout=1200,
              secrets=[modal.Secret.from_name("wandb-secret")], max_containers=1)
def train():
    import torch
    from embedding_rl.config import digest, write_json
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.tracking import Tracker
    from rerank_features import evaluate_weights, train_policy

    torch.set_num_threads(4)
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    data = torch.load(DIRECTORY / "features.pt", weights_only=True)
    if data["protocol_digest"] != digest(protocol) or set(data["train"]["query_ids"]) & set(data["validation"]["query_ids"]):
        raise ValueError("Feature provenance or split boundary differs")
    def evaluate(weights, split):
        subset = data[split]
        return evaluate_weights(weights, subset["features"], subset["gains"], subset["ideal"])
    before = evaluate(torch.tensor([1., 0., 0., 0.]), "validation")
    frozen = evaluate(torch.tensor([0., 0., 0., 1.]), "validation")
    previous = json.loads((ROOT / "experiments/hybrid-reranker-20261106/comparison.json").read_text())
    expected = next(r for r in previous["runs"] if r["seed"] == 42 and r["variant"] == "rrf_first512" and r["max_length"] == 512)
    if not torch.allclose(before, torch.tensor(expected["before"]), atol=1e-6, rtol=0) or not torch.allclose(frozen, torch.tensor(expected["after"]), atol=1e-6, rtol=0):
        raise ValueError("Feature baselines differ from saved frozen controls")
    rows = []
    for seed in protocol["seeds"]:
        folder = DIRECTORY / f"seed-{seed}"
        if (folder / "result.json").exists():
            rows.append(json.loads((folder / "result.json").read_text()))
            continue
        folder.mkdir(exist_ok=True)
        tracker = Tracker(folder, base_config(), "feature-reranker-rl", {"experiment_id": "feature-reranker-20261106", "seed": seed, "protocol": protocol})
        try:
            subset = data["train"]
            weights, history = train_policy(subset["features"], subset["gains"], subset["ideal"], seed,
                steps=protocol["steps"], batch_size=protocol["batch_size"], samples=protocol["samples"], learning_rate=protocol["learning_rate"])
            for row in history:
                tracker.log({k: v for k, v in row.items() if k != "step"}, row["step"])
            after = evaluate(weights, "validation")
            row = {"seed": seed, "weights": weights.tolist(), "steps": protocol["steps"], "before": before.tolist(), "after": after.tolist(),
                "mean_before": before.mean().item(), "mean_after": after.mean().item(),
                "train_before": evaluate(torch.tensor([1., 0., 0., 0.]), "train").mean().item(), "train_after": evaluate(weights, "train").mean().item(),
                "paired_query_bootstrap": paired_bootstrap(before.tolist(), after.tolist()), "wandb_url": tracker.run.url if tracker.run else None,
                "protocol_digest": digest(protocol), "informative_fraction_mean_logged": statistics.mean(r["train/informative_fraction"] for r in history)}
            torch.save({"weights": weights, "step": protocol["steps"], "protocol_digest": digest(protocol)}, folder / "final.pt")
            write_json(folder / "result.json", row)
            tracker.finish(row)
        except Exception:
            tracker.finish(success=False)
            raise
        volume.commit()
        rows.append(row)
        print("POLICY_DONE " + json.dumps({k: v for k, v in row.items() if k not in {"before", "after"}}), flush=True)
    mean_after = [statistics.mean(r["after"][i] for r in rows) for i in range(len(before))]
    result = {"runs": rows, "mean_before": before.mean().item(), "frozen_bge_mean": frozen.mean().item(),
        "mean_after": statistics.mean(mean_after), "paired_query_bootstrap": paired_bootstrap(before.tolist(), mean_after),
        "query_ids": data["validation"]["query_ids"], "api_calls": 0, "browsecomp_final_test_evaluated": False,
        "encoder_updates": 0, "reranker_policy_updates_per_seed": protocol["steps"], "protocol_digest": digest(protocol)}
    write_json(DIRECTORY / "comparison.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(train_only: bool = False, compact_candidates: bool = False):
    if compact_candidates:
        print("COMPACTED " + json.dumps(compact_saved.remote()), flush=True)
        return
    if not train_only:
        print("PREPARED " + json.dumps(prepare.remote()), flush=True)
    result = train.remote()
    print("COMPARISON " + json.dumps({k: v for k, v in result.items() if k not in {"runs", "query_ids"}}), flush=True)


@app.function(image=feature_image, cpu=2, memory=4096, volumes={ROOT: volume}, timeout=600)
def compact_saved():
    """Remove retained full-sort storage from saved top-25 tensor views."""
    import torch
    from embedding_rl.config import write_json

    def clone_tensors(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {k: clone_tensors(v) for k, v in value.items()}
        return value
    checks = []
    for name, key in (("candidates.pt", "candidates"), ("features.pt", "candidate_indices")):
        path = DIRECTORY / name
        original = torch.load(path, weights_only=True)
        compact = clone_tensors(original)
        if not torch.equal(original[key], compact[key]) or original["protocol_digest"] != compact["protocol_digest"]:
            raise ValueError("Compaction changed candidate values or provenance")
        before = path.stat().st_size
        temporary = path.with_suffix(".compact.pt")
        torch.save(compact, temporary)
        temporary.replace(path)
        checks.append({"file": name, "before_bytes": before, "after_bytes": path.stat().st_size,
                       "candidate_values_unchanged": True, "candidate_sha256": hashlib.sha256(compact[key].numpy().tobytes()).hexdigest()})
    result = {"files": checks, "reason": "Torch serialized a top-25 view with the full argsort backing storage; cloning preserves values and removes unused storage"}
    write_json(DIRECTORY / "compaction-verification.json", result)
    volume.commit()
    return result
