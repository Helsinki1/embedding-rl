"""Frozen cross-encoder baseline; deduplicated persistent query/document scoring."""
import json
import statistics

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-frozen-reranker")
MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@app.function(image=image.add_local_python_source("reranking_control"), gpu="L4", cpu=4,
              memory=24576, volumes={ROOT: volume}, timeout=7200, max_containers=1)
def run(include_lexical: bool = False, stronger_hybrids: bool = False):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from reranking_control import reranked_ndcg

    source_directory = ROOT / "experiments/frozen-reranker-20261106"
    directory = ROOT / "experiments/hybrid-reranker-20261106" if stronger_hybrids else source_directory
    config = base_config()
    torch.set_num_threads(4)
    _, manifest, bundle, _ = load_prepared(ROOT, config)
    qids = bundle["splits"]["validation"]
    if len(qids) != 83 or set(qids) & set(bundle["splits"]["test"]):
        raise ValueError("Unexpected validation split")
    protocol = {"model": MODEL, "revision": REVISION, "max_lengths": [512, 2048],
        "dtype": "float16", "candidate_count": 25, "ranking_ties": "preserve initial candidate order",
        "text": "Full scraped text, tokenizer pair truncation=longest_first; no URL/title enrichment",
        "systems": "All14 candidate-coverage dense/feedback systems, plus full-text BM25 and all3 full2000/full-BM25 RRF controls",
        "parameters_selected_before_evaluation": True, "api_calls": 0, "training_updates": 0,
        "validation_queries": 83, "test_evaluated": False, "prepared_digest": digest(manifest),
        "caveats": "Frozen pretrained reranker, not RL training. 2048-token pairs see more document text than the512-token dense index. Tokenizers differ. Benchmark labels are used only for evaluation."}
    source_protocol = dict(protocol)
    if stronger_hybrids:
        protocol["systems"] = "All three final full2000 encoders fused separately with prefix BM25 and passage-max BM25; six systems"
        protocol["adaptive_followup"] = "Selected these retrieval recipes after earlier validation experiments. Both existing reranker lengths are retained; repeated-validation results are exploratory."
        if json.loads((source_directory / "protocol.json").read_text()) != source_protocol:
            raise ValueError("Source score protocol changed")
        if json.loads((source_directory / "complete-candidates.json").read_text())["query_ids"] != qids:
            raise ValueError("Source score query indices changed")
    protocol_digest = digest(protocol)
    if (directory / "protocol.json").exists() and json.loads((directory / "protocol.json").read_text()) != protocol:
        raise ValueError("Reranker protocol changed")
    write_json(directory / "protocol.json", protocol)
    volume.commit()
    saved = torch.load(ROOT / "experiments/candidate-coverage-20261106/validation-vectors.pt", weights_only=True)
    if saved["query_ids"] != qids:
        raise ValueError("Cached validation IDs changed")
    systems = [{"family": m["family"], "seed": m["seed"], "variant": m["variant"],
                "run_id": m["run_id"], "candidates": m["top100"][:, :25]}
               for m in saved["models"]]
    expected_before = {}
    if stronger_hybrids:
        systems = []
        lexical_dir = ROOT / "experiments/lexical-20261106"
        prefix = torch.load(lexical_dir / "lexical-scores.pt", weights_only=True)
        passages = torch.load(ROOT / "experiments/passages-20261106/scores.pt", weights_only=True)
        dense = torch.load(lexical_dir / "dense-rankings.pt", weights_only=True)
        if any(saved["query_ids"] != qids for saved in (prefix, passages, dense)):
            raise ValueError("Hybrid source query IDs changed")
        comparisons = {
            "rrf_first512": json.loads((lexical_dir / "comparison.json").read_text()),
            "rrf_passage_max": json.loads((ROOT / "experiments/passages-20261106/comparison.json").read_text())}
        reciprocal = 1 / (60 + torch.arange(1, 1001, dtype=torch.float32))
        for variant, scores in (("rrf_first512", prefix["scores"]["first512"]), ("rrf_passage_max", passages["scores"])):
            order = scores.argsort(dim=-1, descending=True, stable=True)[:, :1000]
            lexical_rrf = torch.zeros_like(scores).scatter(1, order, reciprocal.expand(len(qids), -1))
            for model in dense["models"]:
                if model["name"] != "binary2000":
                    continue
                fused = lexical_rrf + torch.zeros_like(scores).scatter(1, model["top1000"], reciprocal.expand(len(qids), -1))
                systems.append({"family": "full2000", "seed": model["seed"], "variant": variant,
                    "run_id": model["run_id"], "candidates": fused.argsort(dim=-1, descending=True, stable=True)[:, :25]})
                expected_before[(variant, model["seed"])] = next(r["ndcg"] for r in comparisons[variant]["runs"]
                    if r["model"] == "binary2000" and r["seed"] == model["seed"] and r["variant"] == variant.removeprefix("rrf_"))
    elif include_lexical:
        lexical_dir = ROOT / "experiments/lexical-20261106"
        lexical = torch.load(lexical_dir / "lexical-scores.pt", weights_only=True)
        dense = torch.load(lexical_dir / "dense-rankings.pt", weights_only=True)
        if lexical["query_ids"] != qids or dense["query_ids"] != qids:
            raise ValueError("Lexical query IDs changed")
        scores = lexical["scores"]["full"]
        order = scores.argsort(dim=-1, descending=True, stable=True)[:, :1000]
        systems.append({"family": "bm25", "seed": None, "variant": "full", "run_id": None, "candidates": order[:, :25]})
        reciprocal = 1 / (60 + torch.arange(1, 1001, dtype=torch.float32))
        lexical_rrf = torch.zeros_like(scores).scatter(1, order, reciprocal.expand(len(qids), -1))
        for model in dense["models"]:
            if model["name"] != "binary2000":
                continue
            fused = lexical_rrf + torch.zeros_like(scores).scatter(1, model["top1000"], reciprocal.expand(len(qids), -1))
            systems.append({"family": "full2000", "seed": model["seed"], "variant": "rrf_full",
                            "run_id": model["run_id"], "candidates": fused.argsort(dim=-1, descending=True, stable=True)[:, :25]})
    pair_ids = sorted({(qi, di) for system in systems for qi, ids in enumerate(system["candidates"].tolist()) for di in ids})
    print(f"PAIRS systems={len(systems)} unique={len(pair_ids)}", flush=True)
    complete = include_lexical or stronger_hybrids
    write_json(directory / ("complete-candidates.json" if complete else "dense-candidates.json"),
               {"protocol_digest": protocol_digest, "query_ids": qids,
                "systems": [{**s, "candidates": s["candidates"].tolist()} for s in systems]})
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", revision=bundle["revisions"]["corpus"])
    if [str(x) for x in corpus["docid"]] != [d["id"] for d in bundle["docs"]]:
        raise ValueError("Corpus ordering changed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16).to("cuda").eval()
    grades = torch.zeros(len(qids), len(bundle["docs"]))
    doc_index = {d["id"]: i for i, d in enumerate(bundle["docs"])}
    for i, qid in enumerate(qids):
        for docid, grade in bundle["qrels"][qid].items():
            grades[i, doc_index[docid]] = grade
    rows = []
    for length in (512, 2048):
        cache_file = directory / f"scores-{length}.json"
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {"protocol_digest": protocol_digest, "scores": {}}
        if stronger_hybrids and not cache_file.exists():
            source_cache = json.loads((source_directory / f"scores-{length}.json").read_text())
            if source_cache["protocol_digest"] != digest(source_protocol):
                raise ValueError("Source logits provenance changed")
            cache["scores"] = {f"{qi}:{di}": source_cache["scores"][f"{qi}:{di}"] for qi, di in pair_ids
                               if f"{qi}:{di}" in source_cache["scores"]}
            cache["source_protocol_digest"] = digest(source_protocol)
            cache["reused_pairs"] = len(cache["scores"])
            write_json(cache_file, cache)
            volume.commit()
        if cache["protocol_digest"] != protocol_digest:
            raise ValueError("Cached scores have different provenance")
        pending = [(qi, di) for qi, di in pair_ids if f"{qi}:{di}" not in cache["scores"]]
        batch_size = 16 if length == 512 else 4
        with torch.inference_mode():
            for start in range(0, len(pending), batch_size):
                ids = pending[start:start + batch_size]
                pairs = [[bundle["queries"][qids[qi]], corpus[di]["text"]] for qi, di in ids]
                inputs = tokenizer(pairs, padding=True, truncation=True, max_length=length, return_tensors="pt").to("cuda")
                values = model(**inputs, return_dict=True).logits.reshape(-1).float().cpu()
                if not torch.isfinite(values).all() or len(values) != len(ids):
                    raise ValueError("Nonfinite or malformed reranker outputs")
                for (qi, di), value in zip(ids, values.tolist(), strict=True):
                    cache["scores"][f"{qi}:{di}"] = value
                if (start + len(ids)) % 256 == 0 or start + len(ids) == len(pending):
                    write_json(cache_file, cache)
                    volume.commit()
                    print(f"SCORED length={length} cached={len(cache['scores'])} target={len(pair_ids)}", flush=True)
        for system in systems:
            candidates = system["candidates"]
            logits = torch.tensor([[cache["scores"][f"{qi}:{di}"] for di in ids] for qi, ids in enumerate(candidates.tolist())])
            before = reranked_ndcg(candidates, -torch.arange(25).float().expand(len(qids), -1), grades)
            if stronger_hybrids and not torch.allclose(before, torch.tensor(expected_before[(system["variant"], system["seed"])]), atol=1e-6, rtol=0):
                raise ValueError("Hybrid baseline differs from its completed control")
            after = reranked_ndcg(candidates, logits, grades)
            oracle = reranked_ndcg(candidates, grades.gather(1, candidates), grades)
            if (after > oracle + 1e-6).any():
                raise ValueError("Reranker exceeded candidate oracle")
            rows.append({**{k: v for k, v in system.items() if k != "candidates"}, "max_length": length,
                         "before": before.tolist(), "after": after.tolist(), "oracle": oracle.tolist(),
                         "mean_before": before.mean().item(), "mean_after": after.mean().item()})
    summary = []
    for family, variant, length in sorted({(r["family"], r["variant"], r["max_length"]) for r in rows}):
        chosen = [r for r in rows if (r["family"], r["variant"], r["max_length"]) == (family, variant, length)]
        before = [statistics.mean(r["before"][i] for r in chosen) for i in range(len(qids))]
        after = [statistics.mean(r["after"][i] for r in chosen) for i in range(len(qids))]
        summary.append({"family": family, "variant": variant, "max_length": length,
                        "mean_before": statistics.mean(before), "mean_after": statistics.mean(after),
                        "paired_query_bootstrap": paired_bootstrap(before, after)})
    context_comparisons = []
    for family, variant in sorted({(r["family"], r["variant"]) for r in rows}):
        values = {}
        for length in (512, 2048):
            chosen = [r for r in rows if (r["family"], r["variant"], r["max_length"]) == (family, variant, length)]
            values[length] = [statistics.mean(r["after"][i] for r in chosen) for i in range(len(qids))]
        context_comparisons.append({"family": family, "variant": variant,
                                    "longer_minus_shorter": paired_bootstrap(values[512], values[2048])})
    result = {"summary": summary, "context_comparisons": context_comparisons,
              "runs": rows, "query_ids": qids, "unique_pairs": len(pair_ids),
              "complete": complete, "api_calls": 0, "training_updates": 0, "test_evaluated": False,
              "protocol_digest": protocol_digest}
    write_json(directory / ("comparison.json" if complete else "dense-comparison.json"), result)
    volume.commit()
    return {"summary": summary, "unique_pairs": len(pair_ids), "complete": complete}


@app.local_entrypoint()
def main(include_lexical: bool = False, stronger_hybrids: bool = False, verify_hybrids: bool = False):
    if verify_hybrids:
        print("VERIFIED " + json.dumps(verify_saved_hybrids.remote()), flush=True)
    else:
        print("COMPARISON " + json.dumps(run.remote(include_lexical, stronger_hybrids)), flush=True)


@app.function(image=image.add_local_python_source("reranking_control"), cpu=4,
              memory=16384, volumes={ROOT: volume}, timeout=600)
def verify_saved_hybrids():
    import torch
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config
    from reranking_control import reranked_ndcg

    directory = ROOT / "experiments/hybrid-reranker-20261106"
    _, _, bundle, _ = load_prepared(ROOT, base_config())
    result = json.loads((directory / "comparison.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    saved = json.loads((directory / "complete-candidates.json").read_text())
    qids = bundle["splits"]["validation"]
    if saved["query_ids"] != qids or result["query_ids"] != qids or len(result["runs"]) != 12:
        raise ValueError("Incomplete or mismatched saved evaluation")
    grades = torch.zeros(len(qids), len(bundle["docs"]))
    index = {d["id"]: i for i, d in enumerate(bundle["docs"])}
    for i, qid in enumerate(qids):
        for docid, grade in bundle["qrels"][qid].items():
            grades[i, index[docid]] = grade
    errors, counts = [], {}
    for length in (512, 2048):
        cache = json.loads((directory / f"scores-{length}.json").read_text())
        if cache["protocol_digest"] != digest(protocol) or saved["protocol_digest"] != digest(protocol):
            raise ValueError("Cached protocol differs")
        counts[length] = {"cached": len(cache["scores"]), "reused": cache["reused_pairs"]}
        for system in saved["systems"]:
            candidates = torch.tensor(system["candidates"])
            logits = torch.tensor([[cache["scores"][f"{qi}:{di}"] for di in ids] for qi, ids in enumerate(system["candidates"])])
            actual = {"before": reranked_ndcg(candidates, -torch.arange(25).float().expand(len(qids), -1), grades),
                      "after": reranked_ndcg(candidates, logits, grades),
                      "oracle": reranked_ndcg(candidates, grades.gather(1, candidates), grades)}
            row = next(r for r in result["runs"] if r["seed"] == system["seed"] and r["variant"] == system["variant"] and r["max_length"] == length)
            errors.extend((values - torch.tensor(row[name])).abs().max().item() for name, values in actual.items())
    if max(errors) > 1e-6:
        raise ValueError("Cached scores do not reproduce metrics")
    verification = {"fresh_container_recomputed_metric_max_error": max(errors), "conditions": 12,
                    "queries": len(qids), "pairs": counts, "api_calls": 0, "test_evaluated": False}
    write_json(directory / "verification.json", verification)
    volume.commit()
    return verification
