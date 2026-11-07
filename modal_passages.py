"""One predeclared, no-API lexical passage-scoring control."""
import json
import statistics

import modal

from modal_common import ROOT, volume
from modal_lexical import lexical_image

app = modal.App("embedding-rl-lexical-passages")
passage_image = lexical_image.add_local_python_source("lexical_passages", "modal_lexical")


@app.function(image=passage_image, cpu=8, memory=65536, volumes={ROOT: volume}, timeout=3600, max_containers=1)
def run():
    import bm25s
    import numpy as np
    import Stemmer
    import torch
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.objectives import ndcg
    from lexical_passages import PassageTokenRows, max_page_scores

    torch.set_num_threads(8)
    directory = ROOT / "experiments/passages-20261106"
    source = ROOT / "experiments/lexical-20261106"
    if (directory / "comparison.json").exists():
        return json.loads((directory / "comparison.json").read_text())["summary"]
    _, manifest, bundle, tensors = load_prepared(ROOT, base_config())
    qids = bundle["splits"]["validation"]
    count = len(bundle["docs"])
    if len(qids) != 83 or set(qids) & set(bundle["splits"]["test"]):
        raise ValueError("Unexpected validation split")
    source_protocol = json.loads((source / "protocol.json").read_text())
    progress = json.loads((source / "full-tokens/progress.json").read_text())
    if progress["protocol_digest"] != digest(source_protocol) or progress["done"] != count:
        raise ValueError("Incomplete or mismatched source token cache")
    protocol = {"prepared_digest": digest(manifest), "source_protocol_digest": digest(source_protocol),
        "document_ids_digest": digest([d["id"] for d in bundle["docs"]]), "bm25s": "0.3.16", "pystemmer": "3.1.0",
        "size": 512, "stride": 384, "token_unit": "Retained English-stemmed lexical tokens after stopword removal; not BGE tokens",
        "text": "Full scraped text, no title or URL enrichment", "aggregation": "Maximum passage score per original page",
        "empty_pages": "One empty passage", "k1": 1.5, "b": .75, "method": "lucene",
        "fusion_depth": 1000, "rrf_constant": 60, "fusion_weights": [1, 1],
        "dense_models": "Original and all three final full-corpus 2000-update encoders; cached unchanged vectors",
        "parameters_selected_before_evaluation": True, "api_calls": 0, "training_updates": 0,
        "validation_queries": 83, "browsecomp_final_test_evaluated": False, "external_datasets_evaluated": False,
        "limitations": "Changes scoring unit, term document frequency and length normalization together. Overlapping chunks give long pages more opportunities. An exploratory lexical MaxP adaptation, not a reproduction of neural MaxP or Late Chunking."}
    marker = directory / "protocol.json"
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError("Passage protocol changed")
    write_json(marker, protocol)
    volume.commit()
    offsets = np.load(source / "full-tokens/offsets.npy")
    mapped = np.memmap(source / "full-tokens/tokens.bin", dtype=np.int32, mode="r")
    rows = PassageTokenRows(mapped, offsets, size=512, stride=384)
    print(f"PASSAGES documents={count} chunks={len(rows)} retained_tokens={len(mapped)}", flush=True)
    tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=Stemmer.Stemmer("english"))
    tokenizer.load_vocab(source / "full-tokens")
    index_folder = directory / "index"
    if (index_folder / "provenance.json").exists():
        if json.loads((index_folder / "provenance.json").read_text())["protocol_digest"] != digest(protocol):
            raise ValueError("Index provenance changed")
        retriever = bm25s.BM25.load(str(index_folder), mmap=True, load_corpus=False)
    else:
        retriever = bm25s.BM25(k1=1.5, b=.75, method="lucene", backend="numpy")
        retriever.index(bm25s.tokenization.Tokenized(ids=rows, vocab=tokenizer.get_vocab_dict()), show_progress=False)
        retriever.save(str(index_folder))
        np.savez(directory / "passage-map.npz", starts=rows.starts, ends=rows.ends, documents=rows.documents)
        write_json(index_folder / "provenance.json", {"protocol_digest": digest(protocol), "chunks": len(rows), "documents": count})
        volume.commit()
        print("PASSAGE_INDEX_SAVED", flush=True)
    tokens = bm25s.tokenize([bundle["queries"][q] for q in qids], stopwords="en", stemmer=Stemmer.Stemmer("english"),
                            return_ids=False, show_progress=False)
    lexical = torch.from_numpy(np.stack([max_page_scores(retriever.get_scores(q) if q else np.zeros(len(rows), dtype=np.float32),
                                                         rows.documents, count) for q in tokens]))
    del retriever
    torch.save({"scores": lexical, "query_ids": qids, "protocol_digest": digest(protocol)}, directory / "scores.pt")
    grades = torch.zeros(len(qids), count)
    doc_index = {d["id"]: i for i, d in enumerate(bundle["docs"])}
    for i, qid in enumerate(qids):
        for docid, grade in bundle["qrels"][qid].items():
            grades[i, doc_index[docid]] = grade

    def measure(scores):
        order = scores.argsort(dim=-1, descending=True, stable=True)
        relevant = grades > 0
        recall = relevant.gather(1, order[:, :100]).sum(1) / relevant.sum(1).clamp_min(1)
        values = ndcg(scores, grades, 10)
        return {"ndcg": values.tolist(), "recall100": recall.tolist(), "mean_ndcg": values.mean().item(), "mean_recall100": recall.mean().item()}

    def reciprocal(scores):
        order = scores.argsort(dim=-1, descending=True, stable=True)[:, :1000]
        return torch.zeros_like(scores).scatter(1, order, (1 / (60 + torch.arange(1, 1001, dtype=torch.float32))).expand(len(qids), -1))

    runs = [{"model": "bm25", "seed": None, "variant": "passage_max", **measure(lexical)}]
    dense_saved = torch.load(source / "dense-rankings.pt", map_location="cpu", weights_only=True)
    if dense_saved["query_ids"] != qids:
        raise ValueError("Cached dense query order differs")
    previous = json.loads((source / "comparison.json").read_text())
    lexical_fusion = reciprocal(lexical)
    for model in dense_saved["models"]:
        scores = model["query_vectors"] @ tensors["documents"].T
        before = next(r for r in previous["runs"] if r["model"] == model["name"] and r["seed"] == model["seed"] and r["variant"] == "baseline")
        if np.max(np.abs(np.array(measure(scores)["ndcg"]) - np.array(before["ndcg"]))) > 1e-6:
            raise ValueError("Dense baseline changed")
        runs.append({"model": model["name"], "seed": model["seed"], "run_id": model["run_id"],
                     "variant": "passage_max", **measure(reciprocal(scores) + lexical_fusion)})
    summary = []
    for family in ("bm25", "original", "binary2000"):
        selected = [r for r in runs if r["model"] == family]
        values = [statistics.mean(r["ndcg"][i] for r in selected) for i in range(len(qids))]
        comparisons = {}
        for variant in (("first512", "full") if family == "bm25" else ("baseline", "first512", "full")):
            before = [r for r in previous["runs"] if r["model"] == family and r["variant"] == variant]
            comparisons[variant] = paired_bootstrap([statistics.mean(r["ndcg"][i] for r in before) for i in range(len(qids))], values)
        summary.append({"model": family, "mean_ndcg": statistics.mean(values),
                        "mean_recall100": statistics.mean(r["mean_recall100"] for r in selected), "comparisons": comparisons})
    write_json(directory / "comparison.json", {"summary": summary, "runs": runs, "query_ids": qids,
        "chunks": len(rows), "documents": count, "protocol_digest": digest(protocol), "api_calls": 0,
        "training_updates": 0, "browsecomp_final_test_evaluated": False})
    volume.commit()
    return summary


@app.function(image=passage_image, cpu=4, memory=32768, volumes={ROOT: volume}, timeout=600)
def verify():
    import bm25s
    import numpy as np
    import Stemmer
    import torch
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config
    from lexical_passages import max_page_scores

    directory = ROOT / "experiments/passages-20261106"
    _, _, bundle, _ = load_prepared(ROOT, base_config())
    saved = torch.load(directory / "scores.pt", map_location="cpu", weights_only=True)
    if saved["query_ids"] != bundle["splits"]["validation"]:
        raise ValueError("Query split differs")
    mapping = np.load(directory / "passage-map.npz")
    retriever = bm25s.BM25.load(str(directory / "index"), mmap=True, load_corpus=False)
    tokens = bm25s.tokenize([bundle["queries"][q] for q in saved["query_ids"]], stopwords="en", stemmer=Stemmer.Stemmer("english"),
                            return_ids=False, show_progress=False)
    values = np.stack([max_page_scores(retriever.get_scores(q) if q else np.zeros(len(mapping["documents"]), dtype=np.float32),
                                       mapping["documents"], len(bundle["docs"])) for q in tokens])
    error = float(np.max(np.abs(values - saved["scores"].numpy())))
    if error > 1e-6:
        raise ValueError("Reloaded passage scores differ")
    result = {"fresh_container_reload_max_error": error, "queries": len(tokens), "documents": len(bundle["docs"]),
              "passages": len(mapping["documents"]), "api_calls": 0, "browsecomp_final_test_evaluated": False}
    write_json(directory / "verification.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(verify_only: bool = False):
    if not verify_only:
        print("COMPARISON " + json.dumps(run.remote()), flush=True)
    print("VERIFIED " + json.dumps(verify.remote()), flush=True)
