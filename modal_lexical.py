"""No-API lexical and hybrid controls, including a document-context ablation."""
import json
from pathlib import Path
import statistics

import modal

from modal_common import ROOT, volume

app = modal.App("embedding-rl-lexical-control")
lexical_image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    .pip_install("bm25s==0.3.16", "PyStemmer==3.1.0")
    .env({"HF_HOME": "/data/huggingface", "TOKENIZERS_PARALLELISM": "true", "RAYON_NUM_THREADS": "8"})
    .add_local_python_source("embedding_rl", "modal_common", "lexical_tokens"))


@app.function(image=lexical_image, cpu=8, memory=32768, volumes={ROOT: volume},
              timeout=3600, max_containers=1)
def run():
    import bm25s
    import numpy as np
    import Stemmer
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.objectives import ndcg
    from embedding_rl.train import make_encoder
    from lexical_tokens import tokenize_to_disk

    torch.set_num_threads(8)
    directory = ROOT / "experiments/lexical-20261106"
    if (directory / "comparison.json").exists():
        return json.loads((directory / "comparison.json").read_text())["summary"]
    config = base_config()
    _, manifest, bundle, tensors = load_prepared(ROOT, config)
    qids = bundle["splits"]["validation"]
    if len(qids) != 83 or set(qids) & set(bundle["splits"]["test"]):
        raise ValueError("Unexpected validation split")
    protocol = {"prepared_digest": digest(manifest), "document_ids_digest": digest([d["id"] for d in bundle["docs"]]),
        "bm25s": "0.3.16", "pystemmer": "3.1.0", "stopwords": "en", "stemmer": "english",
        "method": "lucene", "k1": 1.5, "b": .75, "rrf_constant": 60, "fusion_depth": 1000,
        "fusion_weights": [1, 1], "context_modes": ["first512", "full"],
        "first512_definition": "Decode the exact pretrained BGE tokenizer's max_length512 truncated input, omitting special tokens; tokenize that visible text for BM25. This approximates equal visible content, not equal tokenization.",
        "full_definition": "Full scraped text only, without URL/title enrichment",
        "dense_models": "Original encoder and all three longer-20261106 final2000-step models",
        "parameters_selected_before_evaluation": True, "api_calls": 0, "training_updates": 0,
        "validation_queries": 83, "test_evaluated": False,
        "limitations": "Full-text lexical retrieval has a larger document-context budget than dense retrieval. RRF is an inference control, not an RL gain. English stemming may disadvantage non-English pages."}
    write_json(directory / "protocol.json", protocol)
    volume.commit()
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", revision=bundle["revisions"]["corpus"])
    if [str(x) for x in corpus["docid"]] != [d["id"] for d in bundle["docs"]]:
        raise ValueError("Corpus document order changed")
    document_count = len(corpus)
    print(f"CORPUS documents={document_count} arrow_gb={corpus.data.nbytes / 1e9:.3f}", flush=True)
    stemmer = Stemmer.Stemmer("english")
    smoke = bm25s.BM25(k1=1.5, b=.75, method="lucene", backend="numpy")
    smoke.index(bm25s.tokenize(["cats sleep", "fish swim"], stemmer=stemmer, show_progress=False), show_progress=False)
    smoke_query = bm25s.tokenize(["fish"], stemmer=stemmer, return_ids=False, show_progress=False)[0]
    if smoke.get_scores(smoke_query).argmax() != 1:
        raise ValueError("Pinned lexical dependency failed its retrieval smoke check")
    query_tokens = bm25s.tokenize([bundle["queries"][q] for q in qids], stopwords="en",
                                   stemmer=stemmer, return_ids=False, show_progress=False)
    lexical = {}
    for mode in ("first512", "full"):
        folder = directory / f"index-{mode}"
        marker = folder / "provenance.json"
        if marker.exists():
            if json.loads(marker.read_text())["protocol_digest"] != digest(protocol):
                raise ValueError("Cached lexical index differs")
            retriever = bm25s.BM25.load(str(folder), load_corpus=False, mmap=True)
        else:
            print("INDEX_START " + mode, flush=True)
            if mode == "first512":
                tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=manifest["revisions"]["model"])
                visible = []
                prefix_folder = directory / "prefix512"
                for start in range(0, document_count, 2048):
                    prefix_file = prefix_folder / f"{start:06d}.json"
                    if prefix_file.exists():
                        cached = json.loads(prefix_file.read_text())
                        if cached["protocol_digest"] != digest(protocol):
                            raise ValueError("Prefix cache differs")
                        prefix = cached["texts"]
                    else:
                        prefix = []
                        for batch_start in range(start, min(start + 2048, document_count), 16):
                            batch_end = min(batch_start + 16, start + 2048, document_count)
                            encoded = tokenizer(corpus[batch_start:batch_end]["text"], truncation=True, max_length=512)
                            prefix.extend(tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True))
                        write_json(prefix_file, {"protocol_digest": digest(protocol), "texts": prefix})
                        volume.commit()
                        print(f"PREFIX_SAVED {start + len(prefix)}/{document_count}", flush=True)
                    visible.extend(prefix)
                tokens = bm25s.tokenize(visible, stopwords="en", stemmer=stemmer, show_progress=False)
                del visible
            else:
                tokens = tokenize_to_disk(lambda start, end: corpus[start:end]["text"], document_count,
                                          directory / "full-tokens", digest(protocol), volume.commit)
            retriever = bm25s.BM25(k1=1.5, b=.75, method="lucene", backend="numpy")
            retriever.index(tokens, show_progress=False)
            retriever.save(str(folder))
            write_json(marker, {"protocol_digest": digest(protocol), "documents": document_count, "mode": mode})
            del tokens
            volume.commit()
            print("INDEX_DONE " + mode, flush=True)
        lexical[mode] = torch.from_numpy(np.stack([
            retriever.get_scores(q) if q else np.zeros(document_count, dtype=np.float32) for q in query_tokens]))
        del retriever
    torch.save({"scores": lexical, "query_ids": qids, "protocol_digest": digest(protocol)}, directory / "lexical-scores.pt")
    documents = tensors["documents"]
    grades = torch.zeros(len(qids), len(documents))
    index = {d["id"]: i for i, d in enumerate(bundle["docs"])}
    for i, qid in enumerate(qids):
        for docid, grade in bundle["qrels"][qid].items():
            grades[i, index[docid]] = grade

    def measure(scores):
        values = ndcg(scores, grades, 10)
        order = scores.argsort(dim=-1, descending=True, stable=True)
        recall = (grades > 0).gather(1, order[:, :100]).sum(1) / (grades > 0).sum(1).clamp_min(1)
        return {"ndcg": values.tolist(), "recall100": recall.tolist(),
                "mean_ndcg": values.mean().item(), "mean_recall100": recall.mean().item()}

    def reciprocal(scores):
        order = scores.argsort(dim=-1, descending=True, stable=True)[:, :1000]
        values = 1 / (60 + torch.arange(1, 1001, dtype=torch.float32))
        result = torch.zeros_like(scores)
        return result.scatter(1, order, values.expand(len(scores), -1))

    rows = [{"model": "bm25", "seed": None, "variant": mode, **measure(scores)} for mode, scores in lexical.items()]
    fusion = {mode: reciprocal(scores) for mode, scores in lexical.items()}
    models = [{"name": "original", "seed": None, "run_id": None, "expected": .037363700568675995}]
    for seed in (42, 43, 44):
        record = json.loads((ROOT / "experiments/longer-20261106" / f"both-2000-seed{seed}.json").read_text())
        models.append({"name": "binary2000", "seed": seed, "run_id": record["summary"]["run_id"],
                       "expected": record["summary"]["final_validation_ndcg"]})
    dense_rankings = []
    with torch.no_grad():
        for model in models:
            encoder = make_encoder(config, manifest, "cpu")
            if model["run_id"]:
                saved = torch.load(ROOT / "runs" / model["run_id"] / "last.pt", map_location="cpu", weights_only=True)
                encoder.model.load_state_dict(saved["state_dict"])
            queries = encoder.encode_batches([bundle["queries"][q] for q in qids], 32, query=True)
            scores = queries @ documents.T
            baseline = measure(scores)
            if abs(baseline["mean_ndcg"] - model["expected"]) > 1e-5:
                raise ValueError("Dense control changed")
            dense_rankings.append({**model, "query_vectors": queries, "top1000": scores.argsort(dim=-1, descending=True, stable=True)[:, :1000]})
            dense = reciprocal(scores)
            for variant, combined in {"baseline": scores, **{mode: dense + v for mode, v in fusion.items()}}.items():
                measured = baseline if variant == "baseline" else measure(combined)
                rows.append({"model": model["name"], "seed": model["seed"], "run_id": model["run_id"],
                             "variant": variant, **measured})
                print("MEASURED " + json.dumps({k: v for k, v in rows[-1].items() if k not in {"ndcg", "recall100"}}), flush=True)
            del encoder
    summary = []
    for model in ("bm25", "original", "binary2000"):
        variants = ("first512", "full") if model == "bm25" else ("baseline", "first512", "full")
        for variant in variants:
            selected = [r for r in rows if r["model"] == model and r["variant"] == variant]
            values = [statistics.mean(r["ndcg"][i] for r in selected) for i in range(len(qids))]
            baseline_rows = [r for r in rows if r["model"] == model and r["variant"] == ("first512" if model == "bm25" else "baseline")]
            before = [statistics.mean(r["ndcg"][i] for r in baseline_rows) for i in range(len(qids))]
            summary.append({"model": model, "variant": variant, "mean_ndcg": statistics.mean(values),
                            "paired_query_bootstrap": paired_bootstrap(before, values)})
    torch.save({"models": dense_rankings, "query_ids": qids}, directory / "dense-rankings.pt")
    write_json(directory / "comparison.json", {"summary": summary, "runs": rows, "query_ids": qids,
        "test_evaluated": False, "api_calls": 0, "training_updates": 0, "protocol_digest": digest(protocol)})
    volume.commit()
    return summary


@app.local_entrypoint()
def main(verify_only: bool = False):
    if not verify_only:
        print("COMPARISON " + json.dumps(run.remote()), flush=True)
    print("VERIFIED " + json.dumps(verify_saved.remote()), flush=True)


@app.function(image=lexical_image, cpu=4, memory=16384, volumes={ROOT: volume}, timeout=600)
def verify_saved():
    import bm25s
    import numpy as np
    import Stemmer
    import torch
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config

    directory = ROOT / "experiments/lexical-20261106"
    _, _, bundle, _ = load_prepared(ROOT, base_config())
    saved = torch.load(directory / "lexical-scores.pt", map_location="cpu", weights_only=True)
    qids = saved["query_ids"]
    if qids != bundle["splits"]["validation"]:
        raise ValueError("Saved query split differs")
    tokens = bm25s.tokenize([bundle["queries"][q] for q in qids], stopwords="en",
                            stemmer=Stemmer.Stemmer("english"), return_ids=False, show_progress=False)
    checks = []
    for mode in ("first512", "full"):
        model = bm25s.BM25.load(str(directory / f"index-{mode}"), load_corpus=False, mmap=True)
        values = np.stack([model.get_scores(q) if q else np.zeros(len(bundle["docs"]), dtype=np.float32) for q in tokens])
        error = float(np.abs(values - saved["scores"][mode].numpy()).max())
        if error > 1e-6:
            raise ValueError("Reloaded index scores differ")
        checks.append({"mode": mode, "max_score_error": error, "queries": len(qids), "documents": len(bundle["docs"])})
    result = {"fresh_container_reload": checks, "api_calls": 0, "test_evaluated": False}
    write_json(directory / "verification.json", result)
    volume.commit()
    return result


@app.function(image=lexical_image, cpu=4, memory=16384, volumes={ROOT: volume}, timeout=900)
def verify_full_tokenization():
    """Check cached full-text token IDs against independent functional tokenization."""
    from collections import Counter
    import bm25s
    import numpy as np
    import Stemmer
    import torch
    from datasets import load_dataset
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config

    directory = ROOT / "experiments/lexical-20261106"
    _, _, bundle, _ = load_prepared(ROOT, base_config())
    scores = torch.load(directory / "lexical-scores.pt", weights_only=True)["scores"]["full"]
    # Include random pages and high-scoring pages; no relevance labels select this sample.
    selected = set(np.random.default_rng(42).choice(len(bundle["docs"]), 32, replace=False).tolist())
    selected.update(scores[:8].argsort(dim=-1, descending=True, stable=True)[:, :3].flatten().tolist())
    selected = sorted(selected)
    token_folder = directory / "full-tokens"
    progress = json.loads((token_folder / "progress.json").read_text())
    if progress["done"] != len(bundle["docs"]):
        raise ValueError("Token stream is incomplete")
    offsets = np.load(token_folder / "offsets.npy")
    mapped = np.memmap(token_folder / "tokens.bin", dtype=np.int32, mode="r")
    tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=Stemmer.Stemmer("english"))
    tokenizer.load_vocab(token_folder)
    vocabulary = tokenizer.get_vocab_dict()
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", revision=bundle["revisions"]["corpus"])
    reference = bm25s.tokenize([corpus[i]["text"] for i in selected], stopwords="en",
                              stemmer=Stemmer.Stemmer("english"), return_ids=False, show_progress=False,
                              allow_empty=False)
    checked = 0
    for di, terms in zip(selected, reference, strict=True):
        expected = Counter(vocabulary[term] for term in terms)
        actual = Counter(mapped[offsets[di]:offsets[di + 1]].tolist())
        if actual != expected:
            raise ValueError(f"Full-text tokenization differs for document index{di}")
        checked += sum(actual.values())
    result = {"documents": len(selected), "tokens_checked": checked, "all_term_counts_equal": True,
              "selection": "32 seeded random documents plus top3 full-BM25 results for first8 validation queries; no labels",
              "streaming_tokenizer": "bm25s.Tokenizer.streaming_tokenize", "reference_tokenizer": "bm25s.tokenize",
              "api_calls": 0, "test_evaluated": False}
    write_json(directory / "tokenization-verification.json", result)
    volume.commit()
    return result
