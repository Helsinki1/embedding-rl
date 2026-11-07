"""Evaluate frozen interpretation checkpoints on the canonical OR retrieval suite."""
import argparse
import collections
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.quest_or_spec_v2.build import read_rows, sha256, write_json, write_rows
from benchmarks.quest_or_spec_v2.data import load_splits, load_corpus
from benchmarks.quest_or_spec_v2.metrics import evaluate
from benchmarks.quest_or_spec_v2.run import predictions, select_cutoff
from .model import InterpretationModel, MODEL_ID, REVISION, tokenize


def overlap_audit(rows, training, validation):
    def normalized(text):
        return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())

    lookup = collections.defaultdict(list)
    for split, source in (("training", training), ("checkpoint_validation", validation)):
        for row in source:
            for text in [row["query"], *row["interpretations"]]:
                lookup[normalized(text)].append({"split": split, "id": row["id"], "text": text})
    matches = []
    for row in rows:
        found = [{"component": text, "matches": lookup[normalized(text)]}
                 for text in row["components"] if normalized(text) in lookup]
        if found:
            matches.append({"id": row["id"], "source": row["source"], "components": found})
    return {"matching_rule": "casefold, punctuation removed, whitespace collapsed; exact text only, not semantic deduplication",
            "queries_with_overlap": len(matches),
            "by_source": dict(collections.Counter(x["source"] for x in matches)),
            "matches": matches}


@torch.inference_mode()
def retrieve(model, tokenizer, rows, documents, single=False):
    started = time.monotonic()
    rankings, vectors, cosines = [], [], []
    max_tokens = 0
    for start in range(0, len(rows), 16):
        tokens = tokenize(tokenizer, [r["query"] for r in rows[start:start+16]], "cuda")
        max_tokens = max(max_tokens, tokens.input_ids.shape[1])
        if single:
            embedded = F.normalize(model(**tokens, use_cache=False).last_hidden_state[:, -1].float(), dim=-1)[:, None]
        else:
            embedded = model(**tokens)
            pair = embedded @ embedded.transpose(-1, -2)
            cosines.extend(pair[:, [0, 0, 1], [1, 2, 2]].cpu().tolist())
        if not torch.isfinite(embedded).all():
            raise ValueError("Non-finite query embeddings")
        rankings.extend((embedded @ documents.T).topk(100, dim=-1).indices.cpu().tolist())
        vectors.append(embedded.cpu())
    torch.cuda.synchronize()
    return rankings, torch.cat(vectors), {
        "seconds": time.monotonic()-started, "queries": len(rows), "max_tokens": max_tokens,
        "mean_pairwise_slot_cosine": sum(sum(x) for x in cosines)/(3*len(cosines)) if cosines else None}


def main():
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    spec = args.bundle / "benchmarks/quest_or_spec_v2"
    nq = args.bundle / "benchmarks/quest_or_synthetic"
    _, validation, test, split_provenance = load_splits(spec)
    expected_hashes = json.loads((args.corpus / "source_checksums.json").read_text())
    corpus_hashes = {str(f.relative_to(args.corpus)): sha256(f)
                     for f in sorted(args.corpus.rglob("*")) if f.is_file()}
    if any(corpus_hashes[name] != expected for name, expected in expected_hashes.items()):
        raise ValueError("Document vectors differ from source-server checksums")
    documents, titles, corpus_provenance = load_corpus(
        args.corpus / "base", nq, args.corpus / "nq", args.corpus / "msmarco", "cuda", spec)
    expected_ids = json.loads((args.bundle / "reference/corpus_ids.json").read_text())
    if titles != expected_ids or len(titles) != 489188:
        raise ValueError("Corpus identity or order differs from the previous evaluation")
    title_set = set(titles)
    if any(set(r["docs"]) - title_set for r in validation+test):
        raise ValueError("Gold documents missing from corpus")
    ambig = args.bundle / "benchmarks/ambig_interpretations/data"
    overlap = overlap_audit(test, read_rows(ambig / "train.jsonl"), read_rows(ambig / "validation.jsonl"))
    write_json(args.output / "overlap_audit.json", overlap)
    write_json(args.output / "provenance.json", {
        "model": MODEL_ID, "revision": REVISION, "dataset": split_provenance,
        "corpus": corpus_provenance, "document_count": len(titles),
        "corpus_ids_sha256": sha256(args.bundle / "reference/corpus_ids.json"),
        "corpus_file_sha256": corpus_hashes,
        "code_sha256": {str(f.relative_to(args.bundle)): sha256(f)
                        for f in sorted((args.bundle / "benchmarks").rglob("*.py"))},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "training_updates": 0, "test_used_for_selection": False,
        "checkpoint_policy": "Existing AmbigQA validation-loss selections, unchanged",
        "cutoff_policy": "Validation union F1; single k=1..60, three vectors k=1..20 each; deduplicate",
        "fixed_budgets": [10, 20, 60, 100]})
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    base = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to("cuda").eval()
    all_results = {}
    for architecture in ("single_frozen", "mtp", "autoregressive"):
        output = args.output / architecture
        output.mkdir()
        if architecture == "single_frozen":
            model = base
            checkpoint = None
        else:
            if architecture == "mtp":
                base = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
            model = InterpretationModel(base, tokenizer.eos_token_id, architecture)
            checkpoint = args.checkpoints / architecture / "best.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            expected = {n for n, _ in model.named_parameters() if "lora_" in n or n == "slots"}
            if set(state) != expected:
                raise ValueError("Checkpoint parameter mismatch")
            model.load_state_dict(state, strict=False)
        model.eval()
        print(json.dumps({"stage": "validation", "architecture": architecture}), flush=True)
        single = architecture == "single_frozen"
        val_ranks, val_vectors, val_timing = retrieve(model, tokenizer, validation, documents, single)
        selected, curve = select_cutoff(validation, val_ranks, titles, 60 if single else 20)
        selection = {"retrieval": selected, "validation_curve": curve, "selected_at": time.time(),
                     "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                     "checkpoint_selection": json.loads((checkpoint.parent / "selection.json").read_text()) if checkpoint else None}
        # Persist the frozen cutoff before opening the test inference path.
        write_json(output / "selection.json", selection)
        write_json(output / "validation_rankings.json", {"ids": [r["id"] for r in validation], "corpus_indices": val_ranks})
        print(json.dumps({"stage": "test", "architecture": architecture, "selection": selected}), flush=True)
        ranks, vectors, timing = retrieve(model, tokenizer, test, documents, single)
        pred = predictions(test, ranks, titles, k=selected["k"])
        summary, details = evaluate(test, pred)
        fixed = {str(b): evaluate(test, predictions(test, ranks, titles, budget=b))[0] for b in (10, 20, 60, 100)}
        excluded = {r["id"] for r in overlap["matches"]}
        clean_rows = [r for r in test if r["id"] not in excluded]
        clean_pred = [r for r in pred if r["id"] not in excluded]
        summary.update({"architecture": architecture, "selection": selection, "fixed_budget": fixed,
                        "timing": timing, "validation_timing": val_timing,
                        "no_exact_training_or_checkpoint_validation_overlap": evaluate(clean_rows, clean_pred)[0],
                        "training_updates": 0})
        write_rows(output / "test_predictions.jsonl", pred)
        write_rows(output / "test_details.jsonl", details)
        write_json(output / "test_rankings.json", {"ids": [r["id"] for r in test], "corpus_indices": ranks})
        torch.save({"ids": [r["id"] for r in test], "embeddings": vectors}, output / "test_embeddings.pt")
        torch.save({"ids": [r["id"] for r in validation], "embeddings": val_vectors}, output / "validation_embeddings.pt")
        write_json(output / "summary.json", summary)
        all_results[architecture] = summary
        print(json.dumps({"stage": "complete", "architecture": architecture, "overall": summary["overall"], "timing": timing}), flush=True)
    old = json.loads((args.bundle / "reference/single_frozen_summary.json").read_text())
    baseline = all_results["single_frozen"]
    consistency = {"previous_k": old["selection"]["k"], "current_k": baseline["selection"]["retrieval"]["k"],
                   "previous_test_f1": old["overall"]["union"]["f1"], "current_test_f1": baseline["overall"]["union"]["f1"]}
    write_json(args.output / "comparison.json", {"models": all_results, "baseline_reproduction": consistency,
               "overlap_queries": overlap["queries_with_overlap"], "elapsed_seconds": time.monotonic()-started})
    print(json.dumps({"stage": "finished", **consistency, "elapsed_seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
