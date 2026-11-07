"""Persist reranker examples without mixing training, validation, or final test."""
import hashlib
import json
from pathlib import Path

import torch

from .config import Config, digest, write_json
from .data import load_prepared
from .train import make_encoder


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_jsonl(path, rows):
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def export_documents(bundle, config, selected):
    if config.dataset == "browsecomp-plus":
        from datasets import load_dataset
        corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train",
                              revision=bundle["revisions"]["corpus"])
        indices = [i for i, doc in enumerate(bundle["docs"]) if doc["id"] in selected]
        # Check IDs rather than silently relying on dataset row order.
        found = set()
        for row in corpus.select(indices):
            docid = str(row["docid"])
            if docid not in selected:
                raise ValueError("Corpus row order differs from prepared document index")
            found.add(docid)
            yield {"document_id": docid, "url": row["url"], "text": row["text"]}
        if found != selected:
            raise ValueError("Export is missing candidate page text")
    else:
        for doc in bundle["docs"]:
            if doc["id"] in selected:
                yield {"document_id": doc["id"], "title": doc["title"], "text": doc["text"]}


@torch.no_grad()
def export_reranker(root, run_id, k=25, device="cpu", encoder_factory=make_encoder):
    if Path(run_id).name != run_id or k < 1:
        raise ValueError("Require a run directory name and positive candidate count")
    run_dir = root / "runs" / run_id
    if not (run_dir / "summary.json").exists():
        raise ValueError("Only completed training runs can be exported")
    metadata = json.loads((run_dir / "run.json").read_text())
    config = Config(**metadata["config"])
    path, manifest, bundle, tensors = load_prepared(root, config)
    if manifest != metadata["prepared_manifest"]:
        raise ValueError("Run and prepared dataset provenance differ")
    if k > len(bundle["docs"]):
        raise ValueError("Candidate count exceeds corpus size")
    checkpoint_path = run_dir / "best.pt"
    checkpoint_hash = file_sha256(checkpoint_path)
    destination = root / "exports" / "reranker" / run_id / f"top-{k}"
    marker = destination / "manifest.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved["checkpoint_sha256"] != checkpoint_hash
                or saved["prepared_manifest_digest"] != digest(manifest)):
            raise ValueError("Export source changed; preserve the original run artifacts")
        for name, checksum in saved["files_sha256"].items():
            if file_sha256(destination / name) != checksum:
                raise ValueError(f"Export file changed: {name}")
        return {"directory": str(destination), **saved}

    destination.mkdir(parents=True, exist_ok=True)
    encoder = encoder_factory(config, manifest, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder.model.load_state_dict(checkpoint["state_dict"])
    selected = set()
    counts, with_evidence = {}, {}
    for split in ("train", "validation"):
        qids = bundle["splits"][split]
        queries = encoder.encode_batches([bundle["queries"][qid] for qid in qids],
                                         config.encode_batch_size, query=True)
        scores = queries @ tensors["documents"].T
        order = scores.argsort(dim=-1, descending=True, stable=True)[:, :k]
        rows = []
        for row_index, qid in enumerate(qids):
            candidates = []
            for rank, index in enumerate(order[row_index].tolist(), 1):
                doc = bundle["docs"][index]
                selected.add(doc["id"])
                candidates.append({
                    "document_id": doc["id"], "rank": rank,
                    "retrieval_score": scores[row_index, index].item(),
                    "relevance": bundle["qrels"][qid].get(doc["id"], 0),
                })
            rows.append({"query_id": qid, "query": bundle["queries"][qid],
                         "split": split, "candidates": candidates,
                         "known_relevant_count": len(bundle["qrels"][qid])})
        write_jsonl(destination / f"{split}.jsonl", rows)
        counts[split] = len(rows)
        with_evidence[split] = sum(any(c["relevance"] > 0 for c in row["candidates"])
                                   for row in rows)
    write_jsonl(destination / "documents.jsonl", export_documents(bundle, config, selected))
    filenames = ["train.jsonl", "validation.jsonl", "documents.jsonl"]
    result = {
        "schema": 1, "run_id": run_id, "candidate_count": k,
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "checkpoint_step": checkpoint["step"], "checkpoint_sha256": checkpoint_hash,
        "prepared": str(path.relative_to(root)), "prepared_manifest_digest": digest(manifest),
        "revisions": bundle["revisions"], "query_counts": counts,
        "queries_with_retrieved_evidence": with_evidence, "document_count": len(selected),
        "label_source": config.dataset + ("/evidence" if config.dataset == "browsecomp-plus"
                                          else "/qrels"),
        "label_note": "Unjudged documents receive 0; they are not verified negatives.",
        "candidate_note": "Natural top-k results; relevant documents are never injected.",
        "splits": ["train", "validation"], "test_exported": False,
        "files_sha256": {name: file_sha256(destination / name) for name in filenames},
    }
    # Completion marker comes last; partial exports can be safely retried.
    write_json(marker, result)
    return {"directory": str(destination), **result}
