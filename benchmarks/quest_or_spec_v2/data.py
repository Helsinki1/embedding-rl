"""Pinned shared inputs and split-isolation checks for every new run."""
import collections
import json
from pathlib import Path

from .build import COUNTS, HERE, PARTS, normalize, read_rows, sha256


def audit_splits(parts):
    ids, queries, components, positive_hashes = set(), set(), set(), set()
    audit = {}
    for part, rows in zip(PARTS, parts):
        current_ids = {r["id"] for r in rows}
        current_queries = {normalize(r["query"]) for r in rows}
        current_components = {normalize(c) for r in rows for c in r["components"]}
        current_positive = {d.split(":", 1)[1] for r in rows if r["source"] != "quest" for d in r["docs"]}
        if len(current_ids) != len(rows) or len(current_queries) != len(rows):
            raise ValueError("Duplicate query IDs or text")
        if ids & current_ids or queries & current_queries or components & current_components or positive_hashes & current_positive:
            raise ValueError("Cross-partition query, component, or synthetic-positive leakage")
        for r in rows:
            if len(r["components"]) not in (2, 3) or len(r["component_docs"]) != len(r["components"]):
                raise ValueError("Expected two or three labeled OR components")
            if not r["docs"] or set().union(*map(set, r["component_docs"])) != set(r["docs"]):
                raise ValueError("Union labels differ from the deduplicated component union")
        counts = dict(collections.Counter(r["source"] for r in rows))
        if counts != {s: n[PARTS.index(part)] for s, n in COUNTS.items()}:
            raise ValueError("Source mixture differs from the frozen specification")
        audit[part] = {"queries": len(rows), "sources": counts, "distinct_components": len(current_components)}
        ids |= current_ids
        queries |= current_queries
        components |= current_components
        positive_hashes |= current_positive
    return audit


def load_splits(root=HERE):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if sha256(root / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the accepted protocol")
    for name, expected in manifest["files"].items():
        if sha256(root / name) != expected:
            raise ValueError("Frozen dataset changed: " + name)
    parts = [read_rows(root / f"data/{p}.jsonl") for p in PARTS]
    audit = audit_splits(parts)
    return *parts, {"manifest_sha256": sha256(root / "manifest.json"), "protocol_sha256": sha256(root / "protocol.json"), "audit": audit}


def load_corpus(base, nq_dataset, nq_embeddings, ms_embeddings, device, dataset=HERE):
    import torch
    from benchmarks.quest_or_synthetic.data import load_augmented_corpus, load_added_vectors

    original, ids, old_provenance = load_augmented_corpus(base, nq_dataset, nq_embeddings, device)
    added, new_ids, new_provenance = load_added_vectors(dataset, ms_embeddings, device)
    if set(ids) & set(new_ids):
        raise ValueError("Corpus identity collision")
    return torch.cat((original, added)), ids + new_ids, {"existing": old_provenance, "msmarco": new_provenance}
