"""Load the additive dataset and frozen document vectors without changing QUEST."""
import json
from pathlib import Path

from .build import HERE, frozen_base, read_rows, sha256


def load_bundle(dataset=HERE, benchmark=None, splits=None):
    dataset = Path(dataset)
    benchmark = Path(benchmark) if benchmark else HERE.parent / "quest_or"
    splits = Path(splits) if splits else HERE.parent / "quest_or_training/split.json"
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(dataset / name) != expected:
            raise ValueError("Frozen synthetic dataset changed: " + name)
    if sha256(splits) != manifest["base_split_sha256"]:
        raise ValueError("Base QUEST training split changed")
    (original, validation, test), _ = frozen_base(benchmark, splits)
    train = read_rows(dataset / "data/train.jsonl")
    if train[:len(original)] != original:
        raise ValueError("Combined training data must preserve original training rows")
    if (read_rows(dataset / "data/validation.jsonl") != validation
            or read_rows(dataset / "data/test.jsonl") != test):
        raise ValueError("Evaluation query/label rows changed")
    if len(train) != manifest["counts"]["combined_train"]:
        raise ValueError("Combined training count mismatch")
    return train, validation, test, manifest


def load_added_vectors(dataset, embeddings, device, dimension=4096):
    import numpy as np
    import torch
    import torch.nn.functional as F

    dataset, embeddings = Path(dataset), Path(embeddings)
    manifest = json.loads((dataset / "manifest.json").read_text())
    progress = json.loads((embeddings / "progress.json").read_text())
    count = manifest["counts"]["unique_passages"]
    if progress["done"] != count or progress["total"] != count:
        raise ValueError("Added corpus embeddings are incomplete")
    if (progress["revision"] != manifest["corpus"]["model_revision"]
            or progress["document_input"] != "text" or progress["max_length"] != 32768
            or progress["dimension"] != dimension
            or progress["corpus_sha256"] != manifest["files"]["data/corpus_added.jsonl"]):
        raise ValueError("Added corpus embedding provenance mismatch")
    if (sha256(embeddings / "document_ids.json") != progress["ids_sha256"]
            or sha256(embeddings / "documents.npy") != progress["vectors_sha256"]):
        raise ValueError("Added corpus vector/ID checksum mismatch")
    titles = json.loads((embeddings / "document_ids.json").read_text())
    expected = [row["title"] for row in read_rows(dataset / "data/corpus_added.jsonl")]
    if titles != expected:
        raise ValueError("Added corpus vectors and IDs are misaligned")
    source = np.load(embeddings / "documents.npy", mmap_mode="r")
    if source.shape != (count, dimension):
        raise ValueError("Added corpus embedding shape mismatch")
    vectors = torch.empty((count, dimension), dtype=torch.float32, device=device)
    for start in range(0, count, 8192):
        chunk = torch.tensor(np.array(source[start:start + 8192]), device=device).float()
        if not torch.isfinite(chunk).all() or (chunk.norm(dim=-1) == 0).any():
            raise ValueError("Invalid added corpus embeddings")
        vectors[start:start + len(chunk)] = F.normalize(chunk, dim=-1)
    return vectors, titles, progress


def load_augmented_corpus(base_work_dir, dataset, embeddings, device):
    import torch
    from benchmarks.quest_or_training.data import load_corpus

    base, titles, base_status = load_corpus(Path(base_work_dir), device)
    added, added_titles, added_status = load_added_vectors(dataset, embeddings, device)
    if set(titles).intersection(added_titles):
        raise ValueError("Document ID namespace collision")
    return torch.cat((base, added)), titles + added_titles, {
        "base": base_status, "added": added_status,
        "documents": len(titles) + len(added_titles),
    }
