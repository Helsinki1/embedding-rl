"""Prepare immutable corpus vectors, split manifests, and fixed training pools."""
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from .config import Config, digest, prepared_dir, write_json


def grouped_split(qrels, train_count, validation_count, seed):
    """Keep claims sharing positive evidence in the same train/validation group."""
    parent = {qid: qid for qid in sorted(qrels)}

    def find(qid):
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    owners = {}
    for qid, labels in sorted(qrels.items()):
        for doc_id, grade in labels.items():
            if grade > 0:
                if doc_id in owners:
                    parent[find(qid)] = find(owners[doc_id])
                owners[doc_id] = qid
    groups = {}
    for qid in sorted(qrels):
        groups.setdefault(find(qid), []).append(qid)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    validation, train = [], []
    for group in groups:
        if len(validation) < validation_count:
            validation.extend(group)
        else:
            train.extend(group)
    if len(train) < train_count or len(validation) < validation_count:
        raise ValueError("Not enough independent query groups for the requested split")
    # Unused members of the boundary groups are discarded, never moved to the other split.
    return train[:train_count], validation[:validation_count]


def load_scifact(config):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    revisions = {
        "model": api.model_info(config.model_name, revision=config.model_revision).sha,
        "corpus": api.dataset_info("BeIR/scifact", revision=config.corpus_revision).sha,
        "qrels": api.dataset_info("BeIR/scifact-qrels", revision=config.qrels_revision).sha,
    }
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus",
                          revision=revisions["corpus"])
    queries = load_dataset("BeIR/scifact", "queries", split="queries",
                           revision=revisions["corpus"])
    labels = load_dataset("BeIR/scifact-qrels", revision=revisions["qrels"])
    docs = sorted([
        {"id": str(row["_id"]), "title": row["title"], "text": row["text"]}
        for row in corpus
    ], key=lambda row: row["id"])
    query_text = {str(row["_id"]): row["text"] for row in queries}
    qrels = {}
    for split in ("train", "test"):
        qrels[split] = {}
        for row in labels[split]:
            qrels[split].setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(
                row["score"]
            )
    train, validation = grouped_split(
        qrels["train"], config.train_queries, config.validation_queries, config.seed
    )
    test = sorted(qrels["test"])
    if (set(train) | set(validation)) & set(test):
        raise ValueError("Official train and test query IDs overlap")
    return {
        "revisions": revisions, "docs": docs, "queries": query_text,
        "splits": {"train": train, "validation": validation, "test": test},
        "qrels": {**qrels["train"], **qrels["test"]},
    }


def doc_text(doc):
    return doc["title"] + "\n" + doc["text"]


def mixed_pool(dense_scores, sparse_scores, size):
    if size > len(dense_scores):
        raise ValueError("pool_size exceeds corpus size")
    dense = np.argsort(-np.asarray(dense_scores), kind="stable")
    sparse = np.argsort(-np.asarray(sparse_scores), kind="stable")
    selected = dict.fromkeys(int(i) for pair in zip(dense, sparse) for i in pair)
    return list(selected)[:size]


def prepare(root: Path, config: Config, device="cpu"):
    from rank_bm25 import BM25Okapi
    from .model import Encoder

    destination = prepared_dir(root, config)
    if (destination / "manifest.json").exists():
        return str(destination)
    bundle = load_scifact(config)
    encoder = Encoder(config.model_name, bundle["revisions"]["model"], device,
                      config.max_length, config.query_prefix)
    texts = [doc_text(doc) for doc in bundle["docs"]]
    vectors = encoder.encode_batches(texts, config.encode_batch_size)
    # Test queries are not encoded or evaluated during preparation or training.
    query_ids = bundle["splits"]["train"] + bundle["splits"]["validation"]
    query_vectors = encoder.encode_batches(
        [bundle["queries"][qid] for qid in query_ids], config.encode_batch_size, query=True
    )
    tokenize = lambda text: re.findall(r"\w+", text.lower())  # noqa: E731
    bm25 = BM25Okapi([tokenize(text) for text in texts])
    pools = {}
    for index, qid in enumerate(bundle["splits"]["train"]):
        pools[qid] = mixed_pool(
            query_vectors[index] @ vectors.T,
            bm25.get_scores(tokenize(bundle["queries"][qid])), config.pool_size,
        )
    bundle["pools"] = pools
    bundle["query_vector_ids"] = query_ids
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "data.json", bundle)
    torch.save({"documents": vectors, "queries": query_vectors}, destination / "vectors.pt")
    manifest = {
        "schema": 1, "preparation": config.preparation_settings(),
        "revisions": bundle["revisions"], "data_fingerprint": digest(bundle),
    }
    # This completion marker is written last; an interrupted preparation can be retried.
    write_json(destination / "manifest.json", manifest)
    return str(destination)


def load_prepared(root, config):
    path = prepared_dir(root, config)
    manifest = json.loads((path / "manifest.json").read_text())
    bundle = json.loads((path / "data.json").read_text())
    if manifest["data_fingerprint"] != digest(bundle):
        raise ValueError("Prepared data changed; use a fresh work directory")
    tensors = torch.load(path / "vectors.pt", map_location="cpu", weights_only=True)
    return path, manifest, bundle, tensors
