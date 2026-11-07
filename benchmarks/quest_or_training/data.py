"""Read the frozen QUEST package and completed corpus shards without changing them."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DIMENSION = 4096
CORPUS_SIZE = 325427
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_splits(benchmark, split_path):
    spec = json.loads(Path(split_path).read_text())
    for name, expected in spec["source_sha256"].items():
        if sha256(benchmark / name) != expected:
            raise ValueError(f"Frozen query file changed: {name}")
    rows = read_rows(benchmark / "data/validation.jsonl") + read_rows(benchmark / "data/test.jsonl")
    by_id = {row["id"]: row for row in rows}
    partitions = [spec[name] for name in ("train_ids", "validation_ids", "test_ids")]
    all_ids = [qid for partition in partitions for qid in partition]
    if (len(by_id) != len(rows) or len({r["query"] for r in rows}) != len(rows)
            or len(set(all_ids)) != len(all_ids) or set(all_ids) != set(by_id)):
        raise ValueError("Training/validation/test IDs must partition all frozen queries exactly once")
    return (*([by_id[qid] for qid in partition] for partition in partitions), spec)


def corpus_status(root):
    shards = []
    for rank in (0, 1):
        prefix = root / "results" / f"corpus_{rank}"
        progress_path = Path(str(prefix) + "_progress.json")
        if not progress_path.exists():
            return False, {"waiting_for": str(progress_path)}
        progress = json.loads(progress_path.read_text())
        if progress["revision"] != REVISION or progress["max_length"] != 32768:
            raise ValueError("Corpus encoder revision or context limit differs from the baseline")
        shards.append(progress)
        if progress["done"] > progress["total"]:
            raise ValueError("Invalid corpus progress")
        if progress["done"] != progress["total"]:
            return False, {"shard": rank, "done": progress["done"], "total": progress["total"]}
        for suffix in (".npy", "_ids.json"):
            if not Path(str(prefix) + suffix).exists():
                raise ValueError(f"Completed shard is missing {prefix}{suffix}")
    return True, {"shards": shards, "documents": sum(p["total"] for p in shards)}


def wait_for_corpus(root, wait=False, timeout=86400, interval=30):
    deadline = time.monotonic() + timeout
    while True:
        ready, status = corpus_status(root)
        if ready:
            return status
        if not wait:
            raise RuntimeError(f"Corpus is incomplete; use --wait: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for corpus: {status}")
        print(json.dumps({"waiting_for_corpus": status}), flush=True)
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def load_corpus(root, device, dimension=DIMENSION, expected_count=CORPUS_SIZE):
    ready, status = corpus_status(root)
    if not ready:
        raise RuntimeError("Refusing to read an incomplete corpus")
    identities, arrays = [], []
    for rank, progress in enumerate(status["shards"]):
        ids = json.loads((root / f"results/corpus_{rank}_ids.json").read_text())
        digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()
        if digest != progress["identity_sha256"]:
            raise ValueError("Corpus identity checksum mismatch")
        array = np.load(root / f"results/corpus_{rank}.npy", mmap_mode="r")
        if array.shape != (progress["total"], dimension) or len(ids) != len(array):
            raise ValueError("Corpus shape or identity count mismatch")
        identities.extend(ids)
        arrays.append(array)
    titles = [row["title"] for row in identities]
    if (len(titles) != expected_count or len(set(titles)) != len(titles)
            or sorted(row["index"] for row in identities) != list(range(expected_count))):
        raise ValueError("Corpus must contain every unique document exactly once")
    # Keep rank-0 then rank-1 order for BOTH vectors and titles. Source indices interleave.
    vectors = torch.empty((len(titles), dimension), device=device, dtype=torch.float32)
    offset = 0
    for array in arrays:
        for start in range(0, len(array), 8192):
            chunk = torch.tensor(np.array(array[start:start + 8192]), device=device).float()
            if not torch.isfinite(chunk).all() or (chunk.norm(dim=-1) == 0).any():
                raise ValueError("Corpus contains non-finite or zero vectors")
            vectors[offset:offset + len(chunk)] = F.normalize(chunk, dim=-1)
            offset += len(chunk)
    return vectors, titles, status


def gold_indices(rows, title_to_index, device):
    values = []
    for row in rows:
        if not row["docs"]:
            raise ValueError(f"Empty gold answer set: {row['id']}")
        missing = {title for title in row["docs"] if title not in title_to_index}
        if missing:
            raise ValueError(f"Gold documents missing from the full corpus: {row['id']}")
        values.append(torch.tensor(sorted({title_to_index[t] for t in row["docs"]}),
                                   device=device, dtype=torch.long))
    return values
