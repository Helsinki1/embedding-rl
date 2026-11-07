"""Encode only the added NQ passages with the original frozen Qwen checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .build import HERE, read_rows, sha256, write_json

REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE)
    parser.add_argument("--base-work-dir", type=Path, default=Path("/home/ubuntu/quest-or-baseline-20261106"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "embed.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["RAYON_NUM_THREADS"] = "6"
    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    source = args.dataset / "data/corpus_added.jsonl"
    source_hash = sha256(source)
    if source_hash != manifest["files"]["data/corpus_added.jsonl"]:
        raise ValueError("Added corpus changed after generation")
    rows = read_rows(source)
    ids = [row["title"] for row in rows]
    ids_path = args.output / "document_ids.json"
    progress_path = args.output / "progress.json"
    vectors_path = args.output / "documents.npy"
    start = tokens_total = truncated = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if (progress["corpus_sha256"] != source_hash or progress["revision"] != REVISION
                or progress["document_input"] != "text" or progress["max_length"] != 32768
                or progress["total"] != len(rows) or progress["ids_sha256"] != sha256(ids_path)
                or json.loads(ids_path.read_text()) != ids):
            raise ValueError("Embedding resume provenance mismatch")
        start, tokens_total, truncated = progress["done"], progress["tokens"], progress["truncated"]
        if start == len(rows):
            print("Already complete", flush=True)
            return
        vectors = np.lib.format.open_memmap(vectors_path, mode="r+")
        if vectors.shape != (len(rows), 4096):
            raise ValueError("Embedding resume shape mismatch")
    else:
        if vectors_path.exists():
            raise FileExistsError("Uncheckpointed embeddings exist; choose a fresh output")
        write_json(ids_path, ids)
        vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype="float16", shape=(len(rows), 4096))
    model_path = str(args.base_work_dir / "models/Qwen3-Embedding-8B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    # Check every new query against the existing training encoder's context cap.
    max_query_tokens = 0
    query_rows = read_rows(args.dataset / "data/train_synthetic.jsonl")
    for offset in range(0, len(query_rows), 4096):
        prompts = ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
                   + row["query"] for row in query_rows[offset:offset + 4096]]
        lengths = map(len, tokenizer(prompts, add_special_tokens=True, truncation=False)["input_ids"])
        max_query_tokens = max(max_query_tokens, max(lengths))
    if max_query_tokens > 512:
        raise ValueError("Synthetic queries exceed the existing training context cap")
    del query_rows
    llm = LLM(model=model_path, runner="pooling", convert="embed", dtype="bfloat16",
              max_model_len=32768, gpu_memory_utilization=0.55,
              max_num_batched_tokens=32768, max_num_seqs=64,
              enable_chunked_prefill=False, enforce_eager=True, seed=42, trust_remote_code=False)
    began = time.monotonic()
    ids_hash = sha256(ids_path)
    for offset in range(start, len(rows), 2048):
        end = min(offset + 2048, len(rows))
        tokenized = tokenizer([row["text"] for row in rows[offset:end]],
                              add_special_tokens=True, truncation=False)["input_ids"]
        truncated += sum(len(value) > 32768 for value in tokenized)
        tokenized = [value[:32768] for value in tokenized]
        tokens_total += sum(map(len, tokenized))
        outputs = llm.embed([{"prompt_token_ids": value} for value in tokenized],
                            pooling_params=PoolingParams(use_activation=True), use_tqdm=False)
        values = np.asarray([output.outputs.embedding for output in outputs], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if values.shape != (end - offset, 4096) or not np.isfinite(values).all() or (norms == 0).any():
            raise ValueError("Invalid document embeddings")
        vectors[offset:end] = (values / norms).astype(np.float16)
        vectors.flush()
        progress = {"done": end, "total": len(rows), "revision": REVISION,
                    "dimension": 4096, "dtype": "float16", "document_input": "text",
                    "max_length": 32768, "corpus_sha256": source_hash, "ids_sha256": ids_hash,
                    "tokens": tokens_total, "truncated": truncated, "max_query_tokens": max_query_tokens,
                    "elapsed_seconds_this_session": time.monotonic() - began}
        if end == len(rows):
            progress["vectors_sha256"] = sha256(vectors_path)
        temporary = progress_path.with_suffix(".tmp")
        write_json(temporary, progress)
        temporary.replace(progress_path)
        print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
