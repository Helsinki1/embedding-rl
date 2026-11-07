"""Query-only pinned LLM control; no labels or gold components enter API requests."""
import asyncio
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI

from benchmarks.quest_or_llm_union.decompose import MODEL, PROMPT_HASH, request_decomposition
from benchmarks.quest_or_llm_union.union import components_for
from .build import HERE, read_rows, sha256, write_json
from .data import load_splits


async def main():
    _, validation, test, provenance = load_splits()
    queries = list(dict.fromkeys(r["query"] for r in validation + test))
    output = HERE / "decomposition"
    output.mkdir(exist_ok=True)
    cache_path = output / "decompositions.jsonl"
    cache = {}
    for path in (cache_path, HERE.parent / "quest_or/provenance/decompositions.jsonl",
                 HERE.parent / "quest_or_llm_union/runs/shared-v1/decomposition/decompositions.jsonl"):
        if path.exists():
            for record in read_rows(path):
                if record["query"] in queries and record["model"] == MODEL and record["prompt_sha256"] == PROMPT_HASH:
                    cache.setdefault(record["query"], {**record, "reused_from": str(path)})
    # Preserve the cache on interruption; append one completed response at a time.
    cache_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cache.values()))
    env = dotenv_values(HERE.parent.parent / ".env")
    key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_KEY") or env.get("OPENAI_API_KEY")
    semaphore = asyncio.Semaphore(16)

    async def request(query, client):
        async with semaphore:
            record = await request_decomposition(query, client)
            cache[query] = record
            with cache_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(cache) % 25 == 0:
                print(json.dumps({"completed": len(cache), "total": len(queries)}), flush=True)

    missing = [q for q in queries if q not in cache]
    print(json.dumps({"cached": len(cache), "requests": len(missing)}), flush=True)
    if missing:
        if not key:
            raise ValueError("Missing OpenAI API key")
        async with AsyncOpenAI(api_key=key, timeout=90, max_retries=3) as client:
            await request(missing[0], client)
            await asyncio.gather(*(request(q, client) for q in missing[1:]))
    write_json(output / "manifest.json", {**provenance, "model": MODEL, "prompt_sha256": PROMPT_HASH,
        "cache_sha256": sha256(cache_path), "queries": len(queries), "input_fields": ["query"],
        "fallback_query_ids": [r["id"] for r in validation + test if components_for(r["query"], cache[r["query"]])[1]],
        "fallback": "Whole query; never drop an example"})


if __name__ == "__main__":
    asyncio.run(main())
