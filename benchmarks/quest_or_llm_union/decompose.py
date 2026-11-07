"""Run the original query-only LLM prompt on shared validation and test queries."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import dotenv_values
from openai import AsyncOpenAI

from benchmarks.quest_or.scripts.decompose import MODEL, PROMPT, SCHEMA
from benchmarks.quest_or_balanced.data import load_splits
from benchmarks.quest_or_synthetic.build import read_rows, sha256, write_json
from .union import components_for

HERE = Path(__file__).resolve().parent
PROMPT_HASH = hashlib.sha256(PROMPT.encode()).hexdigest()


async def request_decomposition(query, client):
    response = await client.responses.create(
        model=MODEL, instructions=PROMPT, input=query,
        reasoning={"effort": "low"}, max_output_tokens=1500, store=False,
        text={"format": {"type": "json_schema", "name": "or_decomposition", "strict": True,
                         "schema": SCHEMA}},
    )
    if response.status == "completed" and response.output_text:
        try:
            result = json.loads(response.output_text)
        except json.JSONDecodeError:
            result = {"keep": False, "components": [], "reason": "invalid_json"}
    else:
        result = {"keep": False, "components": [], "reason": "incomplete_or_refused_response"}
    return {"query": query, **result, "model": MODEL, "returned_model": response.model,
            "response_id": response.id, "usage": response.usage.model_dump(),
            "prompt_sha256": PROMPT_HASH, "cache_origin": "new_request"}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=HERE.parent / "quest_or_balanced")
    parser.add_argument("--source", type=Path, default=HERE.parent / "quest_or_synthetic")
    parser.add_argument("--env-file", type=Path, default=HERE.parent.parent / ".env")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    _, dev, test, split = load_splits(args.split, args.source)
    queries = list(dict.fromkeys(r["query"] for r in dev + test))
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.output / "decompositions.jsonl"
    cache = {}
    if cache_path.exists():
        for record in read_rows(cache_path):
            if record["model"] != MODEL or record["prompt_sha256"] != PROMPT_HASH:
                raise ValueError("Existing decomposition cache uses a different model or prompt")
            cache[record["query"]] = record
    original_cache = HERE.parent / "quest_or/provenance/decompositions.jsonl"
    for record in read_rows(original_cache):
        if record["query"] in queries and record["query"] not in cache:
            if record["model"] != MODEL or record["prompt_sha256"] != PROMPT_HASH:
                raise ValueError("Historical decomposition model/prompt mismatch")
            record = {**record, "cache_origin": "original_quest_llm_cache"}
            cache[record["query"]] = record
            with cache_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    missing = [q for q in queries if q not in cache]
    print(json.dumps({"total": len(queries), "cached": len(queries) - len(missing), "requests": len(missing)}), flush=True)
    env = dotenv_values(args.env_file)
    key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_KEY") or env.get("OPENAI_API_KEY")
    if missing and not key:
        raise ValueError("OPENAI_KEY or OPENAI_API_KEY is required")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process(query, client):
        async with semaphore:
            record = await request_decomposition(query, client)
            cache[query] = record
            with cache_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(cache) % 25 == 0 or len(cache) == len(queries):
                print(json.dumps({"completed": len(cache), "total": len(queries),
                                  "elapsed_seconds": round(time.time() - started, 1)}), flush=True)
    if missing:
        async with AsyncOpenAI(api_key=key, timeout=90, max_retries=3) as client:
            await process(missing[0], client)
            print("Pinned model access and structured output verified", flush=True)
            await asyncio.gather(*(process(q, client) for q in missing[1:]))
    selected = [cache[q] for q in queries]
    fallbacks = [r["id"] for r in dev + test if components_for(r["query"], cache[r["query"]])[1]]
    write_json(args.output / "decomposition_manifest.json", {
        "model": MODEL, "prompt": PROMPT, "prompt_sha256": PROMPT_HASH, "schema": SCHEMA,
        "split_manifest_sha256": sha256(args.split / "manifest.json"), "split_version": split["version"],
        "query_count": len(queries), "validation_queries": len(dev), "test_queries": len(test),
        "input_fields": ["query"], "fallback_rule": "Whole query if LLM rejects, refuses, or returns invalid decomposition; no query is excluded",
        "fallback_query_ids": fallbacks,
        "cached_original_quest": sum(r["cache_origin"] == "original_quest_llm_cache" for r in selected),
        "new_llm_responses": sum(r["cache_origin"] == "new_request" for r in selected),
        "usage": {key: sum(r["usage"][key] for r in selected) for key in ("input_tokens", "output_tokens")},
        "decompositions_sha256": sha256(cache_path), "source_cache_sha256": sha256(original_cache),
        "elapsed_seconds": time.time() - started,
    })
    print(json.dumps({"stage": "complete", "queries": len(queries), "fallbacks": len(fallbacks),
                      "elapsed_seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
