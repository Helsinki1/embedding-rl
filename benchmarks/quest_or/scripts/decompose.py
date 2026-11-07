"""Query-only eligibility audit and OR decomposition; never send labels to the LLM."""
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(os.environ.get('QUEST_OR_WORK_DIR', '/home/ubuntu/quest-or-baseline-20261106'))
MODEL = 'gpt-5.4-mini-2026-03-17'
PROMPT = '''You audit and decompose entity retrieval queries for an OR-only benchmark.
Read ONLY the supplied natural-language query. Do not answer it, search, or add facts.
Keep the query only if it unambiguously requests the UNION of two or three alternatives.
Return each alternative as a self-contained retrieval query, carrying over the entity
type and every shared modifier (dates, geography, genre, etc.) where its scope is clear.
Do not simplify or broaden any alternative, invent entities, or omit constraints.
Named or compound categories (e.g. Canadian fantasy comedy films) are atomic retrieval
descriptions. Internal modifiers do not by themselves introduce a separate set operation.
EXCLUDE queries with an actual AND/intersection, NOT/difference/exclusion, exclusive OR,
or ambiguous logical scope, even if they also contain OR. In particular, ordinary
conjunctive wording such as 'both', 'also', or 'A and B' must not be reinterpreted as OR.
An 'and' inside a proper name (Saint Kitts and Nevis, Sense and Sensibility, etc.) is not
a logical operation and should be preserved. A conventional noun phrase such as
'avant-garde and experimental films' may stay an atomic description.
For excluded queries, return an empty components array and a concise reason.
For accepted queries, return exactly two or three DISTINCT components and a concise
explanation of the OR structure. There must be no set-level AND or NOT left inside
any component. Be conservative if the natural-language query is ambiguous.
Output only the requested JSON schema.'''
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'keep': {'type': 'boolean'},
        'reason': {'type': 'string'},
        'components': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['keep', 'reason', 'components'],
}


async def main():
    cache_path = ROOT / 'results/decompositions.jsonl'
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            item = json.loads(line)
            assert item['prompt_sha256'] == prompt_hash and item['model'] == MODEL
            cache[item['query']] = item
    rows = []
    source_counts = {}
    for split in ('val', 'test'):
        source = [json.loads(x) for x in (ROOT / f'data/{split}.jsonl').read_text().splitlines()]
        source_counts[split] = dict(Counter(r['metadata']['template'] for r in source))
        for index, row in enumerate(source):
            if row['metadata']['template'] in ('_ or _', '_ or _ or _'):
                rows.append({'id': f'{split}-{index:04}', 'source_index': index,
                             'split': split, **row})
    val_queries = {r['query'] for r in rows if r['split'] == 'val'}
    test_queries = {r['query'] for r in rows if r['split'] == 'test'}
    assert not val_queries & test_queries, 'Validation and test queries overlap'
    client = None
    if any(row['query'] not in cache for row in rows):
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            key = json.loads((ROOT / 'credentials.json').read_text())['OPENAI_API_KEY']
        client = AsyncOpenAI(api_key=key, timeout=90, max_retries=3)
    semaphore = asyncio.Semaphore(8)
    stats = {'completed': len(cache), 'total': len({r['query'] for r in rows})}

    async def process(query):
        if query in cache:
            return
        async with semaphore:
            response = await client.responses.create(
                model=MODEL, instructions=PROMPT, input=query,
                reasoning={'effort': 'low'}, max_output_tokens=1500, store=False,
                text={'format': {'type': 'json_schema', 'name': 'or_decomposition',
                                 'strict': True, 'schema': SCHEMA}})
            if response.status != 'completed':
                raise RuntimeError(f'Incomplete model response: {response.id}')
            result = json.loads(response.output_text)
            components = result['components']
            if result['keep']:
                assert len(components) in (2, 3), (query, result)
                assert len(set(c.casefold().strip() for c in components)) == len(components)
                assert all(c.strip() for c in components)
            else:
                assert components == []
            usage = response.usage.model_dump()
            item = {'query': query, **result, 'model': MODEL,
                    'returned_model': response.model, 'response_id': response.id,
                    'usage': usage, 'prompt_sha256': prompt_hash}
            cache[query] = item
            with cache_path.open('a') as file:
                file.write(json.dumps(item, ensure_ascii=False) + '\n')
            stats['completed'] += 1
            if stats['completed'] % 20 == 0:
                print(json.dumps({**stats, 'kept': sum(x['keep'] for x in cache.values())}), flush=True)

    # A single real request validates model access and schema before parallel calls.
    unique = list(dict.fromkeys(r['query'] for r in rows))
    await process(unique[0])
    print('Model access and structured output verified', flush=True)
    await asyncio.gather(*(process(query) for query in unique[1:]))
    if client is not None:
        await client.close()
    for split in ('val', 'test'):
        kept, excluded = [], []
        for row in rows:
            if row['split'] != split:
                continue
            audit = cache[row['query']]
            record = {**row, 'components': audit['components'], 'cleaning_reason': audit['reason']}
            (kept if audit['keep'] else excluded).append(record)
        for label, records in [('clean', kept), ('excluded', excluded)]:
            with (ROOT / f'results/{split}_{label}.jsonl').open('w') as file:
                for record in records:
                    file.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(split, 'kept', len(kept), 'excluded', len(excluded), flush=True)
    usage = {'input_tokens': sum(r['usage']['input_tokens'] for r in cache.values()),
             'output_tokens': sum(r['usage']['output_tokens'] for r in cache.values())}
    manifest = {'model': MODEL, 'prompt': PROMPT, 'prompt_sha256': prompt_hash,
                'source_counts': source_counts, 'usage': usage,
                'estimated_usd_uncached': usage['input_tokens'] * 0.75 / 1e6 + usage['output_tokens'] * 4.5 / 1e6,
                'finished_at': time.time()}
    (ROOT / 'results/cleaning_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Decomposition complete', usage, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
