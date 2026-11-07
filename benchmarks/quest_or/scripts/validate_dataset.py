"""Validate the frozen split, provenance, OR guard, and content checksums."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    guard = json.loads((ROOT / 'provenance/dataset_manifest.json').read_text())
    audits = {r['query']: r for r in map(json.loads, (ROOT / 'provenance/decompositions.jsonl').read_text().splitlines())}
    queries = {}
    for relative, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    for split, source_split in [('validation', 'val'), ('test', 'test')]:
        rows = list(map(json.loads, (ROOT / f'data/{split}.jsonl').read_text().splitlines()))
        sources = list(map(json.loads, (ROOT / f'provenance/source_{split}.jsonl').read_text().splitlines()))
        assert len(rows) == manifest['splits'][split]['count']
        assert len({r['id'] for r in rows}) == len(rows)
        assert len({r['query'] for r in rows}) == len(rows)
        for row in rows:
            source = sources[row['source_index']]
            assert row['query'] == source['query'] and row['docs'] == source['docs']
            assert row['metadata'] == source['metadata']
            assert row['metadata']['template'] in guard['templates']
            assert row['id'] == f'{source_split}-{row["source_index"]:04}'
            assert re.search(r'\bor\b', row['query'], re.I)
            assert row['components'] == audits[row['query']]['components']
            assert audits[row['query']]['keep'] and len(row['components']) in (2, 3)
            for text in [row['query'], *row['components']]:
                for name in guard['proper_name_allowlist']:
                    text = re.sub(re.escape(name), 'PROPER_NAME', text, flags=re.I)
                assert not re.search(guard['forbidden_pattern'], text, re.I), row['id']
        queries[split] = {row['query'] for row in rows}
        print(f'{split}: {len(rows)} frozen queries, original labels and OR-only guard verified')
    assert not queries['validation'] & queries['test']
    print('All checksums and split separation verified')


if __name__ == '__main__':
    main()
