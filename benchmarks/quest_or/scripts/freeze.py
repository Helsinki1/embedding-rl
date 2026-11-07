"""Freeze conservative OR-only splits using query text, never retrieval outcomes."""
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('QUEST_OR_WORK_DIR', '/home/ubuntu/quest-or-baseline-20261106'))
PROPER_NAMES = [
    'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines',
    'Heard Island and McDonald Islands', 'Heard Island and McDonald Island',
    'Sense and Sensibility', 'Sjöwall and Wahlöö', 'Antigua and Barbuda',
    'George G Harrap and Co',
]
FORBIDDEN = re.compile(r'\b(and|both|also|not|without|except|excluding|neither|nor)\b|\bother than\b', re.I)


def text_guard(query, components):
    if not re.search(r'\bor\b', query, re.I):
        return 'No explicit OR in the natural-language query.'
    for text in [query, *components]:
        scrubbed = text
        for name in PROPER_NAMES:
            scrubbed = re.sub(re.escape(name), 'PROPER_NAME', scrubbed, flags=re.I)
        match = FORBIDDEN.search(scrubbed)
        if match:
            return f'Conservative text guard: {match.group(0)!r} outside an allowed proper name.'
    return None


def main():
    audits = {r['query']: r for r in map(json.loads, (ROOT / 'results/decompositions.jsonl').read_text().splitlines())}
    manifest = {'version': 'quest-or-v1', 'templates': ['_ or _', '_ or _ or _'],
                'proper_name_allowlist': PROPER_NAMES, 'forbidden_pattern': FORBIDDEN.pattern,
                'require_explicit_or': True,
                'selection': 'Official union template, query-only LLM eligibility, then conservative text guard; no labels or retrieval scores used.',
                'splits': {}}
    for split in ('val', 'test'):
        accepted, excluded = [], []
        rows = list(map(json.loads, (ROOT / f'data/{split}.jsonl').read_text().splitlines()))
        for index, row in enumerate(rows):
            record = {'id': f'{split}-{index:04}', 'source_index': index, 'split': split, **row}
            if row['metadata']['template'] not in manifest['templates']:
                excluded.append({**record, 'exclusion_stage': 'template', 'exclusion_reason': 'Not an OR-only template.'})
                continue
            audit = audits[row['query']]
            record.update(components=audit['components'], cleaning_reason=audit['reason'])
            if not audit['keep']:
                excluded.append({**record, 'exclusion_stage': 'llm_query_audit', 'exclusion_reason': audit['reason']})
                continue
            reason = text_guard(row['query'], audit['components'])
            if reason:
                excluded.append({**record, 'exclusion_stage': 'text_guard', 'exclusion_reason': reason})
                continue
            accepted.append(record)
        paths = {}
        for label, data in [('clean', accepted), ('excluded', excluded)]:
            path = ROOT / f'results/{split}_{label}.jsonl'
            path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
            paths[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['splits'][split] = {
            'source_queries': len(rows), 'source_or_queries': sum(r['metadata']['template'] in manifest['templates'] for r in rows),
            'retained': len(accepted), 'exclusion_counts': dict(Counter(r['exclusion_stage'] for r in excluded)),
            'domains': dict(Counter(r['metadata']['domain'] for r in accepted)),
            'template_counts': dict(Counter(r['metadata']['template'] for r in accepted)),
            'component_counts': dict(Counter(len(r['components']) for r in accepted)), 'sha256': paths}
        assert len(accepted) + len(excluded) == len(rows)
        assert len({r['query'] for r in accepted}) == len(accepted)
        print(split, manifest['splits'][split], flush=True)
    (ROOT / 'results/dataset_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
