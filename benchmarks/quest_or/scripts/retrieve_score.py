"""Exact cosine search, validation-only cutoff choice, and frozen test scoring."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate import evaluate

ROOT = Path(os.environ.get('QUEST_OR_WORK_DIR', '/home/ubuntu/quest-or-baseline-20261106'))


def search():
    result_file = ROOT / 'results/top1000.npz'
    if result_file.exists():
        return np.load(result_file)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    shard_vectors, shard_ids = [], []
    for rank in (0, 1):
        progress = json.loads((ROOT / f'results/corpus_{rank}_progress.json').read_text())
        assert progress['done'] == progress['total']
        shard_vectors.append(np.load(ROOT / f'results/corpus_{rank}.npy', mmap_mode='r'))
        shard_ids.extend(json.loads((ROOT / f'results/corpus_{rank}_ids.json').read_text()))
    titles = [row['title'] for row in shard_ids]
    assert len(titles) == 325427 and len(set(titles)) == len(titles)
    (ROOT / 'results/retrieval_document_titles.json').write_text(json.dumps(titles, ensure_ascii=False))
    corpus = torch.from_numpy(np.concatenate(shard_vectors).astype(np.float32)).to('cuda')
    corpus = F.normalize(corpus, dim=1)
    queries = np.load(ROOT / 'results/queries.npy').astype(np.float32)
    indices, scores = [], []
    with torch.inference_mode():
        for start in range(0, len(queries), 64):
            query = F.normalize(torch.from_numpy(queries[start:start + 64]).to('cuda'), dim=1)
            similarity = query @ corpus.T
            values, positions = torch.topk(similarity, 1000, dim=1, sorted=True)
            scores.append(values.cpu().numpy())
            indices.append(positions.cpu().numpy().astype(np.int32))
            print('Searched query vectors', min(start + 64, len(queries)), '/', len(queries), flush=True)
    np.savez_compressed(result_file, scores=np.concatenate(scores), indices=np.concatenate(indices))
    return np.load(result_file)


def main():
    protocol = json.loads((ROOT / 'results/protocol.json').read_text())
    for split in ('val', 'test'):
        actual = hashlib.sha256((ROOT / f'results/{split}_clean.jsonl').read_bytes()).hexdigest()
        assert actual == protocol['split_sha256'][split], 'Frozen split changed'
    cached = search()
    retrieval = {key: cached[key] for key in ('indices', 'scores')}
    cached.close()
    titles = json.loads((ROOT / 'results/retrieval_document_titles.json').read_text())
    query_rows = json.loads((ROOT / 'results/queries_ids.json').read_text())
    query_index = {(r['id'], r['component']): i for i, r in enumerate(query_rows)}
    splits = {s: list(map(json.loads, (ROOT / f'results/{s}_clean.jsonl').read_text().splitlines()))
              for s in ('val', 'test')}
    title_set = set(titles)
    assert all(set(row['docs']) <= title_set for rows in splits.values() for row in rows)

    def predict(rows, method, k):
        predictions = []
        for row in rows:
            components = ['original'] if method == 'original_query' else list(map(str, range(len(row['components']))))
            candidate_scores, component_results = {}, []
            for component in components:
                index = query_index[(row['id'], component)]
                docs = []
                for doc_index, score in zip(retrieval['indices'][index, :k], retrieval['scores'][index, :k]):
                    title = titles[int(doc_index)]
                    candidate_scores[title] = max(candidate_scores.get(title, -float('inf')), float(score))
                    docs.append(title)
                component_results.append({'component': component, 'docs': docs})
            docs = sorted(candidate_scores, key=lambda title: (-candidate_scores[title], title))
            predictions.append({'id': row['id'], 'query': row['query'], 'docs': docs,
                                'scores': [candidate_scores[d] for d in docs],
                                'component_results': component_results,
                                'duplicates_removed': len(components) * k - len(docs)})
        return predictions

    selection = {}
    for method, grid in protocol['cutoff_grids'].items():
        trials = []
        for k in grid:
            metrics, _ = evaluate(splits['val'], predict(splits['val'], method, k))
            trials.append({'k': k, **metrics['overall']})
        best = max(trials, key=lambda trial: (trial['f1'], -trial['k']))
        selection[method] = {'selected_k': best['k'], 'validation': best, 'grid_results': trials}
    # Persist selection before accessing test labels in any evaluation.
    (ROOT / 'results/selection.json').write_text(json.dumps(selection, indent=2))
    results = {'protocol': protocol, 'selection': selection, 'test': {}, 'validation': {}}
    for method, chosen in selection.items():
        for split, result_key in [('val', 'validation'), ('test', 'test')]:
            predictions = predict(splits[split], method, chosen['selected_k'])
            metrics, details = evaluate(splits[split], predictions)
            metrics['mean_duplicates_removed'] = sum(p['duplicates_removed'] for p in predictions) / len(predictions)
            results[result_key][method] = metrics
            for label, rows in [('predictions', predictions), ('per_query', details)]:
                path = ROOT / f'results/{split}_{method}_{label}.jsonl'
                path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    deltas = []
    original = list(map(json.loads, (ROOT / 'results/test_original_query_per_query.jsonl').read_text().splitlines()))
    decomposed = list(map(json.loads, (ROOT / 'results/test_or_union_per_query.jsonl').read_text().splitlines()))
    delta = np.array([b['f1'] - a['f1'] for a, b in zip(original, decomposed)])
    rng = np.random.default_rng(42)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    results['paired_f1_difference'] = {'mean': float(delta.mean()),
                                     'bootstrap_95pct_interval': list(map(float, np.quantile(means, [.025, .975]))),
                                     'resamples': 10000, 'seed': 42}
    (ROOT / 'results/metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'test': results['test'], 'paired_f1_difference': results['paired_f1_difference']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
