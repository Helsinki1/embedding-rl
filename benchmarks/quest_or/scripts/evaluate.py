"""QUEST macro precision/recall/F1, using exact document-title set membership."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def set_metrics(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    if not gold:
        raise ValueError('QUEST gold answer sets must be nonempty')
    correct = len(gold & predicted)
    return {
        'precision': correct / len(predicted) if predicted else 0.0,
        'recall': correct / len(gold),
        'f1': 2 * correct / (len(gold) + len(predicted)),
        'true_positives': correct,
        'false_positives': len(predicted - gold),
        'false_negatives': len(gold - predicted),
        'predicted_count': len(predicted),
        'gold_count': len(gold),
    }


def average(records):
    return {'queries': len(records), **{
        key: sum(r[key] for r in records) / len(records)
        for key in ('precision', 'recall', 'f1', 'predicted_count', 'gold_count')
    }}


def evaluate(gold_rows, predictions):
    if len({r['query'] for r in gold_rows}) != len(gold_rows):
        raise ValueError('Duplicate gold query')
    if len({r['query'] for r in predictions}) != len(predictions):
        raise ValueError('Duplicate prediction query')
    by_query = {r['query']: r for r in predictions}
    if set(by_query) != {r['query'] for r in gold_rows}:
        raise ValueError('Prediction queries must exactly match gold queries')
    details, domains, templates = [], defaultdict(list), defaultdict(list)
    for row in gold_rows:
        result = {'id': row.get('id'), 'query': row['query'],
                  **set_metrics(row['docs'], by_query[row['query']]['docs'])}
        details.append(result)
        domains[row['metadata']['domain']].append(result)
        templates[row['metadata']['template']].append(result)
    summary = {'overall': average(details),
               'by_domain': {key: average(values) for key, values in domains.items()},
               'by_source_template': {key: average(values) for key, values in templates.items()}}
    return summary, details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()
    gold = list(map(json.loads, args.gold.read_text().splitlines()))
    predictions = list(map(json.loads, args.predictions.read_text().splitlines()))
    print(json.dumps(evaluate(gold, predictions)[0], indent=2))
