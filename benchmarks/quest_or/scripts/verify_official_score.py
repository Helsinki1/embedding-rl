"""Cross-check against the official QUEST main scoring function.

The official function is executed from its archived AST. Only file loading and
printing are adapted to stdlib objects, avoiding its TensorFlow/absl dependencies.
The precision/recall/F1 computation and per-example loop are unchanged.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]


def verify(gold_path, prediction_path):
    source = ROOT / 'provenance/official_run_eval.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    captured = []

    def read_examples(path):
        return [SimpleNamespace(query=r['query'], docs=r['docs'])
                for r in map(json.loads, Path(path).read_text().splitlines())]

    globals_ = {'FLAGS': SimpleNamespace(gold=gold_path, pred=prediction_path),
                'example_utils': SimpleNamespace(read_examples=read_examples),
                'eval_utils': SimpleNamespace(
                    print_avg=lambda examples, values: captured.append(sum(values) / len(values)),
                    print_avg_by_template=lambda *args: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), globals_)
    globals_['main'](None)
    gold = list(map(json.loads, gold_path.read_text().splitlines()))
    predictions = list(map(json.loads, prediction_path.read_text().splitlines()))
    ours, _ = evaluate(gold, predictions)
    for key, official in zip(('precision', 'recall', 'f1'), captured):
        assert abs(ours['overall'][key] - official) < 1e-12, key
    print(prediction_path.name, ': exact agreement with official QUEST scoring')
    return dict(zip(('precision', 'recall', 'f1'), captured))


if __name__ == '__main__':
    results = {}
    for method in ('or_union', 'original_query'):
        path = ROOT / f'runs/baseline-v1/test_{method}_predictions.jsonl'
        results[method] = verify(ROOT / 'data/test.jsonl', path)
    (ROOT / 'runs/baseline-v1/official_score_verification.json').write_text(json.dumps(results, indent=2))
