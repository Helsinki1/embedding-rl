"""Collect completed split-vector runs and independently verify every test score."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile

project = Path('/home/davidx/Downloads/embedding-rl')
work = Path(__file__).resolve().parent
ssh_config = '/home/davidx/Documents/Codex/2026-11-06/les/work/ssh/config'
remote = '/home/ubuntu/quest-or-split-r32-5k-20261106-v1'
root = project / 'benchmarks/quest_or_split/runs/r32-5k-v1'
outputs = Path('/home/davidx/Documents/Codex/2026-11-06/les/outputs')
expected_manifest = 'd490fc270ba684d7b0e810265c0221dec2da28ba09f146c15e4455b0a815dbf4'

command = ['ssh', '-F', ssh_config, 'compute',
           f'tar -czf - --exclude="*.pt" --exclude="__pycache__" -C {remote}/results '
           f'frozen infonce rl -C {remote} logs status.json launch.json '
           'snapshot_manifest.json snapshot.tar.gz split_overlap_audit.json '
           'code/benchmarks/quest_or_balanced/data/test.jsonl '
           'code/benchmarks/quest_or_balanced/manifest.json '
           'code/benchmarks/quest_or_balanced/shared_protocol.json '
           'code/benchmarks/quest_or/scripts/evaluate.py '
           'code/benchmarks/quest_or/scripts/verify_official_score.py '
           'code/benchmarks/quest_or/provenance/official_run_eval.py']
with (work / 'artifacts.tar.gz').open('wb') as stream:
    subprocess.run(command, stdout=stream, check=True)
with tarfile.open(work / 'artifacts.tar.gz') as archive:
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if not target.is_relative_to(root.resolve()) or member.issym() or member.islnk():
            raise ValueError('Unexpected archive path')
    archive.extractall(root)

shared = root / 'code/benchmarks/quest_or_balanced'
manifest = json.loads((shared / 'manifest.json').read_text())
assert hashlib.sha256((shared / 'manifest.json').read_bytes()).hexdigest() == expected_manifest
gold_path = shared / 'data/test.jsonl'
assert hashlib.sha256(gold_path.read_bytes()).hexdigest() == manifest['files']['data/test.jsonl']
assert json.loads((root / 'status.json').read_text())['stage'] == 'complete'
sys.path.insert(0, str(root / 'code/benchmarks/quest_or/scripts'))
from verify_official_score import verify

verification, entries, training = {}, [], {}
for method in ('frozen', 'infonce', 'rl'):
    method_root = root / method
    progress = json.loads((method_root / 'progress.json').read_text())
    metadata = json.loads((method_root / 'run.json').read_text())
    assert progress['stage'] == 'complete'
    assert metadata['provenance']['split_manifest_sha256'] == expected_manifest
    assert metadata['training_queries'] == 3600 and metadata['validation_queries'] == 400
    assert metadata['document_count'] == 393140
    assert metadata['config']['lora_rank'] == 32 and metadata['config']['lora_alpha'] == 64
    if method != 'frozen':
        assert progress['step'] == 113 and progress['examples_seen'] == 3600
        history = [json.loads(line) for line in (method_root / 'metrics.jsonl').read_text().splitlines()]
        assert len(history) == 113
        assert [r['step'] for r in history if 'validation' in r] == [40, 80, 113]
        overall = json.loads((method_root / 'summary.json').read_text())
        assert overall['elapsed_seconds'] < 1795
        training[method] = {
            'completed_steps': 113, 'examples_seen': 3600,
            'elapsed_seconds_inside_trainer': overall['elapsed_seconds'],
            'peak_gpu_memory_bytes': overall['peak_gpu_memory_bytes'],
            'trainable_parameters': metadata['trainable_parameters'],
            'last_update': history[-1],
        }
    labels = (None,) if method == 'frozen' else ('selected', 'best_trained', 'final')
    for label in labels:
        directory = method_root / label if label else method_root
        summary = json.loads((directory / 'summary.json').read_text())
        name = method + ('/' + label if label else '')
        score = verify(gold_path, directory / 'test_predictions.jsonl')
        predictions = [json.loads(line) for line in (directory / 'test_predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == 1000
        assert all(len(r['docs']) == len(set(r['docs'])) for r in predictions)
        for key, value in score.items():
            assert abs(summary['test']['overall'][key] - value) < 1e-12
        verification[name] = score
        entries.append({
            'method': method, 'evaluation': label or 'frozen',
            'step': summary['checkpoint_step'], 'k_per_piece': summary['k'],
            'validation_f1': summary['validation']['f1'],
            **summary['test']['overall'], 'breakdowns': summary['test'],
            'diagnostics': summary['diagnostics'],
        })

result = {
    'experiment': 'Three contiguous coordinate pieces of Qwen3-Embedding-8B',
    'dimensions': [1366, 1365, 1365], 'document_count': 393140,
    'training_queries': 3600, 'validation_queries': 400, 'test_queries': 1000,
    'epochs': 1, 'steps': 113, 'lora_rank': 32, 'lora_alpha': 64, 'seed': 42,
    'split_manifest_sha256': expected_manifest,
    'test_composition': {'QUEST': 6, 'NQ_synthetic': 994},
    'remote_run': remote, 'local_run': str(root),
    'results': entries, 'training': training,
    'split_overlap_audit': json.loads((root / 'split_overlap_audit.json').read_text()),
    'notes': [
        'The document encoder is frozen; document slices match query coordinates.',
        'Exact shared split used by the other current approaches; fresh base initialization.',
        'Current split holds out full query combinations; components and positives may recur.',
        'Checkpoint and cutoff selection use validation only.',
        'Primary trained result is best_trained; selected additionally permits step zero.',
        'Single-seed results; synthetic positive labels are not exhaustive.',
        'Each method is enclosed by a 1790-second timeout plus five-second termination grace.',
    ],
}
(root / 'official_score_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
(root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
outputs.mkdir(exist_ok=True)
(outputs / 'qwen-three-piece-rank32-5k-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([{k: r[k] for k in ('method', 'evaluation', 'step', 'k_per_piece',
                                   'precision', 'recall', 'f1')} for r in entries], indent=2))
