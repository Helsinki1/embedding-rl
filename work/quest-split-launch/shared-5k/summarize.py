"""Render the verified run summary and validation learning curve."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path('/home/davidx/Downloads/embedding-rl/benchmarks/quest_or_split/runs/r32-5k-v1')
outputs = Path('/home/davidx/Documents/Codex/2026-11-06/les/outputs')
result = json.loads((root / 'comparison.json').read_text())
primary = [row for row in result['results'] if row['evaluation'] in ('frozen', 'best_trained')]
labels = {'frozen': 'Untrained thirds', 'infonce': 'InfoNCE, rank 32', 'rl': 'RL, rank 32'}
table = ['| Method | Test precision | Test recall | Test F1 | Selected update | k per piece |',
         '|---|---:|---:|---:|---:|---:|']
for row in primary:
    table.append(f"| {labels[row['method']]} | {row['precision']:.2%} | {row['recall']:.2%} | "
                 f"{row['f1']:.2%} | {row['step']} | {row['k_per_piece']} |")

report = '''# Qwen three-piece retrieval: rank-32 results

Qwen3-Embedding-8B's original 4,096-dimensional query embedding is split into
1,366 / 1,365 / 1,365 coordinates. Each piece searches the corresponding coordinates
of the frozen document embeddings; retrieved document IDs are unioned and deduplicated.

'''
report += '\n'.join(table)
report += '''

The trained rows use the best **updated** checkpoint selected on validation.
Both methods completed 113 updates and saw all 3,600 training queries once.
Checkpoint selection and retrieval cutoff selection used only the 400 validation
queries, with evaluation at steps 0, 40, 80, and 113. The shared test contains
1,000 queries. Each method started from fresh base weights with LoRA rank 32,
alpha 64, learning rate 0.0001, and effective batch size 32. Documents and base
weights stayed frozen. The complete search corpus contains 393,140 documents.

The launcher enforced a 1,790-second timeout plus five-second termination grace
for each method, covering loading, training, and final evaluation. All runs
completed before the deadline. All seven saved test evaluations exactly match
the archived official QUEST precision/recall/F1 calculation. The implementation
and shared-entrypoint suite passed 18 tests.

## Shared split and interpretation

These runs reuse the exact current shared split, pinned by manifest SHA-256
`d490fc270ba684d7b0e810265c0221dec2da28ba09f146c15e4455b0a815dbf4`.
Its test partition contains **994 synthetic NQ queries and 6 QUEST queries**.
It holds out full query combinations, but train and test share **200 normalized
component/query texts and 397 synthetic positive passages**. This is different
from the retired component-disjoint split and the earlier proposed selection
of all 519 QUEST queries. The current split was reused to keep methods comparable.
These results do not measure generalization to entirely unseen components or
passages. Synthetic relevance labels are supplied positives, not exhaustive
judgments. This is one seed, so small differences should not be overinterpreted.

## Checkpoint accounting

The JSON result also includes the best validation checkpoint allowing step zero
(`selected`) and the final step-113 checkpoint (`final`). These alternatives were
fixed before test scoring; no checkpoint was selected using its test score.

'''
for method in ('infonce', 'rl'):
    selected = next(r for r in result['results'] if r['method'] == method and r['evaluation'] == 'selected')
    final = next(r for r in result['results'] if r['method'] == method and r['evaluation'] == 'final')
    report += (f"- {labels[method]}: selection allowing step zero chose step {selected['step']} "
               f"(test F1 {selected['f1']:.2%}); final-step F1 was {final['f1']:.2%}.\n")
report += f"\nRaw run artifacts: `{root}`\n\nRemote run: `{result['remote_run']}`\n"
(outputs / 'qwen-three-piece-rank32-5k-report.md').write_text(report)

fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
baseline = primary[0]['validation_f1']
ax.axhline(baseline, color='#64748b', linestyle='--', linewidth=1.5, label='Untrained thirds')
for method, color in [('infonce', '#7551bd'), ('rl', '#147b68')]:
    points = [(0, json.loads((root / method / 'initial_validation.json').read_text())['best']['f1'])]
    for path in sorted((root / method).glob('validation-*.json')):
        points.append((int(path.stem.split('-')[1]), max(r['f1'] for r in json.loads(path.read_text()))))
    ax.plot([s for s, _ in points], [v for _, v in points], '-o',
            color=color, label=labels[method], linewidth=2, markersize=5)
ax.set_title('Three-piece Qwen embeddings: validation F1', loc='left', fontweight='bold', pad=16)
ax.set_xlabel('Optimizer updates')
ax.set_ylabel('Macro answer-set F1')
ax.set_xticks([0, 40, 80, 113])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_ylim(bottom=0)
ax.grid(axis='y', alpha=.2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False)
fig.savefig(outputs / 'qwen-three-piece-rank32-5k-learning-curve.png', dpi=180, facecolor='white')
print('\n'.join(table))
