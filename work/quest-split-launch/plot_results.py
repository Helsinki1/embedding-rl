from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root=Path('/home/davidx/Downloads/embedding-rl/benchmarks/quest_or_split/runs/r32-v1')
output=Path('/home/davidx/Documents/Codex/2026-11-06/les/outputs/qwen-three-piece-rank32-learning-curve.png')
fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
frozen=json.loads((root/'frozen/summary.json').read_text())['validation']['f1']
ax.axhline(frozen,color='#6b7280',linestyle='--',linewidth=1.6,label='Untrained split baseline')
for method,label,color in [('infonce','InfoNCE','#7656c4'),('rl','RL','#147b68')]:
    initial=json.loads((root/method/'initial_validation.json').read_text())['best']['f1']
    series=[(0,initial)]
    for path in sorted((root/method).glob('validation-*.json')):
        series.append((int(path.stem.split('-')[1]),max(r['f1'] for r in json.loads(path.read_text()))))
    ax.plot([s for s,_ in series],[v for _,v in series],'-o',color=color,label=label,linewidth=2,markersize=5)
ax.set_title('Three-piece Qwen embeddings: development F1',loc='left',fontweight='bold',pad=18)
ax.set_xlabel('Optimizer updates (one epoch = 3,137 updates)')
ax.set_ylabel('Macro answer-set F1')
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(bottom=0)
ax.grid(axis='y',alpha=.2)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower right',frameon=False)
fig.text(.5,-.025,'LoRA rank 32 / alpha 64 • 78 development queries • k selected from 1–20 per piece',ha='center',fontsize=9,color='#4b5563')
fig.savefig(output,dpi=180,bbox_inches='tight',facecolor='white')
print(output)
