"""Synthetic check only: conditional averaging for a fixed candidate-pool reward."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from embedding_rl.objectives import ndcg

torch.manual_seed(4821)
dim, candidates, groups, samples = 64, 8, 4000, 32
docs = F.normalize(torch.randn(candidates, dim, dtype=torch.float64), dim=-1)
mean = F.normalize(torch.randn(dim, dtype=torch.float64), dim=0)
basis = torch.linalg.qr(docs.T, mode='reduced').Q
sigma = .45 / math.sqrt(dim)
noise = torch.randn(groups, samples, dim, dtype=torch.float64) * sigma
projected_noise = (noise @ basis) @ basis.T
actions = mean + noise
projected_actions = mean + projected_noise
scores = actions @ docs.T
projected_scores = projected_actions @ docs.T
grades = torch.tensor([3.,2.,1.,0.,0.,0.,0.,0.], dtype=torch.float64)
reward = ndcg(scores, grades, k=3)
projected_reward = ndcg(projected_scores, grades, k=3)
advantage = (reward - reward.mean(-1, keepdim=True)) * samples / (samples-1)
plain = (advantage[...,None] * noise / sigma**2).mean(1)
projected = (advantage[...,None] * projected_noise / sigma**2).mean(1)
# Encoder output is unit-normalized, so compare gradients in its tangent space too.
plain -= (plain @ mean)[:,None] * mean
projected -= (projected @ mean)[:,None] * mean
difference = projected-plain
se = difference.std(0) / math.sqrt(groups)
max_z = (difference.mean(0).abs()/se.clamp_min(1e-12)).max().item()
var_plain = plain.var(0).sum().item()
var_projected = projected.var(0).sum().item()
result = {'scope':'Synthetic fixed eight-document pool; not full-corpus training evidence',
          'dimension':dim,'candidates':candidates,'independent_groups':groups,'samples_per_group':samples,
          'max_candidate_score_error':(scores-projected_scores).abs().max().item(),
          'max_reward_error':(reward-projected_reward).abs().max().item(),
          'unprojected_gradient_variance':var_plain,'projected_gradient_variance':var_projected,
          'variance_reduction_factor':var_plain/var_projected,
          'maximum_component_paired_mean_z':max_z,
          'interpretation':'Candidate scores/rewards should be preserved. Paired gradient means should agree within Monte Carlo noise. Lower variance on this fixture is not a demonstrated retrieval improvement.'}
assert result['max_candidate_score_error'] < 1e-10
assert result['max_reward_error'] == 0
assert max_z < 5
assert var_projected < var_plain
out=Path('/home/davidx/Documents/Codex/2026-10-31/i-x20/outputs/embedding-rl-judge-2026-11-06')
(out/'projection-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
