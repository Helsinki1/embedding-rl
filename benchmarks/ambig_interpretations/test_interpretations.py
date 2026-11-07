import json
from pathlib import Path

import pytest
import torch

from .model import InterpretationModel, matching_loss


def test_matching_ignores_label_and_slot_order_and_masks_third():
    targets = torch.eye(4)[:3].unsqueeze(0)
    predictions = targets[:, [2, 0, 1]].clone().requires_grad_()
    assert matching_loss(predictions, targets, torch.tensor([3])).item() == 0
    assert matching_loss(predictions[:, [1, 0, 2]], targets[:, [2, 1, 0]], torch.tensor([3])).item() == 0
    prediction = torch.tensor([[[.8, .2, 0., 0.], [.1, .9, 0., 0.], [0., 0., 1., 0.]]], requires_grad=True)
    loss = matching_loss(prediction, targets, torch.tensor([2])).mean()
    loss.backward()
    assert prediction.grad[0, 2].abs().sum() == 0
    assert prediction.grad[0, :2].abs().sum() > 0
    collapsed = targets[:, :1].expand(-1, 3, -1)
    assert matching_loss(collapsed, targets, torch.tensor([3])).item() > 0.6


@pytest.mark.parametrize("architecture,passes", [("mtp", 1), ("autoregressive", 3)])
def test_real_qwen_slots_causal_and_feedback_differentiable(architecture, passes):
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(42)
    base = Qwen3Model(Qwen3Config(vocab_size=12, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128))
    model = InterpretationModel(base, 1, architecture).eval()
    calls = []
    hook = base.register_forward_hook(lambda *args: calls.append(1))
    ids, mask = torch.tensor([[3, 4, 2]]), torch.ones(1, 3, dtype=torch.long)
    output, states = model(ids, mask, return_states=True)
    hook.remove()
    assert len(calls) == passes
    assert output.shape == (1, 3, 32)
    assert torch.allclose(output.norm(dim=-1), torch.ones(1, 3), atol=1e-6)
    if architecture == "autoregressive":
        grad = torch.autograd.grad(output[:, 2, 0].sum(), states[0], retain_graph=True)[0]
        assert grad.abs().sum() > 0
    loss = matching_loss(output, torch.nn.functional.normalize(torch.randn_like(output), dim=-1), torch.tensor([3])).mean()
    loss.backward()
    assert (model.slots.grad.norm(dim=-1) > 0).all()
    with torch.no_grad():
        before = model(ids, mask)
        model.slots[2].add_(torch.randn(32))
        after = model(ids, mask)
    assert torch.allclose(before[:, :2], after[:, :2], atol=1e-6)


def test_frozen_split_isolation_and_no_missing_targets():
    here = Path(__file__).parent
    rows = {n: [json.loads(x) for x in (here / "data" / f"{n}.jsonl").read_text().splitlines()]
            for n in ("train", "validation", "test")}
    ids = [{r["id"] for r in rs} for rs in rows.values()]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i))
    texts = [{r["query"].strip().lower() for r in rs} for rs in rows.values()]
    assert all(not texts[i] & texts[j] for i in range(3) for j in range(i))
    assert all(len(r["interpretations"]) in (2, 3) for rs in rows.values() for r in rs)
