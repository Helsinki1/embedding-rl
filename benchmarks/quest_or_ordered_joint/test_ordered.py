import pytest
import torch
import torch.nn.functional as F

from .objectives import ordered_clause_infonce
from .model import OutputHead, HeadModel


def test_order_is_supervised_not_permutation_invariant():
    bank = torch.eye(6)
    ids = torch.tensor([[0, 1, 2]])
    outputs = bank[:3][None].clone().requires_grad_()
    good, _ = ordered_clause_infonce(outputs, bank, ids, .1)
    bad, _ = ordered_clause_infonce(outputs[:, [1, 2, 0]], bank, ids, .1)
    assert bad > good + 1
    good.backward()
    assert outputs.grad is not None and torch.isfinite(outputs.grad).all()
    assert bank.grad is None


def test_masked_slot_and_same_query_clauses():
    bank = torch.eye(5)
    outputs = torch.stack((bank[0], bank[1], bank[4]))[None].requires_grad_()
    ids = torch.tensor([[0, 1, -1]])
    loss, _ = ordered_clause_infonce(outputs, bank, ids, .2)
    # Two positives, each with three unrelated negatives; its sibling is excluded.
    expected = torch.log1p(3 * torch.exp(torch.tensor(-5.)))
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert torch.count_nonzero(outputs.grad[:, 2]) == 0


@pytest.mark.parametrize("variant", ["pooled_mlp", "attention_mlp"])
def test_output_heads_and_padding(variant):
    torch.manual_seed(3)
    head = OutputHead(16, variant, width=8, reader_width=8, reader_heads=2)
    x = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    y = head(x, mask)
    assert y.shape == (2, 3, 16)
    assert torch.allclose(y.norm(dim=-1), torch.ones(2, 3), atol=1e-6)
    changed = x.detach().clone()
    changed[0, :2] = 1000
    assert torch.allclose(head(changed, mask), y, atol=1e-6)
    targets = F.normalize(torch.randn_like(y), dim=-1)
    (y * targets).sum().backward()
    assert head.mlp[-1].weight.grad.norm() > 0
    if variant == "attention_mlp":
        assert torch.all(head.readers.grad.norm(dim=-1) > 0)
        assert x.grad[0, 2:-1].norm() > 0


@pytest.mark.parametrize("variant", ["pooled_mlp", "attention_mlp", "one_pass", "three_pass"])
def test_real_qwen_lora_joint_backward_and_checkpoint(variant, tmp_path):
    from transformers import Qwen3Config, Qwen3Model
    from peft import LoraConfig, get_peft_model
    from benchmarks.quest_or_multivector.model import ThreeSlotModel
    from benchmarks.quest_or_multivector.objectives import multi_vector_infonce
    from benchmarks.quest_or_training.train import save_weights, restore_weights
    from types import SimpleNamespace
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, attention_dropout=0.)
    base = get_peft_model(Qwen3Model(config), LoraConfig(r=4, lora_alpha=8,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    if variant in ("one_pass", "three_pass"):
        model = ThreeSlotModel(base, base.get_input_embeddings().weight[1], variant)
    else:
        model = HeadModel(base, OutputHead(32, variant, width=16, reader_width=16, reader_heads=4))
    model.eval()
    tokens = torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]])
    mask = tokens.ne(0).long()
    vectors = model(tokens, mask)
    bank = F.normalize(torch.randn(8, 32), dim=-1)
    target_ids = torch.tensor([[0, 1, 2], [3, 4, -1]])
    documents = F.normalize(torch.randn(12, 32), dim=-1)
    doc = multi_vector_infonce(vectors, documents, [torch.tensor([1, 2]), torch.tensor([4])], .1)
    clause, _ = ordered_clause_infonce(vectors, bank, target_ids, .1)
    (doc+clause).backward()
    assert all(p.grad is None for p in base.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.norm() > 0 for n, p in base.named_parameters() if "lora_B" in n)
    encoder = SimpleNamespace(model=model)
    save_weights(encoder, tmp_path / "weights.pt")
    old = model(tokens, mask).detach()
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(.1)
    restore_weights(encoder, tmp_path / "weights.pt")
    assert torch.allclose(model(tokens, mask), old, atol=1e-6)


@pytest.mark.parametrize("variant", ["one_pass", "three_pass"])
def test_full_batch_matches_accumulated_joint_gradient(variant):
    from transformers import Qwen3Config, Qwen3Model
    from peft import LoraConfig, get_peft_model
    from benchmarks.quest_or_multivector.model import ThreeSlotModel
    from benchmarks.quest_or_multivector.objectives import multi_vector_infonce
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=40, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, attention_dropout=0.)
    base = get_peft_model(Qwen3Model(config), LoraConfig(r=4, lora_alpha=8,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model = ThreeSlotModel(base, base.get_input_embeddings().weight[1], variant).train()
    tokens = torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8], [0, 0, 9, 10], [11, 12, 13, 14]])
    teachers = F.normalize(torch.randn(16, 32), dim=-1)
    targets = torch.tensor([[0, 1, 2], [3, 4, -1], [5, 6, 7], [8, 9, -1]])
    documents = F.normalize(torch.randn(20, 32), dim=-1)
    gold = [torch.tensor([i, i+4]) for i in range(4)]

    def loss(start, end):
        vectors = model(tokens[start:end], tokens[start:end].ne(0).long())
        doc = multi_vector_infonce(vectors, documents, gold[start:end], .1)
        clause, _ = ordered_clause_infonce(vectors, teachers, targets[start:end], .1)
        return doc + clause

    loss(0, 4).backward()
    full = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    for start in range(0, 4, 2):
        (loss(start, start+2)/2).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, full[name], atol=2e-5, rtol=2e-4)


def test_three_epochs_preserve_first_epoch_and_cover_all_queries():
    from collections import Counter
    from benchmarks.quest_or_training.train import training_batches
    config = {"seed": 42, "epochs": 3, "batch_size": 32}
    batches = list(training_batches(8500, config))
    assert len(batches) == 798
    assert batches[:266] == list(training_batches(8500, {**config, "epochs": 1}))
    for epoch in range(3):
        seen = [i for batch in batches[epoch*266:(epoch+1)*266] for i in batch]
        assert sorted(seen) == list(range(8500))
    assert set(Counter(i for batch in batches for i in batch).values()) == {3}
