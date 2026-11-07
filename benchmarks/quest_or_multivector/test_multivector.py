"""Check causality, feedback gradients, collective retrieval, and complete training."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from .objectives import multi_vector_infonce, multi_vector_reinforce, union_answer_f1
from .train import choose_cutoff, diagnostics, score_rankings, train_one


def row(identifier, query, docs):
    return {"id": identifier, "query": query, "docs": docs,
            "metadata": {"domain": "films", "template": "_ or _"},
            "components": ["never encode this"], "original_query": "never encode this"}


def test_union_loss_is_permutation_invariant_and_rewards_complementary_slots():
    documents = torch.eye(4)
    positives = [torch.tensor([0, 1])]
    distinct = documents[[0, 1, 0]].unsqueeze(0).requires_grad_()
    collapsed = documents[[0, 0, 0]].unsqueeze(0)
    loss = multi_vector_infonce(distinct, documents, positives, 0.1)
    assert loss < multi_vector_infonce(collapsed, documents, positives, 0.1)
    assert torch.allclose(loss, multi_vector_infonce(distinct[:, [2, 0, 1]], documents, positives, 0.1))
    loss.backward()
    assert torch.isfinite(distinct.grad).all()


def test_every_positive_receives_gradient_and_collapsed_case_matches_single_vector():
    vectors = torch.zeros(1, 3, 4, requires_grad=True)
    loss = multi_vector_infonce(vectors, torch.eye(4), [torch.tensor([0, 1])], 1.0)
    loss.backward()
    assert loss.item() == pytest.approx(torch.log(torch.tensor(4.)).item())
    assert (vectors.grad[:, :, :2] < 0).all()
    assert (vectors.grad[:, :, 2:] > 0).all()
    assert torch.allclose(vectors.grad.sum(1), torch.tensor([[-0.25, -0.25, 0.25, 0.25]]))


def test_union_scoring_deduplicates_and_cutoff_ties_prefer_small_k():
    rows = [row("q", "q", ["a", "b"])]
    rankings = [[[0, 2], [1, 2], [0, 1]]]
    summary, predictions, _ = score_rankings(rows, rankings, ["a", "b", "c"], 1)
    assert predictions[0]["docs"] == ["a", "b"]
    assert summary["overall"]["f1"] == 1
    info = diagnostics(rows, rankings, ["a", "b", "c"], 1, [[0.1, 0.2, 0.3]])
    assert info["mean_duplicate_retrievals"] == 1
    assert info["mean_unique_relevant_per_slot"] == [0, 1, 0]
    best, _ = choose_cutoff(rows, [[[2, 2], [2, 2], [2, 2]]], ["a", "b", "c"], 2)
    assert best["k"] == 1


@pytest.fixture
def tiny_qwen(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    from benchmarks.quest_or_training import train as original
    from . import model

    monkeypatch.setattr(original, "DIMENSION", 32)
    monkeypatch.setattr(model, "DIMENSION", 32)
    tokenizer = Tokenizer(WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "short": 2, "long": 3, "[EOS]": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        padding_side="left", model_input_names=["input_ids", "attention_mask"])
    wrapped.save_pretrained(tmp_path)
    torch.manual_seed(1)
    base = Qwen3Model(Qwen3Config(
        vocab_size=5, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=128))
    base.save_pretrained(tmp_path)
    return tmp_path


def make_encoder(path, variant):
    from .model import ThreeVectorEncoder
    torch.manual_seed(42)
    return ThreeVectorEncoder(path, dict(
        max_length=128, lora_rank=16, lora_alpha=32, variant=variant, slot_init_noise=0.01), "cpu")


@pytest.mark.parametrize("variant,passes", [("one_pass", 1), ("three_pass", 3)])
def test_real_qwen_shapes_padding_causality_and_trainable_slots(tiny_qwen, variant, passes):
    encoder = make_encoder(tiny_qwen, variant)
    encoder.model.eval()
    calls = []
    handle = encoder.model.backbone.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        before = encoder.encode(["short", "long long short"])
    assert len(calls) == passes
    handle.remove()
    with torch.no_grad():
        alone = encoder.encode(["short"])
        assert torch.allclose(alone[0], before[0], atol=0.015)
        encoder.model.slots[2].add_(torch.randn(32) * 0.1)
        after = encoder.encode(["short", "long long short"])
    assert before.shape == (2, 3, 32)
    assert torch.allclose(before.norm(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.allclose(before[:, :2], after[:, :2], atol=1e-5)
    assert not torch.allclose(before[:, 2], after[:, 2])
    encoder.model.train()
    vectors = encoder.encode(["short", "long long short"])
    documents = F.normalize(torch.randn(8, 32), dim=-1)
    loss = multi_vector_infonce(vectors, documents, [torch.tensor([0, 1]), torch.tensor([2, 3])], 0.1)
    loss.backward()
    assert (encoder.model.slots.grad.norm(dim=-1) > 0).all()
    adapter_gradients = [p.grad for n, p in encoder.model.named_parameters() if "lora_B" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in adapter_gradients)
    assert all(p.grad is None for p in encoder.model.parameters() if not p.requires_grad)


def test_sequential_feedback_is_differentiable_and_first_output_matches(tiny_qwen):
    parallel = make_encoder(tiny_qwen, "one_pass")
    sequential = make_encoder(tiny_qwen, "three_pass")
    parallel.model.eval()
    sequential.model.eval()
    ids = torch.tensor([[2, 3, 2]])
    mask = torch.ones_like(ids)
    a = parallel.model(ids, mask)
    b, states = sequential.model(ids, mask, return_states=True)
    assert torch.allclose(a[:, 0], b[:, 0], atol=0.01)
    assert torch.equal(parallel.model.slots, sequential.model.slots)
    assert not torch.allclose(a[:, 1:], b[:, 1:], atol=0.001)
    gradient = torch.autograd.grad(b[:, 2, 0].sum(), states[0])[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("variant", ["one_pass", "three_pass"])
def test_tiny_qwen_can_learn_a_small_retrieval_problem(tiny_qwen, variant):
    encoder = make_encoder(tiny_qwen, variant)
    encoder.model.train()
    texts = ["short", "long", "short short", "long long"]
    documents = F.normalize(torch.randn(8, 32), dim=-1)
    positives = [torch.tensor([2 * i, 2 * i + 1]) for i in range(4)]
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.003, weight_decay=0)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = multi_vector_infonce(encoder.encode(texts), documents, positives, 0.1)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
    assert sum(losses[-5:]) / 5 < losses[0] * 0.8


def test_launcher_waits_for_evaluation_and_treats_failures_as_terminal(tmp_path):
    from .launch_when_free import dependencies_complete
    for method in ("infonce", "rl"):
        (tmp_path / f"{method}_status.json").write_text('{"stage":"complete"}')
    assert not dependencies_complete(tmp_path)[0]
    (tmp_path / "infonce_evaluation_status.json").write_text('{"stage":"complete"}')
    assert not dependencies_complete(tmp_path)[0]
    (tmp_path / "rl_evaluation_status.json").write_text('{"stage":"failed"}')
    assert dependencies_complete(tmp_path)[0]


class TinyEncoder:
    def __init__(self, model_path, config, device):
        self.model = torch.nn.Linear(2, 6, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(2).repeat(3, 1))

    def encode(self, texts):
        values = torch.tensor([[float(x) for x in text.split(",")] for text in texts])
        return F.normalize(self.model(values).reshape(-1, 3, 2), dim=-1)


@pytest.mark.parametrize("variant", ["one_pass", "three_pass"])
@pytest.mark.parametrize("method", ["infonce", "rl"])
def test_training_selection_checkpoint_and_partial_microbatch(tmp_path, variant, method):
    train = [row(f"t{i}", f"0.{i+1},0.6", ["b"]) for i in range(5)]
    dev = [row("v", "0.6,0.7", ["b"])]
    test = [row("x", "0.6,0.8", ["b"])]
    documents = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    config = dict(seed=42, epochs=1, batch_size=3, micro_batch_size=2, eval_batch_size=2,
                  eval_every=1, learning_rate=0.05, temperature=0.1, max_k=2, variant=variant,
                  method=method, samples_per_query=32, exploration_radius=0.8, kl_weight=0.001)
    path = tmp_path / variant
    result = train_one(config, None, documents, ["a", "b", "c"], train, dev, test,
                       path, "cpu", {}, TinyEncoder)
    assert result["variant"] == variant
    assert result["method"] == method
    assert result["test"]["overall"]["queries"] == 1
    metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert [m["examples_seen"] for m in metrics] == [3, 5]
    selection = json.loads((path / "selection.json").read_text())
    assert 0 <= selection["checkpoint_step"] <= 2
    assert (path / "test_predictions.jsonl").exists()
    last = torch.load(path / "last.pt", weights_only=True)
    assert not torch.equal(last["weight"], torch.eye(2).repeat(3, 1))
    if method == "rl":
        reference = json.loads((path / "rl_reference.json").read_text())
        assert reference["shape"] == [5, 3, 2]
        assert result["rl_reward_k"] == result["initial_validation"]["k"]
        assert all("reward_f1" in m and "kl" in m for m in metrics)
        assert metrics[0]["kl"] == 0
    with pytest.raises(FileExistsError):
        train_one(config, None, documents, ["a", "b", "c"], train, dev, test,
                  path, "cpu", {}, TinyEncoder)


def test_gradient_accumulation_matches_full_batch(tmp_path):
    rows = [row(str(i), f"0.{i+1},0.6", ["b"]) for i in range(5)]
    documents = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    config = dict(seed=42, epochs=1, batch_size=5, eval_batch_size=2, eval_every=1,
                  learning_rate=0.01, temperature=0.1, max_k=2, variant="one_pass")
    for micro in (2, 5):
        train_one(dict(config, micro_batch_size=micro), None, documents, ["a", "b", "c"],
                  rows, rows[:1], rows[1:2], tmp_path / str(micro), "cpu", {}, TinyEncoder)
    a = torch.load(tmp_path / "2/last.pt", weights_only=True)
    b = torch.load(tmp_path / "5/last.pt", weights_only=True)
    assert torch.allclose(a["weight"], b["weight"], atol=1e-6)


@pytest.mark.parametrize("variant", ["one_pass", "three_pass"])
def test_cached_evaluation_matches_full_prefixes(tiny_qwen, variant):
    encoder = make_encoder(tiny_qwen, variant)
    encoder.model.eval()
    with torch.no_grad():
        expected = encoder.encode(["short", "long long short"])
        encoder.evaluation_kv_cache = True
        actual = encoder.encode(["short", "long long short"])
        alone = encoder.encode(["short"])
    assert torch.allclose(actual, expected, atol=0.01, rtol=0.01)
    assert torch.allclose(alone[0], actual[0], atol=0.015, rtol=0.01)
    encoder.model.train()
    encoder.encode(["short"]).sum().backward()
    assert (encoder.model.slots.grad.norm(dim=-1) > 0).all()


@pytest.mark.parametrize("budget", ["steps", "time"])
def test_short_run_budget_preserves_selection_and_test(tmp_path, budget):
    import time
    rows = [row(str(i), f"0.{i+1},0.6", ["b"]) for i in range(5)]
    documents = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    config = dict(seed=42, epochs=1, batch_size=2, micro_batch_size=2, eval_batch_size=2,
                  eval_every=300, learning_rate=0.01, temperature=0.1, max_k=2,
                  variant="one_pass")
    started = None
    if budget == "steps":
        config["max_steps"] = 1
    else:
        config.update(max_run_seconds=2, finalization_reserve_seconds=1)
        started = time.monotonic() - 2
    result = train_one(config, None, documents, ["a", "b", "c"], rows, rows[:1], rows[1:2],
                       tmp_path / budget, "cpu", {}, TinyEncoder, started)
    assert result["completed_steps"] == 1
    assert result["examples_seen"] == 2
    assert result["stop_reason"] == ("max_steps" if budget == "steps" else "wall_time_budget")
    assert result["test"]["overall"]["queries"] == 1
    assert (tmp_path / budget / "best_trained.pt").exists()


def test_canonical_split_counts_hashes_and_strata():
    from collections import Counter
    from benchmarks.quest_or_balanced.build import group, read_rows
    from benchmarks.quest_or_balanced.data import load_splits
    root = Path(__file__).resolve().parents[1]
    source = root / "quest_or_synthetic"
    train, validation, test, manifest = load_splits(root / "quest_or_balanced", source)
    assert [len(train), len(validation), len(test)] == [3600, 400, 1000]
    assert manifest["training_development_side"] == 4000
    assert manifest["test_fraction"] == 0.2
    population = Counter(group(r) for name in ("train", "validation", "test")
                         for r in read_rows(source / "data" / (name + ".jsonl")))
    for rows in (train, validation, test):
        counts = Counter(map(group, rows))
        assert all(abs(counts[key] - len(rows) * n / sum(population.values())) <= 1.01
                   for key, n in population.items())


def test_rl_reward_deduplicates_across_all_heads():
    indices = torch.tensor([[[[0, 0], [0, 1], [1, 2]], [[0, 0], [1, 1], [0, 1]]]])
    actual = union_answer_f1(indices, [torch.tensor([0, 1])])
    assert torch.allclose(actual, torch.tensor([[0.8, 1.0]]))


def test_rl_policy_gradient_moves_all_heads_toward_rewarded_document():
    means = torch.full((1, 3, 2), 0.7, requires_grad=True)
    loss, metrics = multi_vector_reinforce(
        means, means.detach().clone(), torch.eye(2), [torch.tensor([0])],
        k=1, samples=2048, radius=1.0, kl_weight=0,
        generator=torch.Generator().manual_seed(7))
    loss.backward()
    assert (means.grad[..., 0] < 0).all() and (means.grad[..., 1] > 0).all()
    assert metrics["kl"] == 0 and 0 < metrics["reward_f1"] < 1


def test_rl_joint_kl_and_tied_rewards_have_no_score_gradient():
    means = torch.tensor([[[1., 0.]] * 3], requires_grad=True)
    reference = torch.tensor([[[0., 1.]] * 3], requires_grad=True)
    loss, metrics = multi_vector_reinforce(
        means, reference, torch.eye(2), [torch.tensor([0])],
        k=2, samples=8, radius=1.0, kl_weight=0.5)
    loss.backward()
    assert metrics["policy_loss"] == 0 and metrics["zero_advantage_fraction"] == 1
    assert metrics["kl"] == pytest.approx(6.0)
    assert torch.allclose(means.grad, torch.tensor([[[1., -1.]] * 3]))
    assert reference.grad is None


@pytest.mark.parametrize("variant", ["one_pass", "three_pass"])
def test_real_qwen_rl_updates_slots_and_adapter(tiny_qwen, variant):
    encoder = make_encoder(tiny_qwen, variant)
    encoder.model.train()
    vectors = encoder.encode(["short", "long long short"])
    documents = F.normalize(torch.randn(8, 32), dim=-1)
    loss, metrics = multi_vector_reinforce(
        vectors, vectors.detach().clone(), documents, [torch.tensor([0, 1]), torch.tensor([2, 3])],
        k=1, samples=128, radius=1.0, kl_weight=0.01,
        generator=torch.Generator().manual_seed(7))
    loss.backward()
    assert metrics["reward_std"] > 0
    assert (encoder.model.slots.grad.norm(dim=-1) > 0).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in encoder.model.named_parameters() if "lora_B" in n)
    assert all(p.grad is None for p in encoder.model.parameters() if not p.requires_grad)


@pytest.mark.parametrize("variant", ["one_pass", "three_pass"])
def test_reference_cache_matches_shuffled_training_padding(tiny_qwen, variant):
    from benchmarks.quest_or_training.train import training_batches
    from .train import freeze_reference_batches
    encoder = make_encoder(tiny_qwen, variant)
    encoder.evaluation_kv_cache = True
    encoder.model.train()
    config = dict(seed=42, epochs=1, steps=3, batch_size=3, micro_batch_size=2)
    rows = [{"query": "short " + "long " * i} for i in range(8)]
    references = freeze_reference_batches(encoder, rows, config)
    assert encoder.model.training and encoder.evaluation_kv_cache
    assert [len(batch) for batch in references] == [3, 3, 2]
    for step, batch in enumerate(training_batches(len(rows), config)):
        for start in range(0, len(batch), 2):
            ids = batch[start:start + 2]
            actual = encoder.encode([rows[i]["query"] for i in ids])
            expected = references[step][start:start + len(ids)]
            assert not expected.requires_grad
            assert torch.equal(actual.detach(), expected)
