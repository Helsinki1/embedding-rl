"""Exercise slice retrieval, joint rewards, gradients, and complete small runs."""
import json

import pytest
import torch
import torch.nn.functional as F

from .objectives import infonce, piece_rankings, prepare_documents, reinforce, split_vectors, union_f1
from .train import choose_cutoff, score_rankings, train_one


def row(identifier, query, docs):
    return {"id": identifier, "query": query, "docs": docs,
            "metadata": {"domain": "films", "template": "_ or _"},
            "components": ["not an encoder input"]}


def test_balanced_slices_cover_all_coordinates_and_match_document_coordinates():
    raw = torch.arange(1, 4097).float()[None]
    pieces = split_vectors(raw)
    assert [p.shape[-1] for p in pieces] == [1366, 1365, 1365]
    for piece, start, end in zip(pieces, [0, 1366, 2731], [1366, 2731, 4096]):
        assert torch.equal(piece, F.normalize(raw[:, start:end], dim=-1))
    documents = torch.randn(12, 4096)
    documents[7] = raw[0]
    assert piece_rankings(raw, prepare_documents(documents), 1).tolist() == [[[7], [7], [7]]]
    assert torch.equal(torch.cat(torch.tensor_split(raw, 3, dim=-1), dim=-1), raw)


def test_union_reward_deduplicates_both_true_and_false_positives():
    indices = torch.tensor([[[[0, 2], [1, 2], [0, 1]], [[0, 0], [0, 0], [0, 0]]]])
    reward = union_f1(indices, [torch.tensor([0, 1])])
    assert reward.tolist()[0] == pytest.approx([0.8, 2 / 3])
    rows = [row("q", "q", ["a", "b"])]
    result, predictions, _ = score_rankings(rows, indices[:, 0].tolist(), ["a", "b", "c"], 2)
    assert result["overall"]["f1"] == pytest.approx(reward[0, 0].item())
    assert predictions[0]["docs"] == ["a", "c", "b"]
    best, _ = choose_cutoff(rows, [[[2, 2], [2, 2], [2, 2]]], ["a", "b", "c"], 2)
    assert best["k"] == 1


def test_infonce_equals_explicit_union_probability_and_has_gradients_in_each_piece():
    torch.manual_seed(3)
    means = torch.randn(2, 11, requires_grad=True)
    raw = torch.randn(6, 11)
    documents = prepare_documents(raw)
    positives = [torch.tensor([0, 2]), torch.tensor([1, 3, 5])]
    loss = infonce(means, documents, positives, 0.2)
    mass = sum(torch.exp((query @ doc.T) / 0.2)
               for query, doc in zip(split_vectors(means), documents))
    expected = torch.stack([-(mass[i, ids] / mass[i].sum()).log().mean()
                            for i, ids in enumerate(positives)]).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert all(torch.isfinite(p).all() and p.abs().sum() > 0
               for p in torch.tensor_split(means.grad, 3, dim=-1))


def test_joint_policy_gradient_favors_correct_documents_and_kl_is_exact():
    # Each slice starts halfway between two opposing document choices.
    means = torch.tensor([[0., 1., 0., 1., 0., 1.]], requires_grad=True)
    documents = prepare_documents(torch.tensor([[1., 0., 1., 0., 1., 0.],
                                               [-1., 0., -1., 0., -1., 0.]]))
    loss, info = reinforce(means, means.detach(), documents, [torch.tensor([0])],
                           1, 4096, 0.8, 0.0, torch.Generator().manual_seed(4))
    loss.backward()
    assert (means.grad[0, [0, 2, 4]] < 0).all()
    assert info["kl"] == 0 and info["reward_std"] > 0
    anchor = torch.tensor([[1., 0., 1., 0., 1., 0.]])
    _, info = reinforce(means.detach(), anchor, documents, [torch.tensor([0])],
                         1, 8, 0.2, 0.01, torch.Generator().manual_seed(4))
    assert info["kl"] == pytest.approx(150.0, rel=1e-5)


def test_tied_rewards_have_no_policy_gradient():
    means = torch.randn(2, 9, requires_grad=True)
    documents = prepare_documents(torch.randn(5, 9))
    loss, info = reinforce(means, means.detach(), documents,
                           [torch.arange(5), torch.arange(5)], 5, 8, 0.2, 0.0)
    loss.backward()
    assert torch.count_nonzero(means.grad) == 0
    assert info["zero_advantage_fraction"] == 1


class TinyEncoder:
    def __init__(self, path, config, device):
        self.model = torch.nn.Embedding(5, 11).to(device)
        self.device = device

    def encode(self, texts):
        return F.normalize(self.model(torch.tensor([int(t) for t in texts], device=self.device)), dim=-1)


@pytest.mark.parametrize("method", ["frozen", "infonce", "rl"])
def test_complete_run_preserves_data_and_writes_real_scores(tmp_path, method):
    config = dict(seed=42, epochs=2, batch_size=2, micro_batch_size=1, eval_batch_size=2,
                  eval_every=2, learning_rate=0.01, max_length=64, lora_rank=32,
                  lora_alpha=64, samples_per_query=8, exploration_radius=0.2,
                  kl_weight=0.01, temperature=0.1, max_k=2)
    train = [row(str(i), str(i), ["a", "b"]) for i in range(3)]
    dev, test = [row("dev", "3", ["a"])], [row("test", "4", ["b", "c"])]
    documents = prepare_documents(torch.randn(4, 11))
    root = tmp_path / method
    result = train_one(method, config, None, documents, list("abcd"), train, dev, test,
                       root, "cpu", {}, encoder_factory=TinyEncoder)
    assert json.loads((root / "progress.json").read_text())["stage"] == "complete"
    if method != "frozen":
        history = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        assert len(history) == 4 and history[-1]["examples_seen"] == 6
        assert all(entry["grad_norm"] >= 0 for entry in history)
        assert result["best_trained"]["checkpoint_step"] > 0
        result = result["selected"]
        root = root / "selected"
    predictions = [json.loads(line) for line in (root / "test_predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 1
    assert len(predictions[0]["docs"]) == len(set(predictions[0]["docs"]))
    assert 0 <= result["test"]["overall"]["f1"] <= 1
    with pytest.raises(FileExistsError):
        train_one(method, config, None, documents, list("abcd"), train, dev, test,
                  tmp_path / method, "cpu", {}, encoder_factory=TinyEncoder)


def test_actual_qwen_rank32_updates_adapters_and_keeps_base_frozen(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    from benchmarks.quest_or_training import train as original

    monkeypatch.setattr(original, "DIMENSION", 32)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "short": 2, "long": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                     pad_token="[PAD]", padding_side="left",
                                     model_input_names=["input_ids", "attention_mask"])
    wrapped.save_pretrained(tmp_path)
    Qwen3Model(Qwen3Config(vocab_size=4, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=16)).save_pretrained(tmp_path)
    encoder = original.QwenEncoder(tmp_path, dict(max_length=64, lora_rank=32, lora_alpha=64), "cpu")
    assert encoder.model.peft_config["default"].r == 32
    initial = {name: parameter.detach().clone() for name, parameter in encoder.model.named_parameters()}
    documents = prepare_documents(torch.randn(6, 32))
    optimizer = torch.optim.AdamW([p for p in encoder.model.parameters() if p.requires_grad], lr=0.01)
    encoder.model.train()
    infonce(encoder.encode(["short", "long long"]), documents,
            [torch.tensor([0, 1]), torch.tensor([2, 3])], 0.1).backward()
    optimizer.step()
    changed = []
    for name, parameter in encoder.model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None and torch.equal(parameter, initial[name])
        elif not torch.equal(parameter, initial[name]):
            changed.append(name)
    assert changed and all("lora_" in name for name in changed)
