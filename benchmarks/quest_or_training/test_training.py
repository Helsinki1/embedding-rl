"""CPU checks for objectives, corpus alignment, split isolation, and complete run output."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from .data import REVISION, gold_indices, load_corpus, load_splits, wait_for_corpus
from .objectives import answer_f1, infonce, reinforce
from .train import choose_cutoff, train_one, training_batches
from .frozen_baselines import cached_predictions


def test_answer_set_f1():
    predicted = torch.tensor([[[0, 2], [0, 1]]])
    assert torch.allclose(answer_f1(predicted, [torch.tensor([0, 1])]),
                          torch.tensor([[0.5, 1.0]]))


def test_gold_mapping_uses_membership_lookups_without_scanning_corpus():
    class Index(dict):
        def keys(self):
            raise AssertionError("Do not scan every corpus ID per query")
    mapping = Index({"a": 0, "b": 1})
    actual = gold_indices([{"id": "q", "docs": ["b", "a", "b"]}], mapping, "cpu")
    assert actual[0].tolist() == [0, 1]
    with pytest.raises(ValueError, match="missing"):
        gold_indices([{"id": "q", "docs": ["absent"]}], mapping, "cpu")


def test_epoch_batches_cover_every_example_and_preserve_partial_batch():
    config = {"seed": 42, "epochs": 2, "batch_size": 4}
    batches = list(training_batches(11, config))
    assert batches == list(training_batches(11, config))
    assert list(map(len, batches)) == [4, 4, 3, 4, 4, 3]
    assert sorted(i for batch in batches[:3] for i in batch) == list(range(11))
    assert sorted(i for batch in batches[3:] for i in batch) == list(range(11))
    assert batches[:3] != batches[3:]


def test_reinforce_has_score_gradient_in_reward_direction():
    means = torch.tensor([[0.7, 0.7]], requires_grad=True)
    loss, metrics = reinforce(
        means, means.detach().clone(), torch.eye(2), [torch.tensor([0])],
        k=1, samples=512, radius=1.0, kl_weight=0,
        generator=torch.Generator().manual_seed(7),
    )
    loss.backward()
    assert means.grad[0, 0] < 0 < means.grad[0, 1]
    assert 0 < metrics["reward_f1"] < 1
    assert metrics["kl"] == 0


def test_kl_exact_and_constant_rewards_have_zero_policy_gradient():
    means = torch.tensor([[1.0, 0.0]], requires_grad=True)
    reference = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss, metrics = reinforce(means, reference, torch.eye(2), [torch.tensor([0])],
                              k=2, samples=8, radius=1, kl_weight=0.5)
    loss.backward()
    assert metrics["policy_loss"] == 0
    assert metrics["kl"] == pytest.approx(2.0)
    assert torch.allclose(means.grad, torch.tensor([[1.0, -1.0]]))
    assert reference.grad is None


def test_infonce_trains_every_positive():
    means = torch.zeros(1, 3, requires_grad=True)
    loss = infonce(means, torch.eye(3), [torch.tensor([0, 1])], temperature=1)
    loss.backward()
    assert loss.item() == pytest.approx(np.log(3))
    assert torch.allclose(means.grad, torch.tensor([[-1 / 6, -1 / 6, 1 / 3]]))


def make_corpus(root):
    (root / "results").mkdir()
    for rank in (0, 1):
        ids = [{"index": i, "title": f"d{i}"} for i in range(rank, 4, 2)]
        prefix = root / "results" / f"corpus_{rank}"
        Path(str(prefix) + "_ids.json").write_text(json.dumps(ids))
        np.save(str(prefix) + ".npy", np.eye(4, dtype=np.float16)[rank::2])
        Path(str(prefix) + "_progress.json").write_text(json.dumps({
            "done": 2, "total": 2, "revision": REVISION, "max_length": 32768,
            "identity_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest(),
        }))


def test_corpus_alignment_and_completion_gate(tmp_path):
    make_corpus(tmp_path)
    vectors, titles, _ = load_corpus(tmp_path, "cpu", dimension=4, expected_count=4)
    assert titles == ["d0", "d2", "d1", "d3"]
    assert vectors.argmax(-1).tolist() == [0, 2, 1, 3]
    progress_path = tmp_path / "results/corpus_1_progress.json"
    progress = json.loads(progress_path.read_text())
    progress["done"] = 1
    progress_path.write_text(json.dumps(progress))
    with pytest.raises(RuntimeError, match="incomplete"):
        wait_for_corpus(tmp_path)
    with pytest.raises(TimeoutError):
        wait_for_corpus(tmp_path, wait=True, timeout=0)


def test_corpus_rejects_corrupted_ids(tmp_path):
    make_corpus(tmp_path)
    (tmp_path / "results/corpus_0_ids.json").write_text('[{"index": 0, "title": "wrong"}]')
    with pytest.raises(ValueError, match="checksum"):
        load_corpus(tmp_path, "cpu", dimension=4, expected_count=4)


def test_frozen_split_partitions_all_queries_without_overlap():
    here = Path(__file__).resolve().parent
    train, dev, test, spec = load_splits(here.parent / "quest_or", here / "split.json")
    assert (len(train), len(dev), len(test)) == (363, 78, 78)
    all_ids = spec["train_ids"] + spec["validation_ids"] + spec["test_ids"]
    assert len(set(all_ids)) == len(all_ids) == 519


def test_rejects_overlapping_split_ids(tmp_path):
    here = Path(__file__).resolve().parent
    spec = json.loads((here / "split.json").read_text())
    spec["test_ids"][0] = spec["train_ids"][0]
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="partition"):
        load_splits(here.parent / "quest_or", split_path)


def test_cached_or_union_deduplicates_and_whole_query_remains_single():
    query = row("q", "query", ["a", "b"])
    query["components"] = ["first", "second"]
    lookup = {("q", "original"): 0, ("q", "0"): 1, ("q", "1"): 2}
    indices = np.array([[2, 0], [0, 1], [1, 2]])
    union = cached_predictions([query], "frozen_or_union", 2, lookup, ["a", "b", "c"], indices)
    single = cached_predictions([query], "frozen_single", 1, lookup, ["a", "b", "c"], indices)
    assert union[0]["docs"] == ["a", "b", "c"]
    assert single[0]["docs"] == ["c"]


class TinyEncoder:
    def __init__(self, model_path, config, device):
        self.model = torch.nn.Linear(2, 2, bias=False).to(device)
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(2))
        self.device = device

    def encode(self, texts):
        inputs = torch.tensor([[float(x) for x in text.split(",")] for text in texts],
                              device=self.device)
        return F.normalize(self.model(inputs), dim=-1)


def row(qid, query, docs):
    return {"id": qid, "query": query, "docs": docs,
            "metadata": {"domain": "films", "template": "_ or _"},
            "components": ["THIS MUST NEVER BE ENCODED"],
            "original_query": "THIS MUST NEVER BE ENCODED"}


def test_cutoff_tie_prefers_smaller_k():
    gold = [row("q", "query", ["d0"])]
    best, _ = choose_cutoff(gold, [[1, 2]], ["d0", "d1", "d2"], max_k=2)
    assert best["k"] == 1


@pytest.mark.parametrize("method", ["rl", "infonce"])
def test_end_to_end_run_with_real_gradients_and_held_out_scoring(tmp_path, method):
    train = [row("train0", "0.8,0.6", ["d1"]), row("train1", "0.9,0.4", ["d1"])]
    dev = [row("dev", "0.6,0.8", ["d1"])]
    test = [row("test", "0.6,0.7", ["d1"])]
    config = dict(seed=42, steps=3, batch_size=2, eval_every=1, learning_rate=0.05,
                  max_k=3, samples_per_query=16, exploration_radius=0.5,
                  kl_weight=0.001, temperature=0.1)
    documents = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    directory = tmp_path / method
    result = train_one(method, config, None, documents, ["d0", "d1", "d2"],
                       train, dev, test, directory, "cpu", {}, TinyEncoder)
    assert result["method"] == method
    assert result["test"]["overall"]["queries"] == 1
    assert (directory / "selection.json").exists()
    trained_selection = json.loads((directory / "best_trained_selection.json").read_text())
    assert 1 <= trained_selection["checkpoint_step"] <= config["steps"]
    assert (directory / "test_predictions.jsonl").exists()
    last = torch.load(directory / "last.pt", weights_only=True)
    assert not torch.equal(last["weight"], torch.eye(2))
    assert len((directory / "metrics.jsonl").read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        train_one(method, config, None, documents, ["d0", "d1", "d2"],
                  train, dev, test, directory, "cpu", {}, TinyEncoder)


def test_qwen_lora_pooling_padding_and_backward(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    from . import train as training

    # Exercise the actual Qwen/PEFT path with a randomly initialized miniature model.
    # The production dimension guard remains 4096 outside this test.
    monkeypatch.setattr(training, "DIMENSION", 32)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "short": 2, "long": 3},
                                    unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                      pad_token="[PAD]", padding_side="left",
                                      model_input_names=["input_ids", "attention_mask"])
    wrapped.save_pretrained(tmp_path)
    base = Qwen3Model(Qwen3Config(vocab_size=4, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=16, max_position_embeddings=128))
    base.save_pretrained(tmp_path)
    encoder = training.QwenEncoder(tmp_path, dict(max_length=128, lora_rank=2, lora_alpha=4), "cpu")
    encoder.model.eval()
    with torch.no_grad():
        single = encoder.encode(["short"])
        batch = encoder.encode(["short", "long long short"])
    assert batch.shape == (2, 32)
    assert torch.allclose(batch.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(single[0], batch[0], atol=0.01)
    encoder.model.train()
    encoder.encode(["short", "long long short"])[:, 0].sum().backward()
    adapter_grads = [p.grad for name, p in encoder.model.named_parameters() if "lora_B" in name]
    assert any(g is not None and g.abs().sum() > 0 for g in adapter_grads)
    assert all(p.grad is None for p in encoder.model.parameters() if not p.requires_grad)
