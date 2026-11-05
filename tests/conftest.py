import torch
import torch.nn.functional as F
import pytest

from embedding_rl.config import Config, digest, prepared_dir, write_json


class TinyEncoder:
    """A real differentiable model for offline integration tests; no HF downloads."""
    def __init__(self, config, manifest, device):
        self.model = torch.nn.Linear(2, 2, bias=False).to(device)
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(2, device=device))
        self.device = device

    def encode(self, texts, query=False):
        inputs = torch.tensor([[float(x) for x in text.split(",")] for text in texts],
                              device=self.device)
        return F.normalize(self.model(inputs), dim=-1)

    @torch.no_grad()
    def encode_batches(self, texts, batch_size, query=False):
        return self.encode(texts, query).cpu()


@pytest.fixture
def prepared(tmp_path):
    config = Config(train_queries=2, validation_queries=1, pool_size=4, ndcg_k=2,
                    recall_k=3, steps=3, batch_size=2, eval_every=1, learning_rate=0.01,
                    exploration_radius=0.8, judge_batch_size=2, judge_max_new_pairs=8,
                    wandb_mode="disabled")
    bundle = {
        "docs": [{"id": f"d{i}", "title": f"Document {i}", "text": "Synthetic fixture"}
                 for i in range(4)],
        "queries": {"q0": "0.8,0.6", "q1": "0.7,0.7", "q2": "0.7,0.8", "q3": "0.6,0.8"},
        "splits": {"train": ["q0", "q1"], "validation": ["q2"], "test": ["q3"]},
        "qrels": {f"q{i}": {"d1": 1} for i in range(4)},
        "pools": {"q0": [0, 1, 2, 3], "q1": [0, 1, 2, 3]},
        "query_vector_ids": ["q0", "q1", "q2"],
        "revisions": {"model": "synthetic", "corpus": "synthetic", "qrels": "synthetic"},
    }
    encoder = TinyEncoder(config, {}, "cpu")
    tensors = {
        "documents": torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
        "queries": encoder.encode_batches([bundle["queries"][qid]
                                           for qid in bundle["query_vector_ids"]], 3),
    }
    path = prepared_dir(tmp_path, config)
    manifest = {"schema": 1, "preparation": config.preparation_settings(),
                "revisions": bundle["revisions"], "data_fingerprint": digest(bundle)}
    write_json(path / "data.json", bundle)
    write_json(path / "manifest.json", manifest)
    torch.save(tensors, path / "vectors.pt")
    return tmp_path, config, bundle
