from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Config:
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    corpus_revision: str = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
    qrels_revision: str = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    max_length: int = 512
    seed: int = 42
    train_queries: int = 200
    validation_queries: int = 50
    pool_size: int = 100
    encode_batch_size: int = 32
    steps: int = 100
    batch_size: int = 4
    samples_per_query: int = 8
    learning_rate: float = 1e-6
    exploration_radius: float = 0.15
    kl_weight: float = 0.01
    temperature: float = 0.05
    eval_every: int = 20
    ndcg_k: int = 10
    recall_k: int = 25
    judge_model: str = "gpt-5.4-mini-2026-03-17"
    judge_reasoning_effort: str = "low"
    judge_batch_size: int = 10
    judge_max_new_pairs: int = 20000
    judge_max_chars: int = 12000
    judge_max_output_tokens: int = 4096
    wandb_project: str = "embedding-rl"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    wandb_mode: str = "online"

    def __post_init__(self):
        for key in (
            "train_queries", "validation_queries", "pool_size", "encode_batch_size", "steps",
            "batch_size", "eval_every", "ndcg_k", "recall_k", "judge_batch_size",
            "judge_max_new_pairs", "max_length", "judge_max_chars", "judge_max_output_tokens",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.samples_per_query < 2:
            raise ValueError("RLOO needs at least two samples per query")
        if min(self.learning_rate, self.exploration_radius, self.temperature) <= 0:
            raise ValueError("learning_rate, exploration_radius and temperature must be positive")
        if self.kl_weight < 0 or self.pool_size < self.ndcg_k:
            raise ValueError("Require kl_weight >= 0 and pool_size >= ndcg_k")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.judge_reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("Unsupported judge_reasoning_effort")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)

    def preparation_settings(self):
        keys = (
            "model_name", "model_revision", "corpus_revision", "qrels_revision", "query_prefix",
            "max_length", "seed", "train_queries", "validation_queries", "pool_size",
        )
        return {key: getattr(self, key) for key in keys}


def prepared_dir(root: Path, config: Config) -> Path:
    return root / "prepared" / digest(config.preparation_settings())
