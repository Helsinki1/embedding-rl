"""Two residual output heads; Qwen and the existing slot encoders remain intact."""
import torch
import torch.nn.functional as F

from benchmarks.quest_or_training.data import INSTRUCTION
from benchmarks.quest_or_training.train import QwenEncoder
from benchmarks.quest_or_multivector.model import ThreeVectorEncoder


class OutputHead(torch.nn.Module):
    def __init__(self, dimension, variant, width=512, reader_width=256, reader_heads=8):
        super().__init__()
        if variant not in ("pooled_mlp", "attention_mlp"):
            raise ValueError("Unknown output head")
        self.variant = variant
        self.norm = torch.nn.LayerNorm(dimension)
        if variant == "pooled_mlp":
            self.mlp = torch.nn.Sequential(torch.nn.Linear(dimension, width), torch.nn.GELU(),
                                           torch.nn.Linear(width, 3 * dimension))
        else:
            self.readers = torch.nn.Parameter(torch.randn(3, reader_width) * .02)
            self.attention = torch.nn.MultiheadAttention(
                reader_width, reader_heads, kdim=dimension, vdim=dimension, batch_first=True)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(reader_width, width), torch.nn.GELU(),
                                           torch.nn.Linear(width, dimension))
        # Start close to the existing embedding, with nonidentical, trainable outputs.
        torch.nn.init.normal_(self.mlp[-1].weight, std=1e-4)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, states, mask):
        states = states.float()
        base = F.normalize(states[:, -1], dim=-1)
        if self.variant == "pooled_mlp":
            delta = self.mlp(self.norm(states[:, -1])).reshape(len(states), 3, -1)
        else:
            tokens = self.norm(states)
            queries = self.readers[None].expand(len(states), -1, -1)
            pooled, _ = self.attention(queries, tokens, tokens,
                                       key_padding_mask=~mask.bool(), need_weights=False)
            delta = self.mlp(pooled)
        return F.normalize(base[:, None] + delta, dim=-1)


class HeadModel(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, input_ids, attention_mask):
        positions = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
        states = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=positions, use_cache=False).last_hidden_state
        return self.head(states, attention_mask)


class HeadEncoder(QwenEncoder):
    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
        self.model = HeadModel(self.model, OutputHead(
            self.model.config.hidden_size, config["variant"], config["mlp_width"],
            config["reader_width"], config["reader_heads"]).to(device))

    def encode(self, texts):
        prompts = [f"Instruct: {INSTRUCTION}\nQuery:{text}" for text in texts]
        tokens = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(self.device)
        if tokens.input_ids.shape[-1] > self.max_length:
            raise ValueError("Query exceeds maximum length")
        return self.model(tokens.input_ids, tokens.attention_mask)


def make_encoder(model_path, config, device):
    cls = ThreeVectorEncoder if config["variant"] in ("one_pass", "three_pass") else HeadEncoder
    return cls(model_path, config, device)
