import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class Encoder:
    """BGE encoder: CLS pooling and unit normalization, with gradients on demand."""

    def __init__(self, name, revision, device, max_length=512, prefix=""):
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision).to(device)
        self.device = device
        self.max_length = max_length
        self.prefix = prefix
        # Keep dropout off even during optimization: exploration is explicit in action space.
        # eval() does NOT turn off autograd.
        self.model.eval()

    def encode(self, texts, query=False):
        if query:
            texts = [self.prefix + text for text in texts]
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        return F.normalize(self.model(**tokens).last_hidden_state[:, 0].float(), dim=-1)

    @torch.no_grad()
    def encode_batches(self, texts, batch_size, query=False):
        return torch.cat([
            self.encode(texts[start:start + batch_size], query=query).cpu()
            for start in range(0, len(texts), batch_size)
        ])
