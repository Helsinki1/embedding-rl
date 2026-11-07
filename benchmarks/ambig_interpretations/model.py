"""Three causal output slots, optionally with differentiable output feedback."""
import itertools

import torch
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def tokenize(tokenizer, texts, device, max_length=256):
    prompts = [f"Instruct: {INSTRUCTION}\nQuery:{text}" for text in texts]
    tokens = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    if tokens.input_ids.shape[1] + 3 > max_length:
        raise ValueError("Query exceeds audited context limit; refusing truncation")
    return tokens.to(device)


class InterpretationModel(torch.nn.Module):
    def __init__(self, backbone, eos_id, architecture):
        super().__init__()
        if architecture not in ("mtp", "autoregressive"):
            raise ValueError(architecture)
        self.backbone, self.architecture = backbone, architecture
        initial = backbone.get_input_embeddings().weight[eos_id].detach().float()
        self.register_buffer("feedback_norm", initial.norm().clamp_min(1e-6))
        self.slots = torch.nn.Parameter(initial.repeat(3, 1) +
            0.01 * initial.square().mean().sqrt() * torch.randn(3, len(initial), device=initial.device))

    def hidden(self, inputs, attention_mask, slots):
        mask = torch.cat([attention_mask, attention_mask.new_ones(len(inputs), len(slots))], 1)
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
        output = self.backbone(inputs_embeds=torch.cat([
            inputs, torch.stack(slots, 1).to(inputs.dtype)], 1),
            attention_mask=mask, position_ids=positions, use_cache=False)
        return output.last_hidden_state

    def forward(self, input_ids, attention_mask, return_states=False, **unused):
        inputs = self.backbone.get_input_embeddings()(input_ids)
        slots = [s.expand(len(inputs), -1) for s in self.slots]
        if self.architecture == "mtp":
            output = self.hidden(inputs, attention_mask, slots)[:, -3:]
            states = [output[:, i] for i in range(3)]
        else:
            states, prefix = [], []
            for i, slot in enumerate(slots):
                if i:
                    slot = slot + F.normalize(states[-1].float(), dim=-1) * self.feedback_norm
                prefix.append(slot)
                states.append(self.hidden(inputs, attention_mask, prefix)[:, -1])
        vectors = F.normalize(torch.stack(states, 1).float(), dim=-1)
        return (vectors, states) if return_states else vectors


def matching_loss(predictions, targets, counts):
    """Exact minimum-cost one-to-one matching; labels and slots have no fixed order.

    For two labels, match two distinct slots and mask the unused slot entirely.
    """
    similarities = torch.einsum("bsd,btd->bst", predictions.float(), targets.float())
    losses = []
    for i, n in enumerate(counts.tolist()):
        if n not in (2, 3):
            raise ValueError("This experiment requires two or three labeled interpretations")
        candidates = []
        for slots in itertools.permutations(range(3), n):
            cost = sum(1 - similarities[i, s, j] for j, s in enumerate(slots))
            candidates.append(cost / n)
        losses.append(torch.stack(candidates).min())
    return torch.stack(losses)


def diagnostics(predictions, targets, counts):
    loss = matching_loss(predictions, targets, counts)
    pairwise = predictions @ predictions.transpose(-1, -2)
    upper = torch.triu_indices(3, 3, 1, device=predictions.device)
    target_diversity = []
    for i, n in enumerate(counts.tolist()):
        ix = torch.triu_indices(n, n, 1, device=targets.device)
        sim = targets[i, :n] @ targets[i, :n].T
        target_diversity.append((1 - sim[ix[0], ix[1]]).mean())
    return {
        "matching_cosine_loss": loss.mean().item(),
        "matched_cosine": (1 - loss).mean().item(),
        "prediction_pairwise_cosine": pairwise[:, upper[0], upper[1]].mean().item(),
        "target_pairwise_cosine": 1 - torch.stack(target_diversity).mean().item(),
    }
