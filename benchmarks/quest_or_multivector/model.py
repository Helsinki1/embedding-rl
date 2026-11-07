"""Three learned slots, with optional completed-hidden-state feedback."""
import torch
import torch.nn.functional as F

from benchmarks.quest_or_training.data import DIMENSION, INSTRUCTION
from benchmarks.quest_or_training.train import QwenEncoder


class ThreeSlotModel(torch.nn.Module):
    def __init__(self, backbone, initial_token, variant, noise=0.01):
        super().__init__()
        if variant not in ("one_pass", "three_pass"):
            raise ValueError("Unknown three-slot variant")
        self.backbone, self.variant = backbone, variant
        initial = initial_token.detach().float()
        self.register_buffer("feedback_norm", initial.norm().clamp_min(1e-6))
        self.slots = torch.nn.Parameter(
            initial.expand(3, -1).clone()
            + noise * initial.square().mean().sqrt() * torch.randn(3, len(initial), device=initial.device)
        )

    def hidden(self, inputs, query_mask, slots):
        slot_inputs = torch.stack(slots, dim=1).to(inputs.dtype)
        mask = torch.cat((query_mask, query_mask.new_ones(len(inputs), len(slots))), dim=1)
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
        return self.backbone(
            inputs_embeds=torch.cat((inputs, slot_inputs), dim=1),
            attention_mask=mask, position_ids=positions, use_cache=False,
        ).last_hidden_state

    def forward(self, input_ids, attention_mask, return_states=False):
        inputs = self.backbone.get_input_embeddings()(input_ids)
        initial_slots = [slot.expand(len(inputs), -1) for slot in self.slots]
        if self.variant == "one_pass":
            final = self.hidden(inputs, attention_mask, initial_slots)[:, -3:]
            states = tuple(final[:, i] for i in range(3))
        else:
            states, slot_inputs = [], []
            for index, slot in enumerate(initial_slots):
                if index:
                    # No detach: later losses train earlier completed outputs too.
                    slot = slot + F.normalize(states[-1].float(), dim=-1) * self.feedback_norm
                slot_inputs.append(slot)
                states.append(self.hidden(inputs, attention_mask, slot_inputs)[:, -1])
            states = tuple(states)
        embeddings = F.normalize(torch.stack(states, dim=1).float(), dim=-1)
        return (embeddings, states) if return_states else embeddings

    @torch.no_grad()
    def forward_cached(self, input_ids, attention_mask):
        """Inference-only equivalent of three_pass, reusing unchanged causal prefixes."""
        if self.training:
            raise ValueError("Cached generation is for evaluation only")
        if self.variant == "one_pass":
            return self.forward(input_ids, attention_mask)
        inputs = self.backbone.get_input_embeddings()(input_ids)
        slot = self.slots[0].expand(len(inputs), -1).to(inputs.dtype)
        mask = torch.cat((attention_mask, attention_mask.new_ones(len(inputs), 1)), dim=1)
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
        output = self.backbone(inputs_embeds=torch.cat((inputs, slot[:, None]), dim=1),
                               attention_mask=mask, position_ids=positions, use_cache=True)
        states = [output.last_hidden_state[:, -1]]
        cache = output.past_key_values
        for index in (1, 2):
            slot = (self.slots[index].expand(len(inputs), -1)
                    + F.normalize(states[-1].float(), dim=-1) * self.feedback_norm)
            mask = torch.cat((mask, mask.new_ones(len(inputs), 1)), dim=1)
            position = mask.sum(-1, keepdim=True) - 1
            output = self.backbone(inputs_embeds=slot[:, None].to(inputs.dtype),
                                   attention_mask=mask, position_ids=position,
                                   past_key_values=cache, use_cache=True)
            states.append(output.last_hidden_state[:, -1])
            cache = output.past_key_values
        return F.normalize(torch.stack(states, dim=1).float(), dim=-1)


class ThreeVectorEncoder(QwenEncoder):
    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            token_id = self.tokenizer.unk_token_id
        if token_id is None:
            raise ValueError("Need a real token embedding to initialize slots")
        initial = self.model.get_input_embeddings().weight[token_id]
        self.model = ThreeSlotModel(self.model, initial, config["variant"], config["slot_init_noise"])
        self.evaluation_kv_cache = config.get("evaluation_kv_cache", False)
        if self.model.slots.shape != (3, DIMENSION):
            raise ValueError("Expected three full-size Qwen embeddings")

    def encode(self, texts):
        prompts = [f"Instruct: {INSTRUCTION}\nQuery:{text}" for text in texts]
        tokens = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if tokens.input_ids.shape[-1] + 3 > self.max_length:
            raise ValueError("Query plus three slots exceeds max_length; no silent truncation")
        tokens = tokens.to(self.device)
        if self.evaluation_kv_cache and not self.model.training:
            return self.model.forward_cached(tokens.input_ids, tokens.attention_mask)
        return self.model(tokens.input_ids, tokens.attention_mask)
