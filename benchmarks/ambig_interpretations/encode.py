"""Export three trained query embeddings without opening a document corpus."""
import argparse
import json
from pathlib import Path

import torch

from .model import InterpretationModel, MODEL_ID, REVISION, tokenize


def main():
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", choices=("mtp", "autoregressive"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL objects with query and optional id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.model or snapshot_download(MODEL_ID, revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    base = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to(args.device)
    base = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
    model = InterpretationModel(base, tokenizer.eos_token_id, args.architecture)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Checkpoint does not match this architecture's trainable parameters")
    model.load_state_dict(state, strict=False)
    model.eval()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            tokens = tokenize(tokenizer, [r["query"] for r in rows[start:start+16]], args.device)
            vectors.append(model(**tokens).cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "embeddings": torch.cat(vectors), "architecture": args.architecture,
                "model": MODEL_ID, "revision": REVISION}, args.output)


if __name__ == "__main__":
    main()
