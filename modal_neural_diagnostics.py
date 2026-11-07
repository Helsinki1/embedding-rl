"""Explain frozen-score discrepancies before accepting a neural feature cache."""
import json

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-neural-diagnostics")


@app.function(image=image, gpu="A10G", cpu=2, memory=16384, volumes={ROOT: volume}, timeout=1200)
def probe():
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config

    torch.set_num_threads(2)
    source = ROOT / "experiments/feature-reranker-20261106"
    data = torch.load(source / "features.pt", weights_only=True)
    cache = json.loads((source / "bge-scores.json").read_text())
    _, _, bundle, _ = load_prepared(ROOT, base_config())
    qids = data["train"]["query_ids"]
    pairs = [(qi, di) for qi, ids in enumerate(data["candidate_indices"][:len(qids)].tolist()) for di in ids][:256]
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", revision=bundle["revisions"]["corpus"])
    revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    model_id = "BAAI/bge-reranker-v2-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16).to("cuda").eval()
    inputs = [tokenizer([[bundle["queries"][qids[qi]], corpus[di]["text"]] for qi, di in pairs[start:start + 16]],
                       padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda") for start in range(0, len(pairs), 16)]
    def score():
        with torch.inference_mode():
            return torch.cat([model(**batch, return_dict=True).logits.reshape(-1).float().cpu() for batch in inputs])
    ordinary = score()
    captured = []
    handle = model.classifier.out_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone().cpu()))
    hooked = score()
    handle.remove()
    model.float()
    fp32 = score()
    reference = torch.tensor([cache["scores"][f"{qi}:{di}"] for qi, di in pairs])
    result = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "transformers": transformers.__version__,
        "pairs": len(pairs), "training_only": True, "source_protocol_digest": data["protocol_digest"],
        "max_cache_error": (ordinary - reference).abs().max().item(),
        "mean_cache_error": (ordinary - reference).abs().mean().item(),
        "hook_error": (ordinary - hooked).abs().max().item(),
        "fp32_vs_current_fp16_max": (fp32 - ordinary).abs().max().item(),
        "fp32_vs_cached_fp16_max": (fp32 - reference).abs().max().item(),
        "fp32_vs_current_fp16_mean": (fp32 - ordinary).abs().mean().item(),
        "fp32_vs_cached_fp16_mean": (fp32 - reference).abs().mean().item(),
        "cached_logits": reference.tolist(), "ordinary_logits": ordinary.tolist(), "fp32_logits": fp32.tolist()}
    directory = ROOT / "experiments/neural-reranker-20261106/diagnostics"
    write_json(directory / (result["gpu"].replace(" ", "-") + ".json"), result)
    volume.commit()
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("logits")}), flush=True)
    return result


@app.local_entrypoint()
def main():
    first = probe.spawn()
    second = probe.with_options(gpu="L4").spawn()
    for call in (first, second):
        result = call.get()
        print("DIAGNOSTIC " + json.dumps({k: v for k, v in result.items() if not k.endswith("logits")}), flush=True)
