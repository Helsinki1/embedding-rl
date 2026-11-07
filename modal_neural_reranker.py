"""Cache frozen cross-encoder representations, then train a small residual head."""
import hashlib
import json
from pathlib import Path
import statistics

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-neural-reranker")
neural_image = image.add_local_python_source("rerank_features", "rerank_neural")
DIRECTORY = ROOT / "experiments/neural-reranker-fp32-20261106"
SOURCE = ROOT / "experiments/feature-reranker-20261106"
MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@app.function(image=neural_image, gpu="A10G", cpu=4, memory=24576, volumes={ROOT: volume}, timeout=5400, max_containers=1)
def prepare():
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import rerank_neural
    from embedding_rl.config import digest, write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_protocol = json.loads((SOURCE / "protocol.json").read_text())
    scalar = torch.load(SOURCE / "features.pt", weights_only=True)
    source_cache = json.loads((SOURCE / "bge-scores.json").read_text())
    diagnostic = json.loads((ROOT / "experiments/neural-reranker-20261106/diagnostics/NVIDIA-A10.json").read_text())
    if diagnostic["pairs"] != 256 or not diagnostic["training_only"] or diagnostic["hook_error"] != 0 or max(diagnostic["fp32_vs_current_fp16_max"], diagnostic["fp32_vs_cached_fp16_max"]) > .02:
        raise ValueError("FP16 tolerance diagnostic differs")
    _, manifest, bundle, _ = load_prepared(ROOT, base_config())
    qids = scalar["train"]["query_ids"] + scalar["validation"]["query_ids"]
    if scalar["train"]["query_ids"] != bundle["splits"]["train"] or scalar["validation"]["query_ids"] != bundle["splits"]["validation"] or scalar["candidate_indices"].shape != (664, 25):
        raise ValueError("Source candidate split or shape differs")
    if qids != bundle["splits"]["train"] + bundle["splits"]["validation"] or len(qids) != 664 or set(qids) & set(bundle["splits"]["test"]):
        raise ValueError("Neural cache query boundaries differ")
    if source_protocol["prepared_digest"] != digest(manifest) or scalar["protocol_digest"] != digest(source_protocol) or source_cache["protocol_digest"] != digest(source_protocol):
        raise ValueError("Scalar feature provenance differs")
    pairs = [(qi, di) for qi, ids in enumerate(scalar["candidate_indices"].tolist()) for di in ids]
    protocol = {"source_protocol_digest": digest(source_protocol), "prepared_digest": digest(manifest),
        "query_ids_digest": digest(qids), "pair_ids_digest": digest(pairs), "pairs": len(pairs),
        "model": MODEL, "revision": REVISION, "dtype": "float32", "pair_limit": 512, "gpu_options": ["A10G"], "tf32": False,
        "representation": "Input to classifier.out_proj, captured by forward pre-hook in eval mode", "feature_width": 1024,
        "text": source_protocol["reranker_text"], "old_fp16_logits": "Numerical comparison only; old scalar features remain unchanged. The new neural representation is computed in FP32.",
        "tolerance_diagnostic": "experiments/neural-reranker-20261106/diagnostics/NVIDIA-A10.json; first256 training pairs, hook error0, current/cached FP16 each within0.016014 of FP32",
        "diagnostic_digest": digest(diagnostic), "head_reconstruction_error_limit": .0001,
        "policy": "Four existing normalized features times theta plus raw frozen penultimate features times delta",
        "initial_theta": [1, 0, 0, 0], "initial_delta": "1024 zeros", "seeds": [42, 43, 44],
        "steps": 500, "batch_size": 16, "samples": 32, "theta_learning_rate": .01, "delta_learning_rate": .0001,
        "optimizer": "Adam", "loss": "Whole-ranking unscaled RLOO; benchmark evidence nDCG with global ideal denominator",
        "kl": 0, "final_checkpoint_only": True, "backbone_updates": 0, "api_calls": 0,
        "train_queries": 581, "validation_queries": 83, "browsecomp_final_test_evaluated": False,
        "external_datasets_evaluated": False, "policy_source_sha256": hashlib.sha256(Path(rerank_neural.__file__).read_bytes()).hexdigest(),
        "limitations": "1028 learned scalar parameters using frozen representations; not full cross-encoder fine-tuning. Same fixed upstream retriever and reused validation split. More capacity may overfit."}
    if (DIRECTORY / "protocol.json").exists() and json.loads((DIRECTORY / "protocol.json").read_text()) != protocol:
        raise ValueError("Neural experiment protocol changed")
    write_json(DIRECTORY / "protocol.json", protocol)
    volume.commit()
    if (DIRECTORY / "neural.pt").exists() and (DIRECTORY / "cache-verification.json").exists():
        saved = torch.load(DIRECTORY / "neural.pt", weights_only=True)
        if saved["protocol_digest"] != digest(protocol):
            raise ValueError("Neural tensor provenance differs")
        return json.loads((DIRECTORY / "cache-verification.json").read_text())
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", revision=bundle["revisions"]["corpus"])
    if [str(x) for x in corpus["docid"]] != [d["id"] for d in bundle["docs"]]:
        raise ValueError("Corpus document order differs")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32).to("cuda").eval()
    head = model.classifier.out_proj
    if not isinstance(head, torch.nn.Linear) or head.in_features != 1024 or head.out_features != 1:
        raise ValueError("Unexpected frozen classifier head")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured = []
    hook = head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
    blocks, checks = [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(pairs), 256):
                selected = pairs[start:start + 256]
                path = DIRECTORY / "blocks" / f"{start:06d}.pt"
                if path.exists():
                    saved = torch.load(path, weights_only=True)
                    if saved["protocol_digest"] != digest(protocol) or saved["pairs"] != selected:
                        raise ValueError("Cached neural block differs")
                else:
                    vectors, outputs, reconstruction_errors = [], [], []
                    for offset in range(0, len(selected), 16):
                        batch = selected[offset:offset + 16]
                        inputs = tokenizer([[bundle["queries"][qids[qi]], corpus[di]["text"]] for qi, di in batch],
                            padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                        captured.clear()
                        logits = model(**inputs, return_dict=True).logits.reshape(-1)
                        if len(captured) != 1 or captured[0].shape != (len(batch), 1024) or not torch.isfinite(captured[0]).all() or not torch.isfinite(logits).all():
                            raise ValueError("Invalid captured neural features")
                        reconstructed = F.linear(captured[0], head.weight, head.bias).reshape(-1)
                        reconstruction_errors.append((reconstructed.float() - logits.float()).abs().max().item())
                        vectors.append(captured[0].clone().cpu())
                        outputs.append(logits.float().cpu())
                    neural = torch.cat(vectors)
                    logits = torch.cat(outputs)
                    reference = torch.tensor([source_cache["scores"][f"{qi}:{di}"] for qi, di in selected])
                    error = (logits - reference).abs().max().item()
                    mean_error = (logits - reference).abs().mean().item()
                    reconstruction_error = max(reconstruction_errors)
                    if reconstruction_error > .0001:
                        raise ValueError(f"Frozen representation reconstruction failed: cache_error={error}, mean_error={mean_error}, head_error={reconstruction_error}")
                    saved = {"protocol_digest": digest(protocol), "pairs": selected, "vectors": neural, "gpu": torch.cuda.get_device_name(),
                             "logits": logits, "max_cached_logit_error": error, "mean_cached_logit_error": mean_error, "max_head_error": reconstruction_error}
                    path.parent.mkdir(exist_ok=True)
                    temporary = path.with_suffix(".tmp.pt")
                    torch.save(saved, temporary)
                    temporary.replace(path)
                    volume.commit()
                    print(f"NEURAL_SAVED {start + len(selected)}/{len(pairs)} error={error:.6g}", flush=True)
                blocks.append(saved["vectors"])
                checks.append({k: saved[k] for k in ("max_cached_logit_error", "mean_cached_logit_error", "max_head_error", "gpu")})
    finally:
        hook.remove()
    features = torch.cat(blocks).reshape(len(qids), 25, 1024).clone()
    torch.save({"features": features, "query_ids": qids, "protocol_digest": digest(protocol)}, DIRECTORY / "neural.pt")
    result = {"pairs": len(pairs), "feature_width": 1024, "max_cached_logit_error": max(c["max_cached_logit_error"] for c in checks),
        "max_block_mean_cached_logit_error": max(c["mean_cached_logit_error"] for c in checks),
        "max_head_reconstruction_error": max(c["max_head_error"] for c in checks), "blocks": len(blocks),
        "tensor_file_bytes": (DIRECTORY / "neural.pt").stat().st_size, "gpus_used": sorted({c["gpu"] for c in checks}),
        "api_calls": 0, "browsecomp_final_test_evaluated": False}
    write_json(DIRECTORY / "cache-verification.json", result)
    volume.commit()
    return result


@app.function(image=neural_image, cpu=4, memory=8192, volumes={ROOT: volume}, timeout=1800,
              secrets=[modal.Secret.from_name("wandb-secret")], max_containers=1)
def train():
    import torch
    import rerank_neural
    from embedding_rl.config import digest, write_json
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.tracking import Tracker
    from rerank_neural import evaluate_neural, train_neural_policy

    torch.set_num_threads(4)
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    scalar = torch.load(SOURCE / "features.pt", weights_only=True)
    saved = torch.load(DIRECTORY / "neural.pt", weights_only=True)
    source_result = json.loads((SOURCE / "comparison.json").read_text())
    if saved["protocol_digest"] != digest(protocol) or scalar["protocol_digest"] != protocol["source_protocol_digest"]:
        raise ValueError("Neural training provenance differs")
    if protocol["policy_source_sha256"] != hashlib.sha256(Path(rerank_neural.__file__).read_bytes()).hexdigest():
        raise ValueError("Neural policy implementation changed")
    if saved["query_ids"] != scalar["train"]["query_ids"] + scalar["validation"]["query_ids"] or set(scalar["train"]["query_ids"]) & set(scalar["validation"]["query_ids"]):
        raise ValueError("Training split differs")
    neural = {"train": saved["features"][:581].float(), "validation": saved["features"][581:].float()}
    def evaluate(theta, delta, split):
        subset = scalar[split]
        return evaluate_neural(theta, delta, subset["features"], neural[split], subset["gains"], subset["ideal"])
    before = evaluate(torch.tensor([1., 0., 0., 0.]), torch.zeros(1024), "validation").tolist()
    if max(abs(a - b) for a, b in zip(before, source_result["runs"][0]["before"], strict=True)) > 1e-6:
        raise ValueError("Zero residual does not reproduce the hybrid baseline")
    rows = []
    for seed in protocol["seeds"]:
        folder = DIRECTORY / f"seed-{seed}"
        if (folder / "result.json").exists():
            rows.append(json.loads((folder / "result.json").read_text()))
            continue
        folder.mkdir(exist_ok=True)
        tracker = Tracker(folder, base_config(), "neural-feature-reranker-rl", {"experiment_id": DIRECTORY.name, "seed": seed, "protocol": protocol})
        try:
            subset = scalar["train"]
            theta, delta, history = train_neural_policy(subset["features"], neural["train"], subset["gains"], subset["ideal"], seed)
            for record in history:
                tracker.log({k: v for k, v in record.items() if k != "step"}, record["step"])
            after = evaluate(theta, delta, "validation").tolist()
            row = {"seed": seed, "theta": theta.tolist(), "delta_norm": delta.norm().item(), "before": before, "after": after,
                "mean_before": statistics.mean(before), "mean_after": statistics.mean(after),
                "train_after": evaluate(theta, delta, "train").mean().item(), "protocol_digest": digest(protocol),
                "paired_query_bootstrap": paired_bootstrap(before, after), "wandb_url": tracker.run.url if tracker.run else None}
            torch.save({"theta": theta, "delta": delta, "step": 500, "protocol_digest": digest(protocol)}, folder / "final.pt")
            write_json(folder / "result.json", row)
            tracker.finish(row)
        except Exception:
            tracker.finish(success=False)
            raise
        volume.commit()
        rows.append(row)
        print("POLICY_DONE " + json.dumps({k: v for k, v in row.items() if k not in {"before", "after"}}), flush=True)
    after = [statistics.mean(r["after"][i] for r in rows) for i in range(83)]
    reference = [statistics.mean(r["after"][i] for r in source_result["runs"]) for i in range(83)]
    result = {"runs": rows, "mean_before": statistics.mean(before), "mean_after": statistics.mean(after),
        "mean_train_after": statistics.mean(r["train_after"] for r in rows), "scalar_reference_mean": statistics.mean(reference),
        "versus_hybrid": paired_bootstrap(before, after), "versus_scalar_policy": paired_bootstrap(reference, after),
        "query_ids": scalar["validation"]["query_ids"], "protocol_digest": digest(protocol),
        "api_calls": 0, "backbone_updates": 0, "browsecomp_final_test_evaluated": False}
    write_json(DIRECTORY / "comparison.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(train_only: bool = False):
    if not train_only:
        print("PREPARED " + json.dumps(prepare.remote()), flush=True)
    result = train.remote()
    print("COMPARISON " + json.dumps({k: v for k, v in result.items() if k not in {"runs", "query_ids"}}), flush=True)
