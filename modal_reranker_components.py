"""Component ablations of a fixed feature transform; no new fit or retrieval."""
import hashlib
import json
from pathlib import Path
import statistics

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-reranker-components")
anchor_image = image.add_local_python_source("rerank_features", "rerank_neural", "rerank_anchor", "rerank_components")
DIRECTORY = ROOT / "experiments/reranker-components-20261106"
CONDITIONING = ROOT / "experiments/reranker-conditioning-20261106"
SOURCE = ROOT / "experiments/neural-reranker-fp32-20261106"
SCALAR = ROOT / "experiments/feature-reranker-20261106"


@app.function(image=anchor_image, cpu=4, memory=8192, volumes={ROOT: volume}, timeout=1800,
              secrets=[modal.Secret.from_name("wandb-secret")], max_containers=1)
def run():
    import torch
    import rerank_anchor
    import rerank_components
    import rerank_features
    import rerank_neural
    from embedding_rl.config import digest, write_json
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.tracking import Tracker

    torch.set_num_threads(4)
    scalar = torch.load(SCALAR / "features.pt", weights_only=True)
    saved = torch.load(SOURCE / "neural.pt", weights_only=True)
    source_protocol = json.loads((SOURCE / "protocol.json").read_text())
    reference_result = json.loads((SOURCE / "comparison.json").read_text())
    if saved["protocol_digest"] != digest(source_protocol) or reference_result["protocol_digest"] != digest(source_protocol) or scalar["protocol_digest"] != source_protocol["source_protocol_digest"]:
        raise ValueError("Cached-feature provenance differs")
    if saved["query_ids"] != scalar["train"]["query_ids"] + scalar["validation"]["query_ids"] or len(scalar["train"]["query_ids"]) != 581 or len(scalar["validation"]["query_ids"]) != 83 or set(scalar["train"]["query_ids"]) & set(scalar["validation"]["query_ids"]):
        raise ValueError("Training split differs")
    if source_protocol["policy_source_sha256"] != hashlib.sha256(Path(rerank_neural.__file__).read_bytes()).hexdigest():
        raise ValueError("Unregularized policy source changed")
    parent_protocol = json.loads((CONDITIONING / "protocol.json").read_text())
    fitted = torch.load(CONDITIONING / "transform.pt", weights_only=True)
    combined_result = json.loads((CONDITIONING / "comparison.json").read_text())
    if fitted["protocol_digest"] != digest(parent_protocol) or combined_result["protocol_digest"] != digest(parent_protocol) or fitted["query_ids"] != scalar["train"]["query_ids"] or parent_protocol["source_protocol_digest"] != digest(source_protocol):
        raise ValueError("Parent transform provenance differs")
    protocol = {"parent_protocol_digest": digest(parent_protocol), "modes": ["remove_only", "scale_only"], "source_protocol_digest": digest(source_protocol), "scalar_protocol_digest": scalar["protocol_digest"],
        "query_ids_digest": digest(saved["query_ids"]), "beta": [0., .1], "seeds": [42, 43, 44],
        "steps": 500, "queries_per_batch": 16, "rankings_per_query": 32,
        "theta_learning_rate": .01, "delta_learning_rate": .0001,
        "loss": "Whole-ranking unscaled RLOO plus beta times mean-query first-choice KL(current || initial hybrid)",
        "reference": "Initial hybrid logits, detached and fixed; first-choice marginal, not full-permutation KL",
        "initial_theta": [1, 0, 0, 0], "initial_delta": "1024 zeros", "final_checkpoint_only": True,
        "train_queries": 581, "validation_queries": 83, "api_calls": 0, "new_page_scoring": 0,
        "browsecomp_final_test_evaluated": False, "external_datasets_evaluated": False,
        "source_hashes": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (rerank_anchor, rerank_features, rerank_neural, rerank_components)},
        "predeclared": "research-log.json queued_reranker_components_hypothesis; adaptive ablation after combined conditioning results",
        "conditioning": "Remove scalar-predictable part only, or divide query-centered features by SAME saved residual RMS only. Parent transform fixed; no refitting.",
        "limitations": "Two strengths on reused validation; one upstream seed; exploratory, no model selection or independent final test"}
    if (DIRECTORY / "protocol.json").exists() and json.loads((DIRECTORY / "protocol.json").read_text()) != protocol:
        raise ValueError("Conditioning protocol changed")
    write_json(DIRECTORY / "protocol.json", protocol)
    volume.commit()
    all_scalar = torch.cat([scalar["train"]["features"], scalar["validation"]["features"]])
    neural = {"train": saved["features"][:581], "validation": saved["features"][581:]}
    def evaluate(theta, delta, split):
        subset = scalar[split]
        return rerank_neural.evaluate_neural(theta, delta, subset["features"], neural[split], subset["gains"], subset["ideal"])
    before = evaluate(torch.tensor([1., 0., 0., 0.]), torch.zeros(1024), "validation").tolist()
    if max(abs(a - b) for a, b in zip(before, reference_result["runs"][0]["before"], strict=True)) > 1e-6:
        raise ValueError("Initial policy changed")
    rows = []
    for mode in protocol["modes"]:
        transformed = rerank_components.transform_component(all_scalar, saved["features"], fitted, mode).float()
        if not torch.isfinite(transformed).all():
            raise ValueError("Nonfinite component features")
        neural = {"train": transformed[:581], "validation": transformed[581:]}
        for beta in protocol["beta"]:
            for seed in protocol["seeds"]:
                label = f"{mode}-beta-{beta:g}-seed-{seed}"
                folder = DIRECTORY / label
                if (folder / "result.json").exists():
                    row = json.loads((folder / "result.json").read_text())
                    if row["protocol_digest"] != digest(protocol):
                        raise ValueError("Saved policy provenance differs")
                    rows.append(row)
                    continue
                folder.mkdir(exist_ok=True)
                tracker = Tracker(folder, base_config(), "reranker-components", {"experiment_id": DIRECTORY.name, "mode": mode, "beta": beta, "seed": seed, "protocol": protocol})
                try:
                    subset = scalar["train"]
                    theta, delta, history = rerank_anchor.train_anchored_policy(subset["features"], neural["train"], subset["gains"], subset["ideal"], seed, beta)
                    for record in history:
                        tracker.log({k: v for k, v in record.items() if k != "step"}, record["step"])
                    after = evaluate(theta, delta, "validation").tolist()
                    final_logits = rerank_neural.neural_logits(theta, delta, subset["features"], neural["train"])
                    train_kl = rerank_anchor.first_choice_kl(final_logits, subset["features"][:, :, 0]).item()
                    row = {"mode": mode, "seed": seed, "beta": beta, "theta": theta.tolist(), "delta_norm": delta.norm().item(),
                        "before": before, "after": after, "mean_after": statistics.mean(after),
                        "train_after": evaluate(theta, delta, "train").mean().item(), "train_first_choice_kl": train_kl,
                        "protocol_digest": digest(protocol), "paired_query_bootstrap": paired_bootstrap(before, after),
                        "wandb_url": tracker.run.url if tracker.run else None}
                    original_theta, original_delta = rerank_components.original_component_weights(theta, delta, fitted, mode)
                    torch.save({"theta": theta, "delta": delta, "original_theta": original_theta, "original_delta": original_delta,
                        "step": 500, "beta": beta, "mode": mode, "protocol_digest": digest(protocol)}, folder / "final.pt")
                    write_json(folder / "result.json", row)
                    tracker.finish(row)
                except Exception:
                    tracker.finish(success=False)
                    raise
                volume.commit()
                rows.append(row)
                print("POLICY_DONE " + json.dumps({k: v for k, v in row.items() if k not in {"before", "after"}}), flush=True)
    anchor_result = json.loads((ROOT / "experiments/reranker-anchor-20261106/comparison.json").read_text())
    anchor_protocol = json.loads((ROOT / "experiments/reranker-anchor-20261106/protocol.json").read_text())
    if anchor_result["protocol_digest"] != digest(anchor_protocol) or anchor_protocol["source_protocol_digest"] != digest(source_protocol) or anchor_result["query_ids"] != scalar["validation"]["query_ids"]:
        raise ValueError("Matched KL control query order differs")
    summary = []
    for mode in protocol["modes"]:
        for beta in protocol["beta"]:
            controls = reference_result["runs"] if beta == 0 else [r for r in anchor_result["runs"] if r["beta"] == beta]
            if {r["seed"] for r in controls} != {42, 43, 44}:
                raise ValueError("Missing matched control")
            reference = [statistics.mean(r["after"][i] for r in controls) for i in range(83)]
            selected = [r for r in rows if r["beta"] == beta and r["mode"] == mode]
            combined = [statistics.mean(r["after"][i] for r in combined_result["runs"] if r["beta"] == beta) for i in range(83)]
            after = [statistics.mean(r["after"][i] for r in selected) for i in range(83)]
            summary.append({"mode": mode, "beta": beta, "mean_after": statistics.mean(after), "mean_train_after": statistics.mean(r["train_after"] for r in selected),
                "mean_train_first_choice_kl": statistics.mean(r["train_first_choice_kl"] for r in selected),
                "versus_hybrid": paired_bootstrap(before, after), "matched_control_mean": statistics.mean(reference), "versus_matched_control": paired_bootstrap(reference, after), "combined_mean": statistics.mean(combined), "versus_combined": paired_bootstrap(combined, after)})
    result = {"runs": rows, "summary": summary, "hybrid_mean": statistics.mean(before), "parent_protocol_digest": digest(parent_protocol),
        "query_ids": scalar["validation"]["query_ids"], "protocol_digest": digest(protocol), "api_calls": 0,
        "browsecomp_final_test_evaluated": False, "external_datasets_evaluated": False}
    write_json(DIRECTORY / "comparison.json", result)
    volume.commit()
    return {k: v for k, v in result.items() if k not in {"runs", "query_ids"}}


@app.local_entrypoint()
def main():
    print("COMPARISON " + json.dumps(run.remote()), flush=True)
