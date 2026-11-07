"""Train-only variance gate, then matched ranking-policy estimator controls."""
import hashlib
import json
from pathlib import Path
import statistics

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-ranking-credit")
credit_image = image.add_local_python_source("rerank_features", "rerank_credit")


@app.function(image=credit_image, cpu=4, memory=8192, volumes={ROOT: volume}, timeout=1800,
              secrets=[modal.Secret.from_name("wandb-secret")], max_containers=1)
def run():
    import torch
    import rerank_credit
    import rerank_features
    from embedding_rl.config import digest, write_json
    from embedding_rl.graded_evaluate import paired_bootstrap
    from embedding_rl.judge_experiment import base_config
    from embedding_rl.tracking import Tracker
    from rerank_credit import credit_loss, train_credit_policy
    from rerank_features import evaluate_weights, policy_loss, ranking_reward, sample_rankings

    torch.set_num_threads(4)
    source = ROOT / "experiments/feature-reranker-20261106"
    directory = ROOT / "experiments/rerank-credit-20261106"
    source_protocol = json.loads((source / "protocol.json").read_text())
    source_result = json.loads((source / "comparison.json").read_text())
    data = torch.load(source / "features.pt", weights_only=True)
    if data["protocol_digest"] != digest(source_protocol) or source_result["protocol_digest"] != digest(source_protocol):
        raise ValueError("Frozen feature source differs")
    if hashlib.sha256(Path(rerank_features.__file__).read_bytes()).hexdigest() != source_protocol["policy_source_sha256"]:
        raise ValueError("Baseline policy source changed")
    if any(source_protocol[k] != v for k, v in {"steps": 500, "samples": 32, "batch_size": 16, "learning_rate": .01}.items()):
        raise ValueError("Baseline optimizer settings differ")
    train_data, val_data = data["train"], data["validation"]
    if len(train_data["query_ids"]) != 581 or len(val_data["query_ids"]) != 83 or set(train_data["query_ids"]) & set(val_data["query_ids"]):
        raise ValueError("Unexpected split boundary")
    protocol = {"source_protocol_digest": digest(source_protocol), "seeds": [42, 43, 44],
        "estimators": ["placement", "marginalize_all"], "steps": 500, "samples": 32, "batch_size": 16, "learning_rate": .01,
        "probe_query_ids": train_data["query_ids"][:32], "probe_groups": 128, "probe_seed": 918,
        "probe_weights": [1, 0, 0, 0], "gate": "Train estimator iff summed four-weight gradient variance is below 70% of whole-ranking RLOO, averaged over the first32 training queries",
        "credit_source_sha256": hashlib.sha256(Path(rerank_credit.__file__).read_bytes()).hexdigest(),
        "objective": "Same additive benchmark nDCG@10 with full-corpus ideal; same frozen features and natural candidates",
        "reference": "https://arxiv.org/html/2607.18689v1, sections4.2 and4.4; traditional relevance estimators, not the full exposure-based method",
        "final_checkpoint_only": True, "api_calls": 0, "browsecomp_final_test_evaluated": False, "external_datasets_evaluated": False,
        "limitations": "Variance diagnostic at initialization for four scalar parameters; no guarantee of validation gains. Additive reward assumption does not extend automatically to arbitrary list-level LLM judgments."}
    if (directory / "protocol.json").exists() and json.loads((directory / "protocol.json").read_text()) != protocol:
        raise ValueError("Credit-assignment protocol changed")
    write_json(directory / "protocol.json", protocol)
    volume.commit()
    if (directory / "comparison.json").exists():
        return json.loads((directory / "comparison.json").read_text())
    probe_file = directory / "variance-probe.json"
    if probe_file.exists():
        probe = json.loads(probe_file.read_text())
    else:
        generator = torch.Generator().manual_seed(918)
        rows = []
        for qi in range(32):
            groups = 128
            weights = torch.tensor([1., 0., 0., 0.]).expand(groups, -1).clone().requires_grad_(True)
            logits = weights @ train_data["features"][qi].T
            gains = train_data["gains"][qi:qi + 1].expand(groups, -1)
            ideal = train_data["ideal"][qi:qi + 1].expand(groups)
            orders = sample_rankings(logits, 32, generator)
            rewards = ranking_reward(orders, gains, ideal)
            for estimator in ("ranking", "placement", "marginalize_all"):
                loss = policy_loss(logits, orders, rewards) if estimator == "ranking" else credit_loss(logits, orders, gains, ideal, estimator)
                gradient = -torch.autograd.grad(loss, weights, retain_graph=True)[0].detach().double() * groups
                if not torch.isfinite(gradient).all():
                    raise ValueError("Invalid variance probe")
                rows.append({"query_id": train_data["query_ids"][qi], "estimator": estimator,
                    "variance_trace": gradient.var(0).sum().item(), "mean_gradient": gradient.mean(0).tolist()})
        means = {name: statistics.mean(r["variance_trace"] for r in rows if r["estimator"] == name)
                 for name in ("ranking", "placement", "marginalize_all")}
        ratios = {name: value / means["ranking"] for name, value in means.items()}
        probe = {"rows": rows, "mean_variance": means, "variance_ratio": ratios,
                 "selected": [name for name in protocol["estimators"] if ratios[name] < .7], "protocol_digest": digest(protocol)}
        write_json(probe_file, probe)
        volume.commit()
    if probe["protocol_digest"] != digest(protocol):
        raise ValueError("Saved variance probe differs")
    print("VARIANCE_GATE " + json.dumps({k: v for k, v in probe.items() if k != "rows"}), flush=True)
    def evaluate(weights, split):
        subset = data[split]
        return evaluate_weights(weights, subset["features"], subset["gains"], subset["ideal"])
    before = evaluate(torch.tensor([1., 0., 0., 0.]), "validation").tolist()
    rows = []
    for estimator in probe["selected"]:
        for seed in protocol["seeds"]:
            folder = directory / f"{estimator}-seed-{seed}"
            if (folder / "result.json").exists():
                rows.append(json.loads((folder / "result.json").read_text()))
                continue
            folder.mkdir(exist_ok=True)
            tracker = Tracker(folder, base_config(), "feature-reranker-credit", {"experiment_id": "rerank-credit-20261106", "seed": seed, "estimator": estimator, "protocol": protocol})
            try:
                weights, history = train_credit_policy(train_data["features"], train_data["gains"], train_data["ideal"], seed, estimator)
                for record in history:
                    tracker.log({k: v for k, v in record.items() if k != "step"}, record["step"])
                after = evaluate(weights, "validation").tolist()
                row = {"estimator": estimator, "seed": seed, "weights": weights.tolist(), "before": before, "after": after,
                    "mean_before": statistics.mean(before), "mean_after": statistics.mean(after),
                    "train_after": evaluate(weights, "train").mean().item(), "protocol_digest": digest(protocol),
                    "paired_query_bootstrap": paired_bootstrap(before, after), "wandb_url": tracker.run.url if tracker.run else None}
                torch.save({"weights": weights, "step": 500, "protocol_digest": digest(protocol)}, folder / "final.pt")
                write_json(folder / "result.json", row)
                tracker.finish(row)
            except Exception:
                tracker.finish(success=False)
                raise
            volume.commit()
            rows.append(row)
            print("POLICY_DONE " + json.dumps({k: v for k, v in row.items() if k not in {"before", "after"}}), flush=True)
    reference = [statistics.mean(r["after"][i] for r in source_result["runs"]) for i in range(len(before))]
    summary = []
    for estimator in probe["selected"]:
        selected = [r for r in rows if r["estimator"] == estimator]
        after = [statistics.mean(r["after"][i] for r in selected) for i in range(len(before))]
        summary.append({"estimator": estimator, "mean_after": statistics.mean(after),
            "mean_train_after": statistics.mean(r["train_after"] for r in selected),
            "versus_hybrid": paired_bootstrap(before, after), "versus_ranking_rloo": paired_bootstrap(reference, after)})
    result = {"summary": summary, "runs": rows, "variance_ratio": probe["variance_ratio"],
        "selected_estimators": probe["selected"], "reference_mean": statistics.mean(reference), "hybrid_mean": statistics.mean(before),
        "query_ids": val_data["query_ids"], "protocol_digest": digest(protocol), "api_calls": 0, "browsecomp_final_test_evaluated": False}
    write_json(directory / "comparison.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    result = run.remote()
    print("COMPARISON " + json.dumps({k: v for k, v in result.items() if k not in {"runs", "query_ids"}}), flush=True)
