"""Audit completed held-out vectors, including whether outputs actually separate."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import matching_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    gold = torch.load(args.run / "teacher/test.pt", weights_only=True)
    counts, targets = gold["counts"], gold["targets"]
    original = gold["original_query_vectors"][:, None].expand(-1, 3, -1)
    original_losses = matching_loss(original, targets, counts)
    # A gold-dependent diagnostic: the best single vector for these targets.
    # It is not a deployable baseline and never participates in training/selection.
    centroid = F.normalize(targets.sum(1), dim=-1)[:, None].expand(-1, 3, -1)
    centroid_losses = matching_loss(centroid, targets, counts)
    report = {"test_queries": len(counts),
              "original_repeated_loss": original_losses.mean().item(),
              "gold_dependent_single_vector_oracle_loss": centroid_losses.mean().item(),
              "oracle_note": "Uses held-out labels only for post-training diagnosis; not a deployable method.",
              "architectures": {}}
    for architecture in ("mtp", "autoregressive"):
        p = args.run / architecture
        predictions = torch.load(p / "test_predictions.pt", weights_only=True)
        assert predictions["ids"] == gold["ids"]
        vectors = predictions["embeddings"]
        assert vectors.shape == (len(counts), 3, 4096)
        assert torch.isfinite(vectors).all()
        assert torch.allclose(vectors.norm(dim=-1), torch.ones(len(counts), 3), atol=1e-5)
        losses = matching_loss(vectors, targets, counts)
        assert torch.allclose(losses, predictions["per_query_loss"], atol=1e-6)
        summary = json.loads((p / "summary.json").read_text())
        assert abs(losses.mean().item() - summary["test"]["matching_cosine_loss"]) < 1e-6
        state = torch.load(p / "best.pt", weights_only=True)
        assert state["slots"].shape == (3, 4096)
        assert all(torch.isfinite(t).all() for t in state.values())
        assert any("lora_B" in n and torch.count_nonzero(t) > 0 for n, t in state.items())
        with (p / "best.pt").open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(f.read()).hexdigest()
        report["architectures"][architecture] = {
            "summary": summary,
            "loss_reduction_relative_to_original": (original_losses.mean()-losses.mean()).item()/original_losses.mean().item(),
            "fraction_queries_beating_original": (losses < original_losses).float().mean().item(),
            "fraction_queries_beating_gold_single_vector_oracle": (losses < centroid_losses).float().mean().item(),
            "checkpoint_sha256": digest,
            "prediction_shape": list(vectors.shape),
            "checkpoint_and_prediction_validation": "passed",
        }
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test_queries": report["test_queries"],
        "original_loss": report["original_repeated_loss"],
        "oracle_centroid_loss": report["gold_dependent_single_vector_oracle_loss"],
        "models": {a: {k: v for k, v in item.items() if k != "summary"}
                   for a, item in report["architectures"].items()}}, indent=2))


if __name__ == "__main__":
    main()
