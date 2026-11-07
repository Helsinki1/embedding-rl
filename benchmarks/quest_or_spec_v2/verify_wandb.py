"""Read back all W&B histories and check that no training steps were lost."""
import argparse
import json
import math
import time
from pathlib import Path

from .build import COUNTS, HERE, write_json
from .collect import ORDER


def main():
    import wandb
    from dotenv import dotenv_values

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    while not (args.output / "completion.json").exists():
        if time.monotonic() - started > 10800:
            raise TimeoutError("Final collection did not complete")
        time.sleep(20)
    env = dotenv_values(HERE.parent.parent / ".env")
    api = wandb.Api(api_key=env.get("WANDB_KEY") or env.get("WANDB_API_KEY"))
    records = []
    for architecture, method in ORDER:
        name = architecture + "_" + method
        status_path = HERE / "wandb_sync" / (name + ".json")
        for attempt in range(6):
            status = json.loads(status_path.read_text())
            if status["stage"] != "complete":
                time.sleep(20)
                continue
            path = status["url"].removeprefix("https://wandb.ai/").replace("/runs/", "/")
            run = api.run(path)
            expected = json.loads((HERE / "runs/shared-v2/results" / name / "summary.json").read_text())
            try:
                assert run.state == "finished", (name, run.state)
                assert run.config["lora_rank"] == 32
                assert run.config["dataset_sha256"] == expected["dataset_manifest_sha256"]
                assert math.isclose(run.summary["test/overall/union/f1"], expected["overall"]["union"]["f1"], abs_tol=1e-12)
                history = list(run.scan_history(keys=["train/step", "train/loss"])) if method != "frozen" else []
                if method != "frozen":
                    assert [int(r["train/step"]) for r in history] == list(range(1, 267)), (name, "missing history")
                    assert run.summary["completed_training_examples"] == 8500
                    assert all(run.summary["train/seen_sources/" + source] == counts[0] for source, counts in COUNTS.items())
                    val = list(run.scan_history(keys=["train/step", "validation/overall/union/f1"]))
                    assert [int(r["train/step"]) for r in val] == [0, 89, 178, 266]
                    if method == "rl":
                        rewards = list(run.scan_history(keys=["train/step", "train/reward_f1", "train/kl"]))
                        assert len(rewards) == 266
                records.append({"run": name, "url": status["url"], "state": run.state,
                                "training_steps_verified": len(history), "test_score_matches": True,
                                "source_counts_match": True, "lora_rank": 32})
                print(json.dumps(records[-1]), flush=True)
                break
            except (AssertionError, KeyError):
                if attempt == 5:
                    raise
                time.sleep(20)
                api.flush()
        else:
            raise RuntimeError("Run did not finish syncing: " + name)
    write_json(args.output / "wandb_verification.json", {"runs": records,
               "trained_runs": sum(r["training_steps_verified"] > 0 for r in records),
               "total_training_steps": sum(r["training_steps_verified"] for r in records)})


if __name__ == "__main__":
    main()
