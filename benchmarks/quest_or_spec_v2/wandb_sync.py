"""Stream every remote run to W&B from the desktop; credentials stay local."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .build import HERE, write_json
from .collect import ORDER

PROJECT = "quest-or-spec-v2"
REMOTE = "/home/ubuntu/quest-spec-v2-20261106"
SSH_CONFIG = "/home/davidx/Documents/Codex/2026-11-06/les/work/ssh/config"


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "/"))
        elif isinstance(item, (float, int)):
            result[name] = item
    return result


def snapshot(name):
    # Only fixed experiment names are accepted; no credentials enter shell arguments.
    if name not in {a + "_" + m for a, m in ORDER}:
        raise ValueError("Unknown run")
    script = f'''import json,pathlib
r=pathlib.Path("{REMOTE}/results/{name}")
out={{}}
for key,filename in [("run","run.json"),("initial","validation-0000.json"),("summary","summary.json")]:
 p=r/filename
 if p.exists():out[key]=json.loads(p.read_text())
p=r/"metrics.jsonl"
out["history"]=[]
if p.exists():
 for line in p.read_text().splitlines():
  try:out["history"].append(json.loads(line))
  except json.JSONDecodeError:pass
p=pathlib.Path("{REMOTE}/status/{name}.json")
if p.exists():out["status"]=json.loads(p.read_text())
print(json.dumps(out))
'''
    response = subprocess.run(["ssh", "-F", SSH_CONFIG, "compute", "python3 -"],
                              input=script, text=True, capture_output=True, timeout=30, check=True)
    return json.loads(response.stdout)


def sync_one(name, output):
    import wandb
    from dotenv import dotenv_values

    env = dotenv_values(HERE.parent.parent / ".env")
    key = env.get("WANDB_KEY") or env.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise ValueError("Missing workspace W&B credential")
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_SILENT"] = "true"
    run, last_step, initial_logged = None, -1, False
    began = time.monotonic()
    while time.monotonic() - began < 10800:
        state = snapshot(name)
        if run is None and "run" in state:
            metadata = state["run"]
            run_id = hashlib.sha256((metadata["provenance"]["manifest_sha256"] + ":" + name).encode()).hexdigest()[:16]
            run = wandb.init(project=PROJECT, name=name, id=run_id, resume="allow", dir=str(output),
                             group="10000-queries-rank32", job_type=metadata["method"],
                             config={**metadata["config"], "architecture": metadata["architecture"],
                                     "objective": metadata["method"], "documents": metadata["documents"],
                                     "trainable_parameters": metadata["trainable_parameters"],
                                     "dataset_sha256": metadata["provenance"]["manifest_sha256"],
                                     "protocol_sha256": metadata["provenance"]["protocol_sha256"]})
            run.define_metric("train/step")
            run.define_metric("*", step_metric="train/step")
            run.define_metric("validation/overall/union/f1", summary="max")
            write_json(output / (name + ".json"), {"stage": "syncing", "url": run.url, "id": run.id})
        if run is not None:
            if "initial" in state and not initial_logged:
                run.log({"train/step": 0, **flatten(state["initial"]["metrics"], "validation/")}, step=0)
                initial_logged = True
                last_step = max(last_step, 0)
            for row in state["history"]:
                if row["step"] <= last_step:
                    continue
                values = {"train/step": row["step"]}
                for key, value in row.items():
                    if key == "validation":
                        values.update(flatten(value, "validation/"))
                    elif isinstance(value, dict):
                        values.update(flatten(value, "train/" + key + "/"))
                    elif isinstance(value, (int, float)):
                        values["train/" + key] = value
                    elif key == "slot_grad_norms":
                        values.update({"train/slot_gradient/" + str(i): x for i, x in enumerate(value)})
                run.log(values, step=row["step"])
                last_step = row["step"]
            if "summary" in state:
                summary = state["summary"]
                values = flatten({k: summary[k] for k in ("overall", "by_source", "by_arity", "fixed_budget")}, "test/")
                run.log({"train/step": max(last_step, 0) + 1, **values}, step=max(last_step, 0) + 1)
                for key, value in {**values, **flatten(summary["selection"], "selection/"),
                                   "completed_training_examples": summary["examples_seen"],
                                   "elapsed_seconds": summary["elapsed_seconds"],
                                   "peak_gpu_allocated_gb": summary["peak_gpu_allocated_gb"]}.items():
                    run.summary[key] = value
                url, run_id = run.url, run.id
                run.finish()
                write_json(output / (name + ".json"), {"stage": "complete", "url": url, "id": run_id,
                                                        "last_training_step": last_step})
                return
        if state.get("status", {}).get("stage") in ("failed", "time_limit"):
            if run is not None:
                run.finish(exit_code=1)
            raise RuntimeError("Remote training failed: " + name)
        time.sleep(20)
    raise TimeoutError("W&B sync exceeded three hours")


def comparison(scores, output, results):
    import wandb
    from dotenv import dotenv_values
    env = dotenv_values(HERE.parent.parent / ".env")
    os.environ["WANDB_API_KEY"] = env.get("WANDB_KEY") or env.get("WANDB_API_KEY") or os.environ["WANDB_API_KEY"]
    data = json.loads(Path(scores).read_text())
    run = wandb.init(project=PROJECT, name="Final comparison", job_type="comparison", dir=str(output))
    dashboard = {"comparison_url": run.url, "project_url": run.url.rsplit("/runs/", 1)[0],
                 "runs": {p.stem: json.loads(p.read_text()).get("url") for p in output.glob("*.json")
                          if p.stem in {a + "_" + m for a, m in ORDER}}}
    write_json(Path(scores).parent / "wandb.json", dashboard)
    report = Path(scores).parent / "RESULTS.md"
    report.write_text(report.read_text() + "\nW&B: [comparison charts](" + run.url + ") · [all run histories]("
                      + dashboard["project_url"] + ").\n")
    table = wandb.Table(columns=["Approach", "Union F1", "Union precision", "Union recall", "Component F1", "Coverage@100", "F1@20"],
                        data=[[r["approach"] + " / " + r["objective"]] +
                              [100 * r[k] for k in ("union_f1", "union_precision", "union_recall", "component_f1", "component_coverage_at_100", "union_f1_at_20")]
                              for r in data["scores"]])
    run.log({"results": table, "Union F1 comparison": wandb.plot.bar(table, "Approach", "Union F1", title="Test union F1 (%)"),
             "Component F1 comparison": wandb.plot.bar(table, "Approach", "Component F1", title="Test component F1 (%)"),
             "Coverage comparison": wandb.plot.bar(table, "Approach", "Coverage@100", title="Component coverage@100 (%)"),
             "Fixed budget comparison": wandb.plot.bar(table, "Approach", "F1@20", title="Union F1 at 20 final results (%)")})
    if results:
        histories = {}
        validation = {}
        for architecture, method in ORDER:
            if method == "frozen":
                continue
            name = architecture + "_" + method
            directory = results / name
            histories[name] = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            validation[name] = [(int(p.stem.split("-")[1]), json.loads(p.read_text())["metrics"]["overall"])
                                for p in sorted(directory.glob("validation-*.json"))]
        charts = {}
        for key, title, objective in (("loss", "InfoNCE training loss", "infonce"),
                                       ("loss", "RL total training loss", "rl"),
                                       ("reward_f1", "RL sampled union F1 reward", "rl"),
                                       ("kl", "RL KL to initialization", "rl"),
                                       ("zero_advantage_fraction", "RL fraction with no reward signal", "rl"),
                                       ("grad_norm", "Gradient norms", None)):
            selected = {n: h for n, h in histories.items() if objective is None or n.endswith("_" + objective)}
            charts[title] = wandb.plot.line_series(xs=[[v["step"] for v in h] for h in selected.values()],
                                                   ys=[[v[key] for v in h] for h in selected.values()],
                                                   keys=list(selected), title=title, xname="Training step")
        for metric, title in (("union", "Validation union F1"), ("component_macro", "Validation component F1")):
            charts[title] = wandb.plot.line_series(xs=[[step for step, _ in h] for h in validation.values()],
                                                   ys=[[100 * v[metric]["f1"] for _, v in h] for h in validation.values()],
                                                   keys=list(validation), title=title + " (%)", xname="Training step")
        run.log(charts)
    artifact = wandb.Artifact("verified-comparison", type="evaluation", metadata=data["provenance"])
    for path in Path(scores).parent.iterdir():
        if path.is_file() and path.suffix in (".csv", ".json", ".md"):
            artifact.add_file(str(path))
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    write_json(output / "comparison.json", {"url": url, "project": PROJECT})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "wandb_sync")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.scores:
        comparison(args.scores, args.output, args.results)
    elif args.run:
        sync_one(args.run, args.output)
    else:
        children = []
        for architecture, method in ORDER:
            name = architecture + "_" + method
            log = (args.output / (name + ".log")).open("w")
            child = subprocess.Popen([sys.executable, "-m", "benchmarks.quest_or_spec_v2.wandb_sync",
                                      "--run", name, "--output", str(args.output)], stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child, log))
        failures = []
        for name, child, log in children:
            if child.wait() != 0:
                failures.append(name)
            log.close()
        if failures:
            raise RuntimeError("W&B sync failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
