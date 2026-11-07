"""Collect completed runs, verify checkpoints, then publish the verified W&B charts."""
import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .build import HERE, write_json
from .collect import ORDER
from .wandb_sync import REMOTE, SSH_CONFIG


def remote(script, python="python3"):
    return subprocess.run(["ssh", "-F", SSH_CONFIG, "compute", python + " -"], input=script,
                          text=True, capture_output=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [a + "_" + m for a, m in ORDER]
    began = time.monotonic()
    while time.monotonic() - began < 10800:
        response = remote(f'''import pathlib,json
p=pathlib.Path("{REMOTE}/status")
print(json.dumps({{f.stem:json.loads(f.read_text())["stage"] for f in p.glob("*.json")}}))
''')
        states = json.loads(response.stdout)
        if any(v in ("failed", "time_limit") for v in states.values()):
            raise RuntimeError("Run failure requires inspection: " + json.dumps(states))
        if all(states.get(n) == "complete" for n in names):
            break
        time.sleep(30)
    else:
        raise TimeoutError("Waiting for completed runs exceeded three hours")
    audit_script = f'''import pathlib,json,hashlib,torch
root=pathlib.Path("{REMOTE}")
records=[]
for name in {names!r}:
 p=root/"results"/name
 frozen=name.endswith("_frozen")
 target=p/("initial.pt" if frozen else "best.pt")
 h=hashlib.sha256()
 with target.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 selection=json.loads((p/"selection.json").read_text())
 assert h.hexdigest()==selection["checkpoint_sha256"],name
 state=torch.load(target,map_location="cpu",weights_only=True)
 ranks={{v.shape[0] for k,v in state.items() if "lora_A" in k}}
 assert ranks=={{32}},(name,ranks)
 changed=0;delta=0.0
 if not frozen:
  initial=torch.load(p/"initial.pt",map_location="cpu",weights_only=True)
  for k,v in state.items():
   diff=(v.float()-initial[k].float()).square().sum().item()
   changed+=int(diff>0);delta+=diff
  assert changed>0,name
 records.append({{"run":name,"checkpoint_sha256":h.hexdigest(),"lora_rank":32,"changed_tensors":changed,"squared_parameter_change":delta,"selected_step":selection["step"]}})
(root/"checkpoint_audit.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records))
'''
    audit = remote(audit_script, "/home/ubuntu/venvs/quest-training-v2/bin/python")
    print(audit.stdout, flush=True)
    command = (f"cd {REMOTE} && tar czf audit_bundle.tar.gz --exclude='*.pt' --exclude='*.npy' "
               "--exclude='__pycache__' --exclude='code/benchmarks/quest_or_spec_v2/data' "
               "results logs status corpus_ids.json checkpoint_audit.json code/benchmarks")
    subprocess.run(["ssh", "-F", SSH_CONFIG, "compute", command], check=True)
    destination = HERE / "runs/shared-v2"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "audit_bundle.tar.gz"
    subprocess.run(["scp", "-F", SSH_CONFIG, "compute:" + REMOTE + "/audit_bundle.tar.gz", str(archive)], check=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            resolved = (destination / member.name).resolve()
            if not resolved.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
        tar.extractall(destination)
    subprocess.run([sys.executable, "-m", "benchmarks.quest_or_spec_v2.collect", "--results", str(destination / "results"),
                    "--output", str(args.output)], check=True)
    subprocess.run([sys.executable, "-m", "benchmarks.quest_or_spec_v2.wandb_sync", "--scores", str(args.output / "scores.json"),
                    "--results", str(destination / "results")], check=True)
    write_json(args.output / "completion.json", {"completed": names, "checkpoint_audit": json.loads(audit.stdout),
                                                "raw_results": str(destination), "remote_checkpoints": REMOTE + "/results"})


if __name__ == "__main__":
    main()
