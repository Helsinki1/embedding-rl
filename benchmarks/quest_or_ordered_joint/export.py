"""Export completed run evidence; retain large checkpoints on compute."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", default=["pooled_mlp", "attention_mlp", "one_pass", "three_pass"],
                        choices=("pooled_mlp", "attention_mlp", "one_pass", "three_pass"))
    args = parser.parse_args()
    root = args.root
    variants = args.architectures
    for name in variants:
        state = json.loads((root / "status" / f"{name}.json").read_text())
        if state["stage"] != "complete" or state["returncode"] != 0:
            raise RuntimeError(f"Run is not complete: {name}")
    files = []
    for name in variants:
        files.extend(f for f in (root / "results" / name).iterdir() if f.suffix in (".json", ".jsonl"))
        files.append(root / "status" / f"{name}.json")
        files.append(root / "logs" / f"{name}.log")
    if (root / "teachers" / "manifest.json").exists():
        files.extend(root / "teachers" / name for name in ("manifest.json", "ordered_targets.jsonl"))
    files.append(root / "gold_corpus_indices.json")
    if (root / "profile-three_pass.json").exists():
        files.append(root / "profile-three_pass.json")
    package = root / "code" / "benchmarks" / "quest_or_ordered_joint"
    files.extend(f for f in package.iterdir() if f.suffix in (".py", ".json", ".md"))
    shared = ("quest_or_spec_v2", "quest_or_training", "quest_or_multivector", "quest_or_split",
              "quest_or_synthetic", "quest_or_llm_union", "quest_or_balanced", "quest_or/scripts")
    for name in shared:
        files.extend((root / "code" / "benchmarks" / name).glob("*.py"))
    for name in ("manifest.json", "protocol.json"):
        files.append(root / "code" / "benchmarks" / "quest_or_spec_v2" / name)
    index = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (root / "export_checksums.json").write_text(json.dumps(index, indent=2) + "\n")
    files.append(root / "export_checksums.json")
    archive_path = root / "completed-results.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for file in sorted(files):
            archive.add(file, arcname=str(file.relative_to(root)))
    print(json.dumps({"archive": str(archive_path), "files": len(files),
                      "bytes": archive_path.stat().st_size}))


if __name__ == "__main__":
    main()
