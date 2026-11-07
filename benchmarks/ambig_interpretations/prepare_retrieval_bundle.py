"""Snapshot code and pinned data for the post-training corpus evaluation."""
import argparse
import json
from pathlib import Path
import shutil

from benchmarks.quest_or_spec_v2.build import sha256
from benchmarks.quest_or_spec_v2.data import load_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("/tmp/ambig-retrieval-bundle"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    _, _, _, provenance = load_splits()
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    packages = ("quest_or", "quest_or_training", "quest_or_split", "quest_or_multivector",
                "quest_or_llm_union", "quest_or_balanced", "quest_or_synthetic",
                "quest_or_spec_v2", "ambig_interpretations")
    for name in packages:
        files.extend(f for f in (root / "benchmarks" / name).rglob("*.py")
                     if not {"runs", "source", "__pycache__"} & set(f.parts))
    spec = root / "benchmarks/quest_or_spec_v2"
    manifest = json.loads((spec / "manifest.json").read_text())
    files += [spec / f for f in ["manifest.json", "protocol.json", *manifest["files"]]]
    files += [root / "benchmarks/quest_or_synthetic" / f for f in ("manifest.json", "data/corpus_added.jsonl")]
    files += [root / "benchmarks/ambig_interpretations/data" / f for f in ("train.jsonl", "validation.jsonl")]
    for source in sorted(set(files)):
        target = args.output / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    reference = args.output / "reference"
    reference.mkdir()
    previous = spec / "runs/shared-v2"
    shutil.copy2(previous / "corpus_ids.json", reference / "corpus_ids.json")
    shutil.copy2(previous / "results/single_frozen/summary.json", reference / "single_frozen_summary.json")
    record = {"dataset": provenance, "files_sha256": {
        str(f.relative_to(args.output)): sha256(f) for f in sorted(args.output.rglob("*")) if f.is_file()}}
    (args.output / "bundle_manifest.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"bundle": str(args.output), "files": len(record["files_sha256"]), "dataset": provenance}))


if __name__ == "__main__":
    main()
