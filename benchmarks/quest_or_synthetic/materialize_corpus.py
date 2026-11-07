"""Write a combined raw corpus on compute, retaining the existing corpus unchanged."""
import argparse
import json
from pathlib import Path

from .build import HERE, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=HERE)
    parser.add_argument("--base-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh combined corpus path")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    added_path = args.dataset / "data/corpus_added.jsonl"
    if sha256(added_path) != manifest["files"]["data/corpus_added.jsonl"]:
        raise ValueError("Added corpus checksum mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    seen, counts = set(), []
    with temporary.open("w") as output:
        for index, path in enumerate((args.base_corpus, added_path)):
            count = 0
            with path.open() as source:
                for line in source:
                    row = json.loads(line)
                    if row["title"] in seen:
                        if index == 1:
                            raise ValueError("Added document ID collides with an existing ID")
                        continue  # Original QUEST duplicate titles were already audited as identical.
                    seen.add(row["title"])
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
            counts.append(count)
    if counts != [manifest["counts"]["base_corpus"], manifest["counts"]["unique_passages"]]:
        raise ValueError("Combined raw corpus count mismatch")
    temporary.replace(args.output)
    report = {"base_documents": counts[0], "added_documents": counts[1], "total": sum(counts),
              "combined_corpus": str(args.output), "sha256": sha256(args.output),
              "base_corpus_sha256": sha256(args.base_corpus), "added_corpus_sha256": sha256(added_path)}
    write_json(args.output.with_suffix(".manifest.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
