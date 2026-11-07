"""Read and verify the immutable 5,000-query split."""
import json
from pathlib import Path

from .build import normalized_query, read_rows, sha256


def load_splits(directory, source=None):
    directory = Path(directory)
    protocol = json.loads(Path(__file__).with_name("shared_protocol.json").read_text())
    if sha256(directory / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Shared split manifest changed; every approach must use the pinned split")
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        if sha256(directory / relative) != digest:
            raise ValueError("Frozen 5,000-query split changed: " + relative)
    if source is not None:
        for relative, digest in manifest["source_sha256"].items():
            if sha256(Path(source) / relative) != digest:
                raise ValueError("Source dataset changed: " + relative)
    parts = [read_rows(directory / "data" / (name + ".jsonl"))
             for name in ("train", "validation", "test")]
    if [len(rows) for rows in parts] != [3600, 400, 1000]:
        raise ValueError("Expected the fixed 3600/400/1000 partition")
    rows = sum(parts, [])
    if len({r["id"] for r in rows}) != 5000 or len({normalized_query(r) for r in rows}) != 5000:
        raise ValueError("Split contains duplicate IDs or query text")
    exclusions = json.loads((directory / "provenance/excluded_from_new_holdouts.json").read_text())
    exposed = set(exclusions["training_exposure_ids"])
    prior_eval = set(exclusions["previous_evaluation_ids"])
    if any(r["id"] in exposed for r in parts[1]) or any(r["id"] in exposed | prior_eval for r in parts[2]):
        raise ValueError("Previously exposed queries entered a new holdout")
    return *parts, manifest
