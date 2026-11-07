"""Every approach must reach the same frozen query split before model allocation."""
import importlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from .build import sha256
from .data import load_splits

HERE = Path(__file__).resolve().parent


def test_shared_protocol_is_pinned():
    protocol = json.loads((HERE / "shared_protocol.json").read_text())
    assert sha256(HERE / "manifest.json") == protocol["manifest_sha256"]
    train, dev, test, manifest = load_splits(HERE, HERE.parent / "quest_or_synthetic")
    assert [len(train), len(dev), len(test)] == [3600, 400, 1000]
    assert manifest["files"] == protocol["files"]


@pytest.mark.parametrize("module,arguments", [
    ("quest_or_balanced.run", ["--method", "rl"]),
    ("quest_or_synthetic.train", ["--method", "infonce"]),
    ("quest_or_training.train", ["--method", "rl"]),
    ("quest_or_split.train", ["--method", "rl", "--device", "cpu"]),
    ("quest_or_multivector.train", ["--variant", "one_pass", "--device", "cpu"]),
    ("quest_or_multivector.train", ["--variant", "three_pass", "--device", "cpu"]),
])
def test_entrypoint_uses_shared_split(module, arguments, monkeypatch, tmp_path):
    entry = importlib.import_module("benchmarks." + module)
    owner = (importlib.import_module("benchmarks.quest_or_synthetic.train")
             if module == "quest_or_training.train" else entry)

    class ReachedSharedSplit(Exception):
        pass

    def inspect(directory, source):
        assert Path(directory).resolve() == HERE
        assert Path(source).resolve() == HERE.parent / "quest_or_synthetic"
        raise ReachedSharedSplit

    monkeypatch.setattr(owner, "load_splits", inspect)
    dataset_flag = "--corpus-dataset" if module == "quest_or_balanced.run" else "--dataset"
    monkeypatch.setattr(sys, "argv", [module, *arguments, dataset_flag, str(HERE.parent / "quest_or_synthetic"),
                                     "--work-dir", str(tmp_path), "--embeddings", str(tmp_path),
                                     "--output", str(tmp_path / "result")])
    with pytest.raises(ReachedSharedSplit):
        entry.main()


def test_changed_queries_are_rejected(tmp_path):
    for name in ("data", "provenance"):
        shutil.copytree(HERE / name, tmp_path / name)
    shutil.copyfile(HERE / "manifest.json", tmp_path / "manifest.json")
    with (tmp_path / "data/train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        load_splits(tmp_path)


def test_different_manifest_is_rejected_even_with_valid_file_hashes(tmp_path):
    manifest = json.loads((HERE / "manifest.json").read_text())
    manifest["seed"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned split"):
        load_splits(tmp_path)
