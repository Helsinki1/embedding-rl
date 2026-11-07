"""Build once from pinned sources; never alter the historical benchmark files."""
import collections
import csv
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = ("train", "validation", "test")
COUNTS = {"quest": (441, 26, 52), "msmarco": (4029, 237, 474), "nq": (4030, 237, 474)}
SEED = 20261106


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def normalize(text):
    return " ".join(text.casefold().split())


def groups(rows, keys):
    """Connected components keep shared questions or positive passages together."""
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in keys(row):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    result = collections.defaultdict(list)
    for i, row in enumerate(rows):
        result[root(i)].append(row)
    return list(result.values())


def take_groups(pool, count):
    """Exact subset sum: keep atomic-category groups intact at fixed split sizes."""
    possible = {0: []}
    for i, group in enumerate(pool):
        for total, ids in list(possible.items()):
            new = total + len(group)
            if new <= count and new not in possible:
                possible[new] = ids + [i]
        if count in possible:
            selected = set(possible[count])
            return ([r for j, g in enumerate(pool) if j in selected for r in g],
                    [g for j, g in enumerate(pool) if j not in selected])
    raise ValueError("Cannot meet exact counts without splitting a category group")


def quest():
    categories = {k.removeprefix("Category:").replace("_", " ").casefold():
                  {d.replace("_", " ") for d in v.split("|")}
                  for k, v in csv.reader((HERE / "source/quest_categories.csv").open())}
    rows = []
    for part in ("validation", "test"):
        for old in read_rows(HERE.parent / f"quest_or/data/{part}.jsonl"):
            components = re.findall(r"<mark>(.*?)</mark>", old["original_query"])
            labels = [sorted(categories[c.casefold()] & set(old["docs"])) for c in components]
            if set().union(*map(set, labels)) != set(old["docs"]):
                raise ValueError("Official categories do not cover the accepted QUEST labels")
            rows.append({"id": "quest:" + old["id"], "source": "quest", "query": old["query"],
                         "docs": sorted(set(old["docs"])), "components": components,
                         "component_docs": labels, "component_ids": ["quest-category:" + normalize(c) for c in components],
                         "label_provenance": "Official category membership intersected with accepted QUEST union labels",
                         "original_split": old["split"], "metadata": old["metadata"]})
    pool = groups(rows, lambda r: r["component_ids"] + ["q:" + normalize(r["query"])])
    random.Random(SEED).shuffle(pool)
    test, pool = take_groups(pool, 52)
    val, pool = take_groups(pool, 26)
    return {"train": [r for g in pool for r in g], "validation": val, "test": test}


def atomic_sources():
    import pyarrow.parquet as pq

    atoms = [{**r, "source": "nq"} for r in read_rows(HERE.parent / "quest_or_synthetic/data/source_queries.jsonl")]
    by_query, passages = {}, {}
    for row in pq.read_table(HERE / "source/msmarco_v1.1_train.parquet").to_pylist():
        query = row["query"].strip()
        if not query or len(query) > 256 or re.search(r"\b(and|or|not|without|except)\b", query, re.I):
            continue
        p = row["passages"]
        positive = []
        candidates = []
        for selected, text in zip(p["is_selected"], p["passage_text"]):
            if not text.strip():
                continue
            pid = "msmarco:" + digest(text)
            candidates.append(pid)
            passages[pid] = text
            if selected == 1:
                positive.append(pid)
        if not positive:
            continue
        key = normalize(query)
        atom = by_query.setdefault(key, {"id": "msmarco-query:" + digest(key), "query": query,
                                        "source": "msmarco", "docs": set(), "candidates": set(), "source_rows": []})
        atom["docs"].update(positive)
        atom["candidates"].update(candidates)
        atom["source_rows"].append(row["query_id"])
    for a in by_query.values():
        a["docs"], a["candidates"] = sorted(a["docs"]), sorted(a["candidates"])
        atoms.append(a)
    # A shared query or identical positive text never crosses splits, even across sources.
    grouped = groups(atoms, lambda a: ["q:" + normalize(a["query"])] + ["p:" + d.split(":", 1)[1] for d in a["docs"]])
    pools = {p: {s: [] for s in ("nq", "msmarco")} for p in PARTS}
    for group in grouped:
        value = int(digest(str(SEED) + min(a["id"] for a in group))[:12], 16) / 16**12
        part = "train" if value < .85 else "validation" if value < .90 else "test"
        for a in group:
            pools[part][a["source"]].append(a)
    return pools, passages


def main():
    if (HERE / "manifest.json").exists():
        raise FileExistsError("Frozen dataset already exists; use a new version for any change")
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    partitions = quest()
    pools, passages = atomic_sources()
    used_atoms, selected_passages = [], set()
    for part_index, part in enumerate(PARTS):
        for source in ("msmarco", "nq"):
            pool = sorted(pools[part][source], key=lambda a: a["id"])
            rng = random.Random(f"{SEED}:{part}:{source}")
            rng.shuffle(pool)
            position = 0
            for i in range(COUNTS[source][part_index]):
                arity = 2 + i % 2
                components = pool[position:position + arity]
                if len(components) != arity:
                    raise ValueError("Insufficient disjoint component questions")
                position += arity
                for a in components:
                    used_atoms.append({**a, "split": part})
                    selected_passages.update(a.get("candidates", []))
                partitions[part].append({
                    "id": f"{source}-or:{part}:{i:05d}", "source": source,
                    "query": " OR ".join("(" + a["query"] + ")" for a in components),
                    "components": [a["query"] for a in components],
                    "component_ids": [a["id"] for a in components],
                    "component_docs": [a["docs"] for a in components],
                    "docs": sorted(set().union(*(set(a["docs"]) for a in components))),
                    "label_provenance": ("MS MARCO v1.1 selected passages" if source == "msmarco" else "sentence-transformers NQ positive passages"),
                    "original_split": "train", "synthetic": True})
        random.Random(f"{SEED}:{part}:order").shuffle(partitions[part])
        write_rows(data / f"{part}.jsonl", partitions[part])
    write_rows(data / "atoms.jsonl", used_atoms)
    write_rows(data / "corpus_added.jsonl", ({"title": p, "text": passages[p], "source": "msmarco"} for p in sorted(selected_passages)))
    # Compatibility with the existing provenance-checked additive-corpus encoder.
    write_rows(data / "train_synthetic.jsonl", [r for p in PARTS for r in partitions[p]])
    empty = [{"query_id": r["id"], "component": c} for rows in partitions.values() for r in rows
             for c, docs in zip(r["components"], r["component_docs"]) if not docs]
    manifest = {"version": "quest-msmarco-nq-10000-v2", "seed": SEED,
                "spec_document": "https://docs.google.com/document/d/1xFXwLzTk-bvELbBD_iOYTBYCnPwtd3_rbOb8flYJ6oo/edit",
                "counts": {"splits": {p: len(v) for p, v in partitions.items()},
                           "by_source": {s: dict(zip(PARTS, counts)) for s, counts in COUNTS.items()},
                           "unique_passages": len(selected_passages)},
                "sources": {"quest": "https://storage.googleapis.com/gresearch/quest/all_categories_filtered.csv",
                            "msmarco": {"dataset": "microsoft/ms_marco", "config": "v1.1", "split": "train", "revision": "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"},
                            "nq": {"dataset": "sentence-transformers/natural-questions", "revision": "f9e894e1081e206e577b4eaa9ee6de2b06ae6f17"}},
                "split_policy": "QUEST atomic-category connected components; NQ/MS MARCO shared-question or shared-positive-text connected components assigned before composition; each synthetic atom used once",
                "corpus": {"model_revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
                           "scope": "Full existing 325427 QUEST + 67713 NQ corpus, plus all candidate passages for sampled MS MARCO atoms; not the 8.8M passage leaderboard"},
                "judgments": "Known positives only; unjudged documents score as nonrelevant. Two QUEST components have no accepted positive label and are excluded only from component averages.",
                "unscorable_components": empty,
                "files": {str(p.relative_to(HERE)): sha256(p) for p in sorted(data.glob("*.jsonl"))},
                "source_hashes": {"source/" + name: sha256(HERE / "source" / name)
                                  for name in ("msmarco_v1.1_train.parquet", "quest_categories.csv")}}
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
