"""Prepare, audit, and compare the bounded BrowseComp LLM-reward experiment."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import random
import statistics

import torch

from .config import Config, digest, write_json
from .data import load_prepared
from .graded_judge import DollarBudget, WebJudge, agreement


class CorpusText:
    def __init__(self, bundle):
        from datasets import load_dataset
        self.corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train",
                                   revision=bundle["revisions"]["corpus"])
        self.index = {doc["id"]: i for i, doc in enumerate(bundle["docs"])}

    def __getitem__(self, docid):
        row = self.corpus[self.index[docid]]
        if str(row["docid"]) != docid:
            raise ValueError("Corpus index mismatch")
        return row["text"]


@torch.no_grad()
def prepare_plan(root, config, directory, device="cuda"):
    _, manifest, bundle, tensors = load_prepared(root, config)
    path = directory / "plan-top25.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["prepared_digest"] != digest(manifest):
            raise ValueError("Prepared data changed")
        return result
    docs = tensors["documents"].to(device)
    train_ids = bundle["splits"]["train"]
    references = {qid: tensors["queries"][i] for i, qid in enumerate(bundle["query_vector_ids"])}
    docids = [doc["id"] for doc in bundle["docs"]]
    pools = {}
    for qid in train_ids:
        scores = references[qid].to(device) @ docs.T
        top = scores.argsort(descending=True, stable=True)[:25].tolist()
        pools[qid] = sorted({docids[i] for i in top} | set(bundle["qrels"][qid]))
    rng = random.Random(1307)
    calibration = {}
    for qid in rng.sample(sorted(train_ids), 32):
        scores = references[qid].to(device) @ docs.T
        top = [docids[i] for i in scores.argsort(descending=True, stable=True)[:25].tolist()]
        positive = rng.choice(sorted(bundle["qrels"][qid]))
        retrieved = [d for d in top if d not in bundle["qrels"][qid]][:2]
        random_id = rng.choice(docids)
        calibration[qid] = {positive: "known_evidence", **{d: "retrieved_unjudged" for d in retrieved},
                            random_id: "random"}
    result = {"prepared_digest": digest(manifest), "config": config.to_dict(),
              "train_pools": pools, "calibration": calibration, "initial_candidate_depth": 25,
              "train_pairs": sum(len(p) for p in pools.values()),
              "train_queries": len(train_ids), "validation_queries": len(bundle["splits"]["validation"]),
              "test_queries_untouched": len(bundle["splits"]["test"])}
    write_json(path, result)
    return result


def calibrate(judge, plan, directory):
    requests = {qid: list(docs) for qid, docs in plan["calibration"].items()}
    judge.ensure(requests, workers=12)

    def audit(item):
        qid, docids = item
        reverse = list(reversed(docids))
        repeated = judge.assess(qid, reverse, tag="reverse-repeat")
        stronger = judge.assess(qid, docids, tag="independent-stronger", model="gpt-6.1-sol")
        return [{"query_id": qid, "document_id": docid,
                 "stratum": plan["calibration"][qid][docid],
                 "first": judge.grades[qid][docid]["effective_grade"],
                 "repeat": repeated[docid]["effective_grade"],
                 "stronger": stronger[docid]["effective_grade"],
                 "first_assessment": judge.grades[qid][docid],
                 "repeat_assessment": repeated[docid], "stronger_assessment": stronger[docid]}
                for docid in docids]

    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = [row for group in pool.map(audit, requests.items()) for row in group]
    result = {"repeat": agreement([r["first"] for r in rows], [r["repeat"] for r in rows]),
              "stronger": agreement([r["first"] for r in rows], [r["stronger"] for r in rows]),
              "by_stratum": {kind: {
                  "n": sum(r["stratum"] == kind for r in rows),
                  "primary_mean": statistics.mean(r["first"] for r in rows if r["stratum"] == kind),
                  "stronger_mean": statistics.mean(r["stronger"] for r in rows if r["stratum"] == kind),
              } for kind in ("known_evidence", "retrieved_unjudged", "random")},
              "rows": rows, "api_usd_upper_estimate": judge.budget.total()}
    # These are development checks on training queries; never select the judge on validation.
    result["passed"] = (result["repeat"]["exact"] >= 0.80
                        and result["repeat"]["within_one"] >= 0.95
                        and result["stronger"]["useful_boundary_agreement"] >= 0.80
                        and result["by_stratum"]["random"]["primary_mean"] <= 0.25)
    write_json(directory / f"calibration-{judge.model}.json", result)
    return result


def load_judge(root, config, directory, commit=None):
    _, _, bundle, _ = load_prepared(root, config)
    return WebJudge(directory / "judge-terra-v1", bundle, CorpusText(bundle), commit=commit,
                    budget=DollarBudget(directory / "judge"), model="gpt-6.1-terra"), bundle


def train_settings(base, label, seed=42):
    return replace(base, training_seed=seed, samples_per_query=8, exploration_radius=0.15,
                   kl_weight=0.01, advantage_std_floor=0.05 if label.endswith("normalized") else 0.0)


def base_config():
    return Config(steps=500, eval_every=500, eval_steps=(10, 50, 100, 250, 500),
                  monitor_train_queries=64, training_seed=42)
