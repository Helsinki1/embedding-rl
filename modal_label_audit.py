"""Training-only agreement counts for cached judge and benchmark candidate labels."""
import json

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-label-agreement")


@app.function(image=image, cpu=4, memory=16384, volumes={ROOT: volume}, timeout=600)
def audit_remote():
    from embedding_rl.candidate_policy import CachedPoolReward
    from embedding_rl.config import write_json
    from embedding_rl.data import load_prepared
    from embedding_rl.judge_experiment import base_config

    _, _, bundle, tensors = load_prepared(ROOT, base_config())
    source = ROOT / "experiments/graded-20261106"
    plan = json.loads((source / "plan-top25.json").read_text())
    reward = CachedPoolReward(plan, source / "judge-terra-v1", bundle, tensors["documents"])
    table = {name: {str(grade): 0 for grade in range(4)}
             for name in ("benchmark_evidence", "benchmark_unjudged")}
    no_useful_evidence = 0
    for qid in sorted(reward.allowed):
        pool = plan["train_pools"][qid]
        evidence = {docid for docid, grade in bundle["qrels"][qid].items() if grade > 0}
        if not evidence <= set(pool):
            raise ValueError("Training pool omitted known evidence")
        useful_evidence = 0
        for docid, grade in zip(pool, reward.grades[qid].int().tolist()):
            name = "benchmark_evidence" if docid in evidence else "benchmark_unjudged"
            table[name][str(grade)] += 1
            useful_evidence += docid in evidence and grade >= 2
        no_useful_evidence += useful_evidence == 0
    result = {"training_queries": len(reward.allowed), "candidate_pairs": reward.metadata["candidate_pairs"],
              "counts_by_benchmark_status_and_llm_grade": table,
              "queries_without_grade2plus_on_known_evidence": no_useful_evidence,
              "pool_digest": reward.metadata["pool_digest"], "grades_digest": reward.metadata["grades_digest"],
              "api_calls": 0, "validation_evaluated": False, "test_evaluated": False,
              "caveat": "Benchmark-unjudged is not necessarily irrelevant. Judge grades refer to supplied excerpts and the rubric, while benchmark evidence may span a full page or one reasoning hop. Disagreement counts do not establish which label is correct."}
    write_json(source / "training-label-agreement.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(audit_remote.remote(), indent=2))
