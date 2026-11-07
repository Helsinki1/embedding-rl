"""Cached, fully judged training pools and Gaussian conditional gradient averaging.

This changes the TRAINING environment to a fixed candidate pool. Evaluation still
uses the full corpus. Projection is not valid for a full-corpus top-k reward unless
the projector retains the complete corpus span.
"""
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from .config import digest
from .objectives import ndcg, retrieval_rewards


def candidate_basis(vectors):
    """Keep the candidate column span, including rank-deficient pools."""
    left, singular, _ = torch.linalg.svd(vectors.T.double(), full_matrices=False)
    keep = singular > singular.max() * 1e-10
    return left[:, keep].to(vectors.dtype)


@torch.no_grad()
def project_gradient_actions(means, actions, bases):
    """Conditional mean of Gaussian actions; used only inside the score gradient.

    For fixed candidates D, reward depends on D.T @ action. Orthogonal Gaussian
    noise is independent of that reward. Replace noise by its projection without
    renormalizing. Detaching this surrogate is essential to the score-function loss.
    """
    noise = actions.detach() - means.detach().unsqueeze(1)
    projected = torch.stack([(row @ basis) @ basis.T for row, basis in zip(noise, bases)])
    return means.detach().unsqueeze(1) + projected


class CachedPoolReward:
    """No API client or fallback: every fixed training candidate must have a grade."""
    def __init__(self, plan, cache_directory, bundle, documents, projected=False, grade_source="llm"):
        if grade_source not in {"llm", "benchmark"}:
            raise ValueError("Unknown candidate grade source")
        self.allowed = set(bundle["splits"]["train"])
        pools = plan["train_pools"]
        if set(pools) != self.allowed:
            raise ValueError("Candidate pools must exactly match training queries")
        self.documents = documents
        self.projected = projected
        self.cached = {}
        self.indices, self.grades = {}, {}
        index = {doc["id"]: i for i, doc in enumerate(bundle["docs"])}
        digests = set()
        all_grades = {}
        for qid, ids in pools.items():
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("Candidate pool is empty or contains duplicates")
            if grade_source == "llm":
                cached = json.loads((Path(cache_directory) / "pairs" / f"{qid}.json").read_text())
                if cached["query_id"] != qid:
                    raise ValueError("Cached query identity mismatch")
                digests.add(cached["settings_digest"])
                if not set(ids) <= set(cached["grades"]):
                    raise ValueError("Candidate pool has unjudged pages")
                values = [cached["grades"][docid]["effective_grade"] for docid in ids]
            else:
                # Explicit benchmark convention: unknown documents have zero relevance.
                # This is a matched label-source control, not an LLM-cache fallback.
                values = [int(bundle["qrels"][qid].get(docid, 0) > 0) for docid in ids]
            if any(type(v) is not int or v not in range(4) for v in values):
                raise ValueError("Expected integer grades from zero to three")
            self.indices[qid] = torch.tensor([index[docid] for docid in ids])
            self.grades[qid] = torch.tensor(values, dtype=torch.float32)
            all_grades[qid] = dict(zip(ids, values))
        if grade_source == "llm" and len(digests) != 1:
            raise ValueError("Mixed judge settings in cached pools")
        self.metadata = {
            "source": f"{grade_source}/cached_candidate_pool_ndcg", "full_corpus_training_reward": False,
            "grade_source": grade_source,
            "pool_includes_known_training_evidence": True,
            "pool_digest": digest(pools), "grades_digest": digest(all_grades),
            "judge_settings_digest": next(iter(digests)) if digests else None,
            "gradient_estimator": "Gaussian conditional projection" if projected else "Gaussian RLOO",
            "basis_relative_singular_tolerance": 1e-10,
            "queries": len(pools), "candidate_pairs": sum(map(len, pools.values())),
            "mean_candidates": statistics.mean(map(len, pools.values())),
            "all_zero_grade_queries": sum(not any(g.values()) for g in all_grades.values()),
            "api_calls": 0,
        }

    def candidates(self, qid, device):
        if qid not in self.allowed:
            raise ValueError("Candidate reward requested a non-training query")
        key = (qid, str(device))
        if key not in self.cached:
            indices = self.indices[qid].to(device)
            vectors = self.documents[self.indices[qid]].to(device)
            basis = candidate_basis(vectors) if self.projected else None
            self.cached[key] = (vectors, self.grades[qid].to(device), indices, basis)
        return self.cached[key]

    @torch.no_grad()
    def __call__(self, batch, actions, documents, k):
        rewards, winners = [], []
        for qid, row in zip(batch, actions):
            vectors, grades, indices, _ = self.candidates(qid, actions.device)
            scores = F.normalize(row, dim=-1) @ vectors.T
            rewards.append(ndcg(scores, grades, k))
            winners.append(indices[scores.argmax(-1)])
        return torch.stack(rewards), torch.stack(winners)

    def gradient_actions(self, batch, means, actions):
        if not self.projected:
            raise ValueError("Projection is disabled for this reward provider")
        bases = [self.candidates(qid, actions.device)[3] for qid in batch]
        return project_gradient_actions(means, actions, bases)


class AlternatingPoolReward(CachedPoolReward):
    """Nine full-corpus binary updates, then one cached graded-pool update.

    The projection is applied ONLY on pool steps. Applying it to a full-corpus
    reward would discard directions that can change which documents are retrieved.
    """
    def __init__(self, plan, cache_directory, bundle, documents, projected=False, pool_every=10,
                 grade_source="llm"):
        super().__init__(plan, cache_directory, bundle, documents, projected, grade_source)
        if pool_every < 2:
            raise ValueError("Alternation must retain full-corpus updates")
        self.pool_every = pool_every
        self.step = 0
        self.pool_step = False
        self.counts = {"full_corpus": 0, "candidate_pool": 0}
        index = {doc["id"]: i for i, doc in enumerate(bundle["docs"])}
        self.binary_labels = {q: [(index[d], grade) for d, grade in bundle["qrels"][q].items()]
                              for q in self.allowed}
        self.metadata.update({"source": f"alternating/full_binary_and_cached_pool_{grade_source}",
                              "full_corpus_training_reward": True, "pool_every_updates": pool_every,
                              "projection_scope": "candidate-pool updates only" if projected else "none"})

    @torch.no_grad()
    def __call__(self, batch, actions, documents, k):
        if not set(batch) <= self.allowed:
            raise ValueError("Alternating reward requested a non-training query")
        self.step += 1
        self.pool_step = self.step % self.pool_every == 0
        self.counts["candidate_pool" if self.pool_step else "full_corpus"] += 1
        if self.pool_step:
            return super().__call__(batch, actions, documents, k)
        grades = torch.zeros(len(batch), len(documents), device=actions.device)
        for row, qid in enumerate(batch):
            for index, grade in self.binary_labels[qid]:
                grades[row, index] = grade
        return retrieval_rewards(actions, documents, grades, k)

    def gradient_actions(self, batch, means, actions):
        if not self.pool_step:
            return actions
        return super().gradient_actions(batch, means, actions)
