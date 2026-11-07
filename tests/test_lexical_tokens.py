import json

import numpy as np
import pytest

bm25s = pytest.importorskip("bm25s")
Stemmer = pytest.importorskip("Stemmer")

from lexical_tokens import tokenize_to_disk  # noqa: E402


def test_interrupted_tokenization_resumes_and_preserves_retrieval(tmp_path):
    texts = ["cats sleeping cats", "fish swim in water", "dogs chase cats", "a fish swims"]
    calls = []
    def interrupted(start, end):
        calls.append(start)
        if start == 2:
            raise RuntimeError("simulated preemption")
        return texts[start:end]
    with pytest.raises(RuntimeError, match="preemption"):
        tokenize_to_disk(interrupted, len(texts), tmp_path, "fixed", block_size=2)
    assert json.loads((tmp_path / "progress.json").read_text())["done"] == 2
    # A partially written next block must be discarded on resume.
    with (tmp_path / "tokens.bin").open("ab") as stream:
        np.array([12345], dtype=np.int32).tofile(stream)
    calls.clear()
    def resumed(start, end):
        calls.append(start)
        return texts[start:end]
    cached = tokenize_to_disk(resumed, len(texts), tmp_path, "fixed", block_size=2)
    assert calls == [2]
    reference = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
    first, second = bm25s.BM25(), bm25s.BM25()
    first.index(cached, show_progress=False)
    second.index(reference, show_progress=False)
    query = bm25s.tokenize(["swimming fish"], stemmer=Stemmer.Stemmer("english"), return_ids=False, show_progress=False)[0]
    assert np.allclose(first.get_scores(query), second.get_scores(query))
    with pytest.raises(ValueError, match="protocol differs"):
        tokenize_to_disk(resumed, len(texts), tmp_path, "changed", block_size=2)
