"""Stream overlapping lexical passages and reduce their scores to page scores."""
import numpy as np


class PassageTokenRows:
    def __init__(self, mapped, offsets, size=512, stride=384):
        offsets = np.asarray(offsets)
        if not 0 < stride <= size or offsets[0] != 0 or offsets[-1] != len(mapped) or np.any(np.diff(offsets) < 0):
            raise ValueError("Invalid passage layout")
        self.mapped = mapped
        starts, ends, documents = [], [], []
        for doc, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            start = int(begin)
            while True:
                starts.append(start)
                ends.append(min(start + size, int(end)))
                documents.append(doc)
                if start + size >= end:
                    break
                start += stride
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)
        self.documents = np.asarray(documents, dtype=np.int32)
        self.document_count = len(offsets) - 1

    def __len__(self):
        return len(self.starts)

    def __iter__(self):
        for start, end in zip(self.starts, self.ends, strict=True):
            yield self.mapped[start:end].tolist()


def max_page_scores(passage_scores, documents, document_count):
    scores = np.full(document_count, -np.inf, dtype=np.asarray(passage_scores).dtype)
    np.maximum.at(scores, documents, passage_scores)
    if not np.isfinite(scores).all():
        raise ValueError("Every page must have a finite passage score")
    return scores
