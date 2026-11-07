"""Resumable, memory-mapped tokenization for a large private lexical corpus."""
import json
from pathlib import Path

import numpy as np


class DiskTokenRows:
    """Reiterate disk-backed IDs as one native-integer document at a time."""

    def __init__(self, mapped, offsets):
        self.mapped = mapped
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    def __iter__(self):
        for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            # BM25S intersects with sets of Python ints; numpy scalar equality is costly.
            yield self.mapped[start:end].tolist()


def tokenize_to_disk(read_texts, count, folder, protocol_digest, commit=lambda: None, block_size=2048):
    import bm25s
    import Stemmer
    from embedding_rl.config import write_json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "progress.json"
    tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=Stemmer.Stemmer("english"))
    done, offsets = 0, [0]
    if marker.exists():
        progress = json.loads(marker.read_text())
        if progress["protocol_digest"] != protocol_digest or progress["count"] != count:
            raise ValueError("Token cache protocol differs")
        done = progress["done"]
        offsets = np.load(folder / "offsets.npy")[:done + 1].tolist()
        tokenizer.load_vocab(folder)
    token_path = folder / "tokens.bin"
    with token_path.open("r+b" if token_path.exists() else "w+b") as stream:
        stream.truncate(offsets[-1] * 4)
        stream.seek(0, 2)
        for start in range(done, count, block_size):
            end = min(start + block_size, count)
            texts = read_texts(start, end)
            if len(texts) != end - start:
                raise ValueError("Text block length differs")
            for ids in tokenizer.streaming_tokenize(texts, update_vocab=True):
                np.asarray(ids, dtype=np.int32).tofile(stream)
                offsets.append(offsets[-1] + len(ids))
            stream.flush()
            np.save(folder / "offsets.next.npy", np.array(offsets, dtype=np.int64))
            (folder / "offsets.next.npy").replace(folder / "offsets.npy")
            tokenizer.save_vocab(folder, vocab_name="vocab.next.json")
            (folder / "vocab.next.json").replace(folder / "vocab.tokenizer.json")
            write_json(marker, {"protocol_digest": protocol_digest, "count": count, "done": end,
                                "token_count": offsets[-1]})
            commit()
            print(f"TOKENS_SAVED {end}/{count}", flush=True)
    mapped = np.memmap(token_path, mode="r", dtype=np.int32)
    rows = DiskTokenRows(mapped, offsets)
    return bm25s.tokenization.Tokenized(ids=rows, vocab=tokenizer.get_vocab_dict())
