"""Frozen, full-corpus Qwen3 embedding inference, independently sharded by GPU."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('QUEST_OR_WORK_DIR', '/home/ubuntu/quest-or-baseline-20261106'))
REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
MAX_LENGTH = 32768
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--mode', choices=['corpus', 'queries'], default='corpus')
    args = parser.parse_args()
    rank = args.rank
    import fcntl
    lock = (ROOT / f'results/{args.mode}_{rank}.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another worker owns this shard', flush=True)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['RAYON_NUM_THREADS'] = '6'
    os.environ['OMP_NUM_THREADS'] = '4'
    from transformers import AutoTokenizer
    from vllm import LLM, PoolingParams

    model_path = str(ROOT / 'models/Qwen3-Embedding-8B')
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    start_time = time.monotonic()
    if args.mode == 'corpus':
        docs = []
        all_titles = set()
        count = 0
        with (ROOT / 'data/documents.jsonl').open() as file:
            for index, line in enumerate(file):
                row = json.loads(line)
                if row['title'] in all_titles:
                    continue  # Corpus audit confirmed all duplicate titles have identical text.
                all_titles.add(row['title'])
                index = count
                count += 1
                if index % 2 == rank:
                    docs.append({'index': index, 'title': row['title'], 'text': row['text']})
        entries = [str(doc['title']) + '\n' + doc['text'] for doc in docs]
        identities = [{'index': d['index'], 'title': d['title']} for d in docs]
        print('Corpus count', count, 'shard count', len(entries), flush=True)
    else:
        identities, entries = [], []
        for split in ('val', 'test'):
            for line in (ROOT / f'results/{split}_clean.jsonl').read_text().splitlines():
                row = json.loads(line)
                for component, text in [('original', row['query'])] + [(str(i), c) for i, c in enumerate(row['components'])]:
                    identities.append({'id': row['id'], 'component': component, 'text': text})
                    entries.append(f'Instruct: {INSTRUCTION}\nQuery:{text}')

    prefix = f'corpus_{rank}' if args.mode == 'corpus' else 'queries'
    output = ROOT / f'results/{prefix}.npy'
    checkpoint_path = ROOT / f'results/{prefix}_progress.json'
    identities_path = ROOT / f'results/{prefix}_ids.json'
    identity_hash = hashlib.sha256(json.dumps(identities, ensure_ascii=False).encode()).hexdigest()
    done, token_count, truncated = 0, 0, 0
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        assert checkpoint['identity_sha256'] == identity_hash
        assert checkpoint['revision'] == REVISION and checkpoint['max_length'] == MAX_LENGTH
        done, token_count, truncated = checkpoint['done'], checkpoint['tokens'], checkpoint['truncated']
        if done == len(entries):
            print('Already complete', prefix, flush=True)
            return
        vectors = np.lib.format.open_memmap(output, mode='r+')
    else:
        vectors = np.lib.format.open_memmap(output, mode='w+', dtype='float16', shape=(len(entries), 4096))
        identities_path.write_text(json.dumps(identities, ensure_ascii=False))
    llm = LLM(model=model_path, runner='pooling', convert='embed', dtype='bfloat16',
              max_model_len=MAX_LENGTH, gpu_memory_utilization=0.55,
              max_num_batched_tokens=32768, max_num_seqs=64,
              enable_chunked_prefill=False, enforce_eager=True,
              seed=42, trust_remote_code=False)
    initial_done = done
    pooling = PoolingParams(use_activation=True)
    for start in range(done, len(entries), 2048):
        end = min(start + 2048, len(entries))
        tokenized = tokenizer(entries[start:end], add_special_tokens=True, truncation=False)['input_ids']
        truncated += sum(len(tokens) > MAX_LENGTH for tokens in tokenized)
        tokenized = [tokens[:MAX_LENGTH] for tokens in tokenized]
        token_count += sum(map(len, tokenized))
        prompts = [{'prompt_token_ids': tokens} for tokens in tokenized]
        outputs = llm.embed(prompts, pooling_params=pooling, use_tqdm=False)
        embeddings = np.asarray([o.outputs.embedding for o in outputs], dtype=np.float32)
        assert embeddings.shape == (end - start, 4096)
        assert np.isfinite(embeddings).all()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        assert (norms > 0).all()
        embeddings /= norms
        vectors[start:end] = embeddings.astype(np.float16)
        vectors.flush()
        elapsed = time.monotonic() - start_time
        progress = {'done': end, 'total': len(entries), 'tokens': token_count,
                    'truncated': truncated, 'revision': REVISION, 'max_length': MAX_LENGTH,
                    'identity_sha256': identity_hash, 'elapsed_seconds': elapsed,
                    'docs_per_second_this_session': (end - initial_done) / elapsed}
        temp = checkpoint_path.with_suffix('.tmp')
        temp.write_text(json.dumps(progress, indent=2))
        temp.replace(checkpoint_path)
        print(json.dumps(progress), flush=True)
    print('Embedding complete', prefix, flush=True)


if __name__ == '__main__':
    main()
