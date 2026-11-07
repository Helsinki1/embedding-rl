import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get('QUEST_OR_WORK_DIR', '/home/ubuntu/quest-or-baseline-20261106'))
REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
EXPECTED_SHA256 = {
    'val.jsonl': '7d23798af2d85460941743b1c4ab35a2edeafa7bb3e92479398e8d5f88a44da1',
    'test.jsonl': 'bde8a4f02bcf74d181af7e0659509f70517f9cc6922c014fbaf186f2a498d318',
    'documents.zip': 'e9f14fdf9398a0270468c0b8761c0a56a05ccde639c685c2c39e2a3e5ffb910f',
}


def dataset():
    manifest = {}
    for name in ('val.jsonl', 'test.jsonl', 'documents.zip'):
        path = ROOT / 'data' / name
        if not path.exists():
            temp = path.with_suffix(path.suffix + '.partial')
            urllib.request.urlretrieve('https://storage.googleapis.com/gresearch/quest/' + name, temp)
            temp.rename(path)
        manifest[name] = {'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                          'bytes': path.stat().st_size}
        assert manifest[name]['sha256'] == EXPECTED_SHA256[name], f'Source data changed: {name}'
        print('Downloaded', name, manifest[name], flush=True)
    with zipfile.ZipFile(ROOT / 'data/documents.zip') as archive:
        print('Archive members', archive.namelist(), flush=True)
        for member in archive.namelist():
            if Path(member).name == 'documents.jsonl':
                target = ROOT / 'data/documents.jsonl'
                if not target.exists():
                    import shutil
                    with archive.open(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                break
        else:
            raise RuntimeError('No documents.jsonl in corpus archive')
    manifest['documents.jsonl'] = {'sha256': hashlib.file_digest((ROOT / 'data/documents.jsonl').open('rb'), 'sha256').hexdigest()}
    (ROOT / 'data/source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Dataset ready', flush=True)


def model():
    path = snapshot_download('Qwen/Qwen3-Embedding-8B', revision=REVISION,
                             local_dir=ROOT / 'models/Qwen3-Embedding-8B',
                             max_workers=8)
    print('Model ready', path, REVISION, flush=True)


if __name__ == '__main__':
    for folder in ('data', 'models', 'results', 'logs'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(dataset), pool.submit(model)]
        for future in futures:
            future.result()
    print('Preparation complete', time.time(), flush=True)
