"""Discard unused least-squares backing storage without changing fitted values."""
import json

import modal

from modal_common import ROOT, image, volume

app = modal.App("embedding-rl-compact-reranker")


@app.function(image=image, cpu=2, memory=2048, volumes={ROOT: volume}, timeout=300)
def compact():
    import hashlib
    import torch
    from embedding_rl.config import write_json

    path = ROOT / "experiments/reranker-conditioning-20261106/transform.pt"
    record_path = path.with_name("compaction-verification.json")
    if record_path.exists():
        return json.loads(record_path.read_text())
    original = torch.load(path, weights_only=True)
    tensors = {k: v for k, v in original.items() if isinstance(v, torch.Tensor)}
    hashes = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in tensors.items()}
    before = path.stat().st_size
    output = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in original.items()}
    temporary = path.with_name("transform-compact.tmp.pt")
    torch.save(output, temporary)
    reloaded = torch.load(temporary, weights_only=True)
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            if not torch.equal(value, reloaded[key]) or hashlib.sha256(reloaded[key].numpy().tobytes()).hexdigest() != hashes[key]:
                raise ValueError("Compaction changed fitted values")
            if reloaded[key].untyped_storage().nbytes() != value.numel() * value.element_size():
                raise ValueError("Tensor retains unnecessary backing storage")
        elif value != reloaded[key]:
            raise ValueError("Compaction changed provenance")
    temporary.replace(path)
    result = {"before_bytes": before, "after_bytes": path.stat().st_size, "tensor_hashes": hashes,
        "values_and_metadata_unchanged": True, "protocol_digest": original["protocol_digest"],
        "training_updates": 0, "api_calls": 0}
    write_json(record_path, result)
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(compact.remote()), flush=True)
