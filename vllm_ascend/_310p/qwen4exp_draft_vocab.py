# SPDX-License-Identifier: Apache-2.0
"""Load complete floating-point QFN vocabulary weights for a TP1 draft."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

VOCAB_COPY_ROWS = 4096


@torch.no_grad()
def load_full_vocab_weight(layer, model_path, weight_names, vocab_size, hidden_size):
    """Read only the indexed tensor, bounding temporary host/device storage.

    A TP6 target module cannot be aliased into a TP1 drafter: its vocabulary
    rows and collectives belong to a different group. Load the full source
    tensor into an independently owned FP16 module instead. Target runtime
    quantization and target parameters are never changed here.
    """
    weight = layer.weight
    if weight.dtype != torch.float16 or weight.ndim != 2 or weight.shape[1] != hidden_size:
        raise ValueError("QFN draft vocabulary requires an FP16 matrix with the target hidden width")
    shards = layer.shard_indices
    if (
        getattr(layer, "tp_size", None) != 1
        or shards.org_vocab_start_index != 0
        or shards.org_vocab_end_index != vocab_size
        or weight.shape[0] < vocab_size
        or getattr(layer, "num_added_embeddings", 0)
    ):
        raise ValueError("QFN draft vocabulary requires TP1 and the complete unremapped target vocabulary")
    model_path = Path(model_path)
    matches = set()
    for index in model_path.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text())["weight_map"]
        matches.update((mapping[name], name) for name in weight_names if name in mapping)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one indexed target vocabulary tensor for {weight_names}, got {matches}")
    filename, name = matches.pop()
    path = (model_path / filename).resolve()
    with safe_open(path, framework="pt", device="cpu") as source:
        tensor = source.get_slice(name)
        if tensor.get_shape() != [vocab_size, hidden_size] or tensor.get_dtype() not in {"F16", "BF16", "F32"}:
            raise ValueError("Target vocabulary source must be a complete floating-point tensor")
        for start in range(0, vocab_size, VOCAB_COPY_ROWS):
            end = min(start + VOCAB_COPY_ROWS, vocab_size)
            rows = tensor[start:end].to(dtype=torch.float16)
            if not torch.isfinite(rows).all():
                raise ValueError("Target vocabulary contains nonfinite values after FP16 conversion")
            weight[start:end].copy_(rows)
    weight[vocab_size:].zero_()
    return {"path": str(path), "tensor": name, "shape": [vocab_size, hidden_size], "dtype": "float16"}
