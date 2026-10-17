# SPDX-License-Identifier: Apache-2.0
"""Strict local W4 bundle loading, also usable without a vLLM installation."""

import json

import regex as re
import torch
from safetensors import safe_open

from tools.minimax_m3.export_w4 import EXPERT_NAME, MANIFEST, file_digest, local_file
from vllm_ascend._310p.ops.minimax_m3 import W4_LAYOUT, PackedM3Experts

LAYER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def read_manifest(folder):
    manifest = json.loads((folder / MANIFEST).read_text())
    if manifest.get("complete") is not True or manifest.get("layout") != W4_LAYOUT:
        raise ValueError("M3 W4 export is incomplete or uses an unsupported layout")
    if file_digest(folder / "config.json") != manifest.get("export_config_sha256"):
        raise ValueError("M3 W4 config digest mismatch")
    config = json.loads((folder / "config.json").read_text())
    text = config.get("text_config", config)
    if config.get("ascend_minimax_m3_w4") != W4_LAYOUT or text.get("ascend_minimax_m3_w4") != W4_LAYOUT:
        raise ValueError("M3 W4 config marker missing")
    records = manifest["weights"]
    if file_digest(folder / "source_index.json") != manifest["index_sha256"]:
        raise ValueError("M3 W4 source index digest mismatch")
    source_keys = set(json.loads((folder / "source_index.json").read_text())["weight_map"])
    if set(records) != source_keys:
        raise ValueError("M3 W4 manifest does not cover exactly the source index")
    if len(records) != manifest["tensor_count"]:
        raise ValueError("M3 W4 manifest tensor count mismatch")
    experts = set()
    for name, record in records.items():
        if record["name"] != name:
            raise ValueError("M3 W4 manifest key mismatch")
        local_file(folder, record["file"])
        match = EXPERT_NAME.fullmatch(name)
        if bool(match) != record["expert"]:
            raise ValueError("M3 W4 expert classification mismatch")
        if match:
            identity = (int(match[1]), int(match[2]), match[3])
            if identity in experts:
                raise ValueError("Duplicate M3 W4 expert identity")
            experts.add(identity)
    expected = {
        (layer, expert, projection)
        for layer, sparse in enumerate(text["moe_layer_freq"])
        if sparse
        for expert in range(text["num_local_experts"])
        for projection in ("w1", "w2", "w3")
    }
    if experts != expected:
        raise ValueError("M3 W4 manifest does not cover exactly the configured expert set")
    return manifest


def checked_tensors(folder, record):
    path = local_file(folder, record["file"])
    if file_digest(path) != record["sha256"]:
        raise ValueError(f"M3 W4 payload digest mismatch: {path.name}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(record["tensors"]):
            raise ValueError(f"M3 W4 payload keys mismatch: {path.name}")
        for name, metadata in record["tensors"].items():
            view = handle.get_slice(name)
            if view.get_shape() != metadata["shape"] or view.get_dtype() != metadata["dtype"]:
                raise ValueError(f"M3 W4 payload metadata mismatch: {path.name}")
            yield name, handle.get_tensor(name)


def load_expert_banks(model, folder, manifest):
    by_id = {}
    for name, record in manifest["weights"].items():
        match = EXPERT_NAME.fullmatch(name)
        if match:
            by_id[int(match[1]), int(match[2]), match[3]] = record
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, PackedM3Experts):
                continue
            layer_match = LAYER_NAME.search(module.prefix)
            if layer_match is None:
                raise ValueError(f"Expert module has no layer identity: {module.prefix}")
            layer = int(layer_match[1])
            for local in range(module.num_local_experts):
                expert = module.local_expert_offset + local
                for projection in ("w1", "w2", "w3"):
                    record = by_id[layer, expert, projection]
                    for kind, tensor in checked_tensors(folder, record):
                        target = getattr(module, f"{projection}_{kind}")[local]
                        if target.shape != tensor.shape or target.dtype != tensor.dtype:
                            raise ValueError(
                                f"Expert shape/dtype mismatch: layer {layer}, expert {expert}, {projection}"
                            )
                        if kind == "scales" and (not torch.isfinite(tensor).all() or not (tensor > 0).all()):
                            raise ValueError("M3 W4 scales must be finite and positive")
                        target.copy_(tensor)
                        count += 1
    if not count:
        raise ValueError("No resident M3 W4 expert banks found")
    return count


def dense_weights(folder, manifest, *, start_layer, end_layer, first_stage, last_stage):
    for name, record in manifest["weights"].items():
        if record["expert"]:
            continue
        # Only language decoder layers are partitioned; vision has its own
        # layer namespace and is loaded by the existing multimodal wrapper.
        match = LAYER_NAME.search(name) if name.startswith(("language_model.model.", "model.layers.")) else None
        if match and not start_layer <= int(match[1]) < end_layer:
            continue
        if name.endswith("embed_tokens.weight") and not first_stage:
            continue
        if (name.endswith("lm_head.weight") or name.endswith("model.norm.weight")) and not last_stage:
            continue
        yield from checked_tensors(folder, record)
