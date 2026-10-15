# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read only one hyperconnection's named tensors from a local checkpoint."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_ascend.models.qwen4_exp.model import _GatedResidual

PARAMETER_SUFFIXES = (
    ("hc_norm_weight", "hc_norm.weight"),
    ("input_mix_weight_down", "input_mix_weight_down.weight"),
    ("input_mix_weight_up", "input_mix_weight_up.weight"),
    ("block_inject_weight", "block_inject_weight.weight"),
)


@torch.no_grad()
def load_hyperconnection(checkpoint: Path, prefix: str):
    """Return a CPU module with serving FP16 storage and the exact source keys."""
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    weights = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    keys = {attribute: f"{prefix}.{suffix}" for attribute, suffix in PARAMETER_SUFFIXES}
    missing = [key for key in keys.values() if key not in weights]
    if missing:
        raise ValueError(f"checkpoint has no complete combinable hyperconnection {prefix}: {missing}")
    module = _GatedResidual(
        hc_count=config["hc_count"],
        hidden_size=config["hidden_size"],
        lowrank=config["hc_lowrank"],
        eps=config["rms_norm_eps"],
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
    )
    for attribute, key in keys.items():
        with safe_open(checkpoint / weights[key], framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(key)
        parameter = getattr(module, attribute)
        if tensor.shape != parameter.shape:
            raise ValueError(f"{key}: expected {tuple(parameter.shape)}, got {tuple(tensor.shape)}")
        parameter.copy_(tensor)
    return module, keys
