# SPDX-License-Identifier: Apache-2.0
"""Header-only parameter ledger; excludes runtime pools and cannot admit hardware."""

import argparse
import json
import math
from pathlib import Path

import regex as re

from tools.glm_w2.convert_full import _load_headers, route_tensor

from .deployment_profiles import W4_LAYERS, profile

LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def parameter_ledger(config, weight_map, headers, selected):
    text = config.get("text_config", config)
    layers, experts = text["num_hidden_layers"], text["n_routed_experts"]
    if experts % selected.expert_partition_size:
        raise ValueError("expert count does not divide profile partition")
    per_expert = [0] * (layers + 1)
    dense = [0] * selected.pipeline_parallel_size
    dense_tensor_count = [0] * selected.pipeline_parallel_size
    for name, filename in weight_map.items():
        meta = headers[filename][name]
        route = route_tensor(
            name, meta["dtype"], selected.default_expert_bits, frozenset(W4_LAYERS), frozenset(selected.w3_layers)
        )
        if route.target in ("SCALE", "EXCLUDE"):
            continue
        match = LAYER.search(name)
        layer = int(match[1]) if match else None
        if layer is not None and layer >= layers and not selected.mtp_tokens:
            continue
        if route.target.startswith("W"):
            # Sum all experts, then divide exactly by the authoritative EP/TP group.
            n, k = meta["shape"]
            per_expert[layer] += n * k * int(route.target[1:]) // 8 + (n // 32) * (k // 32) * 2
            continue
        # Conservative semantic parameter upper bound: every dense parameter
        # is FP32 and replicated, even where the actual model shards/casts it.
        stage = (
            min(layer * selected.pipeline_parallel_size // layers, selected.pipeline_parallel_size - 1)
            if layer is not None
            else 0
        )
        if layer is None and ("lm_head." in name or name.startswith("model.language_model.norm.")):
            stage = selected.pipeline_parallel_size - 1
        dense[stage] += math.prod(meta["shape"]) * 4
        dense_tensor_count[stage] += 1
    result = []
    for stage in range(selected.pipeline_parallel_size):
        first = (layers * stage + selected.pipeline_parallel_size - 1) // selected.pipeline_parallel_size
        stop = (layers * (stage + 1) + selected.pipeline_parallel_size - 1) // selected.pipeline_parallel_size
        banks = per_expert[first:stop]
        if selected.mtp_tokens and stage == selected.pipeline_parallel_size - 1:
            banks += per_expert[layers:]
        routed = sum(banks) // selected.expert_partition_size
        result.append(
            dict(
                stage=stage,
                layers=[first, stop],
                routed_codes_and_fp16_scales_bytes=routed,
                dense_parameter_fp32_replicated_upper_bound_bytes=dense[stage],
                semantic_parameter_upper_bound_bytes=routed + dense[stage],
                dense_tensor_count=dense_tensor_count[stage],
            )
        )
    return dict(
        profile=selected.name,
        stages=result,
        hardware_admission=False,
        excludes=[
            "allocator and parameter padding",
            "HCCL",
            "graph pools",
            "KV and recurrent state",
            "prefill/route scratch",
            "temporary load and conversion buffers",
        ],
        assumption="Illustrative uniform PP layer partition; actual vLLM layer boundaries must match",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cards", type=int, choices=(2, 3), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    config = json.loads((source / "config.json").read_text())
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    result = parameter_ledger(config, weight_map, _load_headers(source, weight_map), profile(args.cards))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
