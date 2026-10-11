# SPDX-License-Identifier: Apache-2.0
"""Offline GLM bytes and work ledger; never equates reductions with speedup."""

import argparse
import json
from pathlib import Path

from .channel_scale_math import CUBE_K, OUTPUT_TILE
from .channel_scale_runtime import CORES, DECODE_ROWS, LANES, Geometry, scratch_shapes, shape_bytes

HIDDEN = 4096
INTERMEDIATE = 2048
GLOBAL_EXPERTS = 288
MOE_LAYERS = 42
PIPELINE_MOE_LAYERS = (12, 15, 15)
LOCAL_EXPERTS = 144
GATE_GROUP = 256
DOWN_GROUP = 256
PHYSICAL_ROWS = 64


def ledger():
    codes = 3 * HIDDEN * INTERMEDIATE // 2
    factors = (2 * INTERMEDIATE * HIDDEN // GATE_GROUP + HIDDEN * INTERMEDIATE // DOWN_GROUP) * 8
    diagonal = (HIDDEN + INTERMEDIATE) * 4
    projection_ub = (
        PHYSICAL_ROWS * CUBE_K // 2
        + 4 * PHYSICAL_ROWS * OUTPUT_TILE * 4
        + 2 * OUTPUT_TILE * 4
        + PHYSICAL_ROWS * LANES * 4
        + PHYSICAL_ROWS * OUTPUT_TILE * 2
    )
    combine_ub = 2 * OUTPUT_TILE * 4 + (LANES + LANES * LANES) * 4
    per_expert = codes + factors + diagonal
    legacy_widened = codes + 3 * HIDDEN * INTERMEDIATE // (32 * 32) * 2
    # From the hardware-qualified r10 weight-load log, before these candidates.
    qualified_weights_gib = (24.3452, 27.9371, 28.5279)
    stages = []
    for layers, existing in zip(PIPELINE_MOE_LAYERS, qualified_weights_gib):
        replacement = layers * LOCAL_EXPERTS * (codes + factors)
        old_experts = layers * LOCAL_EXPERTS * legacy_widened
        inferred_dense = existing * 1024**3 - old_experts
        stages.append(
            {
                "moe_layers": layers,
                "new_expert_bytes_per_rank_identity": replacement,
                "inferred_nonexpert_bytes_from_r10": int(inferred_dense),
                "projected_total_weight_bytes_per_rank": int(replacement + inferred_dense),
                "qualification": "projection using prior measured weights; new weights not hardware loaded",
            }
        )
    scratch = []
    for tokens in (1, 4, 640, 1280, 2560):
        for name, fused in (("unfused_new_math", False), ("fully_fused_new_math", True)):
            geometry = Geometry(LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, GATE_GROUP, DOWN_GROUP, fused, fused, fused)
            shapes = scratch_shapes(tokens, 8, geometry, 8)
            scratch.append(
                {
                    "tokens": tokens,
                    "schedule": name,
                    "live_bytes": shape_bytes(shapes),
                    "buffers": {
                        key: {"shape": list(shape), "dtype": str(dtype)} for key, (shape, dtype) in shapes.items()
                    },
                }
            )
    packets = []
    for tokens in (1, 4, 640, 1280):
        payload = tokens * (HIDDEN * 2 + 4 * HIDDEN * 4 + (4 + 16) * 4)
        packets.append(
            {
                "tokens": tokens,
                "payload_bytes": payload,
                "packed_transport_temporary_bytes": 3 * payload,
                "receive_storage_retained_bytes": 2 * payload,
                "collectives_per_handoff": 1,
                "qualification": "adds packing copies; latency unmeasured",
            }
        )
    return {
        "qualification": "offline source-derived bytes/counts, not NPU peak memory or speed measurements",
        "expert_params": 3 * HIDDEN * INTERMEDIATE,
        "expert_bytes": per_expert,
        "expert_code_bytes": codes,
        "factor_bytes": factors,
        "all_experts_disk_tensor_bytes": per_expert * GLOBAL_EXPERTS * MOE_LAYERS,
        "all_experts_device_bytes_identity": (codes + factors) * GLOBAL_EXPERTS * MOE_LAYERS,
        "stages_tp2_pp3": stages,
        "scratch": scratch,
        "pipeline_packets": packets,
        "on_chip_requests": {
            "qualification": (
                "source-derived TPipe reservations; compiler/runtime overhead "
                "and actual device resource admission remain unmeasured"
            ),
            "cube_physical_rows": PHYSICAL_ROWS,
            "projection_ub_bytes": projection_ub,
            "gate_up_fused_ub_bytes_a8": projection_ub + (PHYSICAL_ROWS // 2) * 2 * DOWN_GROUP * 2 + 32,
            "gate_up_fused_ub_bytes_a4": projection_ub + PHYSICAL_ROWS * 2 * DOWN_GROUP * 2 + 32,
            "decode_down_fused_ub_bytes": projection_ub + DECODE_ROWS * LANES * OUTPUT_TILE * 2 + combine_ub,
            "prefill_down_fused_ub_bytes": projection_ub + combine_ub,
            "projection_l1_bytes": PHYSICAL_ROWS * CUBE_K // 2 + OUTPUT_TILE * CUBE_K // 2,
            "projection_l0a_bytes": PHYSICAL_ROWS * CUBE_K // 2,
            "projection_l0b_bytes": OUTPUT_TILE * CUBE_K // 2,
            "projection_l0c_bytes": PHYSICAL_ROWS * OUTPUT_TILE * 4,
            "cores": CORES,
            "optional_source_row_cache_extra_ub_bytes": PHYSICAL_ROWS * 4,
        },
        "work_per_n128_tile": {
            "gate_k256_updates": HIDDEN // 256,
            "gate_g256_product_readbacks": HIDDEN // GATE_GROUP,
            "gate_block32_scale_groups": HIDDEN // 32,
            "down_k256_updates": INTERMEDIATE // 256,
            "down_g256_product_readbacks": INTERMEDIATE // DOWN_GROUP,
            "down_block32_scale_groups": INTERMEDIATE // 32,
            "old_a8_bias_cube_rows": "replaced by permanent weight sums",
            "numerical_scalar_factor_getvalues": 0,
            "native_launches_full_fusion": 3,
        },
        "context_note": (
            "MoE scratch follows incoming token rows, not max_model_len; KV/state/dense/HCCL need separate admission"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger(), indent=2) + "\n")


if __name__ == "__main__":
    main()
