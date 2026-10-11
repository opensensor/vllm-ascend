# SPDX-License-Identifier: Apache-2.0
"""Source-derived memory and route occupancy; operation counts are not latency."""

import argparse
import json
from pathlib import Path

import torch

from .channel_scale_memory import ledger
from .channel_scale_runtime import Geometry, scratch_shapes, shape_bytes
from .indexer_heads import memory_plan

TOKEN_WINDOW = 16
CUBE_LOGICAL_ROWS = 32
LOCAL_EXPERTS = 144
TOTAL_EXPERTS = 288


def work(ids, weights, *, offset=0):
    valid = (ids >= offset) & (ids < offset + LOCAL_EXPERTS) & (weights != 0)

    def blocks(route_ids, local):
        counts = torch.bincount(route_ids[local].reshape(-1) - offset, minlength=LOCAL_EXPERTS)
        return int(((counts + CUBE_LOGICAL_ROWS - 1) // CUBE_LOGICAL_ROWS).sum())

    bulk = blocks(ids, valid)
    token = sum(
        blocks(ids[start : start + TOKEN_WINDOW], valid[start : start + TOKEN_WINDOW])
        for start in range(0, ids.shape[0], TOKEN_WINDOW)
    )
    # Complete down matrix W4 bytes per expert block, summed across N tiles.
    bytes_per_block = 4096 * 2048 // 2
    return dict(
        valid_routes=int(valid.sum()),
        bulk_projection_blocks=bulk,
        token_projection_blocks=token,
        bulk_packed_weight_bytes=bulk * bytes_per_block,
        token_packed_weight_bytes=token * bytes_per_block,
        weight_reload_ratio=token / bulk if bulk else 0,
        removed_route_write_read_bytes=int(valid.sum()) * 4096 * 2 * 2,
    )


def audit():
    original = ledger()["on_chip_requests"]
    projection = original["projection_ub_bytes"]
    token_ub = projection + 128 * 128 * 2 + 128 * (8 + 4 + 4) + 2 * 128 * 4 + (128 + 64) * 4
    torch.manual_seed(53)
    ids = torch.stack([torch.randperm(TOTAL_EXPERTS)[:8] for _ in range(640)])
    uniform = work(ids, torch.ones_like(ids, dtype=torch.float32))
    skewed = work(torch.arange(8).expand(640, 8), torch.ones(640, 8))
    plans = {}
    for size, stages in ((2, (3, 4, 4)), (4, (11,))):
        plans[f"tp{size}"] = dict(
            router_payload_bytes=640 * 8 * 12,
            router_total_storage_bytes=(size + 1) * 640 * 8 * 12,
            indexer_head_stages=[
                memory_plan(size=size, max_tokens=640, max_model_len=8192, score_elements=1 << 25, indexers=layers)
                for layers in stages
            ],
        )
    geometries = {}
    for tokens in (17, 640, 1280):
        for token_mode in (False, True):
            geometry = Geometry(144, 4096, 2048, 256, 256, True, True, True, token_mode)
            geometries[f"rows{tokens}-{'token' if token_mode else 'bulk'}"] = shape_bytes(
                scratch_shapes(tokens, 8, geometry, 8)
            )
    return dict(
        qualification="offline counts and reservations; no NPU execution or latency measurement",
        on_chip_bytes=dict(
            stream_extra_ub=0,
            stream_l1=original["projection_l1_bytes"] + 16384,
            stream_l0b=original["projection_l0b_bytes"] + 16384,
            bulk_route_cache_ub=original["prefill_down_fused_ub_bytes"] + 640 * 8 * 8,
            token_down_ub=token_ub,
            optional_source_row_cache_extra=256,
        ),
        routes=dict(uniform=uniform, skewed=skewed),
        profiles=plans,
        prefill_scratch_bytes=geometries,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit(), indent=2) + "\n")


if __name__ == "__main__":
    main()
