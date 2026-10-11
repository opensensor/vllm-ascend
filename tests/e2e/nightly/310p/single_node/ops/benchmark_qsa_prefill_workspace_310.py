# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure bounded QSA prefill workspace at the long-prefix OOM geometry.

Run on an idle 310P with the hardware environment sourced. This weight-free
operator regression complements the real-weight cached-extension service gate.
It never invokes the legacy full-row top-k allocation that killed the server.
"""

import argparse
import json
import time

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import (
    _QSA_SELECTION_MAX_TILE_ELEMENTS,
    QSA_SELECTION_FAST_TOPK,
    QSA_SELECTION_POLICIES,
    _qsa_matmul_select_groups,
    _repair_native_group_indices,
    _select_topk_indices,
)

GROUPS_PER_BLOCK = 32
COMPRESSION_RATIO = 4
INDEX_HEADS = 4
HEAD_DIM = 128
SELECTED_GROUPS = 512
MAX_EXTRA_BYTES = 256 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=2560)
    parser.add_argument("--cached-prefix", type=int, default=223872)
    parser.add_argument("--policy", choices=sorted(QSA_SELECTION_POLICIES), default=QSA_SELECTION_FAST_TOPK)
    args = parser.parse_args()
    if args.tokens <= 0 or args.cached_prefix < COMPRESSION_RATIO * SELECTED_GROUPS:
        parser.error("tokens must be positive and cached-prefix must cover the selection budget")
    torch_npu.npu.set_device(args.device)
    torch.manual_seed(729)
    device = f"npu:{args.device}"
    raw_tokens_per_block = GROUPS_PER_BLOCK * COMPRESSION_RATIO
    table_width = (args.cached_prefix + args.tokens + raw_tokens_per_block - 1) // raw_tokens_per_block
    capacity = GROUPS_PER_BLOCK * table_width
    query = torch.randn(args.tokens, INDEX_HEADS, HEAD_DIM, device=device, dtype=torch.float16)
    cache = torch.randn(table_width, GROUPS_PER_BLOCK + 3, HEAD_DIM, device=device, dtype=torch.float16)
    table = torch.arange(table_width, device=device, dtype=torch.int32).unsqueeze(0)
    positions = torch.arange(args.cached_prefix, args.cached_prefix + args.tokens, device=device, dtype=torch.int32)
    visible = ((positions + 1) // COMPRESSION_RATIO).clamp_max_(capacity)
    torch_npu.npu.synchronize()
    baseline = torch_npu.npu.memory_allocated()
    torch_npu.npu.reset_peak_memory_stats()
    start = time.monotonic()
    selected = _qsa_matmul_select_groups(
        query, cache, table, visible, (args.tokens,), GROUPS_PER_BLOCK, SELECTED_GROUPS, args.policy
    )
    torch_npu.npu.synchronize()
    elapsed = time.monotonic() - start
    peak = torch_npu.npu.max_memory_allocated() - baseline
    keys = cache[:, :GROUPS_PER_BLOCK].reshape(capacity, HEAD_DIM).float().t()
    groups = torch.arange(capacity, device=device)
    tile_rows = max(1, _QSA_SELECTION_MAX_TILE_ELEMENTS // capacity)
    checked_rows = sorted({0, min(tile_rows - 1, args.tokens - 1), min(tile_rows, args.tokens - 1), args.tokens - 1})
    for row in checked_rows:
        scores = torch.matmul(query[row : row + 1].float(), keys).relu_().sum(dim=1)
        scores.masked_fill_(groups.unsqueeze(0) >= visible[row : row + 1].unsqueeze(1), -torch.inf)
        chosen = _select_topk_indices(scores, SELECTED_GROUPS, args.policy).to(torch.int32)
        expected = _repair_native_group_indices(chosen, visible[row : row + 1])
        if not torch.equal(expected.cpu(), selected[row : row + 1].cpu()):
            raise AssertionError(f"selection differs at row {row}")
    if peak >= MAX_EXTRA_BYTES:
        raise AssertionError(f"selection workspace exceeded {MAX_EXTRA_BYTES} bytes: {peak}")
    print(
        json.dumps(
            {
                "passed": True,
                "query_rows": args.tokens,
                "cached_prefix_tokens": args.cached_prefix,
                "capacity_groups": capacity,
                "tile_rows": tile_rows,
                "policy": args.policy,
                "elapsed_s": elapsed,
                "peak_extra_bytes": peak,
                "checked_rows": checked_rows,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
