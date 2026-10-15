# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline general QSA capture helper with safe unused page-table entries.

This does not install a replacement or own graph dispatch. The caller must
admit the existing SelectionPlan before replay and retain input storage. NPU
capture/parity qualification of this additional masking operation is pending.
"""

import torch

from tools.qwen4exp.resident_candidates.qsa_graph_selection import select_with_plan


def mask_unused_pages(block_table, logical_positions, *, plan):
    """Replace only causally unused physical page IDs with zero on device.

    The matmul scorer gathers its entire fixed capacity before masking scores.
    A bucket can therefore access uninitialized page-table suffixes that the
    native causal scorer never reads. Compute the used extent from current
    device positions, independently for each packed request. Query lengths are
    immutable admitted host geometry; no tensor value is read by Python.

    Live page IDs are deliberately unchanged, including invalid IDs: allocation
    validation remains the caller's responsibility and must not be hidden by
    clamping them. Zero-query requests need no pages. Input tables are never
    mutated, including their negative padding sentinels.
    """
    if block_table.ndim != 2 or block_table.shape[0] != len(plan.query_lens):
        raise ValueError("page table does not match admitted requests")
    if block_table.shape[1] < plan.capacity_blocks:
        raise ValueError("page table does not cover admitted capacity")
    if logical_positions.shape != (sum(plan.query_lens),):
        raise ValueError("logical positions do not match admitted query rows")
    if block_table.dtype not in (torch.int32, torch.int64) or logical_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("page IDs and logical positions must be integers")
    if block_table.device != logical_positions.device:
        raise ValueError("page IDs and logical positions must share a device")
    used_blocks = []
    start = 0
    for length in plan.query_lens:
        if length:
            last_position = logical_positions[start : start + length].max()
            groups = torch.div(last_position + 1, plan.compress_ratio, rounding_mode="floor").clamp_min(0)
            blocks = torch.div(groups + plan.groups_per_block - 1, plan.groups_per_block, rounding_mode="floor")
        else:
            blocks = logical_positions.new_zeros(())
        used_blocks.append(blocks)
        start += length
    extent = torch.stack(used_blocks).unsqueeze(1)
    page_columns = torch.arange(plan.capacity_blocks, dtype=logical_positions.dtype, device=block_table.device)
    table = block_table[:, : plan.capacity_blocks]
    return torch.where(page_columns.unsqueeze(0) < extent, table, 0)


def select_general_with_plan(query, compressed_cache, block_table, query_start_loc, logical_positions, *, plan):
    """Use qualified score/selection math, masking padding only before GEMM.

    Native scoring already respects causal extents before fetching pages, and
    prefix selection does not fetch pages. Their existing paths are unchanged.
    """
    if plan.backend == "matmul":
        block_table = mask_unused_pages(block_table, logical_positions, plan=plan)
    return select_with_plan(query, compressed_cache, block_table, query_start_loc, logical_positions, plan=plan)
