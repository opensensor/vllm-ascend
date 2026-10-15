# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-only aligned Mamba commit decisions, pending NPU graph qualification.

This computes the existing CPU fallback's decisions, not the state copies.
The consumer must validate destination storage and predicate accesses on
copy_required. No tensor value is inspected by Python; inputs remain owned by
the caller and can be updated in place between graph replays.
"""

from typing import NamedTuple

import torch


class MambaCommitPlan(NamedTuple):
    destination_blocks: torch.Tensor
    accepted_token_bias: torch.Tensor
    copy_required: torch.Tensor
    next_accepted_counts: torch.Tensor


def plan_aligned_commit(source_blocks, computed_tokens, scheduled_tokens, draft_tokens, accepted_counts, *, block_size):
    """Match aligned CPU commit for admitted nonnegative, bounded INT32 rows.

    Admission must bound intermediate token arithmetic to INT32 and require
    scheduled_tokens > draft_tokens for each active row (running_tokens >= 1).
    Inactive copy rows use zero indices/biases; consumers must obey the predicate.
    Unmodified acceptance is retained unless alignment keeps the same state block.
    """
    tensors = (source_blocks, computed_tokens, scheduled_tokens, draft_tokens, accepted_counts)
    if type(block_size) is not int or not 0 < block_size < (1 << 31):
        raise ValueError("block size must be a positive INT32 host constant")
    if source_blocks.ndim != 1 or any(
        value.shape != source_blocks.shape or value.dtype != torch.int32 or value.device != source_blocks.device
        for value in tensors
    ):
        raise ValueError("commit inputs must be equally shaped INT32 vectors on one device")
    running_tokens = computed_tokens + scheduled_tokens - draft_tokens
    new_computed = running_tokens + accepted_counts - 1
    destination_blocks = torch.div(new_computed, block_size, rounding_mode="floor") - 1
    aligned_tokens = (destination_blocks + 1) * block_size
    reaches_boundary = aligned_tokens >= running_tokens
    same_block = source_blocks == destination_blocks
    bias = aligned_tokens - running_tokens
    copy_required = reaches_boundary & (~same_block | (bias != 0))
    next_counts = torch.where(reaches_boundary & same_block, 1, accepted_counts)
    return MambaCommitPlan(
        torch.where(copy_required, destination_blocks, 0),
        torch.where(copy_required, bias, 0),
        copy_required,
        next_counts,
    )
