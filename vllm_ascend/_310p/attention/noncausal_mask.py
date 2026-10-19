# SPDX-License-Identifier: Apache-2.0
"""Logical masks for eager, paged, bidirectional draft attention."""

from collections.abc import Sequence

import torch


def noncausal_draft_mask(
    query_lengths: Sequence[int],
    kv_lengths: Sequence[int],
    *,
    device: torch.device,
    sliding_window: int | None = None,
    max_key_length: int,
) -> torch.Tensor:
    """Bottom-right aligned queries attend within their own KV sequence.

    A window W is symmetric with W-1 keys on either side, matching upstream
    FlashAttention's noncausal window convention. Prefix cache entries are
    included; future draft queries are visible unless outside that window.
    Lengths are host metadata from an eager call, not device scalar reads.
    """
    if len(query_lengths) != len(kv_lengths) or not query_lengths:
        raise ValueError("Expected matching, nonempty query and KV length lists")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError("Sliding window must be positive")
    if any(q < 1 or kv < q or kv > max_key_length for q, kv in zip(query_lengths, kv_lengths)):
        raise ValueError("Expected 0 < query length <= KV length <= mask capacity")
    width = max(kv_lengths)
    keys = torch.arange(width, device=device)
    rows = []
    for query_length, kv_length in zip(query_lengths, kv_lengths):
        query_positions = torch.arange(kv_length - query_length, kv_length, device=device)
        allowed = (keys[None, :] < kv_length).expand(query_length, -1)
        if sliding_window is not None:
            allowed = allowed & ((keys[None, :] - query_positions[:, None]).abs() < sliding_window)
        rows.append(torch.where(allowed, 0.0, -float("inf")).to(torch.float16))
    return torch.cat(rows, dim=0)
