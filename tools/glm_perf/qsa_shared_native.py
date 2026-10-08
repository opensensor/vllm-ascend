# SPDX-License-Identifier: Apache-2.0
"""Direct QSA benchmark adapter with the production 310P tiling arithmetic."""

import math

import torch

CORES = 8
Q24 = 1 << 24
MAX_HEAD_ELEMENTS_PER_TASK = 6144


def tiling_header(query_shape, cache_shape, group_shape, block_shape, starts_shape, scale, logical_heads=0):
    if not (
        len(query_shape) == 3
        and len(cache_shape) == 4
        and len(group_shape) == len(block_shape) == 2
        and len(starts_shape) == 1
    ):
        raise ValueError("invalid QSA descriptor ranks")
    tokens, heads, dim = query_shape
    blocks, channels, page, inner = cache_shape
    if (
        tokens <= 0
        or heads <= 0
        or dim not in (256, 512)
        or inner != 16
        or channels % (dim // 16)
        or not 0 < page <= 65535
        or page % 4
        or blocks <= 0
    ):
        raise ValueError("invalid QSA query/cache geometry")
    physical = channels // (dim // 16)
    kv_heads = logical_heads or physical
    if not 0 < kv_heads <= physical or heads % kv_heads or heads // kv_heads > 64:
        raise ValueError("invalid logical QSA head count")
    if (
        group_shape[0] != tokens
        or not 0 < group_shape[1] <= 512
        or starts_shape[0] < 2
        or block_shape[0] != starts_shape[0] - 1
        or block_shape[1] <= 0
    ):
        raise ValueError("invalid QSA selection geometry")
    if not math.isfinite(scale):
        raise ValueError("QSA scale must be finite")
    base = tokens * kv_heads
    desired = max(1, CORES // base)
    per_head = heads // kv_heads
    per_task = min((per_head + desired - 1) // desired, max(1, MAX_HEAD_ELEMENTS_PER_TASK // dim))
    tiles = (per_head + per_task - 1) // per_task
    tasks = base * tiles
    launch_blocks = min(tasks, CORES)
    raw_scale = scale * Q24
    scale_q24 = math.floor(raw_scale + 0.5) if raw_scale >= 0 else math.ceil(raw_scale - 0.5)
    header = (
        tokens,
        heads,
        kv_heads,
        per_task,
        tiles,
        dim,
        page,
        channels,
        block_shape[1],
        group_shape[1],
        starts_shape[0] - 1,
        (tasks + launch_blocks - 1) // launch_blocks,
        tasks,
        scale_q24,
    )
    return header, launch_blocks


class NativeQsaShared:
    def __init__(self, kernel, launch, device):
        self.kernel, self.launch, self.device = kernel, launch, device
        self.configs = {}
        self.workspace = torch.empty(1, dtype=torch.uint8, device=device)
        self.calls, self.shared_calls = 0, 0

    def __call__(
        self,
        query,
        key,
        value,
        indices,
        counts,
        tail_starts,
        tail_counts,
        blocks,
        starts,
        scale,
        compress_ratio=4,
        logical_kv_heads=0,
    ):
        inputs = (query, key, value, indices, counts, tail_starts, tail_counts, blocks, starts)
        if compress_ratio != 4 or any(t.device != self.device or not t.is_contiguous() for t in inputs):
            raise ValueError("QSA arguments must be contiguous on one device with compression ratio 4")
        if (
            query.dtype != torch.float16
            or key.dtype != torch.float16
            or value.dtype != torch.float16
            or key.shape != value.shape
            or any(t.dtype != torch.int32 for t in inputs[3:])
        ):
            raise ValueError("QSA requires matching FP16 caches and INT32 metadata")
        if any(t.shape != (query.shape[0],) for t in (counts, tail_starts, tail_counts)):
            raise ValueError("QSA row counts must match query rows")
        header, launch_blocks = tiling_header(
            query.shape, key.shape, indices.shape, blocks.shape, starts.shape, scale, logical_kv_heads
        )
        if header not in self.configs:
            self.configs[header] = torch.tensor(header, dtype=torch.int64, device=self.device)
        output = torch.empty_like(query)
        self.launch(self.kernel, [*inputs, output, self.workspace, self.configs[header]], launch_blocks)
        self.calls += 1
        self.shared_calls += int(key.data_ptr() == value.data_ptr())
        return output
