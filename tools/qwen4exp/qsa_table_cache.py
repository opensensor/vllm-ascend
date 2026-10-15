# SPDX-License-Identifier: Apache-2.0
"""Owned eager Qwen QSA entry with a per-core cache of the current block-table row.

The direct binary changes address lookup only. Decode graphs and unsupported
geometries keep their original operator and ownership.
"""

from collections import OrderedDict

import torch

MIN_PREFILL_TOKENS = 128
MAX_RECORDS = 64
CORE_COUNT = 8
MAX_TABLE_BLOCKS = 4096


def tiling_values(query, key, value, selection, table, bounds, scale, compress_ratio):
    if query.ndim != 3:
        return None
    tokens = query.shape[0]
    if (
        query.ndim != 3
        or tuple(query.shape[1:]) != (6, 256)
        or not MIN_PREFILL_TOKENS <= tokens <= 2048
        or key.ndim != 4
        or tuple(key.shape[1:]) != (16, 64, 16)
        or value.shape != key.shape
        or table.ndim != 2
        or not 1 <= table.shape[0] <= 4
        or not 8 <= table.shape[1] <= MAX_TABLE_BLOCKS
        or table.shape[1] % 8
        or tuple(bounds.shape) != (table.shape[0] + 1,)
        or selection.group_indices.ndim != 2
        or selection.group_indices.shape[0] != tokens
        or not 1 <= selection.group_indices.shape[1] <= 512
        or compress_ratio != 4
        or scale not in (None, 0.0625)
    ):
        return None
    floats = (query, key, value)
    integers = (
        selection.group_indices,
        selection.group_counts,
        selection.tail_starts,
        selection.tail_counts,
        table,
        bounds,
    )
    if any(tuple(t.shape) != (tokens,) for t in integers[1:4]):
        return None
    if any(t.dtype != torch.float16 for t in floats) or any(
        t.dtype not in (torch.int32, torch.int64) for t in integers
    ):
        return None
    if any(t.device != query.device or not t.is_contiguous() for t in (*floats, *integers)):
        return None
    return (
        tokens,
        6,
        1,
        6,
        1,
        256,
        64,
        16,
        table.shape[1],
        selection.group_indices.shape[1],
        table.shape[0],
        (tokens + CORE_COUNT - 1) // CORE_COUNT,
        tokens,
        1 << 20,
    )


class QSATableCache:
    def __init__(self, kernel, original, device):
        self.kernel, self.original, self.device = kernel, original, device
        self.stream = torch.npu.current_stream(device).npu_stream
        self.metadata = OrderedDict()
        self.enabled = True
        self.verify = False
        self.failed = False
        self.calls = self.checked = self.fallbacks = 0

    def __call__(self, query, key, value, selection, table, bounds, *, scale=None, compress_ratio=4):
        values = tiling_values(query, key, value, selection, table, bounds, scale, compress_ratio)
        if (
            values is None
            or not self.enabled
            or self.failed
            or query.device != self.device
            or torch.npu.is_current_stream_capturing()
            or torch.npu.current_stream(self.device).npu_stream != self.stream
        ):
            self.fallbacks += 1
            return self.original(
                query, key, value, selection, table, bounds, scale=scale, compress_ratio=compress_ratio
            )
        if values not in self.metadata:
            self.metadata[values] = torch.tensor(values, dtype=torch.int64, device=self.device)
            if len(self.metadata) > MAX_RECORDS:
                self.metadata.popitem(last=False)
        self.metadata.move_to_end(values)
        output = torch.empty_like(query)
        # Match the production wrapper's integer normalization. The learned
        # indexer returns int64 selections; native address metadata is int32.
        integers = [
            tensor.to(dtype=torch.int32).contiguous()
            for tensor in (
                selection.group_indices,
                selection.group_counts,
                selection.tail_starts,
                selection.tail_counts,
                table,
                bounds,
            )
        ]
        torch.ops.qwen_checkpoint_preparation_v1.launch(
            self.kernel, [query, key, value, *integers, output, self.metadata[values]], CORE_COUNT
        )
        self.calls += 1
        if self.verify:
            expected = self.original(
                query, key, value, selection, table, bounds, scale=scale, compress_ratio=compress_ratio
            )
            if torch.equal(output.view(torch.int16), expected.view(torch.int16)):
                self.checked += 1
            else:
                self.failed = True
            return expected
        return output

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            calls=self.calls,
            checked=self.checked,
            fallbacks=self.fallbacks,
            metadata_entries=len(self.metadata),
        )
