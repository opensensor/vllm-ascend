# SPDX-License-Identifier: Apache-2.0
"""Isolated aligned-mask GDN prefill output for the qualified TP4 geometry.

Own the direct binary, metadata and scratch space without changing OPP lookup.
Only eager FP16/FP32, padded 64-row chunks on one stream are admitted.
"""

import hashlib
import inspect
import struct
import textwrap
from collections import OrderedDict

import torch

CHUNK_SIZE = 64
BLOCKS = 8
WORKSPACE_BYTES = 2 * 1024 * 1024
WORKSPACE_OFFSETS = (0, 524288, 1048576, 1310720, 1572864)
MAX_METADATA = 64


def metadata_key(q, k, v, h, scale, g, cu, chunks, chunk_size, transpose):
    if chunk_size != CHUNK_SIZE or transpose or q.ndim != 4:
        return None
    tokens = q.shape[2]
    if not CHUNK_SIZE <= tokens <= 2048 or tokens % CHUNK_SIZE:
        return None
    shapes = (
        (1, 4, tokens, 128),
        (1, 4, tokens, 128),
        (1, 12, tokens, 128),
        (1, 12, tokens // CHUNK_SIZE, 128, 128),
        (1, 12, tokens),
    )
    if g is None:
        return None
    for tensor, shape, dtype in zip((q, k, v, h, g), shapes, (torch.float16,) * 4 + (torch.float32,)):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return None
    if cu is None:
        if chunks is not None:
            return None
        return tokens, float(scale), None, None
    # These lists come from the existing host padding plan, never a device readback.
    if not isinstance(cu, (list, tuple)) or not isinstance(chunks, (list, tuple)):
        return None
    cu = tuple(cu)
    if len(cu) < 2 or cu[0] != 0 or cu[-1] != tokens:
        return None
    if any(b <= a or (b - a) % CHUNK_SIZE for a, b in zip(cu, cu[1:])):
        return None
    expected = tuple(
        value for i, (a, b) in enumerate(zip(cu, cu[1:])) for j in range((b - a) // CHUNK_SIZE) for value in (i, j)
    )
    if tuple(chunks) != expected:
        return None
    return tokens, float(scale), cu, expected


class GDNOutputNative:
    def __init__(self, kernel, original, device):
        self.kernel, self.original, self.device = kernel, original, device
        self.enabled = True
        self.calls = self.fallbacks = 0
        self.records = OrderedDict()
        self.workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.dummy = torch.zeros(1, dtype=torch.int64, device=device)
        self.stream_id = torch.npu.current_stream(device).npu_stream
        self.original_function = self.replacement = None

    def __call__(
        self,
        q,
        k,
        v,
        h,
        scale,
        *,
        g=None,
        g_gamma=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=64,
        transpose_state_layout=False,
    ):
        key = metadata_key(q, k, v, h, scale, g, cu_seqlens, chunk_indices, chunk_size, transpose_state_layout)
        if (
            not self.enabled
            or key is None
            or g_gamma is not None
            or q.device != self.device
            or torch.npu.is_current_stream_capturing()
            or torch.npu.current_stream(self.device).npu_stream != self.stream_id
        ):
            self.fallbacks += 1
            return self.original(
                q,
                k,
                v,
                h,
                scale,
                g=g,
                g_gamma=g_gamma,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                transpose_state_layout=transpose_state_layout,
            )
        if key not in self.records:
            tokens, scale, cu, chunks = key
            raw = struct.pack(
                "<16qf",
                1,
                tokens,
                4,
                12,
                128,
                128,
                CHUNK_SIZE,
                int(cu is not None),
                1 if cu is None else len(cu) - 1,
                0,
                2,
                *WORKSPACE_OFFSETS,
                scale,
            )
            tiling = torch.tensor(list(raw), dtype=torch.uint8, device=self.device)
            cu_tensor = self.dummy if cu is None else torch.tensor(cu, dtype=torch.int64, device=self.device)
            chunks_tensor = (
                self.dummy if chunks is None else torch.tensor(chunks, dtype=torch.int64, device=self.device)
            )
            self.records[key] = tiling, cu_tensor, chunks_tensor
            if len(self.records) > MAX_METADATA:
                self.records.popitem(last=False)
        self.records.move_to_end(key)
        tiling, cu_tensor, chunks_tensor = self.records[key]
        output = torch.empty_like(v)
        torch.ops.qwen_checkpoint_preparation_v1.launch(
            self.kernel, [q, k, v, h, g, cu_tensor, chunks_tensor, output, self.workspace, tiling], BLOCKS
        )
        self.calls += 1
        return output

    def status(self):
        return dict(
            enabled=self.enabled,
            calls=self.calls,
            fallbacks=self.fallbacks,
            metadata_entries=len(self.records),
            workspace_bytes=WORKSPACE_BYTES,
        )


def build_forward(original, owner, source_sha256):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != source_sha256:
        raise RuntimeError("GDN output source differs from qualified runtime")
    seam = "torch.ops._C_ascend.chunk_fwd_o_vllm("
    if source.count(seam) != 1:
        raise RuntimeError("GDN output seam differs")
    source = source.replace(seam, "_output(")
    namespace = dict(original.__globals__, _output=owner)
    exec(compile(source, "<qfn-aligned-gdn-output>", "exec"), namespace)
    return namespace[original.__name__]
