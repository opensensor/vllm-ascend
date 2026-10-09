# SPDX-License-Identifier: Apache-2.0
"""Explicit native prefill resources; importing this module submits no work."""

import torch

CHUNK_SIZE = 64
HEAD_DIM = 128
MAX_PADDED_TOKENS = 8192
MAX_ROUTES = 25600
MAX_CONFIGS = 256


def _on_npu(device):
    return device.type == "npu"


def _capturing(device):
    return _on_npu(device) and torch.npu.is_current_stream_capturing()


class NativeResource:
    def __init__(self, kernel, launch):
        self.kernel = kernel
        self.launch = launch
        self._configs = {}

    def config(self, values, device):
        key = (tuple(values), device)
        if key not in self._configs:
            if _capturing(device):
                raise RuntimeError("prewarm native shape before graph capture")
            if len(self._configs) >= MAX_CONFIGS:
                raise RuntimeError("native configuration cache full; create a new resource offline")
            # Retain config tensors for captured addresses; never evict a live
            # entry or rewrite one while a kernel can still reference it.
            self._configs[key] = torch.tensor(values, dtype=torch.int64, device=device)
        return self._configs[key]


def _tensors_on_one_device(tensors, device):
    if not _on_npu(device) or any(t.device != device or not t.is_contiguous() for t in tensors):
        raise ValueError("native inputs must be contiguous on one NPU")


class NativeWY(NativeResource):
    """Fuse lower decay, triangular solve, U/W products and final casts.

    Grouped Gram, Q/K layout and FP32 cumulative gates remain torch operations.
    The solve changes FP32 summation order relative to the blocked reference;
    recurrent-state and full-model quality gates are mandatory.
    """

    def __call__(self, q, k, v, g, beta, chunk_size):
        if chunk_size != CHUNK_SIZE or k.ndim != 4 or q.shape != k.shape or v.ndim != 4:
            raise ValueError("native WY requires matching BTHD Q/K and 64-token chunks")
        batch, tokens, key_heads, dim = k.shape
        value_heads = v.shape[2]
        if (
            not 1 <= batch <= 4
            or not 0 < tokens <= MAX_PADDED_TOKENS
            or tokens % CHUNK_SIZE
            or dim != HEAD_DIM
            or v.shape[:2] != k.shape[:2]
            or v.shape[-1] != HEAD_DIM
            or key_heads < 1
            or value_heads < 1
            or value_heads % key_heads
            or g.shape != (batch, tokens, value_heads)
            or beta.shape != g.shape
            or any(t.dtype != torch.float16 for t in (q, k, v))
            or any(t.dtype != torch.float32 for t in (g, beta))
        ):
            raise ValueError("unsupported native WY geometry or dtype")
        _tensors_on_one_device((q, k, v, g, beta), k.device)
        q_kernel = q.transpose(1, 2).contiguous()
        k_kernel = k.transpose(1, 2).contiguous()
        key = k_kernel.float().reshape(batch, key_heads, tokens // CHUNK_SIZE, CHUNK_SIZE, dim)
        gram = (key @ key.transpose(-1, -2)).contiguous()
        cumulative = (
            g.transpose(1, 2)
            .contiguous()
            .reshape(batch, value_heads, tokens // CHUNK_SIZE, CHUNK_SIZE)
            .cumsum(-1)
            .reshape(batch, value_heads, tokens)
            .contiguous()
        )
        shape = (batch, value_heads, tokens, dim)
        w = torch.empty(shape, dtype=torch.float16, device=k.device)
        u = torch.empty_like(w)
        config = self.config((batch, tokens, key_heads, value_heads), k.device)
        self.launch(self.kernel, [k_kernel, v, gram, cumulative, beta, w, u, config], 8)
        return q_kernel, k_kernel, w, u, cumulative


class NativeLocalRouteGather(NativeResource):
    """Fixed-capacity outputs, with only the device-counted local prefix written.

    Inactive rows are uninitialized and must never be consumed. The native INT4
    grouped projection skips them using the same group ends and zeros its own
    inactive outputs. This reduces executed traffic, not allocated capacity.
    """

    def __call__(self, prepared, sorted_tokens, group_ends):
        if len(prepared) != 4:
            raise ValueError("route gather requires four packed operands")
        low, high, scale, total = prepared
        if (
            low.ndim != 2
            or low.shape[0] <= 0
            or high.shape != low.shape
            or low.dtype != torch.int8
            or high.dtype != torch.int8
            or not 128 <= low.shape[1] <= 1280
            or low.shape[1] % 64
            or scale.shape != (low.shape[0], low.shape[1] // 64, 8)
            or total.shape != scale.shape
            or scale.dtype != torch.float32
            or total.dtype != torch.float32
            or sorted_tokens.ndim != 1
            or sorted_tokens.dtype != torch.int32
            or not 0 < sorted_tokens.numel() <= MAX_ROUTES
            or group_ends.ndim != 1
            or not 0 < group_ends.numel() <= 128
            or group_ends.dtype != torch.int64
        ):
            raise ValueError("unsupported local route gather shape/dtype")
        _tensors_on_one_device((*prepared, sorted_tokens, group_ends), low.device)
        result = tuple(t.new_empty((sorted_tokens.numel(), *t.shape[1:])) for t in prepared)
        config = self.config((low.shape[1], scale.shape[1] * 8, group_ends.numel()), low.device)
        self.launch(self.kernel, [*prepared, sorted_tokens, group_ends, *result, config], 8)
        return result


class NativeLocalSwigluPack(NativeResource):
    def __call__(self, gate_up, group_ends):
        if (
            gate_up.ndim != 2
            or gate_up.dtype != torch.float16
            or not 0 < gate_up.shape[0] <= MAX_ROUTES
            or not 512 <= gate_up.shape[1] <= 5120
            or gate_up.shape[1] % 256
            or group_ends.ndim != 1
            or not 0 < group_ends.numel() <= 128
            or group_ends.dtype != torch.int64
        ):
            raise ValueError("unsupported local SwiGLU pack shape/dtype")
        _tensors_on_one_device((gate_up, group_ends), gate_up.device)
        rows, width = gate_up.shape[0], gate_up.shape[1] // 2
        low = torch.empty((rows, width // 2), dtype=torch.int8, device=gate_up.device)
        high = torch.empty_like(low)
        scale = torch.empty((rows, width // 128, 8), dtype=torch.float32, device=gate_up.device)
        total = torch.empty_like(scale)
        config = self.config((width // 128, group_ends.numel()), gate_up.device)
        self.launch(self.kernel, [gate_up, group_ends, low, high, scale, total, config], 8)
        return low, high, scale, total
