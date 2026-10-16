# SPDX-License-Identifier: Apache-2.0
"""Explicit native dispatch/reduction resources; no default serving binding."""

import torch

from tools.qwen4exp.native_prefill import NativeResource, _tensors_on_one_device

MAX_FIXED_ROUTES = 60
MAX_EXPERTS = 128
MAX_TOP_K = 10
COLUMN_TILE = 256
MAX_FINALIZE_ROUTES = 25600
MAX_COLUMNS = 2560
FINALIZE_BLOCKS = 8


class NativeFixedRouteDispatch(NativeResource):
    def __call__(self, ids, *, num_local_experts, expert_offset):
        if (
            ids.ndim != 2
            or ids.dtype != torch.int64
            or not 0 < ids.numel() <= MAX_FIXED_ROUTES
            or not 0 < ids.shape[1] <= MAX_TOP_K
            or not 0 < num_local_experts <= MAX_EXPERTS
            or expert_offset < 0
        ):
            raise ValueError("unsupported fixed route dispatch geometry/dtype")
        _tensors_on_one_device((ids,), ids.device)
        tokens = torch.empty(ids.numel(), dtype=torch.int32, device=ids.device)
        inverse = torch.empty_like(tokens)
        ends = torch.empty(num_local_experts, dtype=torch.int64, device=ids.device)
        config = self.config((ids.shape[0], ids.shape[1], num_local_experts, expert_offset), ids.device)
        self.launch(self.kernel, [ids, tokens, inverse, ends, config], 1)
        return tokens, inverse, ends


class NativeOwnedFinalize(NativeResource):
    """Skip peer rows; the selected binary defines the rounding boundaries."""

    def __call__(self, routed, weights, inverse, ends):
        if (
            routed.ndim != 2
            or routed.dtype != torch.float16
            or weights.ndim != 2
            or weights.dtype != torch.float32
            or not 0 < weights.numel() <= MAX_FINALIZE_ROUTES
            or not 0 < weights.shape[1] <= MAX_TOP_K
            or routed.shape[0] != weights.numel()
            or not 0 < routed.shape[1] <= MAX_COLUMNS
            or routed.shape[1] % COLUMN_TILE
            or inverse.shape != (weights.numel(),)
            or inverse.dtype != torch.int32
            or ends.ndim != 1
            or not 0 < ends.numel() <= MAX_EXPERTS
            or ends.dtype != torch.int64
        ):
            raise ValueError("unsupported owned finalize geometry/dtype")
        _tensors_on_one_device((routed, weights, inverse, ends), routed.device)
        output = torch.empty(weights.shape[0], routed.shape[1], dtype=torch.float32, device=routed.device)
        config = self.config((weights.shape[0], weights.shape[1], routed.shape[1], ends.numel()), routed.device)
        self.launch(self.kernel, [routed, weights, inverse, ends, output, config], FINALIZE_BLOCKS)
        return output
