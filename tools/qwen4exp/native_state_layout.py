# SPDX-License-Identifier: Apache-2.0
"""FP32 cache state IO for the optional native-layout chunk GDN seam."""

import torch

from tools.qwen4exp.native_prefill import HEAD_DIM, NativeResource, _tensors_on_one_device

MAX_SEQUENCES = 4
MAX_VALUE_HEADS = 48


class NativeStateLayout(NativeResource):
    def __init__(self, gather_kernel, scatter_kernel, launch):
        super().__init__(gather_kernel, launch)
        self.scatter_kernel = scatter_kernel

    def _validate(self, cache, slots, initialized):
        if (
            cache.ndim != 4
            or cache.dtype != torch.float32
            or cache.shape[-2:] != (HEAD_DIM, HEAD_DIM)
            or not 1 <= cache.shape[1] <= MAX_VALUE_HEADS
            or slots.ndim != 1
            or slots.dtype != torch.int32
            or not 1 <= slots.numel() <= MAX_SEQUENCES
            or initialized.shape != slots.shape
            or initialized.dtype != torch.bool
        ):
            raise ValueError("state IO requires FP32 [slots,H,128,128], int32 slots and bool initialization")
        _tensors_on_one_device((cache, slots, initialized), cache.device)
        # The runner supplies in-range, unique destination slots. Reading device
        # IDs to verify them here would defeat this path; qualify that contract.

    def gather(self, cache, slots, initialized):
        self._validate(cache, slots, initialized)
        output = cache.new_empty((slots.numel(), *cache.shape[1:]))
        config = self.config((slots.numel(), cache.shape[1]), cache.device)
        self.launch(self.kernel, [cache, slots, initialized, output, config], 8)
        return output

    def scatter(self, cache, slots, initialized, state):
        self._validate(cache, slots, initialized)
        if state.shape != (slots.numel(), *cache.shape[1:]) or state.dtype != torch.float32:
            raise ValueError("final state must preserve FP32 native layout")
        _tensors_on_one_device((state,), cache.device)
        config = self.config((slots.numel(), cache.shape[1]), cache.device)
        self.launch(self.scatter_kernel, [cache, slots, initialized, state, config], 8)
