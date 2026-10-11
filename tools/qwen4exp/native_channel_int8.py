# SPDX-License-Identifier: Apache-2.0
"""Isolated whole-K INT8 resource; requires a separately qualified weight recipe."""

import torch

from tools.qwen4exp.native_prefill import NativeResource, _tensors_on_one_device

OUTPUT_TILE = 160
GROUP_SIZE = 128
MAX_K = 2560
MAX_ROWS = 25600


def tile_channel_weights(weight):
    """Pack raw [E,N,K] INT8 codes into the Cube's N16/K32 strips."""
    if weight.ndim != 3 or weight.dtype != torch.int8:
        raise ValueError("raw INT8 expert bank required")
    experts, outputs, inputs = weight.shape
    if (
        not 0 < experts <= 128
        or not outputs > 0
        or outputs % OUTPUT_TILE
        or not 0 < inputs <= MAX_K
        or inputs % GROUP_SIZE
    ):
        raise ValueError("unsupported whole-K weight geometry")
    return (
        weight.reshape(experts, outputs // 16, 16, inputs // GROUP_SIZE, GROUP_SIZE // 32, 32)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .reshape_as(weight)
    )


class NativeChannelInt8(NativeResource):
    def __init__(self, kernel, launch, column_tile=OUTPUT_TILE):
        super().__init__(kernel, launch)
        if column_tile not in (64, OUTPUT_TILE):
            raise ValueError("unsupported whole-K column tile")
        self.column_tile = column_tile

    def __call__(self, x, weights, scales, input_scales, ends):
        if weights.ndim != 3 or x.ndim != 2:
            raise ValueError("whole-K projection requires matrices and an expert bank")
        experts, outputs, inputs = weights.shape
        rows = x.shape[0]
        if (
            not 0 < rows <= MAX_ROWS
            or not 0 < experts <= 128
            or not outputs > 0
            or outputs % OUTPUT_TILE
            or outputs % self.column_tile
            or not 0 < inputs <= MAX_K
            or inputs % GROUP_SIZE
            or x.shape != (rows, inputs)
            or weights.dtype != torch.int8
            or x.dtype != torch.int8
            or scales.shape != (experts, outputs)
            or scales.dtype != torch.float32
            or input_scales.shape != (rows, 8)
            or input_scales.dtype != torch.float32
            or ends.shape != (experts,)
            or ends.dtype != torch.int64
        ):
            raise ValueError("unsupported whole-K projection geometry/dtype")
        _tensors_on_one_device((x, weights, scales, input_scales, ends), x.device)
        # The kernel owns peer padding and active writes on the same cores.
        # No separate memset or host group-count synchronization is submitted.
        output = torch.empty((rows, outputs), dtype=torch.float16, device=x.device)
        config = self.config((rows, inputs, outputs, experts, 1), x.device)
        self.launch(self.kernel, [x, weights, scales, input_scales, ends, output, config], 8)
        return output


class NativeChannelActivation(NativeResource):
    def __init__(self, kernel, launch, lookup):
        super().__init__(kernel, launch)
        self.lookup = lookup

    def __call__(self, prepared):
        if len(prepared) != 4:
            raise ValueError("four native activation operands required")
        low, high, scales, _ = prepared
        rows = low.shape[0] if low.ndim == 2 else 0
        inputs = low.shape[1] * 2 if low.ndim == 2 else 0
        if (
            not 0 < rows <= MAX_ROWS
            or not 0 < inputs <= MAX_K
            or inputs % GROUP_SIZE
            or low.dtype != torch.int8
            or high.shape != low.shape
            or high.dtype != torch.int8
            or scales.shape != (rows, inputs // GROUP_SIZE, 8)
            or scales.dtype != torch.float32
            or self.lookup.shape != (256, 2)
            or self.lookup.dtype != torch.float16
        ):
            raise ValueError("unsupported packed-to-channel activation geometry")
        _tensors_on_one_device((low, high, scales, self.lookup), low.device)
        output = torch.empty((rows, inputs), dtype=torch.int8, device=low.device)
        row_scales = torch.empty((rows, 8), dtype=torch.float32, device=low.device)
        config = self.config((rows, inputs), low.device)
        self.launch(self.kernel, [low, high, scales, self.lookup, output, row_scales, config], 8)
        return output, row_scales


class NativeDirectChannelActivation(NativeResource):
    """Quantize tokens once; optional group ends skip inactive routed rows."""

    def __init__(self, kernel, launch, local_kernel=None):
        super().__init__(kernel, launch)
        self.local_kernel = local_kernel

    def __call__(self, inputs, ends=None):
        if (
            inputs.ndim != 2
            or inputs.dtype != torch.float16
            or not 0 < inputs.shape[0] <= MAX_ROWS
            or not 0 < inputs.shape[1] <= MAX_K
            or inputs.shape[1] % GROUP_SIZE
        ):
            raise ValueError("unsupported direct channel activation geometry/dtype")
        _tensors_on_one_device((inputs,), inputs.device)
        rows, width = inputs.shape
        output = torch.empty((rows, width), dtype=torch.int8, device=inputs.device)
        scales = torch.empty((rows, 8), dtype=torch.float32, device=inputs.device)
        if ends is None:
            config = self.config((rows, width), inputs.device)
            self.launch(self.kernel, [inputs, output, scales, config], 8)
        else:
            if self.local_kernel is None or ends.ndim != 1 or not 0 < ends.numel() <= 128 or ends.dtype != torch.int64:
                raise ValueError("local activation requires a kernel and device expert boundaries")
            _tensors_on_one_device((ends,), inputs.device)
            config = self.config((rows, width, ends.numel()), inputs.device)
            # Only the device-counted local prefix is initialized. The grouped
            # projection owns output padding and never reads inactive operands.
            self.launch(self.local_kernel, [inputs, ends, output, scales, config], 8)
        return output, scales


class NativePackedChannelInt8(NativeResource):
    """Expand resident W4 to the qualified channel recipe in L1, per tile."""

    def __init__(self, kernel, launch, lookup):
        super().__init__(kernel, launch)
        self.lookup = lookup

    def __call__(self, x, bank, channel_scales, input_scales, ends):
        if bank.weight.ndim != 3 or x.ndim != 2:
            raise ValueError("packed channel projection requires an expert bank and rows")
        experts, outputs, half_inputs = bank.weight.shape
        inputs = half_inputs * 2
        rows = x.shape[0]
        if (
            not 0 < rows <= MAX_ROWS
            or not 0 < experts <= 128
            or not outputs > 0
            or outputs % OUTPUT_TILE
            or not 0 < inputs <= MAX_K
            or inputs % GROUP_SIZE
            or x.shape != (rows, inputs)
            or x.dtype != torch.int8
            or bank.weight.dtype != torch.int8
            or bank.weight_scale.shape != (experts, outputs, inputs // GROUP_SIZE)
            or bank.weight_scale.dtype != torch.float16
            or bank.weight_offset.shape != bank.weight_scale.shape
            or bank.weight_offset.dtype != torch.float16
            or channel_scales.shape != (experts, outputs)
            or channel_scales.dtype != torch.float32
            or input_scales.shape != (rows, 8)
            or input_scales.dtype != torch.float32
            or ends.shape != (experts,)
            or ends.dtype != torch.int64
            or self.lookup.shape != (256, 2)
            or self.lookup.dtype != torch.float16
        ):
            raise ValueError("unsupported packed channel projection geometry/dtype")
        operands = (
            x,
            bank.weight,
            bank.weight_scale,
            bank.weight_offset,
            channel_scales,
            input_scales,
            ends,
            self.lookup,
        )
        _tensors_on_one_device(operands, x.device)
        output = torch.empty((rows, outputs), dtype=torch.float16, device=x.device)
        config = self.config((rows, inputs, outputs, experts, 1), x.device)
        self.launch(self.kernel, [*operands, output, config], 8)
        return output
