# SPDX-License-Identifier: Apache-2.0
"""Opt-in HC operand reuse. No global cache, padding, or weight conversion."""

import torch
import torch.nn.functional as F

DECODE_ROWS = (3, 6, 12, 18)
WIDTH = 2560
HC_COUNT = 4
LOWRANK = 320


def _on_npu(device):
    return device.type == "npu"


def pack_nz(value):
    # Called only in an explicitly bound NPU experiment, never at import.
    import torch_npu

    return torch_npu.npu_format_cast(value.to(torch.float16), 29)


def eligible(module, value):
    weights = (module.input_mix_weight_down, module.input_mix_weight_up, module.block_inject_weight)
    return (
        not torch.is_grad_enabled()
        and _on_npu(value.device)
        and value.dtype == module.params_dtype == torch.float16
        and module.compute_dtype == torch.float32
        and module.use_combine
        and module.hc_count == HC_COUNT
        and module.hidden_size == WIDTH
        and value.shape in tuple((rows, WIDTH * HC_COUNT) for rows in DECODE_ROWS)
        and all(w.device == value.device and w.dtype == torch.float16 for w in weights)
        and tuple(w.shape for w in weights)
        == ((LOWRANK, WIDTH * HC_COUNT), (WIDTH * HC_COUNT, LOWRANK), (HC_COUNT, WIDTH * HC_COUNT))
    )


def make_mix(fallback, *, pack=pack_nz):
    """Carry one FP16/NZ tensor from down projection to unchanged combine.

    Borrow the module's normalization and weights. Keep the FP32 normalized
    tensor for the gated mean. The residual tuple owns the converted operand
    only for this forward; never reuse it after input or batch changes.
    """

    def mix(self, hyper_input):
        if not eligible(self, hyper_input):
            return fallback(self, hyper_input)
        normalized = self._normalize(hyper_input)
        operand = pack(normalized)
        gate = F.silu(F.linear(operand, self.input_mix_weight_down) / self.hc_count)
        gate = torch.sigmoid(F.linear(gate, self.input_mix_weight_up))
        mixed = (gate.view(-1, HC_COUNT, WIDTH) * normalized.view(-1, HC_COUNT, WIDTH)).mean(-2)
        return mixed.to(self.params_dtype), (hyper_input, operand)

    return mix
