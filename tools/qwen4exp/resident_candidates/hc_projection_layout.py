# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental 16-row HC gate chain, retaining FP16 operands and weights.

The qualified C1/C2 calls have 3/6 rows. This candidate pads the gate chain's
row dimension once and carries those rows through down/SILU/up/sigmoid. Only
valid rows feed the existing FP32 gated mean. Unlike older projection probes,
it does not transpose/cache weights or change the four-row injection GEMM.
Changing the matmul geometry can change CANN rounding: hardware bitwise gates
are mandatory before installation. No serving defaults select this module.
"""

import torch
import torch.nn.functional as F

from vllm_ascend.models.qwen4_exp.model import _linear, _linear_operand_dtype

CUBE_ROWS = 16
DECODE_ROWS = (3, 6)
HC_WIDTH = 10240
HC_LOWRANK = 320


def _supports_device(device):
    return device.type == "npu"


def eligible(module, normalized):
    return (
        not torch.is_grad_enabled()
        and _supports_device(normalized.device)
        and normalized.ndim == 2
        and tuple(normalized.shape) in ((rows, HC_WIDTH) for rows in DECODE_ROWS)
        and normalized.dtype == module.compute_dtype == torch.float32
        and module.params_dtype == torch.float16
        and module.hc_count == 4
        and module.hidden_size == HC_WIDTH // module.hc_count
        and module.input_mix_weight_down.dtype == module.input_mix_weight_up.dtype == torch.float16
        and module.input_mix_weight_down.device == module.input_mix_weight_up.device == normalized.device
        and tuple(module.input_mix_weight_down.shape) == (HC_LOWRANK, HC_WIDTH)
        and tuple(module.input_mix_weight_up.shape) == (HC_WIDTH, HC_LOWRANK)
    )


def padded_gate(normalized, down_weight, up_weight, hc_count):
    """Pad once; no learned weight copies and no persistent scratch buffers."""
    if normalized.ndim != 2:
        raise ValueError("padded gate requires a matrix with 1 through 15 rows")
    rows = normalized.shape[0]
    if rows <= 0 or rows >= CUBE_ROWS:
        raise ValueError("padded gate requires a matrix with 1 through 15 rows")
    operand = F.pad(normalized.to(down_weight.dtype), (0, 0, 0, CUBE_ROWS - rows))
    lowrank = F.silu(F.linear(operand, down_weight) / hc_count)
    return torch.sigmoid(F.linear(lowrank, up_weight))[:rows]


def mix(self, hyper_input):
    normalized = self._normalize(hyper_input)
    projection_input = normalized
    saved_input = normalized
    # Preserve the newer local opt-in operand policy, while remaining usable
    # with the qualified runtime, whose module has no such attribute.
    if getattr(self, "share_projection_operand", False) and self.use_combine and not torch.is_grad_enabled():
        down_dtype = _linear_operand_dtype(normalized.device.type, self.input_mix_weight_down.dtype, self.compute_dtype)
        injection_dtype = _linear_operand_dtype(
            normalized.device.type, self.block_inject_weight.dtype, self.compute_dtype
        )
        if down_dtype == injection_dtype:
            projection_input = normalized.to(down_dtype)
            saved_input = projection_input
    if eligible(self, normalized):
        gate = padded_gate(projection_input, self.input_mix_weight_down, self.input_mix_weight_up, self.hc_count)
    else:
        gate = F.silu(_linear(projection_input, self.input_mix_weight_down, self.compute_dtype) / self.hc_count)
        gate = torch.sigmoid(_linear(gate, self.input_mix_weight_up, self.compute_dtype))
    rows = hyper_input.shape[0]
    mixed = (
        gate.view(rows, self.hc_count, self.hidden_size) * normalized.view(rows, self.hc_count, self.hidden_size)
    ).mean(dim=-2)
    return mixed.to(self.params_dtype), (hyper_input, saved_input)


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix": mix}
