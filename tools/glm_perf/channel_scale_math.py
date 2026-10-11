# SPDX-License-Identifier: Apache-2.0
"""New W4 channel scales: integer accumulation across K, then one epilogue.

This is a new quantization contract, not an exact rewrite of block32 scales.
CPU references establish arithmetic/layout correctness, not GLM model quality.
Optional diagonal smoothing preserves the unquantized product exactly before
rounding. Original W2/W3/W4 checkpoints must not be admitted as this format.
"""

from dataclasses import dataclass

import torch

from .fused_weight_layout import pack_cube, unpack_cube

LAYOUT = "glm_channel_w4_int32_accum_v1"
SUPPORTED_GROUPS = (256, 512, 1024, 2048, 4096)
CUBE_K = 256
OUTPUT_TILE = 128
FP16_MAX = 65504.0


@dataclass(frozen=True)
class ChannelWeightBank:
    codes: torch.Tensor
    scales: torch.Tensor
    sums: torch.Tensor
    smoothing: torch.Tensor
    group_size: int
    layout: str = LAYOUT

    @property
    def shape(self):
        experts, channels, packed_k = self.codes.shape
        return experts, channels, 2 * packed_k

    def validate(self):
        if self.layout != LAYOUT or self.codes.dtype != torch.int8 or self.codes.ndim != 3:
            raise ValueError("requires new channel-scale native W4 storage")
        e, n, k = self.shape
        if (
            not 1 <= e <= 288
            or not 0 < n <= 8192
            or n % OUTPUT_TILE
            or not 256 <= k <= 4096
            or self.group_size not in SUPPORTED_GROUPS
            or k % self.group_size
        ):
            raise ValueError("channel-scale bank has unsupported geometry")
        if self.scales.shape != (e, n, k // self.group_size) or self.scales.dtype != torch.float32:
            raise ValueError("channel-scale weight factors must be FP32 [E,N,groups]")
        if self.sums.shape != self.scales.shape or self.sums.dtype != torch.int32:
            raise ValueError("INT8 limb correction requires permanent integer weight sums")
        if self.smoothing.shape != (e, k) or self.smoothing.dtype != torch.float32:
            raise ValueError("diagonal smoothing must be FP32 [E,K]")
        if any(
            t.device.type != "cpu" or not t.is_contiguous()
            for t in (self.codes, self.scales, self.sums, self.smoothing)
        ):
            raise ValueError("CPU reference bank requires contiguous CPU tensors")
        if not torch.isfinite(self.scales).all() or not (self.scales > 0).all():
            raise ValueError("weight scales must be positive and finite")
        if not torch.isfinite(self.smoothing).all() or not (self.smoothing > 0).all():
            raise ValueError("smoothing must be positive and finite")
        return self


def smooth_diagonal(weights, activation_max, alpha=0.5):
    """Shared gate/up transform from real calibration maxima, not fake statistics."""
    if weights.device.type != "cpu" or weights.ndim != 3 or activation_max.shape != (weights.shape[-1],):
        raise ValueError("smoothing requires CPU [E,N,K] and activation maxima [K]")
    if not 0 <= alpha <= 1 or not torch.isfinite(weights).all() or not torch.isfinite(activation_max).all():
        raise ValueError("smoothing inputs must be finite and alpha in [0,1]")
    if (activation_max < 0).any():
        raise ValueError("activation maxima must be nonnegative")
    maximum = weights.float().abs().amax((0, 1))
    usable = (maximum > 0) & (activation_max > 0)
    safe_weight = torch.where(usable, maximum, 1.0)
    safe_activation = torch.where(usable, activation_max.float(), 1.0)
    value = safe_activation.pow(alpha) / safe_weight.pow(1 - alpha)
    return torch.where(usable, value, 1.0).contiguous()


def quantize_weights(weights, group_size=None, smoothing=None, clipping=1.0):
    """Offline per-channel W4, with optional wider K groups and calibrated smoothing."""
    if weights.device.type != "cpu" or weights.ndim != 3 or not torch.isfinite(weights).all():
        raise ValueError("weight preparation requires finite CPU [E,N,K]")
    e, n, k = weights.shape
    group_size = k if group_size is None else group_size
    if type(group_size) is not int or group_size not in SUPPORTED_GROUPS or k % group_size or n % OUTPUT_TILE:
        raise ValueError("weight K groups require complete K256 Cube tiles")
    if not 0 < clipping <= 1:
        raise ValueError("weight clipping ratio must be in (0,1]")
    if smoothing is None:
        smoothing = torch.ones(e, k, dtype=torch.float32)
    elif smoothing.shape == (k,):
        smoothing = smoothing.float().expand(e, k).clone()
    else:
        smoothing = smoothing.float().contiguous()
    if smoothing.shape != (e, k) or not (smoothing > 0).all() or not torch.isfinite(smoothing).all():
        raise ValueError("smoothing must be finite positive [K] or [E,K]")
    grouped = (weights.float() * smoothing[:, None]).reshape(e, n, k // group_size, group_size)
    positive = grouped.amax(-1).clamp_min(0) / 7
    negative = (-grouped.amin(-1)).clamp_min(0) / 8
    scale = torch.maximum(positive, negative) * clipping
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).contiguous()
    signed = (grouped / scale[..., None]).round().clamp(-8, 7).to(torch.int8).reshape(e, n, k)
    sums = signed.reshape(e, n, k // group_size, group_size).int().sum(-1, dtype=torch.int32).contiguous()
    return ChannelWeightBank(pack_cube(signed, 4), scale, sums, smoothing.contiguous(), group_size).validate()


def dequantize_weights(bank):
    bank.validate()
    e, n, k = bank.shape
    signed = unpack_cube(bank.codes, k).float().reshape(e, n, k // bank.group_size, bank.group_size)
    return (signed * bank.scales[..., None]).reshape(e, n, k) / bank.smoothing[:, None]


def quantize_activations(inputs, group_size, bits, *, reciprocal=False):
    if inputs.device.type != "cpu" or inputs.ndim != 2 or bits not in (4, 8):
        raise ValueError("activation quantization requires CPU [rows,K], A4 or A8")
    if group_size not in SUPPORTED_GROUPS or inputs.shape[-1] % group_size or not torch.isfinite(inputs).all():
        raise ValueError("activation quantization requires finite complete groups")
    rows, k = inputs.shape
    limit = (1 << (bits - 1)) - 1
    grouped = inputs.float().reshape(rows, k // group_size, group_size)
    maximum = grouped.abs().amax(-1)
    scales = torch.where(maximum > 0, maximum / limit, 1.0)
    normalized = grouped * scales.reciprocal()[..., None] if reciprocal else grouped / scales[..., None]
    codes = normalized.round().clamp(-limit, limit).to(torch.int32)
    return codes, scales


def int8_limbs(codes):
    """Exact signed INT4 limbs; static weight sums replace every bias Cube row."""
    if codes.dtype != torch.int32 or (codes < -127).any() or (codes > 127).any():
        raise ValueError("requires symmetric INT8 activation codes")
    high = torch.div(codes, 16, rounding_mode="floor")
    low = codes - 16 * high - 8
    return low, high


def project(inputs, bank, expert, bits=8, *, reciprocal=False):
    """Mirror native INT32 accumulation and a single scale epilogue per K group."""
    bank.validate()
    e, n, k = bank.shape
    if type(expert) is not int or not 0 <= expert < e or inputs.shape[-1] != k:
        raise ValueError("projection input/expert differs from bank")
    transformed = inputs.float() / bank.smoothing[expert]
    activation, scales = quantize_activations(transformed, bank.group_size, bits, reciprocal=reciprocal)
    signed = unpack_cube(bank.codes[expert : expert + 1], k)[0].int()
    result = torch.zeros(inputs.shape[0], n, dtype=torch.float32)
    for group in range(k // bank.group_size):
        start = group * bank.group_size
        weights = signed[:, start : start + bank.group_size]
        values = activation[:, group]
        limbs = int8_limbs(values) if bits == 8 else (values,)
        dots = [torch.zeros(inputs.shape[0], n, dtype=torch.int32) for _ in limbs]
        # No product conversion/readback/scaling inside the K256 loop.
        for first in range(0, bank.group_size, CUBE_K):
            for limb, dot in zip(limbs, dots):
                dot.add_(limb[:, first : first + CUBE_K] @ weights[:, first : first + CUBE_K].T)
        integer = dots[0] if bits == 4 else dots[0] + 16 * dots[1] + 8 * bank.sums[expert, :, group]
        contribution = integer.float() * bank.scales[expert, :, group]
        contribution *= scales[:, group, None]
        result += contribution
    return result


def fused_moe_reference(
    inputs, gate_up, down, route_weights, route_ids, *, expert_offset=0, bits=8, shared=None, reciprocal=False
):
    """Complete route -> gate/up -> SwiGLU -> hidden quant -> down -> stable combine."""
    gate_up.validate()
    down.validate()
    experts, twice_intermediate, hidden = gate_up.shape
    intermediate = twice_intermediate // 2
    if down.shape != (experts, hidden, intermediate) or inputs.shape[-1] != hidden:
        raise ValueError("MoE projection banks disagree")
    if route_weights.shape != route_ids.shape or route_ids.shape[0] != inputs.shape[0] or route_ids.ndim != 2:
        raise ValueError("MoE route geometry differs")
    routes = {}
    for expert in range(experts):
        pairs = ((route_ids == expert + expert_offset) & (route_weights != 0)).nonzero()
        if not pairs.numel():
            continue
        projected = project(inputs[pairs[:, 0]], gate_up, expert, bits, reciprocal=reciprocal).half().float()
        gate, up = projected[:, :intermediate], projected[:, intermediate:]
        hidden_values = (torch.nn.functional.silu(gate) * up).clamp(-FP16_MAX, FP16_MAX).half()
        outputs = project(hidden_values, down, expert, bits, reciprocal=reciprocal).half().float()
        for pair, value in zip(pairs.tolist(), outputs):
            routes[tuple(pair)] = value
    output = torch.zeros_like(inputs, dtype=torch.float32)
    for token in range(inputs.shape[0]):
        for slot in range(route_ids.shape[1]):
            if (token, slot) in routes:
                output[token] += routes[token, slot] * route_weights[token, slot]
    if shared is not None:
        output += shared(inputs).float()
    return output


def projection_cost(k, count, bits, group_size, baseline_group_size=32):
    """Source-derived work counts, not measured Cube occupancy or runtime."""
    if k % CUBE_K or not 1 <= count <= 32 or bits not in (4, 8) or k % group_size:
        raise ValueError("invalid projection work-count geometry")
    physical_rows = ((count * (2 if bits == 8 else 1) + 15) // 16) * 16
    groups = k // group_size
    return {
        "cube_updates": k // CUBE_K,
        "product_readbacks": groups,
        "useful_product_casts": count * OUTPUT_TILE * (2 if bits == 8 else 1) * groups,
        "scale_epilogue_elements": count * OUTPUT_TILE * groups,
        "physical_cube_rows": physical_rows,
        "integer_product_bound": group_size * ((1 << (bits - 1)) - 1) * 8,
        "baseline_block32_scale_epilogue_elements": count * OUTPUT_TILE * (k // baseline_group_size),
        "measurement": "offline static counts only",
    }
