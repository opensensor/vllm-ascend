# SPDX-License-Identifier: Apache-2.0
"""Opt-in resident W8A8 for shared experts and attention output projections."""

import torch

INT8_MAX = 127
NZ_FORMAT = 29
MAX_PROJECTION_NRMSE = 0.03


def selected_projection(name):
    """Keep routing, QKV/indexer, vision and vocabulary projections untouched."""
    return (
        ".layers." in name
        and (
            name.endswith(".self_attn.o_proj")
            or name.endswith(".shared_experts.gate_up_proj")
            or name.endswith(".shared_experts.down_proj")
        )
        and "vision" not in name
    )


def quantize_weight(weight):
    value = weight.float()
    scale = value.abs().amax(1).clamp_min(1e-12) / INT8_MAX
    packed = (value / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return packed, scale


class DenseInt8Method:
    """A prepared layer owns its scale; apply never copies tensors to the host."""

    def __init__(self, scale):
        self.scale = scale

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("M3 dense INT8 requires bias-free projections")
        if x.device.type == "cpu":
            scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / INT8_MAX
            q = (x.float() / scale).round().clamp(-INT8_MAX, INT8_MAX)
            return ((q @ layer.weight.float().T) * scale * self.scale).to(x.dtype)
        # Worker-only import: offline conversion and ownership tests use CPU.
        import torch_npu

        q, scale = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(q, layer.weight.T, self.scale, pertoken_scale=scale, output_dtype=x.dtype)


def device_weight(weight, device, storage_format):
    result = weight.to(device)
    if device.type != "cpu":
        import torch_npu

        result = torch_npu.npu_format_cast(result, storage_format)
    return result


@torch.inference_mode()
def restore(state):
    """Called only after draining requests and releasing all captured graphs."""
    for name in list(state):
        record = state[name]
        layer = record["layer"]
        layer.weight.data = device_weight(record["weight"], layer.weight.device, record["format"])
        layer.quant_method = record["method"]
        del state[name]


@torch.inference_mode()
def convert(model, state):
    """Replace selected weights with CPU-backed recovery and bounded scratch."""
    if state:
        raise ValueError("dense INT8 conversion is already active")
    reports = []
    try:
        for name, layer in model.named_modules():
            if not selected_projection(name):
                continue
            weight = layer.weight
            if weight.dtype != torch.float16 or weight.ndim != 2 or getattr(layer, "bias", None) is not None:
                raise ValueError(f"unsupported dense projection: {name}")
            storage_format = NZ_FORMAT
            if weight.device.type != "cpu":
                import torch_npu

                storage_format = torch_npu.get_npu_format(weight)
            # Keep exact FP16 recovery in host RAM, not a second device copy.
            original = weight.detach().cpu().contiguous().clone()
            packed, scale = quantize_weight(original)
            x = torch.randn(4, weight.shape[1], generator=torch.Generator().manual_seed(71)).half().to(weight.device)
            expected = layer.quant_method.apply(layer, x).float()
            record = {"layer": layer, "weight": original, "format": storage_format, "method": layer.quant_method}
            state[name] = record
            layer.weight.requires_grad_(False)
            layer.weight.data = device_weight(packed, weight.device, NZ_FORMAT)
            layer.quant_method = DenseInt8Method(scale.to(weight.device))
            actual = layer.quant_method.apply(layer, x).float()
            error = float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
            if not torch.isfinite(actual).all() or error > MAX_PROJECTION_NRMSE:
                raise RuntimeError(f"dense INT8 projection {name} failed: NRMSE {error}")
            reports.append(
                {
                    "name": name,
                    "shape": list(weight.shape),
                    "nrmse": error,
                    "saved_bytes": original.numel() - scale.numel() * scale.element_size(),
                }
            )
        if not reports:
            raise ValueError("no supported M3 dense projections found")
    except Exception:
        restore(state)
        raise
    return reports
