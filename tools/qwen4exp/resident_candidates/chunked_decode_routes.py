# SPDX-License-Identifier: Apache-2.0
"""Bounded experiment avoiding grouped dispatch for small verification batches."""

import torch


def _routed_builtin(module, inputs, weights, ids):
    import torch_npu

    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device

    routes = (ids - module.expert_offset).to(torch.int32).flatten().contiguous()
    projected = module.projections["gate_up_proj"].native_linear(pack_activation_device(inputs), routes)
    # Keep the grouped path's FP16 activation rounding before quantization.
    prepared = pack_activation_device(torch_npu.npu_swiglu(projected, dim=-1))
    output = module.projections["down_proj"].native_linear(prepared, routes)
    if module.grouped_finalize == "cann_v2":
        # Routed rows already have token/slot order. Preserve both FP16 scales
        # and the finalizer's FP16 output boundary from grouped dispatch.
        order = torch.arange(routes.numel(), dtype=torch.int32, device=routes.device)
        return torch_npu.npu_moe_finalize_routing(
            output, None, None, None, weights.to(output.dtype).contiguous(), order, None, 2
        ).to(module.compute_dtype)
    output = output.to(module.compute_dtype)
    return (output.reshape(inputs.shape[0], module.top_k, inputs.shape[1]) * weights.unsqueeze(-1)).sum(1)


def make_grouped(original, *, chunk_tokens=8, max_tokens=24):
    if type(chunk_tokens) is not int or chunk_tokens not in (6, 8, 12):
        raise ValueError("chunk_tokens must be 6, 8, or 12")
    if type(max_tokens) is not int or not chunk_tokens < max_tokens <= 24:
        raise ValueError("max_tokens must be bounded by 24")

    def grouped(self, inputs, weights, ids):
        tokens = inputs.shape[0]
        if (
            not self.native_int4
            or not self.device_routing
            or self.top_k != 10
            or inputs.shape[1] != 2560
            or inputs.device.type != "npu"
            or self.grouped_activation not in ("cann_swiglu_pack", "cann_builtin_fp16")
            or tokens * self.top_k <= self.max_routed_rows
            or tokens > max_tokens
            or chunk_tokens * self.top_k > self.max_routed_rows
        ):
            return original(self, inputs, weights, ids)
        routed = (
            self._forward_routed
            if self.grouped_activation == "cann_swiglu_pack"
            else lambda inputs, weights, ids: _routed_builtin(self, inputs, weights, ids)
        )
        return torch.cat(
            [
                routed(
                    inputs[start : start + chunk_tokens],
                    weights[start : start + chunk_tokens],
                    ids[start : start + chunk_tokens],
                )
                for start in range(0, tokens, chunk_tokens)
            ],
            dim=0,
        )

    return grouped
