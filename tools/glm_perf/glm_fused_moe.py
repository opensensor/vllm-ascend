# SPDX-License-Identifier: Apache-2.0
"""Two native MoE stages with UB SwiGLU/quantization and weighted down epilogue.

Routing metadata remains device-resident. No FP16 gate/up, hidden activation,
or routed-down tensor is materialized between the native stages.
"""

from dataclasses import dataclass

import torch

from .glm_int4 import MAX_GROUPED_ROUTES, packed_weight_bits, pipeline_gather_offsets


@dataclass(frozen=True)
class FusedGeometry:
    tokens: int
    top_k: int
    experts: int
    hidden: int
    intermediate: int
    gate_bits: int
    down_bits: int
    activation_bits: int

    def __post_init__(self):
        if any(type(value) is not int for value in self.__dict__.values()):
            raise ValueError("fused geometry requires integers")
        if not 1 <= self.tokens * self.top_k <= MAX_GROUPED_ROUTES or self.tokens <= 0 or self.top_k <= 0:
            raise ValueError("invalid fused route count")
        if not 1 <= self.experts <= 288 or self.hidden not in (256, 512, 2048, 4096):
            raise ValueError("unsupported fused bank geometry")
        if not 256 <= self.intermediate <= 4096 or self.intermediate % 256:
            raise ValueError("fused intermediate requires complete NZ tiles")
        if self.gate_bits not in (2, 3, 4) or self.down_bits not in (2, 3, 4) or self.activation_bits not in (4, 8):
            raise ValueError("unsupported fused precision")


class NativeFusedMoE:
    input_dtype = torch.float16

    def __init__(self, root, *, namespace, activation_bits, kernel_factory=None, launch=None):
        if activation_bits not in (4, 8):
            raise ValueError("activation bits must be 4 or 8")
        self.activation_bits = activation_bits
        self.launch = launch or getattr(torch.ops, namespace).launch
        factory = kernel_factory or getattr(torch.classes, namespace).Kernel
        self.gate_kernel = factory(str(root / "glm_fused_gate_up.bin"), "glm_fused_gate_up_v1")
        self.down_kernel = factory(str(root / "glm_fused_down.bin"), "glm_fused_down_v1")
        self.pack_kernel = factory(str(root / "glm_fused_pack.bin"), "glm_fused_pack_v1")
        self.device = torch.device("npu", torch.npu.current_device())
        self.offsets = {
            bits: pipeline_gather_offsets(128, bits).to(torch.int32).view(torch.int64) for bits in (2, 3, 4)
        }
        self.configs = {}

    def geometry(self, x, gate_codes, gate_scales, down_codes, down_scales, weights, ids):
        tensors = (x, gate_codes, gate_scales, down_codes, down_scales, weights, ids)
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("fused tensors must be contiguous on the prepared NPU")
        if (
            x.ndim != 2
            or gate_codes.ndim != 3
            or down_codes.ndim != 3
            or weights.ndim != 2
            or ids.shape != weights.shape
        ):
            raise ValueError("invalid fused tensor ranks")
        if x.dtype != torch.float16 or weights.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("fused input, routes require FP16/FP32/integer")
        if gate_codes.dtype not in (torch.int8, torch.uint8) or down_codes.dtype not in (torch.int8, torch.uint8):
            raise ValueError("fused banks require byte storage")
        if gate_scales.dtype != torch.float32 or down_scales.dtype != torch.float32:
            raise ValueError("fused bank scales require FP32")
        tokens, hidden = x.shape
        experts, twice_inter, packed_hidden = gate_codes.shape
        inter = twice_inter // 2
        geometry = FusedGeometry(
            tokens,
            weights.shape[1],
            experts,
            hidden,
            inter,
            packed_weight_bits(hidden, packed_hidden),
            packed_weight_bits(inter, down_codes.shape[-1]),
            self.activation_bits,
        )
        if weights.shape[0] != tokens or twice_inter != 2 * inter or down_codes.shape[:2] != (experts, hidden):
            raise ValueError("fused banks and routes disagree")
        if gate_scales.shape != (experts, twice_inter // 32, hidden // 32) or down_scales.shape != (
            experts,
            hidden // 32,
            inter // 32,
        ):
            raise ValueError("fused scale geometry differs")
        return geometry

    def __call__(self, x, gate_codes, gate_scales, down_codes, down_scales, weights, ids, expert_offset=0):
        # Worker-only import preserves CPU planning and reference tests.
        from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

        geometry = self.geometry(x, gate_codes, gate_scales, down_codes, down_scales, weights, ids)
        sentinel = expert_offset + geometry.experts
        selected = torch.where(weights != 0, ids, torch.full_like(ids, sentinel))
        dispatch = build_grouped_expert_dispatch(
            weights,
            selected,
            num_local_experts=geometry.experts,
            expert_offset=expert_offset,
            weight_dtype=torch.float32,
        )
        return self.grouped(
            x,
            gate_codes,
            gate_scales,
            down_codes,
            down_scales,
            weights,
            dispatch.order.contiguous(),
            dispatch.group_list.to(torch.int64).contiguous(),
            geometry,
        )

    def grouped(self, x, gate_codes, gate_scales, down_codes, down_scales, weights, order, ends, geometry):
        # Metadata depends only on CPU-visible shapes and precision. Preparing
        # a first prefill geometry never reads routes or expert counts on host.
        if geometry not in self.configs:
            common = (geometry.tokens * geometry.top_k, geometry.experts)
            suffix = (geometry.activation_bits, geometry.tokens, geometry.top_k)
            gate = (*common, 2 * geometry.intermediate, geometry.hidden, geometry.gate_bits, *suffix)
            down = (*common, geometry.hidden, geometry.intermediate, geometry.down_bits, *suffix)
            projection_configs = tuple(
                torch.cat((torch.tensor(header, dtype=torch.int64), self.offsets[bits])).to(self.device)
                for header, bits in ((gate, geometry.gate_bits), (down, geometry.down_bits))
            )
            pack_config = torch.tensor(
                (geometry.tokens * geometry.hidden // 32, geometry.activation_bits), dtype=torch.int64
            ).to(self.device)
            self.configs[geometry] = (*projection_configs, pack_config)
        gate_config, down_config, pack_config = self.configs[geometry]
        input_shape = (geometry.tokens, geometry.hidden // 32, 32)
        input_low = torch.empty(input_shape, dtype=torch.int8, device=self.device)
        input_high = torch.empty(
            input_shape if self.activation_bits == 8 else (1,), dtype=torch.int8, device=self.device
        )
        input_scales = torch.empty((geometry.tokens, geometry.hidden // 32, 8), dtype=torch.float32, device=self.device)
        routes = geometry.tokens * geometry.top_k
        shape = (routes, geometry.intermediate // 32, 32)
        low = torch.empty(shape, dtype=torch.int8, device=self.device)
        high = torch.empty(shape if self.activation_bits == 8 else (1,), dtype=torch.int8, device=self.device)
        scales = torch.empty((routes, geometry.intermediate // 32, 8), dtype=torch.float32, device=self.device)
        output = torch.empty((geometry.tokens, geometry.hidden), dtype=torch.float32, device=self.device)
        self.launch(self.pack_kernel, [x, input_low, input_high, input_scales, pack_config], 8)
        self.launch(
            self.gate_kernel,
            [input_low, input_high, input_scales, order, gate_codes, gate_scales, ends, low, high, scales, gate_config],
            8,
        )
        self.launch(
            self.down_kernel, [low, high, scales, down_codes, down_scales, ends, order, weights, output, down_config], 8
        )
        return output
