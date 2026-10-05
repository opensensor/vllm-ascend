# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in packed W4A16 storage with reference and 310P Cube projections.

No persistent FP16 bank or W8 shadow is created. ``cube_310_grouped`` keeps
prefill and decode routing on device; ``cube_310_int4_a8`` is an experimental
activation-quantizing backend, not a performance-qualified W4A16 replacement.
The reference and group-only backends require eager execution.
"""

from __future__ import annotations

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy
from .grouped_expert_dispatch import GroupedExpertDispatch, build_grouped_expert_dispatch
from .moe import route_topk
from .w4a8_int4 import (
    NATIVE_INT4_BACKEND,
    pack_activation_device,
    pack_native_metadata,
    pack_native_weight,
    swiglu_pack_activation_device,
)
from .weight_mapping import local_expert_range

FORMAT = "qwen4exp_w4a16_group_v1"
EXPERT_NAME = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_offset)$"
)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
KINDS = ("weight", "weight_scale", "weight_offset")
MAX_EAGER_TOKENS = 256
MAX_CUBE_TOKENS = 128
CUBE_GROUP_SIZE = 128
CUBE_MAX_INPUTS = 2560
CUBE_TILE_OUTPUTS = 32
CUBE_ZERO_POINT_BIAS = 8
CUBE_FRACTAL_SIZE = 16
CUBE_DEVICE_ROUTED_BACKENDS = ("cube_310_routed", "cube_310_grouped", NATIVE_INT4_BACKEND)
CUBE_BACKENDS = ("cube_310", "cube_310_tiled", *CUBE_DEVICE_ROUTED_BACKENDS)
CUBE_TILED_BACKENDS = ("cube_310_tiled", "cube_310_routed", "cube_310_grouped")
# Four requests with MTP k=2 reach 120 routes. The native INT4 operator
# accepts 128 rows, while the W4A16 routed operator accepts only 80. The
# grouped W4A16 backend handles larger shapes on device instead.
MAX_CUBE_ROUTES = 128
MAX_W4A16_ROUTES = 80
# The fused down-projection epilogue stages one FP16 tile per route in UB.
# Model c1 decode has at most three tokens times ten routes.
MAX_FUSED_DOWN_ROUTES = 30
FUSED_DOWN_INPUTS = 640
FUSED_DOWN_OUTPUTS = 2560
# Bound route expansion and the device count matrix independently of the
# configured context window. Keep the established W4A16 workspace unchanged.
# Native INT4 streams the route dimension through bounded on-chip tiles; its
# pack and grouped-matmul operators expose a 20,480-route prefill contract.
MAX_GROUPED_W4A16_TOKENS = 512
MAX_GROUPED_W4A16_ROUTES = 5120
# The 1,536-token split avoids the native large-tile schedule cliff while
# retaining the 20,480-route operator capacity for other callers.
MAX_GROUPED_NATIVE_TOKENS = 1536
MAX_GROUPED_NATIVE_ROUTES = 20480
# Concurrent shared/routed GEMMs help only while decode leaves Cube headroom.
# Multi-request MTP can split work into two-row MoE calls, where the concurrent
# GEMMs contend and regress aggregate throughput. Restrict overlap to one row.
MAX_SHARED_EXPERT_OVERLAP_TOKENS = 1
SHARED_EXPERT_EXECUTIONS = (
    "tp_sharded",
    "tp_sharded_overlap",
    "replicated",
    "replicated_overlap",
    "replicated_deferred",
)
LM_HEAD_EXECUTIONS = ("float16", "w8a8_dynamic")
PLE_PROJECTION_EXECUTIONS = ("float16", "w8a8_dynamic")
MTP_EXPERT_EXECUTIONS = ("w8a16_routed", "w8a8_grouped")
GROUPED_FINALIZE_METHODS = ("torch", "cann_v2")
GROUPED_ACTIVATION_METHODS = ("torch", "cann_swiglu_pack", "cann_builtin_fp16")
DEFAULT_NATIVE_GROUPED_ACTIVATION = "cann_builtin_fp16"


class DeferredReduceStream:
    """Lazily own one communication stream shared by all model layers."""

    def __init__(self) -> None:
        self._stream = None

    def get(self) -> torch.npu.Stream:
        if self._stream is None:
            # HCCL graph capture requires the communication stream to use the
            # default priority on 310P. A high-priority stream aborts the HCCL
            # watchdog while the first decode graph is captured.
            self._stream = torch.npu.Stream()
        return self._stream


def w4_config(config: object) -> dict | None:
    metadata = getattr(config, "ascend_expert_quantization", None)
    if metadata is None:
        return None
    expected = {
        "format": FORMAT,
        "bits": 4,
        "symmetric": False,
        "packing": "signed_int4_low_nibble_first_in_axis",
        "scale_dtype": "float16",
        "offset_dtype": "int8",
    }
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported Qwen4Exp expert quantization metadata; refusing W8 fallback")
    backend = metadata.get("backend")
    if backend not in ("eager_dequant", *CUBE_BACKENDS):
        raise ValueError(f"W4 backend must be eager_dequant or one of {CUBE_BACKENDS}")
    activation = metadata.get("activation_quantization", "float16")
    if activation not in ("float16", "int8_per_group"):
        raise ValueError("unsupported W4 activation_quantization policy")
    if backend == NATIVE_INT4_BACKEND and activation != "int8_per_group":
        raise ValueError("native INT4 requires explicit activation_quantization=int8_per_group permission")
    shared_execution = metadata.get("shared_expert_execution", "tp_sharded")
    if shared_execution not in SHARED_EXPERT_EXECUTIONS:
        raise ValueError(f"shared_expert_execution must be one of {SHARED_EXPERT_EXECUTIONS}")
    lm_head_execution = metadata.get("lm_head_execution", "float16")
    if lm_head_execution not in LM_HEAD_EXECUTIONS:
        raise ValueError(f"lm_head_execution must be one of {LM_HEAD_EXECUTIONS}")
    if lm_head_execution == "w8a8_dynamic" and backend != NATIVE_INT4_BACKEND:
        raise ValueError("dynamic-W8A8 LM head requires the hardware-qualified native INT4 backend")
    ple_projection_execution = metadata.get("ple_projection_execution", "float16")
    if ple_projection_execution not in PLE_PROJECTION_EXECUTIONS:
        raise ValueError(f"ple_projection_execution must be one of {PLE_PROJECTION_EXECUTIONS}")
    mtp_expert_execution = metadata.get("mtp_expert_execution", "w8a16_routed")
    if mtp_expert_execution not in MTP_EXPERT_EXECUTIONS:
        raise ValueError(f"mtp_expert_execution must be one of {MTP_EXPERT_EXECUTIONS}")
    grouped_finalize = metadata.get("grouped_finalize", "torch")
    if grouped_finalize not in GROUPED_FINALIZE_METHODS:
        raise ValueError(f"grouped_finalize must be one of {GROUPED_FINALIZE_METHODS}")
    if grouped_finalize == "cann_v2" and backend != NATIVE_INT4_BACKEND:
        raise ValueError("experimental cann_v2 grouped finalization requires native INT4")
    grouped_activation = metadata.get(
        "grouped_activation", DEFAULT_NATIVE_GROUPED_ACTIVATION if backend == NATIVE_INT4_BACKEND else "torch"
    )
    if grouped_activation not in GROUPED_ACTIVATION_METHODS:
        raise ValueError(f"grouped_activation must be one of {GROUPED_ACTIVATION_METHODS}")
    if grouped_activation == "cann_swiglu_pack" and backend != NATIVE_INT4_BACKEND:
        raise ValueError("experimental grouped SwiGLU pack requires native INT4")
    if grouped_activation == "cann_builtin_fp16" and backend != NATIVE_INT4_BACKEND:
        raise ValueError("experimental built-in SwiGLU requires native INT4")
    group = metadata.get("group_size")
    if type(group) is not int or group <= 0 or group % 2:
        raise ValueError("W4 group_size must be a positive even integer")
    for field in ("hidden_size", "moe_intermediate_size"):
        if int(getattr(config, field)) % group:
            raise ValueError(f"W4 group_size must divide {field}")
        if backend in CUBE_BACKENDS and not 256 <= int(getattr(config, field)) <= CUBE_MAX_INPUTS:
            raise ValueError(f"W4 cube_310 requires 256 <= {field} <= {CUBE_MAX_INPUTS}")
    if backend in CUBE_BACKENDS and group != CUBE_GROUP_SIZE:
        raise ValueError("W4 cube_310 requires group_size=128")
    return metadata


def require_eager_w4(model_config: object, config: object) -> None:
    metadata = w4_config(config)
    if metadata is not None and metadata["backend"] not in CUBE_DEVICE_ROUTED_BACKENDS:
        if not getattr(model_config, "enforce_eager", False):
            raise ValueError("Qwen4Exp W4 host routing requires --enforce-eager; use cube_310_routed for decode graphs")


def finalize_grouped_routes(
    routed: torch.Tensor,
    dispatch: GroupedExpertDispatch,
    weights: torch.Tensor,
    accumulation_dtype: torch.dtype,
    method: str = "torch",
) -> torch.Tensor:
    """Combine sorted expert rows in original token and route order.

    ``cann_v2`` is an opt-in 310P experiment. Its FP16 output is widened for
    the existing FP32 TP reduction, but the earlier FP16 rounding can change
    model numerics and must pass a real-weight gate before serving use.
    """
    if routed.ndim != 2 or weights.ndim != 2 or routed.shape[0] != weights.numel():
        raise ValueError("routed rows and [tokens, top_k] weights do not match")
    if dispatch.inverse_order.numel() != routed.shape[0]:
        raise ValueError("inverse expert order must cover every routed row")
    tokens, top_k = weights.shape
    if method == "torch":
        ordered_weights = dispatch.route_weights.index_select(0, dispatch.order)
        weighted = routed.to(accumulation_dtype).mul_(ordered_weights)
        return weighted.index_select(0, dispatch.inverse_order).reshape(tokens, top_k, routed.shape[1]).sum(1)
    if method != "cann_v2":
        raise ValueError(f"unknown grouped finalization method: {method}")

    if routed.dtype != torch.float16:
        raise ValueError("310P grouped finalization requires FP16 routed rows")
    # Keep torch_npu lazy so host-only loader and model tests do not initialize
    # the NPU runtime. The installed 310P CANN build requires the scales to
    # match the routed-row dtype, so this conversion belongs in the timing gate.
    import torch_npu

    inverse_order = dispatch.inverse_order.to(torch.int32).contiguous()
    combined = torch_npu.npu_moe_finalize_routing(
        routed,
        None,
        None,
        None,
        weights.to(routed.dtype).contiguous(),
        inverse_order,
        None,
        2,
    )
    return combined.to(accumulation_dtype)


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int8:
        raise ValueError("W4 packed storage must be int8")
    # Signed shifts are deliberate; mask recovers the high two's-complement nibble.
    nibbles = torch.stack((packed & 15, (packed >> 4) & 15), dim=-1).flatten(-2)
    return torch.where(nibbles >= 8, nibbles - 16, nibbles)


def dequantize(packed: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, group_size: int) -> torch.Tensor:
    dtype = ASCEND_QWEN4EXP_DTYPE_POLICY.accumulation_dtype
    quant = unpack_signed_int4(packed).to(dtype).reshape(*scale.shape, group_size)
    return ((quant - offset.to(dtype).unsqueeze(-1)) * scale.to(dtype).unsqueeze(-1)).flatten(-2)


def pack_cube_tiles(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    """Lossless one-expert load-time encoding; preserve shape/dtype/bytes.

    Codes: [N/32, K/128, 8, 16, 16], with the low/high nibbles holding
    output channels n and n+16. Each unpacked plane is already Cube NZ.
    Metadata: [N/32, K/128, 32]. Bias both codes and offsets by eight, so subtraction
    gives exactly the original signed q-offset without GPU sign extension.
    Public tensor shapes remain canonical for loader validation; the tiled
    operator must be selected explicitly to interpret their physical layout.
    """
    if tensor.ndim != 2 or tensor.shape[0] % CUBE_TILE_OUTPUTS:
        raise ValueError("W4 tile packing requires a matrix with N divisible by 32")
    rows = tensor.shape[0] // CUBE_TILE_OUTPUTS
    if kind == "weight":
        packed_group = CUBE_GROUP_SIZE // 2
        if tensor.shape[1] % packed_group:
            raise ValueError("W4 tile packing requires K divisible by 128")
        biased = unpack_signed_int4(tensor) + CUBE_ZERO_POINT_BIAS
        planes = biased.reshape(rows, 2, CUBE_FRACTAL_SIZE, -1, CUBE_FRACTAL_SIZE).permute(0, 3, 4, 1, 2)
        packed = planes[..., 0, :] | (planes[..., 1, :] << 4)
    elif kind in ("weight_scale", "weight_offset"):
        values = tensor + CUBE_ZERO_POINT_BIAS if kind == "weight_offset" else tensor
        packed = values.reshape(rows, CUBE_TILE_OUTPUTS, -1).transpose(1, 2)
    else:
        raise ValueError(f"unknown W4 projection field: {kind}")
    return packed.contiguous().view_as(tensor)


class PackedExpertBank(nn.Module):
    def __init__(
        self, experts: int, outputs: int, inputs: int, group_size: int, *, backend: str = "eager_dequant"
    ) -> None:
        super().__init__()
        if backend not in ("eager_dequant", *CUBE_BACKENDS):
            raise ValueError("unsupported W4 projection backend")
        self.group_size = group_size
        self.backend = backend
        dtype = ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype
        self.weight = nn.Parameter(torch.zeros(experts, outputs, inputs // 2, dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(
            torch.zeros(experts, outputs, inputs // group_size, dtype=dtype), requires_grad=False
        )
        offset_dtype = dtype if backend == NATIVE_INT4_BACKEND else torch.int8
        self.weight_offset = nn.Parameter(torch.zeros(self.weight_scale.shape, dtype=offset_dtype), requires_grad=False)
        if backend == NATIVE_INT4_BACKEND:
            self.register_buffer("weight_sum", torch.zeros_like(self.weight_scale), persistent=False)

    def linear(self, inputs: torch.Tensor, expert: int) -> torch.Tensor:
        if self.backend == NATIVE_INT4_BACKEND:
            raise RuntimeError("native INT4 has no Python expert fallback; use grouped_linear")
        if self.backend in CUBE_BACKENDS:
            if inputs.device.type != "npu" or self.group_size != CUBE_GROUP_SIZE:
                raise ValueError("W4 cube_310 requires NPU inputs and group_size=128")
            if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_group_matmul_310"):
                raise RuntimeError("W4 cube_310 requires the rebuilt Qwen W4 custom operator; refusing silent fallback")
            return torch.ops._C_ascend.npu_qwen_w4_group_matmul_310(
                inputs,
                self.weight[expert],
                self.weight_scale[expert],
                self.weight_offset[expert],
                self.backend in CUBE_TILED_BACKENDS,
            )
        weight = dequantize(self.weight[expert], self.weight_scale[expert], self.weight_offset[expert], self.group_size)
        return F.linear(inputs, weight.to(inputs.dtype))

    def routed_linear(self, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if self.backend == NATIVE_INT4_BACKEND:
            if inputs.device.type != "npu":
                raise ValueError("native INT4 requires NPU inputs")
            return self.native_linear(pack_activation_device(inputs), expert_ids)
        if self.backend not in ("cube_310_routed", "cube_310_grouped") or inputs.device.type != "npu":
            raise ValueError("W4 routed projection requires NPU inputs and cube_310_routed")
        if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_routed_matmul_310"):
            raise RuntimeError("W4 routed projection requires the rebuilt custom operator; refusing silent fallback")
        return torch.ops._C_ascend.npu_qwen_w4_routed_matmul_310(
            inputs, self.weight, self.weight_scale, self.weight_offset, expert_ids
        )

    def grouped_linear(self, inputs: torch.Tensor, group_ends: torch.Tensor) -> torch.Tensor:
        """Project sorted routes; peer rows after the last group become zero.

        Group ends remain on device. No host counts, unpacked weight bank, or
        single-expert Python dispatch is permitted in this backend.
        """
        if self.backend == NATIVE_INT4_BACKEND:
            if inputs.device.type != "npu":
                raise ValueError("native INT4 requires NPU inputs")
            return self.native_linear(pack_activation_device(inputs), group_ends)
        if self.backend != "cube_310_grouped" or inputs.device.type != "npu":
            raise ValueError("W4 grouped projection requires NPU inputs and cube_310_grouped")
        if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_grouped_matmul_310"):
            raise RuntimeError("W4 grouped projection requires the rebuilt custom operator; refusing silent fallback")
        return torch.ops._C_ascend.npu_qwen_w4_grouped_matmul_310(
            inputs, self.weight, self.weight_scale, self.weight_offset, group_ends
        )

    def native_linear(self, prepared: tuple[torch.Tensor, ...], group_ends: torch.Tensor) -> torch.Tensor:
        """Consume activations packed once and shared across expert routes."""
        if self.backend != NATIVE_INT4_BACKEND:
            raise ValueError("packed activation inputs require native INT4")
        if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_a8_int4_matmul_310"):
            raise RuntimeError("native INT4 requires the rebuilt custom operator; refusing silent fallback")
        return torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
            *prepared, self.weight, self.weight_scale, self.weight_offset, self.weight_sum, group_ends
        )

    def native_down_reduce(
        self, prepared: tuple[torch.Tensor, ...], route_ids: torch.Tensor, route_weights: torch.Tensor
    ) -> torch.Tensor:
        """Project route rows, then apply FP32 route weights in slot order."""
        if self.backend != NATIVE_INT4_BACKEND:
            raise ValueError("packed activation inputs require native INT4")
        if not hasattr(torch.ops._C_ascend, "npu_qwen_w4_a8_int4_down_reduce_310"):
            raise RuntimeError("native INT4 down-reduce requires the rebuilt custom operator; refusing silent fallback")
        return torch.ops._C_ascend.npu_qwen_w4_a8_int4_down_reduce_310(
            *prepared,
            self.weight,
            self.weight_scale,
            self.weight_offset,
            self.weight_sum,
            route_ids,
            route_weights.contiguous(),
        )


class W4SparseMoE(nn.Module):
    """Same router/shared/TP contract as W8, but separate packed weight banks."""

    def __init__(
        self,
        *,
        config: object,
        dtype_policy: Qwen4ExpDtypePolicy,
        expert_sharding=(0, 1),
        deferred_reduce_stream: DeferredReduceStream | None = None,
    ) -> None:
        super().__init__()
        metadata = w4_config(config)
        if metadata is None:
            raise ValueError("W4SparseMoE requires explicit checkpoint metadata")
        hidden, intermediate = int(config.hidden_size), int(config.moe_intermediate_size)
        self.intermediate_size = intermediate
        self.max_chunk_tokens = MAX_CUBE_TOKENS if metadata["backend"] in CUBE_BACKENDS else MAX_EAGER_TOKENS
        self.device_routing = metadata["backend"] in CUBE_DEVICE_ROUTED_BACKENDS
        self.native_int4 = metadata["backend"] == NATIVE_INT4_BACKEND
        self.max_routed_rows = MAX_CUBE_ROUTES if self.native_int4 else MAX_W4A16_ROUTES
        self.fused_native_down_reduce = (
            self.native_int4 and intermediate == FUSED_DOWN_INPUTS and hidden == FUSED_DOWN_OUTPUTS
        )
        self.grouped_routing = metadata["backend"] in ("cube_310_grouped", NATIVE_INT4_BACKEND)
        self.grouped_finalize = metadata.get("grouped_finalize", "torch")
        self.grouped_activation = metadata.get(
            "grouped_activation", DEFAULT_NATIVE_GROUPED_ACTIVATION if self.native_int4 else "torch"
        )
        self.fused_gate_up = self.device_routing
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        if self.top_k <= 0 or self.top_k > int(config.num_experts):
            raise ValueError("W4 top_k must be positive and no larger than num_experts")
        grouped_token_limit, grouped_route_limit = (
            (MAX_GROUPED_NATIVE_TOKENS, MAX_GROUPED_NATIVE_ROUTES)
            if self.native_int4
            else (MAX_GROUPED_W4A16_TOKENS, MAX_GROUPED_W4A16_ROUTES)
        )
        self.grouped_chunk_tokens = min(grouped_token_limit, grouped_route_limit // self.top_k)
        if self.grouped_routing and self.grouped_chunk_tokens == 0:
            raise ValueError("W4 grouped top_k exceeds the bounded route workspace")
        self.expert_tp_rank, self.expert_tp_size = expert_sharding
        if (
            self.expert_tp_size < 1
            or not 0 <= self.expert_tp_rank < self.expert_tp_size
            or self.num_experts < self.expert_tp_size
        ):
            raise ValueError("invalid W4 expert TP ownership")
        self.expert_offset, stop = local_expert_range(self.num_experts, self.expert_tp_size, self.expert_tp_rank)
        self.num_local_experts = stop - self.expert_offset
        self.params_dtype, self.compute_dtype = dtype_policy.main_dtype, dtype_policy.accumulation_dtype
        self.shared_expert_execution = metadata.get("shared_expert_execution", "tp_sharded")
        self.shared_expert_replicated = self.shared_expert_execution.startswith("replicated")
        self.overlap_shared_expert = self.shared_expert_execution.endswith("_overlap")
        self.defer_shared_expert_sync = self.shared_expert_execution == "replicated_deferred"
        self.deferred_reduce_stream = None
        if self.defer_shared_expert_sync:
            self.deferred_reduce_stream = deferred_reduce_stream or DeferredReduceStream()
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0) or 1.0)
        self.gate = nn.Parameter(torch.zeros(self.num_experts, hidden, dtype=self.params_dtype))
        # Gate/up have identical route ownership and input activations. Store
        # adjacent N tiles in one packed bank, without a shadow or load-time
        # full-bank concatenation. Other backends keep their reference layout.
        projection_shapes = (
            {"gate_up_proj": (2 * intermediate, hidden), "down_proj": (hidden, intermediate)}
            if self.fused_gate_up
            else {
                "gate_proj": (intermediate, hidden),
                "up_proj": (intermediate, hidden),
                "down_proj": (hidden, intermediate),
            }
        )
        self.projections = nn.ModuleDict(
            {
                name: PackedExpertBank(
                    self.num_local_experts,
                    outputs,
                    inputs,
                    metadata["group_size"],
                    backend=metadata["backend"],
                )
                for name, (outputs, inputs) in projection_shapes.items()
            }
        )
        shared = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        self.has_shared_expert = shared > 0
        if not self.shared_expert_replicated and shared % self.expert_tp_size:
            raise ValueError("shared expert intermediate dimension must divide TP")
        self.local_shared_inter = shared if self.shared_expert_replicated else shared // self.expert_tp_size
        if self.has_shared_expert:
            self.shared_gate_up = nn.Parameter(
                torch.zeros(2 * self.local_shared_inter, hidden, dtype=self.params_dtype)
            )
            self.shared_down = nn.Parameter(torch.zeros(hidden, self.local_shared_inter, dtype=self.params_dtype))
            self.shared_expert_gate = nn.Parameter(torch.zeros(1, hidden, dtype=self.params_dtype))
        self._tp_reduce = None
        if self.expert_tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            self._tp_reduce = tensor_model_parallel_all_reduce

    def load_projection(self, expert: int, projection: str, kind: str, tensor: torch.Tensor) -> str | None:
        if not 0 <= expert < self.num_experts:
            raise ValueError(f"W4 expert id out of range: {expert}")
        local = expert - self.expert_offset
        if not 0 <= local < self.num_local_experts:
            return None
        bank_name = "gate_up_proj" if self.fused_gate_up and projection in ("gate_proj", "up_proj") else projection
        bank = self.projections[bank_name]
        target = getattr(bank, kind)[local]
        if bank_name == "gate_up_proj":
            start = self.intermediate_size if projection == "up_proj" else 0
            target = target[start : start + self.intermediate_size]
        expected_dtype = torch.int8 if kind in ("weight", "weight_offset") else target.dtype
        if tensor.dtype != expected_dtype or tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"W4 {projection}.{kind}: expected {target.dtype} {tuple(target.shape)}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        if kind != "weight":
            if not torch.isfinite(tensor).all() or (kind == "weight_scale" and not (tensor > 0).all()):
                raise ValueError("W4 scales must be positive and quantization parameters finite")
            if kind == "weight_offset" and not ((tensor >= -8) & (tensor <= 7)).all():
                raise ValueError("W4 offsets must be signed-int4 zero points")
        with torch.no_grad():
            if bank.backend == NATIVE_INT4_BACKEND:
                if kind == "weight":
                    tensor, sums = pack_native_weight(tensor)
                    sum_target = bank.weight_sum[local]
                    if bank_name == "gate_up_proj":
                        sum_target = sum_target[start : start + self.intermediate_size]
                    sum_target.copy_(sums)
                else:
                    tensor = pack_native_metadata(tensor)
            elif bank.backend in CUBE_TILED_BACKENDS:
                tensor = pack_cube_tiles(tensor, kind)
            target.copy_(tensor)
        return f"projections.{bank_name}.{kind}"

    def _forward_shared(self, block_input: torch.Tensor) -> torch.Tensor:
        # Match production W8's NPU projection policy. Converting NZ FP16
        # weights to FP32 on every call both copies weights and selects an
        # aclop Cast that cannot be captured. Keep the CPU/eager reference
        # unchanged; the routed backend uses its resident FP16 weights.
        operand_dtype = (
            self.params_dtype if block_input.device.type == "npu" and self.fused_gate_up else self.compute_dtype
        )
        inputs = block_input.to(operand_dtype)
        gate, up = F.linear(inputs, self.shared_gate_up.to(operand_dtype)).chunk(2, -1)
        shared = F.linear(F.silu(gate) * up, self.shared_down.to(operand_dtype))
        return shared * torch.sigmoid(F.linear(inputs, self.shared_expert_gate.to(operand_dtype)))

    def _start_shared_overlap(
        self, block_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.npu.Stream, torch.npu.Event]:
        # Keep the NPU runtime optional for host-side model and loader tests.
        from vllm_ascend.utils import current_stream, npu_stream_switch, shared_experts_calculation_stream

        main_stream = current_stream()
        input_ready = main_stream.record_event()
        shared_stream = shared_experts_calculation_stream()
        block_input.record_stream(shared_stream)
        with npu_stream_switch(shared_stream):
            shared_stream.wait_event(input_ready)
            shared = self._forward_shared(block_input)
            shared_done = shared_stream.record_event()
        shared.record_stream(main_stream)
        return shared, main_stream, shared_done

    def _start_deferred_reduce(self, routed: torch.Tensor) -> tuple[torch.Tensor, torch.npu.Stream, torch.npu.Event]:
        """Enqueue the routed reduction before independent shared compute."""
        from vllm_ascend.utils import current_stream, npu_stream_switch

        if self._tp_reduce is None:
            raise RuntimeError("W4 expert TP requires all-reduce")
        main_stream = current_stream()
        routed_ready = main_stream.record_event()
        if self.deferred_reduce_stream is None:
            raise RuntimeError("deferred all-reduce stream was not configured")
        reduce_stream = self.deferred_reduce_stream.get()
        routed.record_stream(reduce_stream)
        with npu_stream_switch(reduce_stream):
            reduce_stream.wait_event(routed_ready)
            reduced = self._tp_reduce(routed)
            reduce_done = reduce_stream.record_event()
        reduced.record_stream(main_stream)
        return reduced, main_stream, reduce_done

    def _should_overlap_shared_expert(self, block_input: torch.Tensor) -> bool:
        return (
            self.has_shared_expert
            and self.overlap_shared_expert
            and block_input.device.type == "npu"
            and (
                self.shared_expert_execution != "tp_sharded_overlap"
                or block_input.shape[0] <= MAX_SHARED_EXPERT_OVERLAP_TOKENS
            )
        )

    def forward(self, block_input: torch.Tensor) -> torch.Tensor:
        shared = None
        main_stream = None
        shared_done = None
        reduce_done = None
        if self._should_overlap_shared_expert(block_input):
            shared, main_stream, shared_done = self._start_shared_overlap(block_input)

        weights, ids = route_topk(
            F.linear(block_input, self.gate),
            self.top_k,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        if self.device_routing and block_input.shape[0] * self.top_k <= self.max_routed_rows:
            result = self._forward_routed(block_input, weights, ids)
        elif self.grouped_routing:
            result = self._forward_grouped(block_input, weights, ids)
        else:
            if self.device_routing and torch.npu.is_current_stream_capturing():
                raise RuntimeError(f"W4 decode graph exceeds {self.max_routed_rows} routes; reduce capture sizes")
            result = self._forward_host_routed(block_input, weights, ids)
        if self.has_shared_expert and not self.shared_expert_replicated:
            if shared is None:
                shared = self._forward_shared(block_input)
            else:
                main_stream.wait_event(shared_done)
            result += shared
        if self.expert_tp_size > 1:
            if self._tp_reduce is None:
                raise RuntimeError("W4 expert TP requires all-reduce")
            if self.defer_shared_expert_sync and self.has_shared_expert and block_input.device.type == "npu":
                result, main_stream, reduce_done = self._start_deferred_reduce(result)
            else:
                result = self._tp_reduce(result)
        if self.has_shared_expert and self.shared_expert_replicated:
            if shared is None:
                shared = self._forward_shared(block_input)
            else:
                main_stream.wait_event(shared_done)
            if reduce_done is not None:
                main_stream.wait_event(reduce_done)
            result += shared
        return result.to(self.params_dtype)

    def _forward_routed(self, block_input: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # Static route slots, dynamic device expert IDs. Peer routes produce
        # exact zero rows and must also overwrite rows on every graph replay.
        tokens, hidden = block_input.shape
        local_ids = (ids - self.expert_offset).to(torch.int32).flatten().contiguous()
        if self.native_int4:
            # Each token has top_k routes, but its gate/up input is identical.
            # The native op broadcasts packed rows by the fixed route factor.
            gate_up = self.projections["gate_up_proj"].native_linear(pack_activation_device(block_input), local_ids)
        else:
            inputs = block_input[:, None, :].expand(-1, self.top_k, -1).reshape(-1, hidden).contiguous()
            gate_up = self.projections["gate_up_proj"].routed_linear(inputs, local_ids)
        if self.native_int4:
            prepared_activation = swiglu_pack_activation_device(gate_up)
            if self.fused_native_down_reduce and tokens * self.top_k <= MAX_FUSED_DOWN_ROUTES:
                return self.projections["down_proj"].native_down_reduce(prepared_activation, local_ids, weights)
            output = self.projections["down_proj"].native_linear(prepared_activation, local_ids).to(self.compute_dtype)
        else:
            gate, up = gate_up.to(self.compute_dtype).chunk(2, -1)
            activation = (F.silu(gate) * up).to(self.params_dtype)
            output = self.projections["down_proj"].routed_linear(activation, local_ids).to(self.compute_dtype)
        return (output.reshape(tokens, self.top_k, hidden) * weights.unsqueeze(-1)).sum(dim=1)

    def _forward_host_routed(self, block_input: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(block_input, dtype=self.compute_dtype)
        # One routing sync per bounded token chunk. Intentionally eager-only.
        for start in range(0, block_input.shape[0], self.max_chunk_tokens):
            routes: dict[int, list[tuple[int, int]]] = {}
            for token, token_ids in enumerate(ids[start : start + self.max_chunk_tokens].cpu().tolist()):
                for slot, global_id in enumerate(token_ids):
                    local = global_id - self.expert_offset
                    if 0 <= local < self.num_local_experts:
                        routes.setdefault(local, []).append((start + token, slot))
            for expert, selected in routes.items():
                indices = torch.tensor(selected, dtype=torch.long, device=block_input.device)
                tokens, slots = indices.unbind(-1)
                inputs = block_input.index_select(0, tokens)
                if self.fused_gate_up:
                    gate, up = (
                        self.projections["gate_up_proj"].linear(inputs, expert).to(self.compute_dtype).chunk(2, -1)
                    )
                else:
                    gate = self.projections["gate_proj"].linear(inputs, expert).to(self.compute_dtype)
                    up = self.projections["up_proj"].linear(inputs, expert).to(self.compute_dtype)
                activation = (F.silu(gate) * up).to(self.params_dtype)
                output = self.projections["down_proj"].linear(activation, expert).to(self.compute_dtype)
                result.index_add_(0, tokens, output * weights[tokens, slots].unsqueeze(-1))
        return result

    def _forward_grouped(self, block_input: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(block_input, dtype=self.compute_dtype)
        for start in range(0, block_input.shape[0], self.grouped_chunk_tokens):
            stop = start + self.grouped_chunk_tokens
            inputs = block_input[start:stop]
            tokens, hidden = inputs.shape
            dispatch = build_grouped_expert_dispatch(
                weights[start:stop],
                ids[start:stop],
                num_local_experts=self.num_local_experts,
                expert_offset=self.expert_offset,
                weight_dtype=self.compute_dtype,
            )
            sorted_tokens = dispatch.token_indices.index_select(0, dispatch.order)
            group_ends = dispatch.group_list.contiguous()
            gate_up_bank = self.projections["gate_up_proj"]
            if self.native_int4 and tokens * self.top_k > self.max_routed_rows:
                # Quantization belongs to the token, not its top-k copies.
                # Share it across prefill routes; small decode avoids four
                # gather launches because its packing work is already tiny.
                prepared = tuple(value.index_select(0, sorted_tokens) for value in pack_activation_device(inputs))
                projected = gate_up_bank.native_linear(prepared, group_ends)
            else:
                inputs = inputs.index_select(0, sorted_tokens).contiguous()
                projected = gate_up_bank.grouped_linear(inputs, group_ends)
            if self.grouped_activation == "cann_swiglu_pack":
                # Opt-in until full prefill rows pass exact packing parity and
                # a real-weight service gate on the coherent 310P OPP package.
                packed_activation = swiglu_pack_activation_device(projected)
                output = self.projections["down_proj"].native_linear(packed_activation, group_ends)
            elif self.grouped_activation == "cann_builtin_fp16":
                # Native INT4 prefill default; preserve an explicit torch
                # override for numerical comparisons and other workloads.
                # Keep the import lazy for host-only model configuration.
                import torch_npu

                activation = torch_npu.npu_swiglu(projected, dim=-1)
                output = self.projections["down_proj"].grouped_linear(activation, group_ends)
            else:
                gate, up = projected.to(self.compute_dtype).chunk(2, -1)
                activation = (F.silu(gate) * up).to(self.params_dtype)
                output = self.projections["down_proj"].grouped_linear(activation, group_ends)
            if self.grouped_finalize == "torch":
                output = output.to(self.compute_dtype)
                output *= dispatch.route_weights.index_select(0, dispatch.order)
                result[start:stop] = (
                    output.index_select(0, dispatch.inverse_order).reshape(tokens, self.top_k, hidden).sum(1)
                )
            else:
                result[start:stop] = finalize_grouped_routes(
                    output,
                    dispatch,
                    weights[start:stop],
                    self.compute_dtype,
                    self.grouped_finalize,
                )
        return result


def validate_w4_inventory(layers: nn.ModuleList, names: set[str]) -> None:
    expected = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{kind}"
        for layer, module in enumerate(layers)
        if isinstance(module.mlp, W4SparseMoE)
        for expert in range(module.mlp.expert_offset, module.mlp.expert_offset + module.mlp.num_local_experts)
        for projection in PROJECTIONS
        for kind in KINDS
    }
    if names != expected:
        raise ValueError(
            f"incomplete W4 expert checkpoint: missing={len(expected - names)}, extra={len(names - expected)}"
        )
