#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""310P fused-MoE method for DeepSeek V4.1 2-bit (W2) routed experts (E1.3).

This is the device-facing wrapper around the validated E1.2 active-expert
unpack path (``vllm_ascend/models/deepseek_v41/w2_unpack.py``). It mirrors the
surface of the 310P W8A8 dynamic fused-MoE method
(:class:`~vllm_ascend._310p.quantization.methods.w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod310`)
so the DeepSeek W2 weight loader (E3.4) can reuse the same param-creation and
weight-loading hooks:

  * :meth:`get_weight` creates the packed W2 code params (``w13_codes`` /
    ``w2_codes``; ``uint8[out, in // 4]`` per the E1.1 pack contract).
  * :meth:`get_dynamic_quant_param` creates the per-``[32, 32]`` block scales
    (``w13_scale`` / ``w2_scale``; ``float32[out // 32, in // 32]``).
  * :meth:`get_shared_expert_weight` / :meth:`get_shared_expert_dynamic_quant_param`
    create the always-on shared-expert params in the same layout.

``apply`` dispatches on whether the INT8 grouped-matmul kernel is available:

  * **Device path** (``_apply_device``, import-guarded): the ``<= top_k`` active
    experts are widened from packed W2 by E1.2 :func:`unpack_active_experts`,
    then the resulting INT8 codes + block scales are handed to
    ``torch_npu.npu_quant_grouped_matmul_dequant`` (as the W8 method's
    ``apply_gmm1_act_quant`` / ``apply_gmm2`` do). ``torch_npu`` is absent
    host-side, so this branch is guarded and never runs in the CPU UT.
  * **Host path** (``_apply_host``): re-expresses the identical math through the
    E1.2 grouped primitives (:func:`unpack_active_experts`,
    :func:`w2_group_qdq_linear`, :func:`swiglu_gate_up`), driven by the router's
    pre-selected ``topk_ids`` / ``topk_weights``. :meth:`moe_forward` is the
    router-logits entrypoint, a thin wrapper over E1.2
    :func:`w2_active_moe_forward`.

Device-wave note (for D1.5)
---------------------------
The W2 -> INT8 widen runs on **only the active experts** (``<= top_k``, i.e. at
most 6 for DeepSeek V4.1), never the full 384-expert bank -- that is the whole
point of :func:`unpack_active_experts`. Whether the widen itself is a single
fused device op or an elementwise widen must be verified against the pinned CANN
op surface: vllm-ascend today exposes no fused W2->INT8 unpack kernel (the
nearest, ``npu_convert_weight_to_int4pack``, only *packs* INT4 for the INT4
matmul path), so absent a fused op the widen is an elementwise unpack over the
``<= top_k`` active experts before ``npu_quant_grouped_matmul_dequant``. The
per-``[32, 32]`` block scale is applied *into* the weight before the matmul (it
varies along the input axis every 32 columns and cannot be reduced to a single
per-output-channel factor); D1.5 confirms the exact block-scale layout the
pinned kernel ingests. These flags record that decision for D1.5.
"""

from typing import Any

import torch

from tools.deepseek_w2.w2_format import (
    NVFP4_BLOCK_COLS,
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W2_CODES_PER_BYTE,
    unpack_codes,
    unpack_nvfp4_codes,
)
from vllm_ascend.models.deepseek_v41.w2_unpack import (
    swiglu_gate_up,
    unpack_active_experts,
    w2_active_moe_forward,
    w2_group_qdq_linear,
)
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType

from .registry import register_scheme


def _infer_bits(packed: torch.Tensor, in_features: int) -> int:
    """Code width (2 or 4) from a packed operand: W2 packs 4 codes/byte
    (last dim = in//4), W4 packs 2 codes/byte (in//2). Lets the runtime handle
    mixed-precision W2/W4 expert banks with no config plumbing."""
    codes_per_byte = in_features // int(packed.shape[-1])
    return 8 // codes_per_byte


# The device INT8 grouped-matmul kernel lives in torch_npu, which is absent
# host-side. Import it guarded so this module is importable on CPU and the host
# path is taken whenever the fused kernel is unavailable.
try:  # pragma: no cover - trivially exercised by the host UT via absence
    import torch_npu  # type: ignore
except ImportError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]

# Device-wave decision record for D1.5 (see module docstring).
W2_UNPACK_IS_FUSED = False
W2_ACTIVE_UNPACK_ONLY = True

# Name of the fused INT8 grouped-matmul + dequant kernel on the device wave.
_W2_DEVICE_KERNEL = "npu_quant_grouped_matmul_dequant"

# The current 310P packed-W2 Cube kernel corrupts the down projection when its
# M dimension exceeds 48 (the first failing model shape is [49, 2048] x
# [4096, 2048]^T). Keep the fast path for decode and small expert groups while
# prefill groups above the hardware-validated boundary use exact eager math.
W2_CUBE_MAX_TOKENS = 48
W2_CUBE_OUTPUT_TILE = 128
W2_CUBE_INPUT_TILE = 128
W2_CUBE_MIN_INPUT_DIM = 256


def _device_kernel_available() -> bool:
    """True only on a real NPU wave that exposes the INT8 grouped-matmul kernel.

    A host-side ``torch_npu`` stub without the fused symbol resolves to False,
    so the CPU UT deterministically takes the host math path.
    """
    return torch_npu is not None and hasattr(torch_npu, _W2_DEVICE_KERNEL)


def _w2_dequant_fp32(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_f: int,
    in_f: int,
) -> torch.Tensor:
    """Dequantize one packed W2 projection to a dense fp32 weight ``[out_f, in_f]``.

    Widens the 2-bit codes (``{-2,-1,0,1}``) and multiplies by the COMPACT
    ``[out_f//32, in_f//32]`` per-block scale via a tiled view-multiply, i.e.
    each ``[32, 32]`` code tile is scaled by its single block scale. This is
    bit-identical to ``codes * broadcast_block_scales(block_scale, ...)`` but
    never materializes the full-size scale and never touches float64 -- 310P has
    no native fp64, so the old broadcast path emitted an emulated cast per op and
    dominated the MoE step time.
    """
    bits = _infer_bits(packed, in_f)  # 2 (W2) or 4 (W4); mixed banks supported
    codes = unpack_codes(packed, in_f, bits).to(torch.float32)  # [out_f, in_f]
    bs = block_scale.to(torch.float32)  # [out_f // 32, in_f // 32]
    tiled = codes.view(out_f // W2_BLOCK_ROWS, W2_BLOCK_ROWS, in_f // W2_BLOCK_COLS, W2_BLOCK_COLS)
    scaled = tiled * bs.view(out_f // W2_BLOCK_ROWS, 1, in_f // W2_BLOCK_COLS, 1)
    return scaled.reshape(out_f, in_f)


def _is_nvfp4(block_scale: torch.Tensor, out_f: int, in_f: int) -> bool:
    """True when a packed operand is NVFP4 (E2M1 + block-16) rather than W2/W4.

    W2/W4 scales tile ``[out_f // 32, in_f // 32]``; NVFP4's folded scale is
    ``[out_f, in_f // 16]`` (per-output-row x per-16-input-col). The codes width
    alone cannot distinguish NVFP4 from W4 (both pack 2 codes/byte), so the scale
    grid is the discriminator.
    """
    if block_scale.ndim != 2:
        return False
    return block_scale.shape[0] == out_f and block_scale.shape[1] == in_f // NVFP4_BLOCK_COLS


def _nvfp4_dequant_fp32(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    out_f: int,
    in_f: int,
) -> torch.Tensor:
    """Dequantize one packed NVFP4 operand to fp32 ``[out_f, in_f]`` (no fp64).

    Decodes the E2M1 nibbles and multiplies by the folded per-``[1, 16]`` block
    scale via a tiled view-multiply (never materializing the full-size scale and
    never touching float64, mirroring :func:`_w2_dequant_fp32` for 310P).
    """
    val = unpack_nvfp4_codes(packed, in_f).to(torch.float32)  # [out_f, in_f]
    bs = block_scale.to(torch.float32)  # [out_f, in_f // 16]
    tiled = val.view(out_f, in_f // NVFP4_BLOCK_COLS, NVFP4_BLOCK_COLS)
    scaled = tiled * bs.unsqueeze(-1)
    return scaled.reshape(out_f, in_f)


_W2_BLOCKED_MM_OP: Any = None


def _w2_blocked_mm_op():
    """The fused 310P Cube kernel ``npu_w2_blocked_dequant_matmul_310`` if built.

    Computes ``out[T,N] = x[T,K] @ (codes ⊙ block_scale)^T`` on the Cube with the
    per-[32,32] block dequant fused into the weight load (arch20 catlass MMAD).
    The kernel expands one bounded output tile into an already-NZ per-core
    workspace, rather than materializing the complete fp16 weight or asking the
    matmul path to perform an ND-to-NZ conversion. A bounded per-core FP32 tile
    preserves accumulator correctness before the final FP16 cast. Resolved
    lazily (the custom-op vendor lib is loaded during worker init); returns
    ``None`` when the op is unavailable so the eager fp32 path stays a correct
    fallback.
    """
    global _W2_BLOCKED_MM_OP
    if _W2_BLOCKED_MM_OP is None:
        try:
            _W2_BLOCKED_MM_OP = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310
        except (AttributeError, RuntimeError):
            _W2_BLOCKED_MM_OP = None
    return _W2_BLOCKED_MM_OP


def _can_use_w2_cube(
    w2_op: Any,
    packed: torch.Tensor,
    in_features: int,
    num_tokens: int,
    is_nvfp4: bool,
) -> bool:
    """Whether the packed integer Cube kernel is safe for this expert group."""
    return (
        w2_op is not None
        and num_tokens <= W2_CUBE_MAX_TOKENS
        and packed.shape[-2] % W2_CUBE_OUTPUT_TILE == 0
        and in_features % W2_CUBE_INPUT_TILE == 0
        and in_features >= W2_CUBE_MIN_INPUT_DIM
        and _infer_bits(packed, in_features) in (2, 4)
        and not is_nvfp4
    )


@register_scheme("W2A8_DYNAMIC", "moe")
class AscendW2DynamicFusedMoEMethod310(AscendMoEScheme):
    """310P-only FusedMoE method for DeepSeek V4.1 2-bit (W2) routed experts.

    The weight *storage* is packed W2 (2-bit signed codes + per-``[32, 32]``
    block scale); once the ``<= top_k`` active experts are widened to INT8 codes
    the device grouped matmul is the same INT8 dynamic QDQ path as W8A8, hence
    ``quant_type = QuantType.W8A8`` (weights land in the INT8 domain, activations
    are per-token symmetric INT8).

    Notes:
      - Discovered via the 310P-local registry (``W2A8_DYNAMIC`` / ``moe``).
      - Mirrors the W8A8 dynamic method's param / weight-loading surface so the
        DeepSeek W2 loader (E3.4) reuses it.
    """

    # The device grouped matmul consumes INT8 codes + INT8 per-token activation.
    quant_type: QuantType = QuantType.W8A8
    act_quant_type: torch.dtype = torch.int8
    # SwiGLU is fused into the device gmm1 path (npu_swiglu), as in the W8 method.
    fused_activations = frozenset({"silu"})

    def __init__(self):
        # Mirror the W8 method's construction surface, but stay host-constructible
        # (the CPU UT builds params + runs host math without a live vLLM runtime).
        try:  # pragma: no cover - runtime-only branch
            from vllm.distributed import get_ep_group

            self.ep_group = get_ep_group()
        except Exception:
            self.ep_group = None
        try:  # pragma: no cover - runtime-only branch
            from vllm.config import get_current_vllm_config

            self.in_dtype = get_current_vllm_config().model_config.dtype
        except Exception:
            self.in_dtype = torch.float16

    # -- param creation (mirrors the W8 method: codes + block scales) ---------

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Packed W2 code params (``uint8[E, out, in // 4]``) per the E1.1 pack.

        ``w13`` fuses gate/up (``out = 2 * inter``, ``in = hidden``); ``w2`` is
        the down projection (``out = hidden``, ``in = inter``).
        """
        inter = intermediate_size_per_partition
        hidden = hidden_sizes
        param_dict: dict[str, Any] = {}
        # Fused gate_up_proj (column parallel): [E, 2*inter, hidden // 4] uint8.
        param_dict["w13_codes"] = torch.empty(num_experts, 2 * inter, hidden // W2_CODES_PER_BYTE, dtype=torch.uint8)
        # down_proj (row parallel): [E, hidden, inter // 4] uint8.
        param_dict["w2_codes"] = torch.empty(num_experts, hidden, inter // W2_CODES_PER_BYTE, dtype=torch.uint8)
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Per-``[32, 32]`` block scales (``float32[E, out // 32, in // 32]``)."""
        inter = intermediate_size_per_partition
        hidden = hidden_sizes
        param_dict: dict[str, Any] = {}
        param_dict["w13_scale"] = torch.empty(
            num_experts, (2 * inter) // W2_BLOCK_ROWS, hidden // W2_BLOCK_COLS, dtype=torch.float32
        )
        param_dict["w2_scale"] = torch.empty(
            num_experts, hidden // W2_BLOCK_ROWS, inter // W2_BLOCK_COLS, dtype=torch.float32
        )
        return param_dict

    def get_shared_expert_weight(
        self, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Always-on shared-expert packed W2 code params (single expert)."""
        inter = intermediate_size_per_partition
        hidden = hidden_sizes
        param_dict: dict[str, Any] = {}
        param_dict["shared_w13_codes"] = torch.empty(2 * inter, hidden // W2_CODES_PER_BYTE, dtype=torch.uint8)
        param_dict["shared_w2_codes"] = torch.empty(hidden, inter // W2_CODES_PER_BYTE, dtype=torch.uint8)
        return param_dict

    def get_shared_expert_dynamic_quant_param(
        self, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Always-on shared-expert per-``[32, 32]`` block scales (single expert)."""
        inter = intermediate_size_per_partition
        hidden = hidden_sizes
        param_dict: dict[str, Any] = {}
        param_dict["shared_w13_scale"] = torch.empty(
            (2 * inter) // W2_BLOCK_ROWS, hidden // W2_BLOCK_COLS, dtype=torch.float32
        )
        param_dict["shared_w2_scale"] = torch.empty(
            hidden // W2_BLOCK_ROWS, inter // W2_BLOCK_COLS, dtype=torch.float32
        )
        return param_dict

    # -- forward --------------------------------------------------------------

    def moe_forward(
        self,
        x: torch.Tensor,
        experts: list,
        router_logits: torch.Tensor,
        top_k: int,
        shared_expert: Any | None = None,
        *,
        renormalize: bool = True,
        cache: dict | None = None,
    ) -> torch.Tensor:
        """Router-logits host-math entrypoint: thin wrapper over E1.2.

        Delegates straight to :func:`w2_active_moe_forward` -- routes the batch,
        unpacks only the active experts (bounded cache), runs the grouped W2->INT8
        QDQ MoE, and adds the shared expert. This is the exact host re-expression
        of ``apply``'s device grouped matmul.
        """
        return w2_active_moe_forward(
            x, experts, router_logits, top_k, shared_expert, renormalize=renormalize, cache=cache
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fused W2 MoE forward over the router-selected experts.

        The routing (softmax -> top-k -> renormalize) is done upstream by the
        routed-experts layer, so ``apply`` receives the selected ``topk_ids`` /
        ``topk_weights``. The packed W2 expert bank and the optional shared
        expert are carried on ``layer`` (populated by the E3.4 loader).
        """
        experts = getattr(layer, "w2_experts", None)
        if experts is None:
            raise ValueError(
                "AscendW2DynamicFusedMoEMethod310.apply requires the packed W2 expert bank on "
                "`layer.w2_experts` (populated by the DeepSeek W2 loader, E3.4)."
            )
        shared_expert = getattr(layer, "w2_shared_expert", None)
        if _device_kernel_available():  # pragma: no cover - device-only wave (D1.5)
            return self._apply_device(experts, x, topk_weights, topk_ids, shared_expert)
        return self._apply_host(experts, x, topk_weights, topk_ids, shared_expert)

    def _apply_host(
        self,
        experts: list,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_expert: Any | None,
    ) -> torch.Tensor:
        """Host re-expression of the device grouped matmul via E1.2 primitives.

        Unpacks only the active experts, then runs one grouped INT8 QDQ MLP per
        active expert (fused ``w13`` -> SwiGLU -> ``w2``), scales by the router
        weight, and scatters back -- the same grouping the device kernel does,
        with :func:`w2_group_qdq_linear` in place of the INT8 grouped matmul.
        """
        x = x.double()
        num_tokens, hidden = x.shape
        top_k = topk_ids.shape[1]
        cache = unpack_active_experts(experts, topk_ids)

        pair_expert = topk_ids.reshape(-1)
        pair_weight = topk_weights.reshape(-1, 1).double()
        pair_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
        pair_x = x[pair_token]

        order = torch.argsort(pair_expert, stable=True)
        sorted_expert = pair_expert[order]
        sorted_x = pair_x[order]
        sorted_weight = pair_weight[order]
        sorted_token = pair_token[order]

        uniq_expert, counts = torch.unique_consecutive(sorted_expert, return_counts=True)
        out = torch.zeros(num_tokens, hidden, dtype=torch.float64, device=x.device)

        start = 0
        for expert_id, count in zip(uniq_expert.tolist(), counts.tolist()):
            stop = start + count
            weight = cache[expert_id]
            group_x = sorted_x[start:stop]
            gate_up = w2_group_qdq_linear(group_x, weight.w13_codes, weight.w13_scale)
            hidden_act = swiglu_gate_up(gate_up)
            y = w2_group_qdq_linear(hidden_act, weight.w2_codes, weight.w2_scale)
            y = y * sorted_weight[start:stop]
            out.index_add_(0, sorted_token[start:stop], y)
            start = stop

        if shared_expert is not None:
            out = out + shared_expert.forward(x)
        return out

    def _apply_device(
        self,
        experts: list,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_expert: Any | None,
    ) -> torch.Tensor:  # pragma: no cover - device-only wave (D1.5)
        """Device grouped matmul over the ``<= top_k`` active experts.

        Widens only the active experts from packed W2 (E1.2), then runs each
        active-expert group eagerly on the NPU: the packed 2-bit codes are
        dequantized to fp32 (``codes * per-block scale``) and matmul'd in fp32.

        The pinned CANN fused kernel
        (``torch_npu.npu_quant_grouped_matmul_dequant``) requires the quantized
        weight pre-tiled into a 5-D fractal-NZ layout ``(G, K//32, N//16, 16, 32)``
        that the W2 unpack does not yet emit (raised ``EZ1001`` /
        ``aclnnQuantGroupedMatmulDequant`` error ``161002`` on 310P); producing
        that exact layout is the D1.5 optimization. Until then this eager fp32
        path is the working device path -- same math as :meth:`_apply_host` but
        without the ``.double()`` (310P matmul supports only fp16/fp32, not fp64
        or bf16). The weight is reconstructed exactly; activations stay fp32
        (strictly >= the fused kernel's per-token INT8 activation quant).
        """
        num_tokens = x.shape[0]
        hidden = x.shape[1]
        top_k = topk_ids.shape[1]
        pair_expert = topk_ids.reshape(-1)
        pair_weight = topk_weights.reshape(-1, 1).to(torch.float32)
        pair_token = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
        pair_x = x[pair_token].to(torch.float32)

        # ArgSort has no int32/int64 AiCore kernel on 310P (falls back to AiCPU,
        # which dominated the eager MoE cost); sort on an fp32 key instead -- the
        # expert ids (< 2**24) are exact in fp32, so the ordering is identical.
        order = torch.argsort(pair_expert.to(torch.float32), stable=True)
        sorted_expert = pair_expert[order]
        sorted_x = pair_x[order]
        sorted_weight = pair_weight[order]
        sorted_token = pair_token[order]

        uniq_expert, counts = torch.unique_consecutive(sorted_expert, return_counts=True)
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)

        w2_op = _w2_blocked_mm_op()
        start = 0
        for expert_id, count in zip(uniq_expert.tolist(), counts.tolist()):
            stop = start + count
            e = experts[expert_id]
            group_x = sorted_x[start:stop]
            inter = int(e.inter)
            nvfp4 = _is_nvfp4(e.gate_scale, inter, hidden)
            # The Cube kernel unpacks signed W2/W4 codes on-chip. NVFP4 uses an
            # E2M1 floating-point codebook and must take its eager path below.
            use_cube = _can_use_w2_cube(
                w2_op,
                e.gate_packed,
                int(e.hidden),
                int(group_x.shape[0]),
                nvfp4,
            )
            if use_cube:
                # Fast path: fused 310P Cube kernel (arch20 catlass MMAD with the
                # per-[32,32] block dequant fused into the weight load). It
                # expands only a reusable 128-column tile directly into NZ,
                # avoiding a full fp16 weight and the ND-to-NZ matmul conversion.
                # Its Cube epilogue writes FP16 output directly. Packed width
                # selects signed W2 (four codes/byte) or W4 (two codes/byte)
                # on-chip.
                gx = group_x.to(torch.float16)
                gate = w2_op(gx, e.gate_packed, e.gate_scale.to(torch.float32))
                up = w2_op(gx, e.up_packed, e.up_scale.to(torch.float32))
                hidden_act = (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.float16)
                y = w2_op(hidden_act, e.down_packed, e.down_scale.to(torch.float32)).to(torch.float32)
            elif nvfp4:
                # NVFP4 (E2M1 float codes + block-16 scale) has no Cube kernel;
                # dequantize to fp32 and matmul natively, exact against the golden
                # GPU NVFP4 weights.
                gate_w = _nvfp4_dequant_fp32(e.gate_packed, e.gate_scale, inter, hidden)
                up_w = _nvfp4_dequant_fp32(e.up_packed, e.up_scale, inter, hidden)
                gate = torch.matmul(group_x, gate_w.t())
                up = torch.matmul(group_x, up_w.t())
                hidden_act = torch.nn.functional.silu(gate) * up
                down_w = _nvfp4_dequant_fp32(e.down_packed, e.down_scale, hidden, inter)
                y = torch.matmul(hidden_act, down_w.t())
            else:
                # Fallback: native-fp32 dequant from the packed bank (compact
                # [out//32,in//32] block scale via tiled view-multiply, no fp64),
                # then fp32 matmul. Correct but ~2.4x slower than the Cube kernel.
                gate_w = _w2_dequant_fp32(e.gate_packed, e.gate_scale, inter, hidden)
                up_w = _w2_dequant_fp32(e.up_packed, e.up_scale, inter, hidden)
                gate = torch.matmul(group_x, gate_w.t())
                up = torch.matmul(group_x, up_w.t())
                hidden_act = torch.nn.functional.silu(gate) * up
                down_w = _w2_dequant_fp32(e.down_packed, e.down_scale, hidden, inter)
                y = torch.matmul(hidden_act, down_w.t())
            y = y * sorted_weight[start:stop]
            out.index_add_(0, sorted_token[start:stop], y)
            start = stop

        if shared_expert is not None:
            out = out + shared_expert.forward(x).to(torch.float32)
        return out
