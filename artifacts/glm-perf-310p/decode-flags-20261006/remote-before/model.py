# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P GLM-5.3-Flash W2 model classes (G3 -- ADAPT of shipped glm5next).

This is an *adaptation* of the shipped, config-driven, ``torch_npu``-native
``glm5next`` model (``vllm_ascend.models.glm5next.model``), NOT a green-field
port. The shipped ``Glm5NextForCausalLM`` / ``Glm5NextModel`` /
``Glm5NextDecoderLayer`` already read the whole GLM-5.3-Flash *shape* (45
layers, hidden 4096, 288 routed experts top-8, ``first_k_dense_replace=3``,
``routed_scaling_factor=2.5``, HYBRID ``layer_types`` = 34 KDA linear-attn + 11
DSA sparse-attn ``full_attn_layers``, MTP-1) off the HF config, so most of the
geometry is *data*, not new code. This module therefore SUBCLASSES the shipped
``Glm5NextForCausalLM`` and, at this G3 gate, only:

* attaches the authoritative :data:`ASCEND_GLM5NEXT_W2_DTYPE_POLICY`;
* plumbs the GLM text config (nested under ``text_config`` in the multimodal
  wrapper) through to the shipped constructor;
* leaves the genuine 310P W2 deltas -- the KDA Triton->eager swap (G4), the DSA
  indexer 310P path (G5), and the MoE->W2 swap (G6) -- as clean, clearly-tagged
  hooks/TODOs.

Triton hygiene (grep-gate)
--------------------------
The shipped ``glm5next/model.py`` top-level pulls in ``FusedMoEFactory`` and,
via ``glm5next.kda``, the Triton KDA op at ``vllm_ascend.ops.triton.kda.kda``.
To keep *this* package's import path Triton-free on the 310P flag, the shipped
base class is resolved **lazily** (see :func:`_shipped_causal_lm_base` and the
module ``__getattr__`` below): merely importing ``glm5next_w2.model`` does not
pull Triton. The base is only resolved when the W2 class is actually built
(i.e. when vLLM instantiates the arch on device). G4 replaces the Triton KDA op
with stateful 310P AscendC conv/KDA operators, and G6 swaps ``FusedMoEFactory`` for
the E1.3 ``AscendW2DynamicFusedMoEMethod310``, at which point even that deferred
path is Triton-free.

This module itself contains ZERO ``triton`` references in code.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .dtype_policy import (
    ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
    Glm5NextW2DtypePolicy,
)

if TYPE_CHECKING:  # keep runtime imports light (no heavy vLLM / Triton pull)
    from vllm.config import VllmConfig


# ===========================================================================
# GLM-5.3-Flash config contract (data-driven; the shipped model reads these
# off config). Verified 2026-09-16 against the shipped glm5next config/patch
# and the source config at
# /run/media/matteius/20TB-drive/models/GLM-5.3-Flash-FP8/config.json.
# ===========================================================================

GLM5NEXT_NUM_HIDDEN_LAYERS = 45
GLM5NEXT_HIDDEN_SIZE = 4096
N_ROUTED_EXPERTS = 288
NUM_EXPERTS_PER_TOK = 8  # top-8
GLM_GROUPED_ROUTE_EXPERIMENT_LIMIT = 32768
FIRST_K_DENSE_REPLACE = 3  # first 3 layers are dense MLP, the rest MoE
N_SHARED_EXPERTS = 1
MOE_INTERMEDIATE_SIZE = 2048
ROUTED_SCALING_FACTOR = 2.5
VOCAB_SIZE = 154880
NUM_NEXTN_PREDICT_LAYERS = 1  # MTP-1

# HYBRID attention: layer_types alternates 3x linear_attention (KDA) then 1x
# deepseek_sparse_attention (DSA), repeating. The 11 DSA layers are the
# full_attn_layers; the other 34 are KDA (kda_layers).
FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)
KDA_LAYERS = tuple(i for i in range(GLM5NEXT_NUM_HIDDEN_LAYERS) if i not in FULL_ATTN_LAYERS)

# KDA (Kimi Delta Attention) linear-attention config (config.linear_attn_config).
KDA_NUM_HEADS = 64
KDA_HEAD_DIM = 128
KDA_SHORT_CONV_KERNEL_SIZE = 4


# KDA config keys on the HF text config (fall back to the frozen constants).
KDA_NUM_HEADS_CONFIG_KEY = "linear_num_heads"
KDA_HEAD_DIM_CONFIG_KEY = "linear_head_dim"
KDA_CONV_KERNEL_CONFIG_KEY = "linear_conv_kernel_dim"

# Virtual prefetch-offload parameter selecting the packed expert bank. Packed
# W2/W4 tensors are plain attributes rather than nn.Parameters, so upstream's
# module offloader cannot discover them through ``named_parameters()``.
PACKED_EXPERTS_OFFLOAD_PARAM = "packed_experts"
GLM_NZ_OUTPUT_TILE = 16
GLM_NZ_INPUT_TILE = 256


def _with_fp16_recurrent_state_dtype(
    state_dtypes: tuple[torch.dtype, ...],
) -> tuple[torch.dtype, ...]:
    """Keep the three conv-state dtypes and select FP16 for recurrent KDA."""
    if not state_dtypes:
        raise ValueError("KDA state dtype tuple must include a recurrent state")
    return (*state_dtypes[:-1], torch.float16)


# ===========================================================================
# fp8-MoE OOM prevention (the GLM-specific device delta -- see module docstring)
# ---------------------------------------------------------------------------
# The shipped ``Glm5NextMoE.__init__`` builds ``self.experts =
# FusedMoEFactory(...)`` which, via ``get_quant_method`` on the fp8 checkpoint,
# calls ``AscendFp8BlockFusedMoEMethod.create_weights`` and allocates ~1.13
# GiB/layer/rank of fp8 expert buffers. On 310P at TP4 that OOMs during
# ``super().__init__()`` -- BEFORE any override hook can run -- so, unlike the
# DeepSeek V4.1 template (which can attach ``mlp_w2`` alongside the fp8 bank),
# GLM must ensure the fp8 experts are *never allocated*.
#
# The least-invasive prevention (task option (b)) is to swap the shipped
# module's ``FusedMoEFactory`` reference for a weightless stub for the *duration
# of construction only*. The router ``gate`` (GateLinear) and the FP16
# ``shared_experts`` (Glm5NextMLP) are still built normally by
# ``Glm5NextMoE.__init__`` (both small); only the giant fp8 routed-expert bank
# is skipped. ``Glm5NextMoE`` identity is preserved, so the shipped
# ``isinstance(self.mlp, Glm5NextMoE)`` / ``_mlp_is_moe`` fast-path stays
# correct. ``_swap_moe_to_w2`` then binds each stub to the real
# :class:`~vllm_ascend.models.glm5next_w2.moe.Glm5NextW2MoE`, so the shipped
# ``Glm5NextMoE.forward`` (``self.experts(hidden_states=, router_logits=)``)
# transparently drives the W2 path and the fp8 ``create_weights`` is never
# reached. Option (a) (routing the MoE quant_type to ``W2A8_DYNAMIC`` via
# ``get_quant_type_for_layer``) was rejected: it is coupled to the checkpoint's
# ``quant_description`` and would also re-route the linear layers, whereas this
# stub is scoped to construction and touches nothing else.
# ===========================================================================


class _NoFp8FusedMoEExperts(nn.Module):
    """Weightless stand-in for the shipped ``FusedMoEFactory`` (fp8 experts).

    Constructed in place of ``FusedMoEFactory`` while the shipped
    ``Glm5NextMoE`` builds its decoder layers, so ``create_weights`` (the
    ~1.13 GiB/layer fp8 allocation) is never called. It allocates **no** expert
    parameters. After construction, :func:`_install_w2_moe` binds a
    :class:`~vllm_ascend.models.glm5next_w2.moe.Glm5NextW2MoE` via
    :meth:`bind_w2_delegate`; the shipped ``Glm5NextMoE.forward`` then calls
    this stub (``self.experts(hidden_states=, router_logits=)``) and the call is
    forwarded to the W2 routed-expert path.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        # The shipped factory is called with a ``shared_experts=`` kwarg (an
        # nn.Module already registered on the owning Glm5NextMoE) -- do NOT store
        # it here (that would double-register the submodule). We only remember
        # the routed geometry for diagnostics.
        self.num_experts = kwargs.get("num_experts")
        self.top_k = kwargs.get("top_k")
        # Bound by _install_w2_moe (kept as a plain attr, not a submodule, so the
        # W2 MoE is owned by ``layer.mlp_w2`` and appears once in the module tree).
        self._w2_delegate: Any = None

    def bind_w2_delegate(self, w2_moe: Any) -> None:
        """Route this stub's forward through the W2 routed-expert MoE."""
        object.__setattr__(self, "_w2_delegate", w2_moe)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, **_: object) -> torch.Tensor:
        delegate = self._w2_delegate
        if delegate is None:  # pragma: no cover - defensive (swap always runs)
            raise RuntimeError(
                "_NoFp8FusedMoEExperts.forward called before _swap_moe_to_w2 bound the "
                "Glm5NextW2MoE delegate (the fp8 experts were suppressed at construction)."
            )
        # The W2 combine returns promoted-accumulation precision; the shipped
        # forward expects main-dtype activations, so cast back to the input dtype.
        return delegate(hidden_states, router_logits).to(hidden_states.dtype)


@contextlib.contextmanager
def _suppress_fp8_expert_allocation(shipped_module: Any) -> Iterator[None]:
    """Temporarily replace ``shipped_module.FusedMoEFactory`` with the stub.

    Scoped to the shipped constructor call only; the original factory is always
    restored (even on error), so nothing else in the process is affected.
    """
    original = shipped_module.FusedMoEFactory
    shipped_module.FusedMoEFactory = _NoFp8FusedMoEExperts
    try:
        yield
    finally:
        shipped_module.FusedMoEFactory = original


# ===========================================================================
# Packed W2 expert bank (streamed from the artifact by :meth:`load_weights`)
# ===========================================================================


class _PackedW2Expert:
    """One routed expert holding the *already-packed* W2 tensors from disk.

    Unlike the ``W2Expert`` test reference (which re-quantizes float weights),
    this holder is filled directly from the artifact's packed ``_codes`` (uint8)
    and ``_scale`` (fp32) tensors by :meth:`AscendGlm5NextW2ForCausalLM.load_weights`.
    It exposes exactly the attributes the E1.3 method's active-expert unpack
    (:func:`~vllm_ascend.models.deepseek_v41.w2_unpack.unpack_active_experts`)
    reads: ``gate_packed`` / ``gate_scale`` / ``up_packed`` / ``up_scale`` /
    ``down_packed`` / ``down_scale`` plus ``hidden`` / ``inter``.
    """

    __slots__ = (
        "hidden",
        "inter",
        "offload_to_cpu",
        "gate_packed",
        "gate_scale",
        "up_packed",
        "up_scale",
        "down_packed",
        "down_scale",
    )

    def __init__(self, hidden: int, inter: int, *, offload_to_cpu: bool = False) -> None:
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.offload_to_cpu = bool(offload_to_cpu)
        self.gate_packed: torch.Tensor | None = None
        self.gate_scale: torch.Tensor | None = None
        self.up_packed: torch.Tensor | None = None
        self.up_scale: torch.Tensor | None = None
        self.down_packed: torch.Tensor | None = None
        self.down_scale: torch.Tensor | None = None


def _pack_codes_nz(codes: torch.Tensor, in_features: int) -> torch.Tensor:
    """Repack one canonical expert matrix into 16×256 Cube NZ tiles.

    Each byte combines values from separate contiguous quarters/halves of the
    final NZ tile. The integer codes and the total byte count are unchanged.
    """
    if codes.dtype != torch.uint8 or codes.ndim != 2 or not codes.is_contiguous():
        raise ValueError("NZ packing requires a contiguous uint8 [N, packedK] matrix")
    n, packed_k = codes.shape
    if packed_k == 0 or in_features % packed_k:
        raise ValueError("packed K must divide input features")
    codes_per_byte = in_features // packed_k
    if codes_per_byte not in (2, 4) or n % GLM_NZ_OUTPUT_TILE or in_features % GLM_NZ_INPUT_TILE:
        raise ValueError("NZ packing requires W2/W4 codes and 16×256 tiles")

    n_tiles = n // GLM_NZ_OUTPUT_TILE
    k_tiles = in_features // GLM_NZ_INPUT_TILE
    tile_bytes = GLM_NZ_OUTPUT_TILE * GLM_NZ_INPUT_TILE // codes_per_byte
    bits = 8 // codes_per_byte
    rows = codes.view(n_tiles, GLM_NZ_OUTPUT_TILE, k_tiles, GLM_NZ_INPUT_TILE // codes_per_byte)
    rows = rows.permute(0, 2, 1, 3)
    fields = torch.stack(
        [(rows >> (bits * field)) & ((1 << bits) - 1) for field in range(codes_per_byte)],
        dim=-1,
    )
    nz = fields.reshape(n_tiles, k_tiles, GLM_NZ_OUTPUT_TILE, GLM_NZ_INPUT_TILE)
    nz = nz.permute(0, 1, 3, 2).reshape(n_tiles, k_tiles, codes_per_byte, tile_bytes)
    packed = nz[:, :, 0, :].clone()
    for field in range(1, codes_per_byte):
        packed |= nz[:, :, field, :] << (bits * field)
    return packed.reshape(n, packed_k).contiguous()


def _pack_codes_nz_w3(codes: torch.Tensor, in_features: int) -> torch.Tensor:
    """Repack canonical W3 into plane-major 16×256 Cube NZ tiles.

    Each tile holds 512 little-endian groups of eight 3-bit codes. Its three
    byte planes are contiguous, so the device decodes each field with vector
    reads and writes the resulting signed values directly in NZ order.
    """
    if codes.dtype != torch.uint8 or codes.ndim != 2 or not codes.is_contiguous():
        raise ValueError("NZ W3 packing requires a contiguous uint8 [N, packedK] matrix")
    n, packed_k = codes.shape
    if n % GLM_NZ_OUTPUT_TILE or in_features % GLM_NZ_INPUT_TILE or packed_k * 8 != in_features * 3:
        raise ValueError("NZ W3 packing requires W3 codes and 16×256 tiles")

    n_tiles = n // GLM_NZ_OUTPUT_TILE
    k_tiles = in_features // GLM_NZ_INPUT_TILE
    groups = codes.view(n, in_features // 8, 3)
    byte0, byte1, byte2 = groups.unbind(dim=-1)
    unsigned = torch.stack(
        (
            byte0 & 7,
            (byte0 >> 3) & 7,
            ((byte0 >> 6) | (byte1 << 2)) & 7,
            (byte1 >> 1) & 7,
            (byte1 >> 4) & 7,
            ((byte1 >> 7) | (byte2 << 1)) & 7,
            (byte2 >> 2) & 7,
            byte2 >> 5,
        ),
        dim=-1,
    ).reshape(n, in_features)
    nz = unsigned.view(n_tiles, GLM_NZ_OUTPUT_TILE, k_tiles, GLM_NZ_INPUT_TILE)
    nz = nz.permute(0, 2, 3, 1).reshape(n_tiles, k_tiles, 8, 512)
    word = nz[:, :, 0, :].to(torch.int32)
    for field in range(1, 8):
        word |= nz[:, :, field, :].to(torch.int32) << (3 * field)
    planes = torch.stack([(word >> (8 * byte)) & 255 for byte in range(3)], dim=2)
    return planes.reshape(n, packed_k).to(torch.uint8).contiguous()


class _PackedW2ExpertBank(list[_PackedW2Expert]):
    """Per-expert views plus contiguous local projection banks.

    The streamed checkpoint names experts globally, so the list retains one
    lightweight slot per global expert. After loading, resident local tensors
    are compacted into six contiguous banks. Individual expert attributes then
    become views of those banks, preserving the established eager fallback
    while enabling one device-grouped projection with no host routing sync.
    Host-offloaded layers stay in their pinned per-expert representation so the
    existing selected-expert staging path remains bounded.
    """

    def __init__(
        self,
        hidden: int,
        inter: int,
        num_experts: int,
        *,
        local_expert_offset: int,
        num_local_experts: int,
        offload_to_cpu: bool,
        layer_key: str | None = None,
        nz_packed_codes: bool = False,
        prefill_route_histogram: bool = False,
        fused_route_combine: bool = False,
        empty_peer_rows: bool = False,
        grouped_max_routes: int | None = None,
        fp32_route_combine: bool = False,
        prefill_swiglu: bool = False,
        prefill_fp32_route_combine: bool = False,
    ) -> None:
        if grouped_max_routes is not None and (
            type(grouped_max_routes) is not int or not 0 < grouped_max_routes <= GLM_GROUPED_ROUTE_EXPERIMENT_LIMIT
        ):
            raise ValueError("ascend_glm_grouped_max_routes must be an integer in 1..32768 or None")
        super().__init__(_PackedW2Expert(hidden, inter, offload_to_cpu=offload_to_cpu) for _ in range(num_experts))
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.local_expert_offset = int(local_expert_offset)
        self.num_local_experts = int(num_local_experts)
        self.offload_to_cpu = bool(offload_to_cpu)
        self.layer_key = layer_key
        self.grouped_ready = False
        self.nz_packed_codes = bool(nz_packed_codes and not offload_to_cpu)
        self.prefill_route_histogram = bool(prefill_route_histogram and not offload_to_cpu)
        self.fused_route_combine = bool(fused_route_combine and not offload_to_cpu)
        self.empty_peer_rows = bool(empty_peer_rows and self.fused_route_combine)
        # Explicit opt-in must match the installed OPP's admission limit.
        # None inherits the quantization method's existing default.
        self.grouped_max_routes = grouped_max_routes
        if sum(map(bool, (fp32_route_combine, prefill_fp32_route_combine, fused_route_combine))) > 1:
            raise ValueError("choose only one of all-token FP32, prefill FP32, or CANN route combine")
        self.fp32_route_combine = bool(fp32_route_combine and not offload_to_cpu)
        self.prefill_fp32_route_combine = bool(prefill_fp32_route_combine and not offload_to_cpu)
        self.prefill_swiglu = bool(prefill_swiglu and not offload_to_cpu)

    def place_resident_tensor(
        self,
        expert_id: int,
        name: str,
        tensor: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> None:
        """Copy one checkpoint tensor directly into grouped storage.

        Overlay checkpoints may stream a stale base projection before the
        authoritative replacement. Equal-shape replacements overwrite their
        existing slice. A quantization-width change rebuilds that projection
        bank and clears its completion markers; finalization then verifies that
        every local expert received the replacement shape.
        """
        local = expert_id - self.local_expert_offset
        if not 0 <= local < self.num_local_experts:
            raise IndexError(f"expert {expert_id} is outside the resident expert range")

        if name.endswith("_packed"):
            in_features = self.inter if name == "down_packed" else self.hidden
            if self.nz_packed_codes:
                repack = _pack_codes_nz_w3 if tensor.shape[-1] * 8 == in_features * 3 else _pack_codes_nz
                tensor = repack(tensor.cpu(), in_features).view(torch.int8)

        # Gate and up consume the same activations. Store their rows in one
        # allocation so the grouped Cube operator can project both in one call.
        # The per-projection banks remain views for the eager fallback.
        fused = name.startswith(("gate_", "up_"))
        kind = name.rsplit("_", 1)[-1]
        bank_name = f"gate_up_{kind}_bank" if fused else f"{name}_bank"
        grouped = getattr(self, bank_name, None)
        expected_shape = (
            (self.num_local_experts, 2 * tensor.shape[0], *tensor.shape[1:])
            if fused
            else (self.num_local_experts, *tensor.shape)
        )
        if grouped is None:
            grouped = torch.empty(
                expected_shape,
                dtype=tensor.dtype,
                device=device,
            )
            setattr(self, bank_name, grouped)
        elif grouped.shape != expected_shape or grouped.dtype != tensor.dtype:
            old_device_type = grouped.device.type
            affected = (f"gate_{kind}", f"up_{kind}") if fused else (name,)
            for resident_expert in self[self.local_expert_offset : self.local_expert_offset + self.num_local_experts]:
                for projection in affected:
                    setattr(resident_expert, projection, None)
            for projection in affected:
                projection_bank = f"{projection}_bank"
                if projection_bank != bank_name and hasattr(self, projection_bank):
                    delattr(self, projection_bank)
            delattr(self, bank_name)
            del grouped
            gc.collect()
            if old_device_type == "npu":
                torch.npu.empty_cache()
            grouped = torch.empty(
                expected_shape,
                dtype=tensor.dtype,
                device=device,
            )
            setattr(self, bank_name, grouped)

        if fused:
            rows = tensor.shape[0]
            setattr(self, f"gate_{kind}_bank", grouped[:, :rows])
            setattr(self, f"up_{kind}_bank", grouped[:, rows:])
        expert = self[expert_id]
        destination = getattr(self, f"{name}_bank")[local] if fused else grouped[local]
        destination.copy_(tensor)
        setattr(expert, name, destination)

    def finalize_grouped_storage(self) -> None:
        """Compact resident local tensors without changing expert views."""
        if self.offload_to_cpu:
            return
        start = self.local_expert_offset
        stop = start + self.num_local_experts
        local_experts = self[start:stop]
        for name in _SLOT_KIND_TO_ATTR.values():
            tensors = [getattr(expert, name) for expert in local_experts]
            if any(tensor is None for tensor in tensors):
                raise ValueError(f"packed expert bank is incomplete: missing {name}")
            bank_name = f"{name}_bank"
            grouped = getattr(self, bank_name, None)
            if grouped is None:
                # Compatibility for callers that populate the lightweight
                # expert holders directly. The model loader uses
                # ``place_resident_tensor`` and never takes this allocating
                # fallback on NPU.
                grouped = torch.stack(tensors).contiguous()
                setattr(self, bank_name, grouped)
                for local, expert in enumerate(local_experts):
                    setattr(expert, name, grouped[local])
        self.grouped_ready = True


def _release_grouped_compaction_cache(banks: Iterable[_PackedW2ExpertBank]) -> None:
    """Return superseded expert allocations before HCCL initializes.

    ``finalize_grouped_storage`` replaces six independently loaded tensors per
    expert with views into contiguous projection banks.  The old allocations
    are dead, but the NPU caching allocator otherwise retains them through the
    subsequent lazy HCCL initialization.  Reclaim the cache once, after all
    model weights have loaded.  Host-only tests and offloaded banks do not need
    an accelerator cache operation.
    """
    has_resident_npu_bank = any(bank.grouped_ready and bank.gate_packed_bank.device.type == "npu" for bank in banks)
    if has_resident_npu_bank:
        gc.collect()
        torch.npu.empty_cache()


# (slot, kind) from the G6 weight map -> the packed-expert attribute it fills.
_SLOT_KIND_TO_ATTR = {
    ("w1", "codes"): "gate_packed",
    ("w1", "scale"): "gate_scale",
    ("w3", "codes"): "up_packed",
    ("w3", "scale"): "up_scale",
    ("w2", "codes"): "down_packed",
    ("w2", "scale"): "down_scale",
}


def _expert_geometry_from_config(config: Any) -> dict[str, int]:
    """Frozen W2 expert geometry read off the GLM text config (defaults = G3)."""
    num_layers = int(getattr(config, "num_hidden_layers", GLM5NEXT_NUM_HIDDEN_LAYERS))
    return {
        "hidden_size": int(getattr(config, "hidden_size", GLM5NEXT_HIDDEN_SIZE)),
        "moe_intermediate_size": int(getattr(config, "moe_intermediate_size", MOE_INTERMEDIATE_SIZE)),
        "n_routed_experts": int(getattr(config, "n_routed_experts", N_ROUTED_EXPERTS)),
        "num_hidden_layers": num_layers,
        "first_k_dense_replace": int(getattr(config, "first_k_dense_replace", FIRST_K_DENSE_REPLACE)),
        "num_nextn_predict_layers": int(getattr(config, "num_nextn_predict_layers", NUM_NEXTN_PREDICT_LAYERS)),
        "mtp_layer_index": int(getattr(config, "mtp_layer_index", num_layers)),
    }


def _new_packed_expert_bank(
    geometry: dict[str, int],
    *,
    offload_to_cpu: bool = False,
    layer_key: str | None = None,
) -> _PackedW2ExpertBank:
    """A fresh (unfilled) per-expert packed bank for one MoE layer."""
    hidden = geometry["hidden_size"]
    inter = geometry["moe_intermediate_size"]
    from .moe import _ep_rank_size, ep_expert_range

    num_experts = geometry["n_routed_experts"]
    ep_rank, ep_size = _ep_rank_size()
    lo, hi = ep_expert_range(ep_rank, ep_size, num_experts)
    return _PackedW2ExpertBank(
        hidden,
        inter,
        num_experts,
        local_expert_offset=lo,
        num_local_experts=hi - lo,
        offload_to_cpu=offload_to_cpu,
        layer_key=layer_key,
        nz_packed_codes=bool(geometry.get("nz_packed_codes", 0)),
        prefill_route_histogram=bool(geometry.get("prefill_route_histogram", 0)),
        fused_route_combine=bool(geometry.get("fused_route_combine", 0)),
        empty_peer_rows=bool(geometry.get("empty_peer_rows", 0)),
        grouped_max_routes=geometry.get("grouped_max_routes"),
        fp32_route_combine=bool(geometry.get("fp32_route_combine", False)),
        prefill_swiglu=bool(geometry.get("prefill_swiglu", False)),
        prefill_fp32_route_combine=bool(geometry.get("prefill_fp32_route_combine", False)),
    )


def _should_offload_packed_experts(offload_config: Any | None, layer_idx: int) -> bool:
    """Whether the existing prefetch pattern selects this packed expert layer.

    The feature is explicit: users include ``packed_experts`` in
    ``--offload-params``. An empty parameter set retains the established
    behavior of offloading registered parameters only, avoiding an unexpected
    host-memory increase for existing launch commands.
    """
    if offload_config is None:
        return False
    prefetch = getattr(offload_config, "prefetch", None)
    backend = getattr(offload_config, "offload_backend", "auto")
    if prefetch is None or backend not in ("auto", "prefetch"):
        return False
    group_size = int(getattr(prefetch, "offload_group_size", 0) or 0)
    num_in_group = int(getattr(prefetch, "offload_num_in_group", 0) or 0)
    offload_params = set(getattr(prefetch, "offload_params", set()) or set())
    if group_size <= 0 or PACKED_EXPERTS_OFFLOAD_PARAM not in offload_params:
        return False
    return layer_idx % group_size >= group_size - num_in_group


def _place_streamed_expert(
    layer_banks: dict[str, _PackedW2ExpertBank],
    name: str,
    tensor: torch.Tensor,
    geometry: dict[str, int],
) -> str:
    """Place one artifact expert tensor into its per-expert packed slot.

    ``layer_banks`` maps ``"layers.{L}"`` -> the layer's packed-expert list. The
    G6 :func:`~vllm_ascend.models.glm5next_w2.weight_mapping.map_expert_tensor`
    parses the name into ``(block, expert_id, slot, kind)`` (gate/up fuse into
    ``w13`` at forward time; here they stay split per proj). Returns the block key.
    """
    from .weight_mapping import map_expert_tensor

    mapping = map_expert_tensor(name, geometry)
    bank = layer_banks.get(mapping.block)
    if bank is None:
        # The stubbed MTP-1 draft head (layers.{mtp_layer_index}) is not
        # wired on the 310P W2 path, so no W2 bank is built for it; skip its
        # expert weights. Any OTHER missing bank is a real error.
        mtp_index = geometry.get("mtp_layer_index", geometry.get("num_hidden_layers", 0))
        num_mtp = int(geometry.get("num_nextn_predict_layers", 0) or 0)
        mtp_blocks = {f"layers.{mtp_index + m}" for m in range(num_mtp)}
        if mapping.block in mtp_blocks:
            return mapping.block
        raise KeyError(f"{name}: no W2 MoE bank for block {mapping.block!r} (dense/non-MoE layer?)")
    attr = _SLOT_KIND_TO_ATTR[(mapping.slot, mapping.kind)]
    # Expert-parallel: the 288 2-bit experts do not fit replicated on every 310P
    # chip, so each rank owns only a contiguous slice [lo, hi) (same tiling the
    # routed forward masks to -- see moe.ep_expert_range). Non-local experts are
    # neither moved to NPU nor stored: their bank slot stays the None-initialised
    # placeholder and is never indexed (the router selection is masked to local
    # ids), so they cost no HBM.
    from .moe import _ep_rank_size, ep_expert_range

    ep_rank, ep_size = _ep_rank_size()
    if ep_size > 1:
        lo, hi = ep_expert_range(ep_rank, ep_size, geometry["n_routed_experts"])
        if not (lo <= mapping.expert_id < hi):
            return mapping.block
    expert = bank[mapping.expert_id]
    if expert.offload_to_cpu:
        # Keep the packed bank on host and stage only router-selected experts at
        # runtime. Pinning makes the non-blocking H2D copies real when the
        # platform supports pinned host storage.
        tensor = tensor.cpu()
        from vllm.model_executor.offloader.base import should_pin_memory

        if should_pin_memory() and not tensor.is_pinned():
            tensor = tensor.pin_memory()
    else:
        # Populate the final contiguous allocation immediately. Moving each
        # expert to NPU first and stacking it after the checkpoint load leaves
        # both copies live at the compaction peak, exactly when a full-capacity
        # model has the least HBM headroom.
        bank.place_resident_tensor(
            mapping.expert_id,
            attr,
            tensor,
            device="npu",
        )
        return mapping.block
    setattr(expert, attr, tensor)
    return mapping.block


# ===========================================================================
# Override installers (the G4/G5/G6 hook bodies; module-level so they are
# CPU-testable against stand-in layers without full device construction)
# ===========================================================================


def _iter_model_layers(model: Any) -> list[Any]:
    """The backbone decoder layers (empty when the backbone is not built yet)."""
    return list(getattr(getattr(model, "model", None), "layers", []) or [])


def _is_kda_layer(layer: Any) -> bool:
    kind = getattr(layer, "layer_kind", None)
    if kind is not None:
        return kind == "kda"
    idx = getattr(layer, "layer_idx", None)
    return idx is not None and idx not in FULL_ATTN_LAYERS


def _is_dsa_layer(layer: Any) -> bool:
    kind = getattr(layer, "layer_kind", None)
    if kind is not None:
        return kind == "mla"
    idx = getattr(layer, "layer_idx", None)
    return idx is not None and idx in FULL_ATTN_LAYERS


def _layer_is_moe(layer: Any) -> bool:
    """A layer carries routed experts iff the shipped MoE fast-path flag is set."""
    if getattr(layer, "_mlp_is_moe", False):
        return True
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    return isinstance(experts, _NoFp8FusedMoEExperts)


def _install_w2_moe(
    layers: Iterable[Any],
    config: Any,
    dtype_policy: Glm5NextW2DtypePolicy,
    offload_config: Any | None = None,
    nz_packed_codes: bool = False,
) -> int:
    """G6: attach a ``Glm5NextW2MoE`` seam to every MoE layer + bind its forward.

    Mirrors the DeepSeek V4.1 ``_swap_moe_to_w2`` (one W2 MoE per routed layer,
    attached as ``layer.mlp_w2``), reusing the shipped router ``gate`` (with its
    ``e_score_correction_bias``) and the FP16 ``shared_experts``. The packed
    per-expert bank is allocated empty here and filled by :meth:`load_weights`.
    Additionally binds the suppressed stub's forward to the W2 MoE so the fp8
    path is fully bypassed.
    """
    from .moe import Glm5NextW2MoE

    geometry = _expert_geometry_from_config(config)
    geometry["nz_packed_codes"] = int(nz_packed_codes)
    geometry["prefill_route_histogram"] = int(getattr(config, "ascend_glm_prefill_route_histogram", False))
    geometry["fused_route_combine"] = int(getattr(config, "ascend_glm_fused_route_combine", False))
    geometry["empty_peer_rows"] = int(getattr(config, "ascend_glm_empty_peer_rows", False))
    grouped_max_routes = getattr(config, "ascend_glm_grouped_max_routes", None)
    geometry["fp32_route_combine"] = int(getattr(config, "ascend_glm_fp32_route_combine", False))
    geometry["prefill_swiglu"] = int(getattr(config, "ascend_glm_prefill_swiglu", False))
    geometry["prefill_fp32_route_combine"] = int(getattr(config, "ascend_glm_prefill_fp32_route_combine", False))
    if grouped_max_routes is not None:
        geometry["grouped_max_routes"] = grouped_max_routes
    count = 0
    for layer in layers:
        if not _layer_is_moe(layer):
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate", None)
        shared = getattr(mlp, "shared_experts", None)
        bias = getattr(gate, "e_score_correction_bias", None) if gate is not None else None
        w2_moe = Glm5NextW2MoE.from_config(
            config,
            shared_expert=(shared.forward if shared is not None else None),
            e_score_correction_bias=bias,
            dtype_policy=dtype_policy,
        )
        layer_idx = int(getattr(layer, "layer_idx", count))
        w2_moe.w2_experts = _new_packed_expert_bank(
            geometry,
            offload_to_cpu=_should_offload_packed_experts(offload_config, layer_idx),
            layer_key=f"layers.{layer_idx}",
        )
        layer.mlp_w2 = w2_moe
        experts = getattr(mlp, "experts", None)
        if isinstance(experts, _NoFp8FusedMoEExperts):
            experts.bind_w2_delegate(w2_moe)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Forward redirection: shipped self_attn call signature -> eager core
# ---------------------------------------------------------------------------
# The shipped ``Glm5NextDecoderLayer.forward`` calls
# ``self.self_attn(hidden_states=..., positions=...)`` on every layer (see the
# shipped model.py). On the KDA layers ``self.self_attn`` is a
# ``Glm5NextLinearAttention`` whose ``forward`` runs the Triton gated-delta
# recurrence (``_forward``) + Triton gated RMSNorm (``o_norm``) -- both of which
# raise ``TypeError: 'function' object is not subscriptable`` on 310P (Triton is
# unavailable, so ``@triton.jit`` kernels are plain functions). On the DSA layers
# ``self.self_attn`` is a ``Glm5NextMLAAttention`` whose device MLA kernels +
# CUDA-only ``SparseAttnIndexerKpool`` do not run on 310P either.
#
# We redirect by **overriding the bound instance's ``forward``** (the same
# philosophy as the MoE ``bind_w2_delegate`` seam) rather than replacing the
# ``self.self_attn`` object: keeping the shipped module in place preserves the
# checkpoint weight names (``...self_attn.in_proj_qkvbfg_a.weight`` etc.) that
# ``AutoWeightsLoader`` loads into, and preserves the layer's registration in the
# ``static_forward_context`` (so vLLM still sizes/allocates the hybrid KDA mamba
# state + DSA MLA-latent caches; we do NOT touch ``get_kv_cache_groups``). The
# override closes over the shipped module's already-loaded projections and drives
# the eager, Triton-free core, guaranteeing the Triton/device path is unreachable.


def _merged_kda_conv_weight(self_attn: Any) -> torch.Tensor:
    """Concatenated q|k|v depthwise conv weight ``[3*local_proj, conv_size]``.

    Mirrors the shipped ``Glm5NextLinearAttention._forward`` merged-conv build
    (``_w(m) = m.weight.view(out, width)`` over q/k/v conv1d), so the eager core
    runs the identical single merged causal conv.
    """

    def _w(module: Any) -> torch.Tensor:
        w = module.weight
        return w.view(w.size(0), w.size(-1))

    return torch.cat(
        [_w(self_attn.q_conv1d), _w(self_attn.k_conv1d), _w(self_attn.v_conv1d)],
        dim=0,
    ).contiguous()


def _rms_norm_gated_310(
    x: torch.Tensor,
    gate: torch.Tensor,
    norm: Any,
) -> torch.Tensor:
    """Apply KDA's output RMSNorm and gate without the Triton launcher.

    ``FusedRMSNormGated.forward_oot`` is backed by a Triton-style kernel.  On
    310P that decorated kernel is a plain Python function, so indexing it with
    a launch grid raises ``TypeError: 'function' object is not subscriptable``.
    Use torch-npu's native RMSNorm and retain the upstream activation order.
    """
    import torch_npu

    weight = getattr(norm, "weight", None)
    if weight is None:
        raise RuntimeError("GLM KDA output RMSNorm requires an affine weight")
    normalized, _ = torch_npu.npu_rms_norm(x, weight, float(norm.eps))
    bias = getattr(norm, "bias", None)
    if bias is not None:
        normalized = normalized + bias

    gate_fp32 = gate.float()
    activation = getattr(norm, "activation", "swish")
    if activation in ("silu", "swish"):
        activated_gate = gate_fp32 * torch.sigmoid(gate_fp32)
    elif activation == "sigmoid":
        activated_gate = torch.sigmoid(gate_fp32)
    else:
        raise ValueError(f"Unsupported KDA output gate activation: {activation}")
    return normalized * activated_gate.to(normalized.dtype)


def _project_kda_qkv_310(self_attn: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """Use a graph-safe one-group NZ projection when its weight is prepared."""
    packed_weight = getattr(self_attn, "_glm_kda_qkv_weight_nz", None)
    if packed_weight is None:
        return self_attn.in_proj_qkvbfg_a(hidden_states)[0]

    # Lazy import keeps CPU-only model assembly tests independent of torch_npu.
    import torch_npu

    num_tokens = hidden_states.shape[0]
    group_lists = self_attn._glm_kda_qkv_group_lists
    group_list = group_lists.get(num_tokens)
    if group_list is None:
        # Prefill row counts vary; decode graph sizes are populated at load.
        group_list = torch.tensor([num_tokens], dtype=torch.int64, device=hidden_states.device)
        group_lists[num_tokens] = group_list
    return torch_npu.npu_grouped_matmul(
        x=[hidden_states.contiguous()],
        weight=[packed_weight],
        group_list=group_list,
        split_item=2,
        group_type=0,
    )[0]


def _prepare_kda_qkv_nz_weights_310(layers: Iterable[Any], max_decode_seqs: int) -> int:
    """Replace each KDA input weight with a lossless transposed NZ view.

    Keeping the public parameter's logical [N,K] shape while retaining its
    transposed [K,N] NZ storage avoids a second ~1.7 GiB model copy at TP4.
    The grouped projection uses only the round-trip transposed view. Packing
    [N,K] directly as NZ and then transposing is *not* equivalent on 310P.
    """
    import torch_npu

    prepared = 0
    for layer in layers:
        if not _is_kda_layer(layer):
            continue
        self_attn = getattr(layer, "self_attn", None)
        projection = getattr(self_attn, "in_proj_qkvbfg_a", None)
        if projection is None:
            continue
        weight = projection.weight
        if weight.device.type != "npu" or weight.dtype != torch.float16:
            raise RuntimeError("GLM KDA grouped NZ projection requires a resident FP16 NPU weight")
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

        with torch.no_grad():
            transposed_nz = torch_npu.npu_format_cast(weight.data.T.contiguous(), ACL_FORMAT_FRACTAL_NZ)
            weight.data = transposed_nz.T
        self_attn._glm_kda_qkv_weight_nz = weight.data.T.unsqueeze(0)
        self_attn._glm_kda_qkv_group_lists = {
            count: torch.tensor([count], dtype=torch.int64, device=weight.device)
            for count in range(1, max_decode_seqs + 1)
        }
        prepared += 1
    return prepared


def _bind_eager_kda_forward(self_attn: Any, kda_core: Any, io_dtype: torch.dtype) -> None:
    """Override shipped KDA with the stateful 310P AscendC implementation.

    The closure reproduces the shipped forward's projection stage
    (``in_proj_qkvbfg_a`` -> split q|k|v / beta / f_a / g_a; ``f_b_proj(f_a)`` ->
    raw gate ``g1``; ``g_b_proj(g_a)`` -> output gate ``g2``) and then calls
    :class:`~vllm_ascend.models.glm5next_w2.kda.Glm5NextW2KDA` (conv -> safe gate
    -> gated-delta recurrence -> sigmoid-gated RMSNorm -> ``o_proj``) instead of
    the Triton ``_forward`` + Triton ``o_norm``. Returns the same
    ``[num_tokens, hidden]`` shape/dtype (fp16) the shipped forward returned.

    On NPU, prefill uses ``chunk_kda_fwd`` and decode/spec use the in-place 310P
    recurrent operator's per-key-channel ``gk`` lane.  Both consume vLLM's
    request metadata and update ``self_attn.kv_cache``.  The plain torch core is
    retained only as a CPU parity/profile fallback.
    """
    conv_cache: dict[str, torch.Tensor | None] = {"w": None, "w_310": None}

    def forward(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Projection stage: identical split to the shipped forward.
        projected = _project_kda_qkv_310(self_attn, hidden_states)
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self_attn.local_projection_size,
                self_attn.local_num_heads,
                self_attn.head_dim,
                self_attn.head_dim,
            ],
            dim=-1,
        )
        raw_g = self_attn.f_b_proj(f_a)[0]  # [T, local_num_heads*head_dim]
        g_out = self_attn.g_b_proj(g_a)[0]  # [T, local_num_heads*head_dim]

        if conv_cache["w"] is None:
            conv_cache["w"] = _merged_kda_conv_weight(self_attn)

        o_norm_weight = getattr(self_attn.o_norm, "weight", None)
        conv_bias = getattr(self_attn.q_conv1d, "bias", None)

        if qkv.device.type == "npu":
            from .kda_310 import run_stateful_kda_310

            if conv_cache["w_310"] is None:
                conv_cache["w_310"] = conv_cache["w"].transpose(0, 1).to(qkv.dtype).contiguous()
            core = run_stateful_kda_310(
                self_attn,
                qkv,
                raw_g.reshape(1, -1, self_attn.local_num_heads, self_attn.head_dim),
                beta_raw.unsqueeze(0),
                conv_cache["w_310"],
            )
            normalized = _rms_norm_gated_310(
                core,
                g_out.reshape(-1, self_attn.local_num_heads, self_attn.head_dim),
                self_attn.o_norm,
            )
            out = self_attn.o_proj(normalized.reshape(normalized.shape[1], -1))[0]
        else:
            out = kda_core(
                qkv,
                conv_weight=conv_cache["w"],
                raw_g=raw_g,
                beta_raw=beta_raw,
                a_log=self_attn.A_log,
                dt_bias=self_attn.dt_bias,
                g_out=g_out,
                o_norm_weight=o_norm_weight,
                conv_bias=conv_bias,
                initial_state=None,
                o_proj=lambda core: self_attn.o_proj(core)[0],
                output_dtype=io_dtype,
            )
        # Guarantee the shipped forward's dtype contract regardless of the
        # o_proj impl (the eager core returns the o_proj output verbatim).
        return out.to(io_dtype)

    # Instance-level override: nn.Module.__call__ dispatches to ``self.forward``,
    # so this shadows the shipped Triton ``forward`` without touching the class.
    self_attn.forward = forward


# (shipped MLA proj -> eager DSA param) name pairs used by the best-effort bind.
# Left = attribute path on the shipped ``Glm5NextMLAAttention`` (or a stand-in);
# right = the eager ``AscendGlm5NextW2DSA`` parameter it feeds. Only exact
# shape matches are copied; a mismatch (e.g. a decoupled-RoPE tail the NoPE eager
# core drops) is skipped and recorded, never fatal.
_DSA_WEIGHT_BINDINGS: tuple[tuple[str, str], ...] = (
    ("q_a_layernorm.weight", "q_a_norm"),
    ("kv_a_layernorm.weight", "kv_a_norm"),
    ("q_b_proj.weight", "w_uq"),
    ("kv_b_proj.weight", "w_ukv"),
    ("o_proj.weight", "w_o"),
    ("indexer.wq_b.weight", "indexer.wq_b"),
    ("indexer.wk_weights_proj.weight", "indexer.wk_weights_proj"),
    ("indexer.k_norm.weight", "indexer.k_norm_weight"),
    ("indexer.k_norm.bias", "indexer.k_norm_bias"),
    ("indexer.index_kpool_compress_ape", "indexer.compress_ape"),
    ("indexer.index_kpool_compress_gate", "indexer.compress_gate"),
)


def _resolve_attr(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _bind_shipped_mla_weights(self_attn: Any, dsa_core: Any) -> list[str]:
    """Best-effort storage binding of shipped MLA/indexer weights into eager DSA.

    Returns the list of eager params that could NOT be bound (shape mismatch or
    absent on the shipped module) so the caller can surface them. This is
    best-effort by design: it never raises, and any unbound param keeps its
    deterministic ``reset_parameters`` init. Bindings are shape-checked because the
    shipped GLM MLA keeps decoupled-RoPE rows (``qk_rope_head_dim``) that the NoPE
    eager core does not, so ``q_b_proj`` / fused down-projections may legitimately
    not match; those are reported for the hardware weight-adapter task. Matching
    tensors share storage instead of retaining an ~82 MiB duplicate per DSA layer.
    """
    from .dsa import share_parameter_storage_

    unbound: list[str] = []
    for src_path, dst_path in _DSA_WEIGHT_BINDINGS:
        src = _resolve_attr(self_attn, src_path)
        dst = _resolve_attr(dsa_core, dst_path)
        if src is None or dst is None or tuple(src.shape) != tuple(dst.shape):
            unbound.append(dst_path)
            continue
        share_parameter_storage_(dst, src)
    # Fused ``fused_qkv_a_proj`` -> (w_dq | w_dkv): split by the eager down dims.
    fused = _resolve_attr(self_attn, "fused_qkv_a_proj.weight")
    if fused is not None:
        q_rows = int(getattr(dsa_core, "q_lora_rank", 0))
        kv_rows = int(getattr(dsa_core, "kv_lora_rank", 0))
        if fused.shape[0] >= q_rows + kv_rows and dsa_core.w_dq.shape == (q_rows, fused.shape[1]):
            share_parameter_storage_(dsa_core.w_dq, fused[:q_rows])
            share_parameter_storage_(dsa_core.w_dkv, fused[q_rows : q_rows + kv_rows])
        else:
            unbound.extend(["w_dq", "w_dkv"])
    else:
        unbound.extend(["w_dq", "w_dkv"])
    return unbound


def _bind_eager_dsa_forward(self_attn: Any, dsa_core: Any, io_dtype: torch.dtype) -> None:
    """Override a shipped ``Glm5NextMLAAttention.forward`` with the eager DSA core.

    Routes ``self_attn(hidden_states, positions)`` to
    :class:`~vllm_ascend.models.glm5next_w2.dsa.AscendGlm5NextW2DSA` (kpool
    Lightning-indexer top-k selection + restricted NoPE MLA-latent attention),
    so the shipped device MLA kernels and the CUDA-only ``SparseAttnIndexerKpool``
    (which raises ``NotImplementedError`` on Ascend) are never entered. On first
    call it best-effort binds the shipped, checkpoint-loaded MLA/indexer
    projections into the eager params (:func:`_bind_shipped_mla_weights`).

    Prefill vs. decode-KV contract (verified vs. hardware-inferred)
    --------------------------------------------------------------
    Verified on CPU: the forward runs Triton-free and returns
    ``[num_tokens, hidden]`` in fp16, restricting attention to the indexer
    selection + local pool window over the tokens in THIS forward.

    TODO(hardware, decode/paged MLA-latent KV): the eager core attends only over
    the current forward's tokens (full self-attention among them) -- correct for
    a one-shot prefill, but it does NOT read/write the paged MLA-latent KV cache
    the shipped ``MultiHeadLatentAttentionWrapper`` registers, so multi-step
    decode (query attending to prior cached tokens) is not yet wired. The exact
    remaining contract: (a) write per-token compressed kv-latent
    (``kv_a_layernorm(w_dkv @ h)``) into the layer's MLA KV cache each step; (b)
    at decode, read the cached latents for positions < current and run the indexer
    top-k over the full history. Also, any eager param left unbound by
    :func:`_bind_shipped_mla_weights` (reported on first forward) needs the
    checkpoint weight adapter. Neither (a) nor (b) can be validated without NPU +
    a real KV plan, so DSA is the prefill/dummy path here.

    TP head sharding (fixed)
    ------------------------
    The eager DSA core now sizes its head-axis params (``w_uq`` / ``w_ukv`` /
    ``w_o``) to ``num_heads // tp_size`` and all-reduces the RowParallel ``o_proj``
    output (see :class:`~vllm_ascend.models.glm5next_w2.dsa.AscendGlm5NextW2DSA`).
    Because those shapes now match the shipped, checkpoint-loaded *sharded* MLA
    projections, :func:`_bind_shipped_mla_weights` binds them (previously they were
    full-size, so every bind silently shape-mismatched and the core ran on
    random-init weights). This removes the ~4x non-expert weight replication that
    OOMed HBM at TP4; the all-reduce placement / paged-KV wiring above still needs
    hardware validation.
    """
    state: dict[str, Any] = {"bound": False, "unbound": None}

    def forward(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not state["bound"]:
            state["bound"] = True
            try:
                state["unbound"] = _bind_shipped_mla_weights(self_attn, dsa_core)
            except Exception:  # pragma: no cover - defensive; binding is best-effort
                state["unbound"] = list(_DSA_WEIGHT_BINDINGS)
        out = dsa_core(hidden_states.to(dsa_core.param_dtype), positions)
        return out.to(io_dtype)

    self_attn.forward = forward


def _install_310p_kda(layers: Iterable[Any], config: Any, dtype_policy: Glm5NextW2DtypePolicy) -> int:
    """G4: route every KDA layer through stateful 310P AscendC operators.

    For each of the 34 KDA (``layer_kind == 'kda'``) layers this (1) attaches
    ``layer.kda_w2`` (a :class:`~vllm_ascend.models.glm5next_w2.kda.Glm5NextW2KDA`
    seam, kept for introspection/reporting) and (2) -- when the shipped
    ``Glm5NextLinearAttention`` is present -- overrides its ``forward`` so the
    layer actually runs the AscendC core, never the Triton recurrence. The
    attached eager core remains the CPU parity fallback and is sized to the
    shipped module's PER-TP-RANK head geometry
    (``local_num_heads`` / ``head_dim`` / ``conv_size``) so it is correct at TP>1;
    it falls back to the config globals for CPU stand-in layers (no shipped attn).
    """
    from .kda import Glm5NextW2KDA

    cfg_num_heads = int(getattr(config, KDA_NUM_HEADS_CONFIG_KEY, KDA_NUM_HEADS))
    cfg_head_dim = int(getattr(config, KDA_HEAD_DIM_CONFIG_KEY, KDA_HEAD_DIM))
    cfg_conv = int(getattr(config, KDA_CONV_KERNEL_CONFIG_KEY, KDA_SHORT_CONV_KERNEL_SIZE))
    io_dtype = dtype_policy.cast_site("kda")
    count = 0
    for layer in layers:
        if not _is_kda_layer(layer):
            continue
        self_attn = getattr(layer, "self_attn", None)
        # Per-rank geometry from the shipped module when available (TP-correct).
        num_heads = int(getattr(self_attn, "local_num_heads", cfg_num_heads))
        head_dim = int(getattr(self_attn, "head_dim", cfg_head_dim))
        conv = int(getattr(self_attn, "conv_size", cfg_conv))
        lower_bound = getattr(self_attn, "kda_lower_bound", None)
        kwargs: dict[str, Any] = dict(
            num_heads=num_heads,
            head_dim=head_dim,
            short_conv_kernel_size=conv,
            dtype_policy=dtype_policy,
        )
        if lower_bound is not None:
            kwargs["lower_bound"] = float(lower_bound)
        kda_core = Glm5NextW2KDA(**kwargs)
        layer.kda_w2 = kda_core
        # Redirect the forward only when the shipped projection stack is present
        # (the real device module). Pure CPU stand-ins without projections keep
        # the attached seam for introspection.
        if self_attn is not None and hasattr(self_attn, "in_proj_qkvbfg_a"):
            original_get_state_dtype = self_attn.get_state_dtype

            def get_state_dtype_310(
                original_get_state_dtype: Any = original_get_state_dtype,
            ) -> tuple[torch.dtype, ...]:
                return _with_fp16_recurrent_state_dtype(original_get_state_dtype())

            def get_attn_backend_310() -> type:
                from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
                    GlmW2GDNAttentionBackend310,
                )

                return GlmW2GDNAttentionBackend310

            self_attn.get_state_dtype = get_state_dtype_310
            self_attn.get_attn_backend = get_attn_backend_310
            _bind_eager_kda_forward(self_attn, kda_core, io_dtype)
        count += 1
    return count


def _install_dsa_indexer(layers: Iterable[Any], config: Any, dtype_policy: Glm5NextW2DtypePolicy) -> int:
    """Verify every DSA layer retained native paged MLA and kpool indexing."""
    del dtype_policy
    count = 0
    for layer in layers:
        if not _is_dsa_layer(layer):
            continue
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and not hasattr(self_attn, "mla_attn"):
            raise RuntimeError("GLM W2 DSA layer requires the native Ascend MLA wrapper")
        if self_attn is not None and getattr(self_attn, "indexer", None) is None:
            raise RuntimeError("GLM W2 DSA layer requires its checkpoint-declared kpool indexer")
        if getattr(config, "ascend_glm_mtp_full_graph", False):
            self_attn.indexer.indexer_op.capture_safe_selection = True
            self_attn.indexer.indexer_op.live_kpool_score = bool(getattr(config, "ascend_glm_live_kpool_score", False))
        count += 1
    return count


def _prepare_dsa_indexer_weights(layers: Iterable[Any]) -> None:
    """Materialize indexer FP32 projections after loading, before graph capture.

    The draft's first indexer call can occur inside capture. In particular,
    converting an NZ weight there may dispatch to the uncapturable ACL Cast.
    Refresh both copies on each load so a reload never leaves stale weights.
    """
    for layer in layers:
        indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
        if indexer is not None:
            indexer._wk_weight_f32 = indexer.wk_weights_proj.weight.detach().float()
            indexer._gate_weight_f32 = indexer.index_kpool_compress_gate.detach().float()


def _prepare_kda_gate_weights(layers: Iterable[Any]) -> None:
    # Loading is outside capture; these operands must outlive every graph.
    from .kda_310 import prepare_kda_gate_weights

    for layer in layers:
        if _is_kda_layer(layer):
            prepare_kda_gate_weights(layer.self_attn)


def _resolve_glm_text_config(vllm_config: VllmConfig) -> Any:
    """Return the effective GLM-5.3-Flash text config.

    GLM-5.3-Flash checkpoints nest the language-model fields under
    ``text_config`` (the multimodal wrapper; the top arch is
    ``Glm5NextForConditionalGeneration``). The shipped ``Glm5NextForCausalLM``
    reads its own text config, so this helper prefers the already-flattened
    ``hf_text_config`` when vLLM exposes it and otherwise unwraps a nested
    ``text_config``. Config plumbing only -- no geometry is re-implemented here.
    """
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is not None:
        return text_config
    hf_config = getattr(model_config, "hf_config", None)
    nested = getattr(hf_config, "text_config", None)
    return nested if nested is not None else hf_config


def _mark_dense_mlp_fp16(vllm_config: VllmConfig, text_config: Any) -> None:
    """Route the dense-MLP and shared-expert linears through the fp16 path.

    The GLM-5.3-Flash-W2 artifact stores every ``Glm5NextMLP`` projection --
    the dense MLP (``first_k_dense_replace`` layers) and the shared expert in
    each MoE layer -- as plain fp16 (``F16`` weight, no ``weight_scale_inv``),
    yet the checkpoint's ``quantization_config.modules_to_not_convert`` omits
    them. Left untouched, the generic ``fp8`` config (``AscendFp8Config``) hands
    these linears the block-wise fp8 scheme: it allocates a ``float8_e4m3fn``
    weight plus a ``weight_scale_inv`` that the checkpoint never fills, so
    ``process_weights_after_loading`` dequants ``weight * 0`` and EVERY
    dense/shared FFN outputs exactly 0.0 (the model then emits gibberish).

    Appending their fully-qualified module prefixes to the config's
    ``ignored_layers`` makes ``is_layer_skipped`` fall back to the unquantized
    linear method, so the fp16 checkpoint weights load losslessly. Both the
    fused ``gate_up_proj`` name and its ``gate_proj`` / ``up_proj`` shards are
    added: this model does not register ``gate_up_proj`` in
    ``packed_modules_mapping``, so ``is_layer_skipped`` matches the merged
    prefix by exact name rather than expanding to the shards. Only the
    dense/shared linears are affected; attention (already in
    ``modules_to_not_convert``) and the routed W2 experts (swapped out before
    quant selection) are untouched. Must run before the shipped constructor
    builds the layers.
    """
    quant_config = getattr(vllm_config, "quant_config", None)
    ignored = getattr(quant_config, "ignored_layers", None)
    if quant_config is None or ignored is None:
        return
    num_layers = int(getattr(text_config, "num_hidden_layers", GLM5NEXT_NUM_HIDDEN_LAYERS))
    num_mtp = int(getattr(text_config, "num_nextn_predict_layers", NUM_NEXTN_PREDICT_LAYERS) or 0)
    existing = set(ignored)
    # +num_mtp+1 covers the (currently unwired) MTP draft layer harmlessly;
    # an entry for a module a given layer lacks never matches a real prefix.
    for i in range(num_layers + num_mtp + 1):
        base = f"model.layers.{i}.mlp"
        for parent in (base, f"{base}.shared_experts"):
            for proj in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
                pfx = f"{parent}.{proj}"
                if pfx not in existing:
                    ignored.append(pfx)
                    existing.add(pfx)


def _reject_multimodal(vllm_config: VllmConfig) -> None:
    """First gate: the 310P GLM-5.3-Flash W2 path is text-only.

    GLM-5.3-Flash ships a vision tower (``model.visual.*`` /
    ``vision_config``); the W2 text path excludes it. Reject at construction so
    a ``Glm5NextForConditionalGeneration`` checkpoint cannot silently run
    degraded.
    """
    model_config = getattr(vllm_config, "model_config", None)
    mm_config = getattr(model_config, "multimodal_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    has_vision = getattr(hf_config, "vision_config", None) is not None
    if mm_config is not None or has_vision:
        raise NotImplementedError(
            "AscendGlm5NextW2ForConditionalGeneration: multimodal/vision inputs "
            "are not supported on the Ascend 310P GLM-5.3-Flash W2 path "
            "(text-only). GLM's model.visual.* tower is excluded. "
            "TODO(later): wire the GLM vision tower."
        )


# ===========================================================================
# Lazy base resolution (keeps the package import path Triton-free)
# ===========================================================================

_W2_CAUSAL_LM_CLS: type | None = None
_W2_COND_GEN_CLS: type | None = None


def _shipped_causal_lm_base() -> type:
    """Lazily import and return the shipped ``Glm5NextForCausalLM``.

    TODO(G4/G6): importing the shipped ``glm5next.model`` transitively pulls in
    ``FusedMoEFactory`` and, via ``glm5next.kda``, the Triton KDA op at
    ``vllm_ascend.ops.triton.kda.kda``. G4 swaps the Triton KDA recurrence for
    stateful 310P AscendC operators, and G6 swaps ``FusedMoEFactory`` for the E1.3
    ``AscendW2DynamicFusedMoEMethod310``, removing Triton from the 310P path.
    Until then the base is resolved only at build time (never at package
    import), so the grep-gate stays green.
    """
    from vllm_ascend.models.glm5next.model import Glm5NextForCausalLM

    return Glm5NextForCausalLM


def _build_causal_lm_cls() -> type:
    """Build (once) the GLM W2 causal-LM subclass of the shipped base."""
    global _W2_CAUSAL_LM_CLS
    if _W2_CAUSAL_LM_CLS is not None:
        return _W2_CAUSAL_LM_CLS

    base = _shipped_causal_lm_base()

    class AscendGlm5NextW2ForCausalLM(base):  # type: ignore[valid-type,misc]
        """Text-only Ascend GLM-5.3-Flash W2 causal LM (G3 ADAPT of glm5next).

        Subclasses the shipped ``Glm5NextForCausalLM`` and overrides only what
        the 310P W2 variant needs at this gate. The 45-layer / 288-expert /
        HYBRID-KDA+DSA / MTP-1 geometry flows through the shipped, config-driven
        constructor; the 310P W2 deltas are staged as hooks below.
        """

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            # Authoritative dtype policy (W2 experts / INT8 act, FP16
            # KDA/DSA/dense/shared/LM-head, FP32 accum). Every W2 submodule
            # reads this rather than spelling dtype literals.
            self.dtype_policy: Glm5NextW2DtypePolicy = Glm5NextW2DtypePolicy.from_vllm_config(vllm_config)
            self._w2_offload_config = getattr(vllm_config, "offload_config", None)
            # Config plumbing: expose the flattened GLM text config so the
            # shipped constructor sees 45L / 288-expert / KDA+DSA / MTP-1.
            self._glm_text_config = _resolve_glm_text_config(vllm_config)
            hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
            self._glm_text_config.ascend_glm_mtp_full_graph = bool(
                getattr(hf_config, "ascend_glm_mtp_full_graph", False)
            )
            self._nz_packed_codes = bool(getattr(hf_config, "ascend_glm_nz_packed_codes", False))
            self._kda_nz_grouped = bool(getattr(hf_config, "ascend_glm_kda_nz_grouped", False))
            self._kda_nz_max_decode_seqs = vllm_config.scheduler_config.max_num_seqs
            self._glm_text_config.ascend_glm_fused_sinkhorn = bool(
                getattr(hf_config, "ascend_glm_fused_sinkhorn", False)
            )
            self._glm_text_config.ascend_glm_mhc_fp16_state = bool(
                getattr(hf_config, "ascend_glm_mhc_fp16_state", False)
            )
            self._glm_text_config.ascend_glm_live_kpool_score = bool(
                getattr(hf_config, "ascend_glm_live_kpool_score", False)
            )
            self._glm_text_config.ascend_glm_native_mhc_post = bool(
                getattr(hf_config, "ascend_glm_native_mhc_post", False)
            )
            self._glm_text_config.ascend_glm_prefill_mhc_post = bool(
                getattr(hf_config, "ascend_glm_prefill_mhc_post", False)
            )
            self._glm_text_config.ascend_glm_mhc_batched_round = bool(
                getattr(hf_config, "ascend_glm_mhc_batched_round", False)
            )
            self._glm_text_config.ascend_glm_prefill_route_histogram = bool(
                getattr(hf_config, "ascend_glm_prefill_route_histogram", False)
            )
            self._glm_text_config.ascend_glm_fused_route_combine = bool(
                getattr(hf_config, "ascend_glm_fused_route_combine", False)
            )
            self._glm_text_config.ascend_glm_empty_peer_rows = bool(
                getattr(hf_config, "ascend_glm_empty_peer_rows", False)
            )
            self._glm_text_config.ascend_glm_grouped_max_routes = getattr(
                hf_config,
                "ascend_glm_grouped_max_routes",
                getattr(self._glm_text_config, "ascend_glm_grouped_max_routes", None),
            )
            self._glm_text_config.ascend_glm_prefill_swiglu = bool(
                getattr(hf_config, "ascend_glm_prefill_swiglu", False)
            )
            self._glm_text_config.ascend_glm_fp32_route_combine = bool(
                getattr(hf_config, "ascend_glm_fp32_route_combine", False)
            )
            self._glm_text_config.ascend_glm_prefill_fp32_route_combine = bool(
                getattr(hf_config, "ascend_glm_prefill_fp32_route_combine", False)
            )
            # FP16-in-checkpoint fix: the dense-MLP and shared-expert projections
            # ship as fp16 (no weight_scale_inv) but are absent from the
            # checkpoint's modules_to_not_convert. Mark them so they load through
            # the unquantized fp16 path instead of the fp8 scheme (whose unloaded
            # zero scale makes every Glm5NextMLP FFN output 0.0). Must run before
            # super().__init__ builds the layers.
            _mark_dense_mlp_fp16(vllm_config, self._glm_text_config)
            # CRITICAL (GLM device delta): suppress the shipped fp8 routed-expert
            # allocation for the duration of construction so the ~1.13 GiB/layer
            # fp8 bank is never created (it would OOM 310P/TP4 before any hook
            # runs). See ``_suppress_fp8_expert_allocation`` above.
            from vllm_ascend.models.glm5next import model as _shipped_glm

            # Keep GLM's checkpoint-declared index_topk during construction so
            # its kpool projection weights and paged caches are instantiated.
            self._native_dense_dsa = False
            with _suppress_fp8_expert_allocation(_shipped_glm):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            # 310P W2 delta hooks: KDA->AscendC (G4), DSA indexer (G5), MoE->W2 (G6).
            self._stage_w2_overrides()

        @classmethod
        def get_mamba_state_dtype_from_config(
            cls,
            vllm_config: VllmConfig,
        ) -> tuple[torch.dtype, ...]:
            """Use the state dtype required by the 310P recurrent KDA kernel."""
            return _with_fp16_recurrent_state_dtype(super().get_mamba_state_dtype_from_config(vllm_config))

        # -- 310P W2 delta hooks (clean seams for later tasks) -------------

        def _stage_w2_overrides(self) -> None:
            """Run the 310P W2 override hooks (KDA->AscendC, DSA indexer, MoE->W2)."""
            self._swap_kda_to_eager()  # G4
            self._override_dsa_indexer()  # G5
            self._swap_moe_to_w2()  # G6

        def _swap_kda_to_eager(self) -> None:
            """G4: route every KDA layer through stateful 310P KDA operators.

            Delegates to :func:`_install_310p_kda`, which attaches ``layer.kda_w2``
            as the CPU oracle and overrides the shipped attention forward on the
            34 ``KDA_LAYERS`` so the Triton recurrence is never entered.
            Component wired: ``glm5next_w2.kda`` (G4).
            """
            self._kda_swapped = _install_310p_kda(_iter_model_layers(self), self._glm_text_config, self.dtype_policy)

        def _override_dsa_indexer(self) -> None:
            """G5: keep native paged MLA and its sparse kpool indexer."""
            self._dsa_overridden = _install_dsa_indexer(
                _iter_model_layers(self), self._glm_text_config, self.dtype_policy
            )

        def _swap_moe_to_w2(self) -> None:
            """G6: swap each ``Glm5NextMoE`` routed path to the E1.3 W2 method.

            Delegates to :func:`_install_w2_moe`: builds one
            :class:`~vllm_ascend.models.glm5next_w2.moe.Glm5NextW2MoE` per routed
            layer (GLM sigmoid ``noaux_tc`` top-8 of 288 -> ``W2A8_DYNAMIC``/``moe``
            method -> eager ``routed*2.5 + FP16 shared`` combine), attaches it as
            ``layer.mlp_w2`` (reusing the shipped router ``gate`` bias + FP16
            ``shared_experts``), and binds the fp8-suppressed stub's forward to it
            so the fp8 path is fully bypassed. The packed per-expert bank is filled
            by :meth:`load_weights`. Component wired: ``glm5next_w2.moe`` (G6).
            """
            self._moe_swapped = _install_w2_moe(
                _iter_model_layers(self),
                self._glm_text_config,
                self.dtype_policy,
                self._w2_offload_config,
                self._nz_packed_codes,
            )

        # -- KV-cache report (hybrid KDA + DSA) ----------------------------
        #
        # The shipped ``Glm5NextForCausalLM`` implements ``IsHybrid`` +
        # ``get_mamba_state_{dtype,shape}_from_config``, so vLLM already builds
        # the hybrid KV plan (KDA linear-attn mamba state + DSA MLA-latent /
        # sparse-indexer caches). We therefore do NOT override
        # ``get_kv_cache_groups`` (that is the shipped base's job); we only add a
        # lean observability report mirroring the DeepSeek V4.1 ``kv_group_report``
        # and the GLM assembly's report.

        def kv_group_report(self) -> dict:
            """Semantic hybrid KV report: KDA linear-attn vs DSA full-attn layers."""
            layers = _iter_model_layers(self)
            kda = [getattr(layer, "layer_idx", i) for i, layer in enumerate(layers) if _is_kda_layer(layer)]
            dsa = [getattr(layer, "layer_idx", i) for i, layer in enumerate(layers) if _is_dsa_layer(layer)]
            return {
                "num_hidden_layers": len(layers),
                "kda_layers": kda,
                "dsa_layers": dsa,
                "num_kda_layers": len(kda),
                "num_dsa_layers": len(dsa),
                "kda_spec": "linear_attention(conv_state + recurrent_state)",
                "dsa_spec": "mla_latent + sparse_indexer_cache",
            }

        # -- weight load (G6: stream packed W2 experts) --------------------

        def _expert_geometry(self) -> dict:
            """Frozen W2 expert geometry from the GLM text config."""
            geometry = _expert_geometry_from_config(self._glm_text_config)
            geometry["nz_packed_codes"] = int(self._nz_packed_codes)
            return geometry

        def load_weights(self, weights: Iterable[tuple[Any, ...]]) -> set[str]:
            """Stream the checkpoint, routing packed W2 experts into the W2 banks.

            Routed-expert ``..._codes`` / ``..._scale`` tensors are placed into the
            per-layer packed banks attached by :meth:`_swap_moe_to_w2` (via the G6
            weight map, reusing the CPU assembly's streaming contract); the fp8
            expert path is never touched (it was never allocated). Everything else
            (FP16 attention / dense / shared-expert / router-gate / norms /
            ``lm_head`` / ``embed_tokens``, plus the ``hc_*`` hyper-connection
            params) is delegated to the shipped ``AutoWeightsLoader`` base; the GLM
            vision tower is excluded (text-only).
            """
            from .weight_mapping import WeightClass, classify_tensor

            geometry = self._expert_geometry()
            layer_banks: dict[str, _PackedW2ExpertBank] = {}
            bank_ids: set[int] = set()
            for layer in _iter_model_layers(self):
                w2_moe = getattr(layer, "mlp_w2", None)
                bank = getattr(w2_moe, "w2_experts", None) if w2_moe is not None else None
                if bank is not None:
                    layer_key = f"layers.{getattr(layer, 'layer_idx', len(layer_banks))}"
                    if layer_key in layer_banks:
                        raise ValueError(f"duplicate packed expert layer key: {layer_key}")
                    if id(bank) in bank_ids:
                        raise ValueError(f"packed expert bank is shared by multiple layers: {layer_key}")
                    if bank.layer_key != layer_key:
                        raise ValueError(
                            f"packed expert bank layer mismatch: attached as {layer_key}, created for {bank.layer_key}"
                        )
                    layer_banks[layer_key] = bank
                    bank_ids.add(id(bank))

            loaded: set[str] = set()
            passthrough: list[tuple[Any, ...]] = []
            # The unwired MTP-1 draft head lives at layers.{mtp_layer_index}
            # (=num_hidden_layers); skip ALL of its weights (experts AND the
            # shared_head/attn/norms) until MTP is wired.
            _mtp_index = geometry.get("mtp_layer_index", geometry.get("num_hidden_layers"))
            _num_mtp = int(geometry.get("num_nextn_predict_layers", 0) or 0)
            _mtp_markers = tuple(f".layers.{_mtp_index + m}." for m in range(_num_mtp))
            for args in weights:
                name = args[0]
                if _mtp_markers and any(mk in name for mk in _mtp_markers):
                    loaded.add(name)
                    continue
                if self._native_dense_dsa and ".indexer." in name:
                    # Native dense MLA has no sparse-indexer parameters. Keep
                    # those tensors in the checkpoint artifact, but do not
                    # hand nonexistent module names to the base loader.
                    continue
                cls = classify_tensor(name)
                if cls is WeightClass.EXCLUDE:
                    continue
                if cls is WeightClass.W2_EXPERT:
                    _place_streamed_expert(layer_banks, name, args[1], geometry)
                    loaded.add(name)
                    continue
                # The GLM checkpoint is a multimodal wrapper
                # (model.language_model.*); the text-only ForCausalLM expects
                # model.* param names. Strip the language_model wrapper for the
                # passthrough (FP16) weights (experts keep the wrapped name for
                # the G6 weight map).
                remapped = name.replace("model.language_model.", "model.", 1)
                passthrough.append((remapped, *tuple(args[1:])))
            banks = list(layer_banks.values())
            for bank in banks:
                bank.finalize_grouped_storage()
            loaded |= super().load_weights(passthrough)
            _prepare_kda_gate_weights(_iter_model_layers(self))
            _prepare_dsa_indexer_weights(_iter_model_layers(self))
            _release_grouped_compaction_cache(banks)
            if self._kda_nz_grouped:
                prepared = _prepare_kda_qkv_nz_weights_310(_iter_model_layers(self), self._kda_nz_max_decode_seqs)
                if prepared != self._kda_swapped:
                    raise RuntimeError(f"prepared {prepared} of {self._kda_swapped} GLM KDA NZ projections")
            return loaded

    AscendGlm5NextW2ForCausalLM.__module__ = __name__
    AscendGlm5NextW2ForCausalLM.__qualname__ = "AscendGlm5NextW2ForCausalLM"
    _W2_CAUSAL_LM_CLS = AscendGlm5NextW2ForCausalLM
    return AscendGlm5NextW2ForCausalLM


def _build_cond_gen_cls() -> type:
    """Build (once) the multimodal-rejecting GLM W2 alias."""
    global _W2_COND_GEN_CLS
    if _W2_COND_GEN_CLS is not None:
        return _W2_COND_GEN_CLS

    causal_lm = _build_causal_lm_cls()

    class AscendGlm5NextW2ForConditionalGeneration(causal_lm):  # type: ignore[valid-type,misc]
        """Multimodal-rejecting alias for the GLM-5.3-Flash W2 arch.

        Registered for ``Glm5NextW2ForConditionalGeneration`` so such
        checkpoints route here, then rejected at the first gate: the 310P W2
        path is text-only (GLM's ``model.visual.*`` tower is excluded).
        """

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            _reject_multimodal(vllm_config)  # first gate
            super().__init__(vllm_config=vllm_config, prefix=prefix)

        def get_multimodal_embeddings(self, *args: object, **kwargs: object):
            raise NotImplementedError(
                "AscendGlm5NextW2ForConditionalGeneration: multimodal embeddings "
                "are not supported on the Ascend 310P path (text-only)."
            )

        def embed_multimodal(self, *args: object, **kwargs: object):
            raise NotImplementedError(
                "AscendGlm5NextW2ForConditionalGeneration: multimodal inputs are "
                "not supported on the Ascend 310P path (text-only)."
            )

    AscendGlm5NextW2ForConditionalGeneration.__module__ = __name__
    AscendGlm5NextW2ForConditionalGeneration.__qualname__ = "AscendGlm5NextW2ForConditionalGeneration"
    _W2_COND_GEN_CLS = AscendGlm5NextW2ForConditionalGeneration
    return AscendGlm5NextW2ForConditionalGeneration


# ===========================================================================
# Packed MTP adapter: reuse the shipped predictor and packed expert implementation.
# ===========================================================================


class Glm5NextW2MTP(nn.Module):
    """Experimental packed draft layer with explicit checkpoint ownership.

    Heavy predictor imports remain constructor-local for CPU tooling. The
    target's KDA layers verify drafts; the draft itself uses the shipped MLA
    layer. Existing proposer code shares target embeddings and an absent head.
    """

    num_nextn_predict_layers = NUM_NEXTN_PREDICT_LAYERS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        from vllm_ascend import envs
        from vllm_ascend.models.glm5next import model as shipped
        from vllm_ascend.models.glm5next.mtp import Glm5NextMultiTokenPredictor

        from .mtp_config import validate_packed_glm_mtp

        validate_packed_glm_mtp(vllm_config)
        if envs.VLLM_ASCEND_310P_GLM_HOST_KV:
            raise ValueError("Packed GLM MTP requires resident MLA caches")
        self.config = _resolve_glm_text_config(vllm_config)
        if self.config.num_nextn_predict_layers != 1:
            raise ValueError("Packed GLM MTP initially supports exactly one prediction layer")
        self.quant_config = vllm_config.quant_config
        self.dtype_policy = Glm5NextW2DtypePolicy.from_vllm_config(vllm_config)
        self._nz_packed_codes = bool(getattr(self.config, "ascend_glm_nz_packed_codes", False))
        self.has_own_lm_head = False
        _mark_dense_mlp_fp16(vllm_config, self.config)
        with _suppress_fp8_expert_allocation(shipped):
            self.model = Glm5NextMultiTokenPredictor(
                vllm_config=vllm_config, prefix=f"{prefix}.model" if prefix else "model"
            )
        for layer in self.model.layers.values():
            layer.use_310p_eh_norm = True
        layers = [layer.mtp_block for layer in self.model.layers.values()]
        _install_dsa_indexer(layers, self.config, self.dtype_policy)
        installed = _install_w2_moe(layers, self.config, self.dtype_policy, None, self._nz_packed_codes)
        if installed != self.model.num_mtp_layers:
            raise ValueError("Every packed GLM MTP layer must have a routed expert bank")

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids, positions, hidden_states, intermediate_tensors=None, inputs_embeds=None, spec_step_idx=0
    ):
        return self.model(input_ids, positions, hidden_states, inputs_embeds, spec_step_idx)

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(self, hidden_states, spec_step_idx=0):
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def share_target_lm_head_if_identical(self, target_model):
        """Bind an absent head through the existing proposer sharing hook."""
        if self.has_own_lm_head:
            return False
        head = getattr(target_model, "lm_head", None)
        if head is None:
            raise ValueError("Packed GLM MTP checkpoint has no head and the target exposes no lm_head")
        for layer in self.model.layers.values():
            layer.shared_head.head = head
        return True

    def _rewrite_spec_layer_name(self, spec_layer, name):
        from vllm_ascend.models.glm5next.mtp import Glm5NextMTP

        return Glm5NextMTP._rewrite_spec_layer_name(self, spec_layer, name)

    def _maybe_set_own_lm_head(self, loaded_weights):
        from vllm_ascend.models.glm5next.mtp import Glm5NextMTP

        Glm5NextMTP._maybe_set_own_lm_head(self, loaded_weights)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm_ascend.models.glm5next.mtp import Glm5NextMTP

        from .weight_mapping import WeightClass, classify_tensor

        geometry = _expert_geometry_from_config(self.config)
        banks = {f"layers.{index}": layer.mtp_block.mlp_w2.w2_experts for index, layer in self.model.layers.items()}
        prefixes = tuple(f"model.language_model.{name}." for name in banks)
        packed_loaded: set[str] = set()
        dense_seen: set[str] = set()

        def draft_weights():
            for name, value in weights:
                # Source-style expert mapping is authoritative for W2/W3/W4.
                canonical = (
                    name.replace("model.layers.", "model.language_model.layers.", 1)
                    if name.startswith("model.layers.")
                    else name
                )
                if not canonical.startswith(prefixes):
                    continue
                if classify_tensor(canonical) is WeightClass.W2_EXPERT:
                    if canonical in packed_loaded:
                        raise ValueError(f"duplicate packed MTP tensor: {canonical}")
                    _place_streamed_expert(banks, canonical, value, geometry)
                    packed_loaded.add(canonical)
                else:
                    if canonical in dense_seen:
                        raise ValueError(f"duplicate MTP tensor: {canonical}")
                    dense_seen.add(canonical)
                    yield canonical, value

        loaded = Glm5NextMTP.load_weights(self, draft_weights())
        # Generic GLM checks layer presence; the packed adapter also requires
        # every allocated non-shared parameter and both halves of fused MLPs.
        missing = {
            name
            for name, _ in self.named_parameters()
            if name not in loaded
            and name != "model.embed_tokens.weight"
            and not name.endswith(".shared_head.head.weight")
        }
        for prefix in prefixes:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}mlp.shared_experts.{projection}.weight"
                if name not in dense_seen:
                    missing.add(name)
        if missing:
            raise ValueError(f"Incomplete packed GLM MTP weights: {sorted(missing)}")
        for bank in banks.values():
            bank.finalize_grouped_storage()
        # Drop uninitialized shared allocations before the generic loader's
        # coverage check. The proposer binds the actual target modules before
        # drafting; an absent checkpoint head must never be used for logits.
        self.model.embed_tokens = None
        if not self.has_own_lm_head:
            for layer in self.model.layers.values():
                layer.shared_head.head = None
        _release_grouped_compaction_cache(banks.values())
        _prepare_dsa_indexer_weights(layer.mtp_block for layer in self.model.layers.values())
        return loaded | packed_loaded


# ===========================================================================
# Lazy attribute access -- materialize the heavy subclasses only on demand,
# so `import vllm_ascend.models.glm5next_w2.model` stays Triton-free (G3).
# ===========================================================================

_LAZY_BUILDERS = {
    "AscendGlm5NextW2ForCausalLM": _build_causal_lm_cls,
    "AscendGlm5NextW2ForConditionalGeneration": _build_cond_gen_cls,
}


def __getattr__(name: str):  # PEP 562 module-level lazy attribute
    builder = _LAZY_BUILDERS.get(name)
    if builder is not None:
        return builder()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_BUILDERS.keys()])


# Keep a reference so linters don't flag the authoritative singleton import as
# unused; downstream modules import it directly from dtype_policy.
_DEFAULT_DTYPE_POLICY = ASCEND_GLM5NEXT_W2_DTYPE_POLICY

# NOTE: ``AscendGlm5NextW2ForCausalLM`` and
# ``AscendGlm5NextW2ForConditionalGeneration`` are provided lazily via the
# module ``__getattr__`` (they subclass the shipped, Triton-pulling
# ``glm5next`` base only on first access), so they are intentionally not in
# ``__all__`` -- resolve them by attribute access or the ModelRegistry arch name.
__all__ = [
    "Glm5NextW2MTP",
]
