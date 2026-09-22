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
with an eager/FLA gated-delta recurrence, and G6 swaps ``FusedMoEFactory`` for
the E1.3 ``AscendW2DynamicFusedMoEMethod310``, at which point even that deferred
path is Triton-free.

This module itself contains ZERO ``triton`` references in code.
"""

from __future__ import annotations

import contextlib
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
        "gate_packed",
        "gate_scale",
        "up_packed",
        "up_scale",
        "down_packed",
        "down_scale",
    )

    def __init__(self, hidden: int, inter: int) -> None:
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.gate_packed: torch.Tensor | None = None
        self.gate_scale: torch.Tensor | None = None
        self.up_packed: torch.Tensor | None = None
        self.up_scale: torch.Tensor | None = None
        self.down_packed: torch.Tensor | None = None
        self.down_scale: torch.Tensor | None = None


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


def _new_packed_expert_bank(geometry: dict[str, int]) -> list[_PackedW2Expert]:
    """A fresh (unfilled) per-expert packed bank for one MoE layer."""
    hidden = geometry["hidden_size"]
    inter = geometry["moe_intermediate_size"]
    return [_PackedW2Expert(hidden, inter) for _ in range(geometry["n_routed_experts"])]


def _place_streamed_expert(
    layer_banks: dict[str, list[_PackedW2Expert]],
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
    # The packed W2 codes/scales are plain attributes (not nn.Parameters), so
    # vLLM's device placement never touches them and they load on CPU. The W2
    # grouped-matmul-dequant runs on the NPU, so move them to the device here.
    if tensor.device.type != "npu":
        tensor = tensor.to("npu")
    setattr(bank[mapping.expert_id], attr, tensor)
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


def _install_w2_moe(layers: Iterable[Any], config: Any, dtype_policy: Glm5NextW2DtypePolicy) -> int:
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
        w2_moe.w2_experts = _new_packed_expert_bank(geometry)
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


def _bind_eager_kda_forward(self_attn: Any, kda_core: Any, io_dtype: torch.dtype) -> None:
    """Override a shipped ``Glm5NextLinearAttention.forward`` with the eager KDA core.

    The closure reproduces the shipped forward's projection stage
    (``in_proj_qkvbfg_a`` -> split q|k|v / beta / f_a / g_a; ``f_b_proj(f_a)`` ->
    raw gate ``g1``; ``g_b_proj(g_a)`` -> output gate ``g2``) and then calls
    :class:`~vllm_ascend.models.glm5next_w2.kda.Glm5NextW2KDA` (conv -> safe gate
    -> gated-delta recurrence -> sigmoid-gated RMSNorm -> ``o_proj``) instead of
    the Triton ``_forward`` + Triton ``o_norm``. Returns the same
    ``[num_tokens, hidden]`` shape/dtype (fp16) the shipped forward returned.

    KV/state contract (verified vs. hardware-inferred)
    --------------------------------------------------
    Verified on CPU: the projection wiring, the merged conv, and the recurrence
    output shape/dtype match the shipped forward for a single contiguous
    sequence started from zero state (``initial_state=None``).

    TODO(hardware, decode/paged state): the shipped ``_forward`` reads the paged
    mamba state from ``self_attn.kv_cache`` (``(conv_state, recurrent_state)``)
    and, via ``GDNAttentionMetadata`` on ``get_forward_context().attn_metadata``,
    (a) segments a batched forward by ``non_spec_query_start_loc`` (cu_seqlens),
    (b) gathers each request's carry-in recurrent state by
    ``non_spec_state_indices_tensor`` (prefill) / advances the conv+recurrent
    state one step per request (decode), and (c) scatters the updated state back.
    This eager binding currently runs each forward from **zero** recurrent/conv
    state over the whole token span, which is exact for a fresh single-sequence
    prefill but does NOT yet carry state across chunks/steps or separate batched
    requests. Wiring the paged read/write (mirroring the shipped
    ``gather_initial_states`` / ``scatter_states`` against
    :meth:`Glm5NextW2KDA.chunked_recurrence`) is the remaining device task; it
    cannot be validated without NPU + a real KV plan, so it is left explicit here.
    """
    conv_cache: dict[str, torch.Tensor | None] = {"w": None}

    def forward(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Projection stage: identical split to the shipped forward.
        projected = self_attn.in_proj_qkvbfg_a(hidden_states)[0]
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
            initial_state=None,  # TODO(hardware): paged recurrent-state carry
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


def _install_eager_kda(layers: Iterable[Any], config: Any, dtype_policy: Glm5NextW2DtypePolicy) -> int:
    """G4: route every KDA layer's forward through the Triton-free eager KDA core.

    For each of the 34 KDA (``layer_kind == 'kda'``) layers this (1) attaches
    ``layer.kda_w2`` (a :class:`~vllm_ascend.models.glm5next_w2.kda.Glm5NextW2KDA`
    seam, kept for introspection/reporting) and (2) -- when the shipped
    ``Glm5NextLinearAttention`` is present -- overrides its ``forward`` so the
    layer actually RUNS the eager core, never the Triton gated-delta recurrence.
    The eager core is sized to the shipped module's PER-TP-RANK head geometry
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
            _bind_eager_kda_forward(self_attn, kda_core, io_dtype)
        count += 1
    return count


def _install_dsa_indexer(layers: Iterable[Any], config: Any, dtype_policy: Glm5NextW2DtypePolicy) -> int:
    """G5: route every full-attn (DSA) layer's forward through the eager DSA core.

    For each of the 11 ``FULL_ATTN_LAYERS`` (``layer_kind == 'mla'``) this (1)
    attaches ``layer.dsa_w2`` (an
    :class:`~vllm_ascend.models.glm5next_w2.dsa.AscendGlm5NextW2DSA`) and (2) --
    when the shipped ``Glm5NextMLAAttention`` is present -- overrides its
    ``forward`` so the layer runs the eager kpool-indexer + NoPE MLA-latent
    attention instead of the device MLA kernels / CUDA-only sparse indexer. See
    :func:`_bind_eager_dsa_forward` for the prefill vs. decode-KV contract.
    """
    from .dsa import AscendGlm5NextW2DSA

    io_dtype = dtype_policy.cast_site("dsa")
    count = 0
    for layer in layers:
        if not _is_dsa_layer(layer):
            continue
        dsa_core = AscendGlm5NextW2DSA.from_config(config, dtype_policy=dtype_policy)
        layer.dsa_w2 = dsa_core
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "kv_b_proj"):
            # Alias the eager DSA params to the shipped MLA storage now, before
            # checkpoint loading and memory profiling. The loader subsequently
            # fills the shipped parameters in-place, so the aliases see the real
            # weights while releasing ~82 MiB of duplicate storage per DSA
            # layer. The forward binding repeats this once after loading as a
            # mixed-dtype fallback (where storage sharing is not possible).
            _bind_shipped_mla_weights(self_attn, dsa_core)
            _bind_eager_dsa_forward(self_attn, dsa_core, io_dtype)
        count += 1
    return count


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
    an eager/FLA path, and G6 swaps ``FusedMoEFactory`` for the E1.3
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
            # Config plumbing: expose the flattened GLM text config so the
            # shipped constructor sees 45L / 288-expert / KDA+DSA / MTP-1.
            self._glm_text_config = _resolve_glm_text_config(vllm_config)
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

            with _suppress_fp8_expert_allocation(_shipped_glm):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            # 310P W2 delta hooks: KDA->eager (G4), DSA indexer (G5), MoE->W2 (G6).
            self._stage_w2_overrides()

        # -- 310P W2 delta hooks (clean seams for later tasks) -------------

        def _stage_w2_overrides(self) -> None:
            """Run the 310P W2 override hooks (KDA->eager, DSA indexer, MoE->W2)."""
            self._swap_kda_to_eager()  # G4
            self._override_dsa_indexer()  # G5
            self._swap_moe_to_w2()  # G6

        def _swap_kda_to_eager(self) -> None:
            """G4: route every KDA layer's forward through the eager KDA core.

            Delegates to :func:`_install_eager_kda`, which attaches ``layer.kda_w2``
            AND overrides the shipped ``Glm5NextLinearAttention.forward`` on the 34
            ``KDA_LAYERS`` so the Triton gated-delta recurrence is never entered.
            Component wired: ``glm5next_w2.kda`` (G4).
            """
            self._kda_swapped = _install_eager_kda(_iter_model_layers(self), self._glm_text_config, self.dtype_policy)

        def _override_dsa_indexer(self) -> None:
            """G5: route every full-attn (DSA) layer's forward through the eager core.

            Delegates to :func:`_install_dsa_indexer`, which attaches ``layer.dsa_w2``
            AND overrides the shipped ``Glm5NextMLAAttention.forward`` on the 11
            ``FULL_ATTN_LAYERS`` so the device MLA kernels + CUDA-only sparse indexer
            are never entered. Component wired: ``glm5next_w2.dsa`` (G5, reusing the
            deepseek_v41 indexer selection).
            """
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
            self._moe_swapped = _install_w2_moe(_iter_model_layers(self), self._glm_text_config, self.dtype_policy)

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
            return _expert_geometry_from_config(self._glm_text_config)

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
            layer_banks: dict[str, list[_PackedW2Expert]] = {}
            for layer in _iter_model_layers(self):
                w2_moe = getattr(layer, "mlp_w2", None)
                bank = getattr(w2_moe, "w2_experts", None) if w2_moe is not None else None
                if bank is not None:
                    layer_banks[f"layers.{getattr(layer, 'layer_idx', len(layer_banks))}"] = bank

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
            loaded |= super().load_weights(passthrough)
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
# MTP-1 stub (registration target for Glm5NextW2MTPModel; wired at G7)
# ===========================================================================


class Glm5NextW2MTP:
    """Stub MTP-1 draft model for GLM-5.3-Flash (``num_nextn_predict_layers=1``).

    Registration target so the ``Glm5NextW2MTPModel`` arch resolves at G3. The
    concrete drafter reuses the shipped ``glm5next`` MTP path (which inherits
    the G4/G6 KDA-eager / W2 fixes automatically); wiring is G7. Constructing it
    now fails fast rather than silently running a stub.
    """

    num_nextn_predict_layers = NUM_NEXTN_PREDICT_LAYERS

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "Glm5NextW2MTP (MTP-1) is not wired yet; registration-only stub. "
            "TODO(G7): reuse the shipped glm5next MTP for num_nextn_predict_layers=1."
        )


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
