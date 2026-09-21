# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P Qwen4Exp (Qwen3.8-Flash-Next) full-model assembly (plan T1.5).

This module owns the importable, registerable package skeleton (T1.2) and, from
T1.5, the *full-model assembly* that ties every component together for a
dummy-weight CPU boot:

* the 48-layer decoder stack with the correct per-layer type (GDN linear
  attention / QSA sparse attention / dense full attention);
* the PLE injection layer at its ``ple_layer_ids`` position;
* the MoE block (routed experts + shared expert) or dense MLP;
* the gated-residual / hyperconnection multi-stream wiring;
* ``ParallelLMHead`` / embedding tie, final norm, logits + sampling seam;
* end-to-end KV-cache spec materialization (T1.4).

Wiring policy (what is REAL vs. eager-stubbed)
----------------------------------------------
The assembly uses the CORRECT wired components where they run host-side today:

* **GDN** (``qwen4exp_gdn`` -- T5.x): ``gdn_gating`` + ``gdn_delta_rule`` +
  ``gdn_short_conv`` on the ``eager`` backend.
* **QSA** (``qsa`` / ``indexer_qsa`` / ``ops.*`` -- T6.x): the real weight-free
  indexer feeds the real sparse GQA attention.
* **PLE** (``ple_layer`` -- T4.x): the real gather -> project -> gate ->
  short-conv layer, fed by a host-safe pinned embedding method
  (``ngram_embedding`` -- T4.1) with injected CPU allocator seams.
* **KV specs** (``kv_cache`` -- T1.4): full-attention / GDN / QSA ring +
  compressed groups.
* **state** (``model_state`` -- T1.3): returned by :meth:`get_model_state_cls`.

Pure-eager reference math stands in for internals not yet wired, each tagged:

* **hyperconnection / gated residual** -- eager multi-stream mix/combine
  (no dedicated Ascend component; ported from the fork ``GatedResidual`` math).
* **MoE routed + shared experts** -- eager top-k FFN (real W8A8 fused-expert
  path is TODO(T3.x), checkpoint-blocked).
* **dense full attention** -- eager causal GQA (used only when a layer is
  ``full_attention`` without an indexer config).
* **n-gram id hashing** -- real SplitMix64 hashing via
  ``AscendQwen4ExpNGramEmbedding.compute_ngram_ids`` (T4.2-verified); ids feed the
  real PLE row gather (bounded to a synthetic table on the host boot path).

Every dtype is read from the authoritative :mod:`dtype_policy`. No Triton/CUDA
module is imported on the 310P path (the GDN ``eager`` backend and the
torch-only QSA/PLE ops never pull ``fla`` / Triton).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFunc,
        MambaStateCopyFuncsByType,
    )
    from vllm.v1.attention.backend import AttentionBackend
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    maybe_prefix,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
)

from .dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    Qwen4ExpDtypePolicy,
)
from .indexer_qsa import AscendQwen4ExpQSAIndexer
from .kv_cache import (
    DEFAULT_ATTENTION_BLOCK_SIZE,
    build_qwen4exp_kv_cache_groups,
    make_qsa_compressed_spec,
    make_qsa_raw_ring_spec,
)
from .moe import (
    route_topk,
    swiglu_gate_up,
    w8a8_grouped_experts,
)
from .ngram_embedding import (
    AscendPLELazyShardEmbeddingMethod,
    AscendPLEPinnedHostEmbeddingMethod,
    AscendQwen4ExpNGramEmbedding,
)
from .ple_layer import AscendQwen4ExpPLELayer
from .qsa import (
    AscendQwen4ExpQSAAttention,
    QSADecoderProjections,
    run_qsa_decoder_attention,
)
from .qwen4exp_gdn import (
    QWEN4EXP_GDN_CHUNK_SIZE,
    Qwen4ExpGDNParams,
    gdn_delta_rule,
    gdn_gating,
    gdn_short_conv,
)
from .weight_mapping import (
    TensorDtypeError,
    TensorShapeError,
    WeightMappingError,
    expert_tensor_is_local,
    is_expert_tensor_name,
    map_expert_tensor,
    validate_expert_weight_map,
)

# ``VllmConfig`` is only needed for typing; keep import light.
try:  # pragma: no cover - trivial import guard
    from vllm.config import VllmConfig
except Exception:  # pragma: no cover
    VllmConfig = object  # type: ignore[assignment, misc]

# Layer-type tags mirroring the HF Qwen4Exp ``layer_types`` vocabulary.
_LAYER_TYPE_LINEAR = "linear_attention"
_LAYER_TYPE_FULL = "full_attention"


def _resolve_checkpoint_dir(vllm_config: object) -> str | None:
    """Local filesystem directory holding this model's checkpoint (or ``None``).

    The lazy-shard PLE transport reads the n-gram table straight from the
    checkpoint's safetensors files, so it needs the on-disk directory. Resolve it
    from the model path (``model_config.model`` for a local-dir launch) with a
    ``download_dir`` fallback for downloaded weights; return ``None`` when no
    local directory exists (host dummy boots / remote-only checkpoints).
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return None
    model = getattr(model_config, "model", None)
    if model and os.path.isdir(model):
        return str(model)
    download_dir = getattr(model_config, "download_dir", None)
    if download_dir is None:
        download_dir = getattr(getattr(vllm_config, "load_config", None), "download_dir", None)
    if model and download_dir:
        candidate = os.path.join(str(download_dir), str(model))
        if os.path.isdir(candidate):
            return str(candidate)
    return None


def _remap_non_expert(name: str, config: object) -> list[tuple[str, slice | None, int | None]] | None:
    """Map a prefix-rewritten non-expert tensor name to target param placements.

    ``name`` is the checkpoint tensor name with ``model.language_model.`` already
    rewritten to ``model.``. Returns a list of ``(target_param_name, source_slice,
    target_row_offset)`` tuples (a 1-element list for the common case), or
    ``None`` to skip the tensor:

    * ``source_slice`` slices the source tensor's dim 0 (used to split a packed
      source across two params, e.g. the QSA indexer ``index_qk_proj``).
    * ``target_row_offset`` writes the (sliced) source into a contiguous row
      range of the fused target param (shared-expert gate/up, GDN a/b, PLE
      key/value), instead of a whole-param copy.

    Skipped (``None``) tensors are the checkpoint's derived n-gram buffers
    (recomputed deterministically at init), the 128 lazy-shard PLE rows (read on
    demand by the lazy transport), and the small set of eager stand-in params the
    Ascend modules do not yet own (shared-expert gate, GDN ``in_proj_z``/``norm``,
    QSA indexer layernorms).
    """
    hidden = int(config.hidden_size)
    hc_hidden = int(getattr(config, "hc_count", 2)) * hidden
    index_rows = int(getattr(config, "indexer_n_heads", 4)) * int(getattr(config, "indexer_head_dim", 128))
    qsa_q_rows = int(getattr(config, "num_attention_heads", 24)) * int(getattr(config, "head_dim", 256))

    def plain(target: str) -> list[tuple[str, slice | None, int | None]]:
        return [(target, None, None)]

    # Top-level mixer + per-layer hyperconnections (eager _GatedResidual params).
    if name.endswith(".hc_norm.weight"):
        return plain(name[: -len(".hc_norm.weight")] + ".hc_norm_weight")
    if name.endswith(".input_mix_weight_down.weight"):
        return plain(name[: -len(".weight")])
    if name.endswith(".input_mix_weight_up.weight"):
        return plain(name[: -len(".weight")])
    if name.endswith(".block_inject_weight.weight"):
        return plain(name[: -len(".weight")])

    # PLE n-gram table (buffers + lazy shards): nothing to place.
    if ".ple.ple_embedding." in name:
        return None

    # MoE router gate + fused shared expert.
    if name.endswith(".mlp.gate.weight"):
        return plain(name[: -len(".weight")])
    # Shared-expert gate/up/down are TP-sharded and placed by the dedicated
    # `_place_shared_expert_tensor` handler in load_weights (column/row
    # parallel on the intermediate dim), so skip them here.
    if (
        ".mlp.shared_expert.gate_proj" in name
        or ".mlp.shared_expert.up_proj" in name
        or ".mlp.shared_expert.down_proj" in name
    ):
        return None
    if ".mlp.shared_expert_gate" in name:
        return None  # eager stand-in has no shared-expert gate scalar

    # GDN linear-attention projections (checkpoint `linear_attn` -> model
    # `attention`) are TP-sharded and placed by the dedicated `_place_gdn_tensor`
    # handler in load_weights, so skip them here.
    if ".linear_attn." in name:
        return None

    # QSA sparse attention (checkpoint `self_attn` -> model `attention`).
    if ".self_attn." in name:
        base = name.replace(".self_attn.", ".attention.")
        if base.endswith(".q_proj.weight"):
            # The checkpoint fuses the query + output-gate projections into one
            # [num_heads*(1+gate), hidden] row block; the eager module keeps them
            # as separate q_proj / gate_proj params.
            stem = base[: -len(".q_proj.weight")]
            q_rows = qsa_q_rows
            return [(stem + ".q_proj", slice(0, q_rows), None), (stem + ".gate_proj", slice(q_rows, None), None)]
        if base.endswith(".k_proj.weight"):
            return plain(base[: -len(".weight")])
        if base.endswith(".v_proj.weight"):
            return plain(base[: -len(".weight")])
        if base.endswith(".o_proj.weight"):
            return plain(base[: -len(".weight")])
        if base.endswith(".q_norm.weight"):
            return plain(base[: -len(".q_norm.weight")] + ".attn.q_norm_weight")
        if base.endswith(".k_norm.weight"):
            return plain(base[: -len(".k_norm.weight")] + ".attn.k_norm_weight")
        if base.endswith(".indexer.index_qk_proj.weight"):
            stem = base[: -len(".indexer.index_qk_proj.weight")]
            return [(stem + ".iq_proj", slice(0, index_rows), None), (stem + ".ik_proj", slice(index_rows, None), None)]
        # indexer layernorms are not yet owned by the eager QSA indexer.
        return None

    # PLE projection layer (checkpoint `ple.*` -> model `ple.ple.*`).
    if ".ple." in name:
        base = name.replace(".ple.", ".ple.ple.", 1)
        if base.endswith(".conv1d.weight"):
            return plain(base[: -len(".conv1d.weight")] + ".conv_weight")
        if base.endswith(".key_proj.weight"):
            return [(base[: -len(".key_proj.weight")] + ".kv_proj_weight", None, 0)]
        if base.endswith(".value_proj.weight"):
            return [(base[: -len(".value_proj.weight")] + ".kv_proj_weight", None, hc_hidden)]
        if base.endswith(".norm_query.weight"):
            return plain(base[: -len(".weight")] + "_weight")
        if base.endswith(".norm_key.weight"):
            return plain(base[: -len(".weight")] + "_weight")
        if base.endswith(".norm_conv.weight"):
            return plain(base[: -len(".weight")] + "_weight")
        return None

    return None


def _gdn_params_from_config(config: object) -> Qwen4ExpGDNParams:
    """Build GDN geometry, tolerating a duck-typed tiny config (getattr defaults).

    The authoritative ``Qwen4ExpGDNParams.from_hf_config`` reads mandatory
    ``linear_*`` keys; a minimal random config may omit them, so we read every
    field with a small valid default and validate here (still within model.py --
    no component is edited).
    """
    params = Qwen4ExpGDNParams(
        num_k_heads=int(getattr(config, "linear_num_key_heads", 2)),
        num_v_heads=int(getattr(config, "linear_num_value_heads", 4)),
        head_k_dim=int(getattr(config, "linear_key_head_dim", 8)),
        head_v_dim=int(getattr(config, "linear_value_head_dim", 8)),
        conv_kernel_size=int(getattr(config, "linear_conv_kernel_dim", 4)),
        head_dim=int(getattr(config, "head_dim", 16)),
        partial_rotary_factor=float(getattr(config, "partial_rotary_factor", 0.25)),
    )
    params.validate()
    return params


def _reject_multimodal(vllm_config: object) -> None:
    """First gate: the 310P Qwen4Exp path is text-only.

    Vision/multimodal support is a later milestone; reject at construction so a
    Qwen4ExpForConditionalGeneration checkpoint cannot silently run degraded.
    """
    model_config = getattr(vllm_config, "model_config", None)
    mm_config = getattr(model_config, "multimodal_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    has_vision = getattr(hf_config, "vision_config", None) is not None
    if mm_config is not None or has_vision:
        raise NotImplementedError(
            "AscendQwen4ExpForConditionalGeneration: multimodal/vision inputs "
            "are not supported on the Ascend 310P Qwen4Exp path (text-only). "
            "TODO(S-later): wire the Qwen3-VL vision tower."
        )


# ===========================================================================
# Eager math helpers (Triton-free, deterministic)
# ===========================================================================
def _grouped_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """GemmaRMSNorm applied per contiguous ``group_size`` lane (``*(1+w)``)."""
    num_tokens, channels = x.shape
    xc = x.to(compute_dtype)
    wc = weight.to(compute_dtype)
    grouped = xc.view(num_tokens, channels // group_size, group_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = (grouped * torch.rsqrt(variance + eps)).reshape(num_tokens, channels)
    return normalized * (1.0 + wc)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, compute_dtype: torch.dtype) -> torch.Tensor:
    """Plain RMSNorm (``* weight``) accumulated in ``compute_dtype``."""
    xc = x.to(compute_dtype)
    variance = xc.square().mean(dim=-1, keepdim=True)
    normalized = xc * torch.rsqrt(variance + eps)
    return normalized * weight.to(compute_dtype)


def _linear(x: torch.Tensor, weight: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Dtype-safe eager matmul: compute in ``compute_dtype`` regardless of the
    stored (fp16) parameter dtype, so fp16 weights never clash with fp32 acts."""
    return F.linear(x.to(compute_dtype), weight.to(compute_dtype))


class _GatedResidual(nn.Module):
    """Eager hyperconnection / gated-residual multi-stream mixer (TODO(hc)).

    Ports the fork ``GatedResidual`` math (``common/hyperconnection.py``) to a
    self-contained Triton-free module: ``mix`` produces one block input from the
    ``hc_count`` parallel streams (grouped GemmaRMSNorm -> low-rank sigmoid gate
    -> gated mean), and ``combine`` injects the block output back into every
    stream through a learned per-stream injection weight. No dedicated Ascend
    hyperconnection component exists yet, so this eager math is the assembly's
    stand-in.
    """

    def __init__(
        self,
        *,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        eps: float,
        params_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        use_combine: bool = True,
    ) -> None:
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hyper_hidden = hc_count * hidden_size
        self.eps = eps
        self.params_dtype = params_dtype
        self.compute_dtype = compute_dtype
        self.use_combine = use_combine

        self.hc_norm_weight = nn.Parameter(torch.zeros(self.hyper_hidden, dtype=params_dtype))
        self.input_mix_weight_down = nn.Parameter(torch.zeros(lowrank, self.hyper_hidden, dtype=params_dtype))
        self.input_mix_weight_up = nn.Parameter(torch.zeros(self.hyper_hidden, lowrank, dtype=params_dtype))
        if use_combine:
            self.block_inject_weight = nn.Parameter(torch.zeros(hc_count, self.hyper_hidden, dtype=params_dtype))

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        return _grouped_rms_norm(hyper_input, self.hc_norm_weight, self.eps, self.hidden_size, self.compute_dtype)

    def mix(self, hyper_input: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        num_tokens = hyper_input.shape[0]
        xn = self._normalize(hyper_input)
        gate = F.silu(_linear(xn, self.input_mix_weight_down, self.compute_dtype) / self.hc_count)
        gate = torch.sigmoid(_linear(gate, self.input_mix_weight_up, self.compute_dtype))
        gate = gate.view(num_tokens, self.hc_count, self.hidden_size)
        mixed = (gate * xn.view(num_tokens, self.hc_count, self.hidden_size)).mean(dim=-2)
        return mixed.to(self.params_dtype), (hyper_input, xn)

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_combine:
            raise RuntimeError("combine disabled for this gated residual")
        hyper_input, xn = residuals
        num_tokens = hyper_input.shape[0]
        residual = hyper_input.view(num_tokens, self.hc_count, self.hidden_size).to(self.compute_dtype)
        injection = 2.0 * torch.sigmoid(_linear(xn, self.block_inject_weight, self.compute_dtype) / self.hc_count)
        block = block_output.to(self.compute_dtype).unsqueeze(-2)
        out = residual + block * injection.unsqueeze(-1)
        return out.flatten(-2).to(self.params_dtype)


def _register_in_static_forward_context(prefix: str, module: nn.Module) -> None:
    """Register an eager attention stub in the compile ``static_forward_context``.

    The concrete vLLM ``Attention`` registers itself during ``__init__``, but the
    eager Qwen4Exp attention stubs are bare ``AttentionLayerBase``/``MambaBase``
    subclasses, so they must register manually (mirroring ``Attention.__init__``).
    Registration is what lets the v1 runner resolve backends and bind KV caches
    by layer name (``model.layers.{idx}.attention``).
    """
    if not prefix:
        return
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = module


def _resolve_attn_backend(head_size: int, dtype: torch.dtype) -> type[AttentionBackend]:
    """Select the Ascend attention backend for an eager attention stub.

    Resolved eagerly during ``__init__`` (inside the config context) so that
    :meth:`get_attn_backend` never needs ``get_current_vllm_config()`` at call
    time; the v1 runner queries backends from ``get_supported_kv_cache_layouts``
    before the config context is (re)established on the worker.
    """
    from vllm.v1.attention.selector import get_attn_backend as _select

    return _select(head_size=head_size, dtype=dtype, kv_cache_dtype=None)


class _EagerDenseAttention(nn.Module, AttentionLayerBase):
    """Eager causal GQA (used for ``full_attention`` layers without an indexer).

    A plain softmax attention seam so the graph runs; a real dense-attention
    backend is not part of the 1M QSA path (QSA replaces it when
    ``indexer_n_heads`` is configured). Registered as an ``AttentionLayerBase``
    so the v1 runner can allocate a standard full-attention KV cache for it
    (the eager forward does not yet read/write that cache).
    """

    def __init__(self, *, config: object, dtype_policy: Qwen4ExpDtypePolicy, prefix: str = "") -> None:
        super().__init__()
        _register_in_static_forward_context(prefix, self)
        self.compute_dtype = dtype_policy.attention_accumulation_dtype
        self.params_dtype = dtype_policy.attention_dtype
        hidden = int(config.hidden_size)
        self.num_heads = int(getattr(config, "num_attention_heads", 4))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(config, "head_dim", hidden // self.num_heads))
        self.group_size = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5
        self._attn_backend: type[AttentionBackend] | None = (
            _resolve_attn_backend(self.head_dim, self.params_dtype)
            if get_current_vllm_config_or_none() is not None
            else None
        )
        self.q_proj = nn.Parameter(torch.zeros(self.num_heads * self.head_dim, hidden, dtype=self.params_dtype))
        self.k_proj = nn.Parameter(torch.zeros(self.num_kv_heads * self.head_dim, hidden, dtype=self.params_dtype))
        self.v_proj = nn.Parameter(torch.zeros(self.num_kv_heads * self.head_dim, hidden, dtype=self.params_dtype))
        self.o_proj = nn.Parameter(torch.zeros(hidden, self.num_heads * self.head_dim, dtype=self.params_dtype))

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self._attn_backend is None:
            self._attn_backend = _resolve_attn_backend(self.head_dim, self.params_dtype)
        return self._attn_backend

    def get_kv_cache_spec(self, vllm_config: object) -> FullAttentionSpec:
        del vllm_config
        return FullAttentionSpec(
            block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.params_dtype,
        )

    def forward(self, block_input: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions  # rope omitted on the eager stub path
        seq_len = block_input.shape[0]
        q = _linear(block_input, self.q_proj, self.compute_dtype).view(seq_len, self.num_heads, self.head_dim)
        k = _linear(block_input, self.k_proj, self.compute_dtype).view(seq_len, self.num_kv_heads, self.head_dim)
        v = _linear(block_input, self.v_proj, self.compute_dtype).view(seq_len, self.num_kv_heads, self.head_dim)
        q = q.reshape(seq_len, self.num_kv_heads, self.group_size, self.head_dim)
        scores = torch.einsum("qkgd,vkd->kgqv", q, k) * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("kgqv,vkd->qkgd", probs, v).reshape(seq_len, self.num_heads * self.head_dim)
        return _linear(ctx, self.o_proj, self.compute_dtype).to(self.params_dtype)


class _GDNAttention(nn.Module, MambaBase):
    """Wire the real GDN adapter (T5.x): in_proj -> short conv -> gating ->
    gated delta rule (eager backend) -> out_proj.

    Registered as a ``MambaBase`` so the v1 runner allocates the GDN
    conv + recurrent state; the eager forward uses the GDN state pool (model
    state) rather than this KV cache, so the cache is currently allocated but
    not read/written by the eager math.
    """

    def __init__(
        self,
        *,
        config: object,
        dtype_policy: Qwen4ExpDtypePolicy,
        prefix: str = "",
        expert_sharding: tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__()
        _register_in_static_forward_context(prefix, self)
        self.compute_dtype = dtype_policy.accumulation_dtype
        self.params_dtype = dtype_policy.main_dtype
        self.mamba_conv_dtype = dtype_policy.mamba_conv_cache_dtype
        self.mamba_ssm_dtype = dtype_policy.mamba_ssm_cache_dtype
        self.params = _gdn_params_from_config(config)
        self.tp_rank, self.tp_size = (int(expert_sharding[0]), int(expert_sharding[1]))
        if self.tp_size < 1 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(f"expert_sharding={expert_sharding} out of range")
        hidden = int(config.hidden_size)
        p = self.params
        # TP-shard the GDN heads evenly: the gated delta rule is independent per
        # value head (q/k are expanded to the v heads via repeat_interleave), so
        # splitting both num_k_heads and num_v_heads keeps the 3:1 group ratio
        # and the recurrent state is sharded per rank. out_proj is row-parallel.
        if p.num_k_heads % self.tp_size or p.num_v_heads % self.tp_size:
            raise ValueError(
                f"GDN heads (k={p.num_k_heads}, v={p.num_v_heads}) not divisible by TP {self.tp_size}"
            )
        self.num_k_heads = p.num_k_heads // self.tp_size
        self.num_v_heads = p.num_v_heads // self.tp_size
        self.key_dim = p.head_k_dim * self.num_k_heads
        self.value_dim = p.head_v_dim * self.num_v_heads
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Parameter(torch.zeros(self.conv_dim, hidden, dtype=self.params_dtype))
        self.conv_weight = nn.Parameter(torch.zeros(self.conv_dim, p.conv_kernel_size, dtype=self.params_dtype))
        self.in_proj_ba = nn.Parameter(torch.zeros(2 * self.num_v_heads, hidden, dtype=self.params_dtype))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=self.params_dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, dtype=self.params_dtype))
        self.out_proj = nn.Parameter(torch.zeros(hidden, self.value_dim, dtype=self.params_dtype))
        self._tp_reduce: object | None = None
        if self.tp_size > 1:
            try:
                from vllm.distributed import tensor_model_parallel_all_reduce

                self._tp_reduce = tensor_model_parallel_all_reduce
            except Exception:
                self._tp_reduce = None

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.conv_dim, self.params.conv_kernel_size - 1),
            (self.num_v_heads, self.params.head_v_dim, self.params.head_k_dim),
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (self.mamba_conv_dtype, self.mamba_ssm_dtype)

    def get_kv_cache_spec(self, vllm_config: object) -> MambaSpec:
        # Match the model's per-layer spec: the GDN state is larger than the
        # generic ``MambaBase`` page-size padding, so build the spec without a
        # ``page_size_padded`` (letting it default to the raw state size).
        del vllm_config
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
        )

    def forward(self, block_input: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions  # GDN applies no rotary
        seq_len = block_input.shape[0]
        p = self.params
        mixed = _linear(block_input, self.in_proj_qkv, self.compute_dtype)
        mixed = gdn_short_conv(mixed, self.conv_weight, activation="silu", compute_dtype=self.compute_dtype)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(seq_len, self.num_k_heads, p.head_k_dim)
        k = k.reshape(seq_len, self.num_k_heads, p.head_k_dim)
        v = v.reshape(seq_len, self.num_v_heads, p.head_v_dim)
        ba = _linear(block_input, self.in_proj_ba, self.compute_dtype)
        a, b = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        g, beta = gdn_gating(self.A_log, a, b, self.dt_bias, compute_dtype=self.compute_dtype, backend="eager")
        out, _state = gdn_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            chunked=True,
            chunk_size=QWEN4EXP_GDN_CHUNK_SIZE,
            compute_dtype=self.compute_dtype,
            backend="eager",
        )
        out = out.reshape(seq_len, self.value_dim)
        out = _linear(out, self.out_proj, self.compute_dtype)
        if self.tp_size > 1:
            if self._tp_reduce is None:
                raise RuntimeError(
                    "Qwen4Exp GDN TP needs an all-reduce: vllm.distributed."
                    "tensor_model_parallel_all_reduce was not importable at model init."
                )
            out = self._tp_reduce(out)
        return out.to(self.params_dtype)


class _QSAAttention(nn.Module, AttentionLayerBase):
    """Wire the real QSA indexer (T6.1) + sparse GQA attention (T6.2).

    Owns the layer's projection weights, indexer and attention module, and runs
    the full QSA decoder-layer path through the single shared composition entry
    point :func:`~vllm_ascend.models.qwen4_exp.qsa.run_qsa_decoder_attention`
    (plan T6.4), so every one of the 12 QSA layers exercises identical code.

    Registered as an ``AttentionLayerBase`` so the v1 runner allocates the
    full-attention KV cache for it; the eager indexer/attention manage the QSA
    raw ring + compressed side-cache out of band (T6.x), so the standard KV
    cache is currently allocated but not the eager forward's read/write target.
    """

    def __init__(self, *, config: object, layer_idx: int, dtype_policy: Qwen4ExpDtypePolicy, prefix: str = "") -> None:
        super().__init__()
        _register_in_static_forward_context(prefix, self)
        self.compute_dtype = dtype_policy.accumulation_dtype
        self.params_dtype = dtype_policy.qsa_main_dtype
        hidden = int(config.hidden_size)
        self.num_heads = int(getattr(config, "num_attention_heads", 24))
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", 2))
        self.head_dim = int(getattr(config, "head_dim", 256))
        self.index_n_heads = int(getattr(config, "indexer_n_heads", 4))
        self.index_head_dim = int(getattr(config, "indexer_head_dim", 128))
        self._attn_backend: type[AttentionBackend] | None = (
            _resolve_attn_backend(self.head_dim, self.params_dtype)
            if get_current_vllm_config_or_none() is not None
            else None
        )

        self.q_proj = nn.Parameter(torch.zeros(self.num_heads * self.head_dim, hidden, dtype=self.params_dtype))
        self.k_proj = nn.Parameter(torch.zeros(self.num_kv_heads * self.head_dim, hidden, dtype=self.params_dtype))
        self.v_proj = nn.Parameter(torch.zeros(self.num_kv_heads * self.head_dim, hidden, dtype=self.params_dtype))
        self.gate_proj = nn.Parameter(torch.zeros(self.num_heads * self.head_dim, hidden, dtype=self.params_dtype))
        self.iq_proj = nn.Parameter(
            torch.zeros(self.index_n_heads * self.index_head_dim, hidden, dtype=self.params_dtype)
        )
        self.ik_proj = nn.Parameter(torch.zeros(self.index_head_dim, hidden, dtype=self.params_dtype))
        self.o_proj = nn.Parameter(torch.zeros(hidden, self.num_heads * self.head_dim, dtype=self.params_dtype))

        self.indexer = AscendQwen4ExpQSAIndexer(config=config, layer_idx=layer_idx, dtype_policy=dtype_policy)
        self.attn = AscendQwen4ExpQSAAttention(config=config, layer_idx=layer_idx, dtype_policy=dtype_policy)

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self._attn_backend is None:
            self._attn_backend = _resolve_attn_backend(self.head_dim, self.params_dtype)
        return self._attn_backend

    def get_kv_cache_spec(self, vllm_config: object) -> FullAttentionSpec:
        del vllm_config
        return FullAttentionSpec(
            block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.params_dtype,
        )

    def forward(self, block_input: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return run_qsa_decoder_attention(
            block_input,
            positions,
            projections=QSADecoderProjections(
                q_proj=self.q_proj,
                k_proj=self.k_proj,
                v_proj=self.v_proj,
                gate_proj=self.gate_proj,
                index_q_proj=self.iq_proj,
                index_k_proj=self.ik_proj,
                out_proj=self.o_proj,
            ),
            indexer=self.indexer,
            attention=self.attn,
            num_query_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            store_dtype=self.params_dtype,
            compute_dtype=self.compute_dtype,
        )


class _EagerMLP(nn.Module):
    """Eager dense SwiGLU MLP (non-MoE layers)."""

    def __init__(self, *, hidden_size: int, intermediate_size: int, dtype_policy: Qwen4ExpDtypePolicy) -> None:
        super().__init__()
        self.compute_dtype = dtype_policy.accumulation_dtype
        self.params_dtype = dtype_policy.main_dtype
        self.gate_up_proj = nn.Parameter(torch.zeros(2 * intermediate_size, hidden_size, dtype=self.params_dtype))
        self.down_proj = nn.Parameter(torch.zeros(hidden_size, intermediate_size, dtype=self.params_dtype))

    def forward(self, block_input: torch.Tensor) -> torch.Tensor:
        gate_up = _linear(block_input, self.gate_up_proj, self.compute_dtype)
        gate, up = gate_up.chunk(2, dim=-1)
        return _linear(F.silu(gate) * up, self.down_proj, self.compute_dtype).to(self.params_dtype)


class _EagerSparseMoE(nn.Module):
    """Routed-expert (W8A8) + shared-expert (F16) MoE block (T3.x, stub closed).

    Ports the upstream ``Qwen3NextSparseMoeBlock`` control flow onto the real
    Ascend 310P W8A8_DYNAMIC fused-expert path. The routed experts are held in
    the :class:`AscendW8A8DynamicFusedMoEMethod310` fused layout
    (``w13_*``/``w2_*``, gate+up column-fused) and evaluated with the
    T3.3-validated grouped QDQ math from :mod:`vllm_ascend.models.qwen4_exp.moe`
    (per-token INT8 activation quant, per-channel ``(q - offset) * scale`` weight
    dequant -- real experts are symmetric so ``offset == 0``). The router runs in
    the policy ``router_dtype`` (fp32) with ``norm_topk_prob`` renormalization;
    the shared expert stays non-quantized F16 (per the T3.1 mapping contract) and
    is applied densely + unweighted.

    Expert-dimension TP slicing (the TP4 target): ``expert_sharding=(rank,
    size)`` gives this rank the contiguous slice of experts
    ``[rank*E_local, (rank+1)*E_local)`` (same linear placement the fork's
    ``ep_weight_filter`` loader skip uses). The router gate (``E_global`` rows)
    and the shared expert stay replicated, so every rank selects identical
    top-k; the routed partial is all-reduced before the shared expert is added
    once. The all-reduce defaults to ``vllm.distributed.
    tensor_model_parallel_all_reduce`` when ``size > 1`` (process group must be
    initialized -- it is, by worker ``init_device``, before model load) and can
    be overridden on the instance ``_tp_reduce`` attribute for host tests.
    ``size == 1`` (host bring-up default) is the exact pre-slicing behavior.

    The class name is retained so the assembly imports/isinstance checks keep
    working; the ``experts_*`` eager stub is replaced by the fused W8A8 params.
    """

    def __init__(
        self,
        *,
        config: object,
        dtype_policy: Qwen4ExpDtypePolicy,
        expert_sharding: tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__()
        self.router_dtype = dtype_policy.router_dtype
        self.compute_dtype = dtype_policy.accumulation_dtype
        self.params_dtype = dtype_policy.main_dtype
        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.num_experts = int(getattr(config, "num_experts", 0) or 0)
        self.top_k = min(int(getattr(config, "num_experts_per_tok", 2)), self.num_experts)
        moe_inter = int(getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", hidden)))
        self.moe_intermediate_size = moe_inter
        # Router renorm + optional routed scaling (fork ``norm_topk_prob`` /
        # ``routed_scaling_factor``; default: renorm on, scale 1.0).
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0) or 1.0)

        # Expert-dimension TP slicing: contiguous [expert_offset,
        # expert_offset + num_local_experts) global-id range on this rank.
        # Requiring divisibility keeps local slotting one line everywhere
        # (512 experts / TP4 = 128); the fork loader filter tolerates
        # non-divisible counts and must match this, so reject loudly instead.
        self.expert_tp_rank, self.expert_tp_size = (int(expert_sharding[0]), int(expert_sharding[1]))
        if self.expert_tp_size < 1 or not 0 <= self.expert_tp_rank < self.expert_tp_size:
            raise ValueError(f"expert_sharding={expert_sharding} out of range")
        if self.num_experts and self.num_experts % self.expert_tp_size:
            raise ValueError(
                f"num_experts={self.num_experts} is not divisible by expert TP size {self.expert_tp_size}; "
                "the W8A8 fused bank slices experts contiguously and cannot shard this count"
            )
        self.num_local_experts = self.num_experts // self.expert_tp_size
        self.num_global_experts = self.num_experts
        self.expert_offset = self.expert_tp_rank * self.num_local_experts
        self._tp_reduce: object | None = None
        if self.expert_tp_size > 1:
            try:
                from vllm.distributed import tensor_model_parallel_all_reduce

                self._tp_reduce = tensor_model_parallel_all_reduce
            except Exception:
                self._tp_reduce = None  # forward() fails loudly if never installed

        # Router gate stays F16 (non-quantized), replicated on every rank so the
        # top-k selection is identical across all TP ranks.
        self.gate = nn.Parameter(torch.zeros(self.num_experts, hidden, dtype=self.params_dtype))

        # Routed experts in the AscendW8A8DynamicFusedMoEMethod310 fused layout
        # (LOCAL slice only under expert-dimension TP slicing):
        #   w13_weight        int8    [E_local, 2*moe, hidden]   gate rows [0,moe), up [moe,2moe)
        #   w2_weight         int8    [E_local, hidden, moe]
        #   w13_weight_scale  float32 [E_local, 2*moe, 1]  (offset likewise, symmetric == 0)
        #   w2_weight_scale   float32 [E_local, hidden, 1]
        # int8 params never require grad (only float/complex tensors may).
        self.w13_weight = nn.Parameter(
            torch.zeros(self.num_local_experts, 2 * moe_inter, hidden, dtype=torch.int8), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(self.num_local_experts, hidden, moe_inter, dtype=torch.int8), requires_grad=False
        )
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(self.num_local_experts, 2 * moe_inter, 1, dtype=self.compute_dtype), requires_grad=False
        )
        self.w13_weight_offset = nn.Parameter(
            torch.zeros(self.num_local_experts, 2 * moe_inter, 1, dtype=self.compute_dtype), requires_grad=False
        )
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(self.num_local_experts, hidden, 1, dtype=self.compute_dtype), requires_grad=False
        )
        self.w2_weight_offset = nn.Parameter(
            torch.zeros(self.num_local_experts, hidden, 1, dtype=self.compute_dtype), requires_grad=False
        )

        shared_inter = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        self.has_shared_expert = shared_inter > 0
        self.local_shared_inter = 0
        if self.has_shared_expert:
            # TP-shard the shared expert's intermediate dim: gate/up are
            # column-parallel (split 2*shared_inter) and down is row-parallel
            # (split shared_inter), matching the routed-expert all-reduce.
            tp = self.expert_tp_size
            if shared_inter % tp:
                raise ValueError(f"shared_expert_intermediate_size={shared_inter} not divisible by TP {tp}")
            self.local_shared_inter = shared_inter // tp
            self.shared_gate_up = nn.Parameter(
                torch.zeros(2 * self.local_shared_inter, hidden, dtype=self.params_dtype)
            )
            self.shared_down = nn.Parameter(torch.zeros(hidden, self.local_shared_inter, dtype=self.params_dtype))

    def forward(self, block_input: torch.Tensor) -> torch.Tensor:
        router_logits = F.linear(block_input.to(self.router_dtype), self.gate.to(self.router_dtype))
        topk_weights, topk_ids = route_topk(
            router_logits,
            self.top_k,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        out = w8a8_grouped_experts(
            block_input,
            topk_weights,
            topk_ids,
            self.w13_weight,
            self.w13_weight_scale,
            self.w13_weight_offset,
            self.w2_weight,
            self.w2_weight_scale,
            self.w2_weight_offset,
            expert_offset=self.expert_offset,
            num_global_experts=self.num_global_experts,
        )
        if self.expert_tp_size > 1:
            if self._tp_reduce is None:
                raise RuntimeError(
                    "Qwen4Exp expert-TP MoE needs an all-reduce: vllm.distributed."
                    "tensor_model_parallel_all_reduce was not importable at model init "
                    "and no override is installed on _EagerSparseMoE._tp_reduce."
                )
            out = self._tp_reduce(out)

        if self.has_shared_expert:
            shared_gate_up = _linear(block_input, self.shared_gate_up, self.compute_dtype)
            shared_partial = _linear(swiglu_gate_up(shared_gate_up), self.shared_down, self.compute_dtype)
            if self.expert_tp_size > 1:
                if self._tp_reduce is None:
                    raise RuntimeError(
                        "Qwen4Exp shared-expert TP needs an all-reduce: "
                        "tensor_model_parallel_all_reduce was not importable."
                    )
                shared_partial = self._tp_reduce(shared_partial)
            out = out + shared_partial
        return out.to(self.params_dtype)


def _scalar_eos(config: object) -> int:
    """A single EOS id from a config that may omit it or store a list.

    The real Qwen4Exp text config carries ``eos_token_id`` as an int, but the
    generation config / some duck configs store a ``[primary, alt]`` list, and
    tiny random test configs omit it entirely.
    """
    eos = getattr(config, "eos_token_id", 0)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else 0
    return int(eos or 0)


class _NGramConfigProxy:
    """Config proxy guaranteeing a scalar ``eos_token_id`` for the n-gram hasher.

    ``AscendQwen4ExpNGramEmbedding`` reads ``config.eos_token_id`` directly; this
    proxy supplies a scalar (real configs pass through unchanged) so construction
    is robust to list-valued or omitted EOS on duck/tiny configs.
    """

    def __init__(self, base: object, eos_token_id: int) -> None:
        self._base = base
        self._eos_token_id = eos_token_id

    def __getattr__(self, name: str) -> object:
        if name == "eos_token_id":
            return self._eos_token_id
        return getattr(self._base, name)


class _PLEInjection(nn.Module):
    """Wire the real PLE injection layer (T4.x) with a host PLE table method.

    The n-gram id hashing is REAL: ``AscendQwen4ExpNGramEmbedding.compute_ngram_ids``
    (SplitMix64, T4.2-verified against the checkpoint's ``layer_multipliers``). The
    row gather, projection, gate and dilated short-conv are the real
    ``AscendQwen4ExpPLELayer`` component. The table itself is either:

    * a lazy per-shard mmap over the checkpoint's 128 n-gram shards (transport c,
      ``checkpoint_dir`` provided) -- no 95.43 GiB host table; or
    * a synthetic 4096-row host stub (dummy boot / host tests), where the global
      row ids are reduced modulo the stub table.
    """

    # Rows in the stubbed PLE table. Sized generously above vocab so the stub
    # n-gram ids index a distinct-enough table; the real layout is T1.3.
    _STUB_TABLE_ROWS = 4096

    def __init__(
        self,
        *,
        config: object,
        layer_idx: int,
        dtype_policy: Qwen4ExpDtypePolicy,
        checkpoint_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.dtype_policy = dtype_policy
        self.eos_token_id = _scalar_eos(config)
        self.checkpoint_dir = checkpoint_dir
        self.ple = AscendQwen4ExpPLELayer(config=config, layer_idx=layer_idx, dtype_policy=dtype_policy)
        self.num_ngram_heads = self.ple.num_ngram_heads
        self.per_head_dim = self.ple.per_head_dim
        self.ngram_size = int(config.ngram_size)
        self._ple_method: AscendPLEPinnedHostEmbeddingMethod | AscendPLELazyShardEmbeddingMethod | None = None
        # Real SplitMix64 n-gram hashing (T4.2-verified). Constructed without a
        # ple_method: only ``compute_ngram_ids`` is used here (no gather), so the
        # global row ids match the checkpoint's layer_multipliers exactly. A tiny
        # duck config that omits the full n-gram vocab layout (e.g. the meta-boot
        # smoke) falls back to the deterministic id stub for control flow only.
        try:
            self.ngram: AscendQwen4ExpNGramEmbedding | None = AscendQwen4ExpNGramEmbedding(
                config=_NGramConfigProxy(config, self.eos_token_id),
                ple_dense_layer_id=0,
                dtype_policy=dtype_policy,
            )
        except (AttributeError, ValueError):
            self.ngram = None

    def _ensure_ple_method(self, device: torch.device) -> None:
        if self.ple.ple_method is not None:
            return

        if self.checkpoint_dir is not None:
            # Real checkpoint: read rows on demand from the 128 shard tensors
            # (transport c) -- no 95.43 GiB host table, no host-budget check.
            if self.ngram is None:
                raise RuntimeError("lazy-shard PLE transport requires the real n-gram hasher")
            method = AscendPLELazyShardEmbeddingMethod(
                self.ngram.padded_vocab_size,
                self.per_head_dim,
                checkpoint_dir=self.checkpoint_dir,
                split_ngram_parts=self.ngram.split_ngram_parts,
                dtype_policy=self.dtype_policy,
            )
            self.ple.ple_method = method
            self._ple_method = method
            return

        # Host-safe pinned embedding method: inject a plain CPU allocator + a
        # UVA probe that reports available, so no accelerator/mmap is needed.
        def _cpu_allocator(rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
            generator = torch.Generator().manual_seed(0)
            return torch.randn(rows, dim, generator=generator).to(dtype)

        method = AscendPLEPinnedHostEmbeddingMethod(
            self._STUB_TABLE_ROWS,
            self.per_head_dim,
            uva_probe=lambda: True,
            pinned_allocator=_cpu_allocator,
            dtype_policy=self.dtype_policy,
        )
        self.ple.ple_method = method
        self._ple_method = method

    def _real_ngram_ids(self, input_ids: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        """Real SplitMix64 n-gram row ids for a single eager-boot request.

        Uses ``AscendQwen4ExpNGramEmbedding.compute_ngram_ids`` (T4.2-verified to
        match the checkpoint's ``layer_multipliers``). The eager-boot path has no
        model state, so we present one request (``query_start_loc=[0, seq_len]``)
        with an EOS-padded history (``ngram_context``). The returned ids index the
        full 320M-row padded vocab; when ``reduce`` is set (the stub-table boot
        path) they are reduced modulo the synthetic table.
        """
        seq_len = int(input_ids.shape[0])
        device = input_ids.device
        query_start_loc = torch.tensor([0, seq_len], dtype=torch.int64, device=device)
        ngram_context = torch.full((1, self.ngram_size - 1), self.eos_token_id, dtype=torch.int64, device=device)
        global_ids = self.ngram.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        if reduce:
            return global_ids.remainder(self._STUB_TABLE_ROWS)
        return global_ids

    def _stub_ngram_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Deterministic id fallback for a tiny duck config with no n-gram vocab.

        Only used by the dummy-weight control-flow boot when the real SplitMix64
        hasher could not be constructed; real configs use :meth:`_real_ngram_ids`.
        """
        seq_len = input_ids.shape[0]
        tokens = input_ids.to(torch.int64)
        ids = torch.empty(seq_len, self.num_ngram_heads, dtype=torch.int64, device=input_ids.device)
        for head in range(self.num_ngram_heads):
            order = head % (self.ngram_size - 1) + 1
            mixed = tokens.clone()
            for shift in range(1, order + 1):
                shifted = torch.full_like(tokens, self.eos_token_id)
                if seq_len > shift:
                    shifted[shift:] = tokens[:-shift]
                mixed = mixed * 1000003 + shifted + (head + 1)
            ids[:, head] = mixed.remainder(self._STUB_TABLE_ROWS)
        return ids

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        self._ensure_ple_method(hidden_states.device)
        if self.ngram is not None:
            # Real hasher: full padded-vocab ids for the lazy-shard table, or
            # reduced to the synthetic stub table on the host dummy-boot path.
            ngram_ids = self._real_ngram_ids(input_ids, reduce=self.checkpoint_dir is None)
        else:
            ngram_ids = self._stub_ngram_ids(input_ids)
        return self.ple(hidden_states, ngram_ids)


# ===========================================================================
# Decoder layer
# ===========================================================================
class AscendQwen4ExpDecoderLayer(nn.Module):
    """One Qwen4Exp decoder layer: PLE (optional) -> attention block -> MoE/MLP
    block, all wired through the gated-residual multi-stream state."""

    def __init__(
        self,
        *,
        config: object,
        layer_type: str,
        layer_idx: int,
        dtype_policy: Qwen4ExpDtypePolicy,
        prefix: str = "",
        expert_sharding: tuple[int, int] = (0, 1),
        checkpoint_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.dtype_policy = dtype_policy
        self.prefix = prefix

        hidden = int(config.hidden_size)
        hc_count = int(getattr(config, "hc_count", 2))
        lowrank = int(getattr(config, "hc_lowrank", 16))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        main_dtype = dtype_policy.main_dtype
        accum_dtype = dtype_policy.accumulation_dtype

        def _make_hc() -> _GatedResidual:
            return _GatedResidual(
                hc_count=hc_count,
                hidden_size=hidden,
                lowrank=lowrank,
                eps=eps,
                params_dtype=main_dtype,
                compute_dtype=accum_dtype,
            )

        self.attn_hyper_connection = _make_hc()
        self.mlp_hyper_connection = _make_hc()

        # PLE injects on the layer whose absolute id (layer_idx + 1) is listed.
        ple_layer_ids = getattr(config, "ple_layer_ids", None) or []
        self.has_ple = (layer_idx + 1) in ple_layer_ids
        self.ple: _PLEInjection | None = (
            _PLEInjection(
                config=config,
                layer_idx=layer_idx,
                dtype_policy=dtype_policy,
                checkpoint_dir=checkpoint_dir,
            )
            if self.has_ple
            else None
        )

        # Attention: GDN linear attention, QSA sparse attention, or dense.
        attn_prefix = f"{prefix}.attention"
        self.uses_qsa = False
        if layer_type == _LAYER_TYPE_LINEAR:
            self.attention: nn.Module = _GDNAttention(
                config=config,
                dtype_policy=dtype_policy,
                prefix=attn_prefix,
                expert_sharding=expert_sharding,
            )
        else:
            if getattr(config, "indexer_n_heads", None) is not None:
                self.uses_qsa = True
                self.attention = _QSAAttention(
                    config=config, layer_idx=layer_idx, dtype_policy=dtype_policy, prefix=attn_prefix
                )
            else:
                self.attention = _EagerDenseAttention(config=config, dtype_policy=dtype_policy, prefix=attn_prefix)

        # MoE (routed + shared expert) vs. dense MLP.
        num_experts = int(getattr(config, "num_experts", 0) or 0)
        decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1) or 1)
        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        is_moe = layer_idx not in mlp_only_layers and num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        if is_moe:
            self.mlp: nn.Module = _EagerSparseMoE(
                config=config, dtype_policy=dtype_policy, expert_sharding=expert_sharding
            )
        else:
            self.mlp = _EagerMLP(
                hidden_size=hidden,
                intermediate_size=int(getattr(config, "intermediate_size", 4 * hidden)),
                dtype_policy=dtype_policy,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # PLE injects into the multi-stream state before the attention block.
        if self.ple is not None:
            hidden_states = self.ple(hidden_states, input_ids)

        block_input, residual = self.attn_hyper_connection.mix(hidden_states)
        attn_out = self.attention(block_input, positions)
        hidden_states = self.attn_hyper_connection.combine(attn_out, residual)

        block_input, residual = self.mlp_hyper_connection.mix(hidden_states)
        mlp_out = self.mlp(block_input)
        hidden_states = self.mlp_hyper_connection.combine(mlp_out, residual)
        return hidden_states


# ===========================================================================
# Backbone
# ===========================================================================
def _resolve_expert_sharding(vllm_config: VllmConfig) -> tuple[int, int]:
    """This process's ``(expert_tp_rank, expert_tp_size)`` for the W8A8 bank.

    The size is the configured tensor-parallel size (expert-dimension slicing,
    one contiguous expert range per rank -- the same linear placement the fork's
    ``ep_weight_filter`` weight loader uses). The rank comes from the initialized
    tensor-model-parallel group; on device runs the worker initializes parallel
    state in ``init_device`` before the model is constructed, so it is always
    valid there. Host bring-up (UT / dummy boots) has no process group: rank
    falls back to 0, which only the ``tp_size == 1`` path consumes.
    """
    parallel = getattr(vllm_config, "parallel_config", None)
    size = int(getattr(parallel, "tensor_parallel_size", 1) or 1)
    if size <= 1:
        return (0, 1)
    rank = 0
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = int(get_tensor_model_parallel_rank())
    except Exception:
        rank = 0
    return (rank % size, size)


class AscendQwen4ExpModel(nn.Module):
    """Backbone: embeddings + decoder-layer stack + final mixer + final norm."""

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = getattr(vllm_config, "quant_config", None)
        self.dtype_policy = Qwen4ExpDtypePolicy.from_vllm_config(vllm_config)

        self.hc_count = int(getattr(config, "hc_count", 2))
        self.hidden_size = int(config.hidden_size)
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=self.dtype_policy.embedding_dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        layer_types = self._resolve_layer_types(config)
        # Expert-dimension TP slicing shared by every MoE block and the loader
        # (T3.1b): contiguous expert range per rank; rank 0 / size 1 host boot.
        self.expert_sharding = _resolve_expert_sharding(vllm_config)
        # Local checkpoint dir feeds the lazy per-shard PLE transport (T4.1c):
        # when present, the PLE layer reads n-gram rows on demand instead of
        # materializing the 95.43 GiB host table.
        self.checkpoint_dir = _resolve_checkpoint_dir(vllm_config)
        self.layers = nn.ModuleList(
            AscendQwen4ExpDecoderLayer(
                config=config,
                layer_type=layer_types[idx],
                layer_idx=idx,
                dtype_policy=self.dtype_policy,
                prefix=maybe_prefix(prefix, f"layers.{idx}"),
                expert_sharding=self.expert_sharding,
                checkpoint_dir=self.checkpoint_dir,
            )
            for idx in range(config.num_hidden_layers)
        )
        self.layer_types = layer_types

        # Final HC mixer collapses the hc_count streams into one sampled stream.
        self.hyper_connection_mixer = _GatedResidual(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            lowrank=int(getattr(config, "hc_lowrank", 16)),
            eps=self.eps,
            params_dtype=self.dtype_policy.main_dtype,
            compute_dtype=self.dtype_policy.accumulation_dtype,
            use_combine=False,
        )
        # Eager final RMSNorm (avoids the vLLM CustomOp config-context dependency
        # on the host boot path); weight loaded as a dummy parameter.
        self.norm_weight = nn.Parameter(torch.ones(self.hidden_size, dtype=self.dtype_policy.main_dtype))

        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self._mtp_hidden_buffer: torch.Tensor | None = None

    @staticmethod
    def _resolve_layer_types(config: object) -> list[str]:
        num_layers = config.num_hidden_layers
        layer_types = getattr(config, "layer_types", None)
        if not layer_types:
            return [_LAYER_TYPE_FULL] * num_layers
        resolved = list(layer_types)
        if len(resolved) < num_layers:
            resolved += [_LAYER_TYPE_FULL] * (num_layers - len(resolved))
        return resolved[:num_layers]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.embed_input_ids(input_ids)
        # Expand to the hc_count parallel streams ([T, H] -> [T, hc*H]).
        hidden_states = hidden_states.repeat(1, self.hc_count)

        raw_input_ids = (
            input_ids
            if input_ids is not None
            else torch.zeros(hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, raw_input_ids)

        # Retain the multi-stream state for the MTP drafter (scheme A).
        self._mtp_hidden_buffer = hidden_states
        # Final mixer collapses the streams to the sampled single-stream state.
        sample_hidden, _residual = self.hyper_connection_mixer.mix(hidden_states)
        sample_hidden = _rms_norm(sample_hidden, self.norm_weight, self.eps, self.dtype_policy.accumulation_dtype)
        return sample_hidden.to(self.dtype_policy.main_dtype)


# ===========================================================================
# Causal LM
# ===========================================================================
class AscendQwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    MixtureOfExperts,
    IsHybrid,
):
    """Text-only Ascend Qwen4Exp causal LM (full-model assembly, T1.5)."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = getattr(vllm_config, "quant_config", None)
        self.config = config
        # Authoritative dtype policy (PRD §5.3 / R4): every submodule reads it.
        self.dtype_policy = Qwen4ExpDtypePolicy.from_vllm_config(vllm_config)

        cache_config = getattr(vllm_config, "cache_config", None)
        if getattr(cache_config, "mamba_cache_mode", None) == "all":
            raise NotImplementedError(
                "Qwen4Exp does not support 'all' mamba prefix caching; use '--mamba-cache-mode=align'."
            )

        self.model = AscendQwen4ExpModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            params_dtype=self.dtype_policy.lm_head_dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)

    # -- state / hooks -----------------------------------------------------

    @staticmethod
    def get_model_state_cls():
        """Return the concrete 310P Qwen4Exp hybrid model-state class (T1.3)."""
        from vllm_ascend._310p.worker.v2.model_state import (
            Ascend310PQwen4ExpModelState,
        )

        return Ascend310PQwen4ExpModelState

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        policy = Qwen4ExpDtypePolicy.from_vllm_config(vllm_config)
        return (policy.mamba_conv_cache_dtype, policy.mamba_ssm_cache_dtype)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, ...]:
        policy = Qwen4ExpDtypePolicy.from_vllm_config(vllm_config)
        return (policy.mamba_conv_cache_dtype,)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_gdn_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_text_config
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        params = _gdn_params_from_config(config)
        conv_dim = params.conv_dim // tp_size
        conv_state = (conv_dim, params.conv_kernel_size - 1)
        recurrent_state = (params.num_v_heads // tp_size, params.head_v_dim, params.head_k_dim)
        return (conv_state, recurrent_state)

    @classmethod
    def get_ple_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_text_config
        conv_kernel_size = int(config.ple_conv_kernel_size)
        short_conv_dilation = int(config.ngram_size)
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        hc_hidden = int(config.hidden_size) * int(config.hc_count)
        return ((hc_hidden, conv_state_len),)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types}

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig) -> tuple[MambaSpec, ...]:
        """All MambaSpecs: GDN layers + the PLE short-conv layer."""
        return (
            MambaSpec(
                shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
                tp_replicated=True,
            ),
        )

    # -- KV-cache spec materialization (T1.4) ------------------------------

    def get_kv_cache_spec(self, vllm_config: VllmConfig | None = None) -> dict[str, KVCacheSpec]:
        """Per-layer KV-cache spec dict for the v1 runner.

        The eager attention modules are plain ``nn.Module`` (not vLLM
        ``AttentionLayerBase`` subclasses), so the standard runner's
        ``get_kv_cache_spec`` finds nothing. This builds the per-layer spec dict
        keyed by the eager module path (``model.layers.{idx}.attention``): GDN
        layers contribute a ``MambaSpec``; QSA / dense layers contribute a
        ``FullAttentionSpec`` (the QSA ring + compressed side-caches are tracked
        out-of-band by the model state, not the attention KV cache).
        """
        del vllm_config  # specs are built from self.config / self.dtype_policy
        config = self.config
        policy = self.dtype_policy
        head_dim = int(getattr(config, "head_dim", 0)) or (
            int(config.hidden_size) // int(getattr(config, "num_attention_heads", 1))
        )
        num_kv_heads = int(getattr(config, "num_key_value_heads", 1))

        try:
            gdn_params = _gdn_params_from_config(config)
        except Exception:  # pragma: no cover - GDN geometry may be absent
            gdn_params = None

        spec: dict[str, KVCacheSpec] = {}
        for idx, layer_type in enumerate(self.model.layer_types):
            name = f"model.layers.{idx}.attention"
            if layer_type == _LAYER_TYPE_LINEAR and gdn_params is not None:
                conv_dim = gdn_params.conv_dim
                spec[name] = MambaSpec(
                    shapes=(
                        (conv_dim, gdn_params.conv_kernel_size - 1),
                        (gdn_params.num_v_heads, gdn_params.head_v_dim, gdn_params.head_k_dim),
                    ),
                    dtypes=(policy.mamba_conv_cache_dtype, policy.mamba_ssm_cache_dtype),
                    block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
                )
            else:
                spec[name] = FullAttentionSpec(
                    block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
                    num_kv_heads=num_kv_heads,
                    head_size=head_dim,
                    dtype=policy.kv_cache_dtype,
                )
        return spec

    def get_kv_cache_groups(self, *, num_speculative_tokens: int = 0) -> list[KVCacheGroupSpec]:
        """Materialize the hybrid KV-cache groups for this model's layers.

        One group per cache class (full attention / GDN mamba / QSA raw ring /
        QSA compressed index), packaged through the T1.4 ``kv_cache`` helpers.
        """
        config = self.config
        policy = self.dtype_policy
        head_dim = int(getattr(config, "head_dim", 0)) or (
            int(config.hidden_size) // int(getattr(config, "num_attention_heads", 1))
        )
        num_kv_heads = int(getattr(config, "num_key_value_heads", 1))

        full_attention_layers: dict[str, object] = {}
        mamba_layers: dict[str, object] = {}
        qsa_raw_ring_layers: dict[str, object] = {}
        qsa_compressed_layers: dict[str, object] = {}

        uses_qsa = getattr(config, "indexer_n_heads", None) is not None
        try:
            gdn_params = _gdn_params_from_config(config)
        except Exception:  # pragma: no cover - GDN geometry may be absent
            gdn_params = None

        for idx, layer_type in enumerate(self.model.layer_types):
            name = f"model.layers.{idx}"
            if layer_type == _LAYER_TYPE_LINEAR and gdn_params is not None:
                conv_dim = gdn_params.conv_dim
                mamba_layers[f"{name}.linear_attn"] = MambaSpec(
                    shapes=(
                        (conv_dim, gdn_params.conv_kernel_size - 1),
                        (gdn_params.num_v_heads, gdn_params.head_v_dim, gdn_params.head_k_dim),
                    ),
                    dtypes=(policy.mamba_conv_cache_dtype, policy.mamba_ssm_cache_dtype),
                    block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
                )
            elif layer_type == _LAYER_TYPE_FULL and uses_qsa:
                qsa_raw_ring_layers[f"{name}.self_attn.qsa.ring"] = make_qsa_raw_ring_spec(
                    dtype_policy=policy,
                    num_speculative_tokens=num_speculative_tokens,
                )
                qsa_compressed_layers[f"{name}.self_attn.qsa.compressed"] = make_qsa_compressed_spec(
                    dtype_policy=policy,
                )
            elif layer_type == _LAYER_TYPE_FULL:
                full_attention_layers[f"{name}.self_attn.attn"] = FullAttentionSpec(
                    block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
                    num_kv_heads=num_kv_heads,
                    head_size=head_dim,
                    dtype=policy.kv_cache_dtype,
                )

        return build_qwen4exp_kv_cache_groups(
            full_attention_layers=full_attention_layers or None,
            mamba_layers=mamba_layers or None,
            qsa_raw_ring_layers=qsa_raw_ring_layers or None,
            qsa_compressed_layers=qsa_compressed_layers or None,
        )

    def kv_group_report(self, *, num_speculative_tokens: int = 0) -> dict[str, int]:
        """KV-cache report: total layers per cache class (merged into groups).

        Keyed by spec class name -> number of layers of that class. Layers that
        share an identical spec are packaged into a single ``KVCacheGroupSpec``
        (T1.4), so the group count per class is <= the layer count reported here.
        """
        groups = self.get_kv_cache_groups(num_speculative_tokens=num_speculative_tokens)
        report: dict[str, int] = {}
        for group in groups:
            key = type(group.kv_cache_spec).__name__
            report[key] = report.get(key, 0) + len(group.layer_names)
        return report

    # -- MoE / weight-load hooks ------------------------------------------

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Fused-expert weight-mapping hook.

        The 310P path streams the real per-expert W8A8 tensors through the T3.1
        mapper directly inside :meth:`load_weights` (rather than the generic vLLM
        FusedMoE ``expert_params_mapping`` remap), so this hook is intentionally
        empty; the mapping authority is :mod:`weight_mapping`.
        """
        return []

    def _expert_geometry(self) -> dict[str, int]:
        """Frozen W8A8 geometry driven from the model config (T3.1 contract)."""
        config = self.config
        hidden = int(config.hidden_size)
        moe_inter = int(getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", hidden)))
        return {
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_experts": int(getattr(config, "num_experts", 0) or 0),
            "moe_intermediate_size": moe_inter,
            "hidden_size": hidden,
        }

    def _place_expert_tensor(self, params, mapping, weight: torch.Tensor) -> str:
        """Copy one source expert tensor into its fused-MoE param slot (streamed).

        ``mapping`` is the T3.1 :class:`ExpertTensorMapping`; the destination is
        ``model.layers.{L}.mlp.{target_param}`` at ``[expert_index,
        row_start:row_stop]`` (the same slice for weight/scale/offset, and for w2
        the slice spans the full output dim). Rejects a dtype/shape mismatch with
        the mapper's own error taxonomy before any copy.
        """
        target_name = f"model.layers.{mapping.layer}.mlp.{mapping.target_param}"
        param = params.get(target_name)
        if param is None:
            raise WeightMappingError(
                f"no fused-MoE parameter {target_name!r} for expert tensor "
                f"{mapping.source_name!r}; layer {mapping.layer} is not a W8A8 MoE layer",
                tensors=[mapping.source_name],
            )
        if weight.dtype != mapping.expected_dtype:
            kind_label = "quantized weight" if mapping.kind == "weight" else mapping.kind.replace("_", " ")
            raise TensorDtypeError(
                f"{mapping.source_name!r}: expert {kind_label} must be {mapping.expected_dtype}, got {weight.dtype}",
                tensors=[mapping.source_name],
            )
        if tuple(weight.shape) != mapping.expected_shape:
            raise TensorShapeError(
                f"{mapping.source_name!r}: expected shape {mapping.expected_shape} for "
                f"{mapping.proj}.{mapping.kind}, got {tuple(weight.shape)}",
                tensors=[mapping.source_name],
            )
        with torch.no_grad():
            param[mapping.expert_index, mapping.row_start : mapping.row_stop].copy_(weight.to(param.dtype))
        return target_name

    def _place_shared_expert_tensor(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> str | None:
        """Place one TP-sliced shared-expert tensor into its local param slot.

        gate/up are column-parallel (split the intermediate dim across the rows
        of the fused ``shared_gate_up``); down is row-parallel (split across the
        columns of ``shared_down``). Returns the target param name, or ``None``
        to fall through to the generic remap path.
        """
        del tp_size
        if name.endswith(".mlp.shared_expert.gate_proj.weight"):
            target_name = name[: -len(".mlp.shared_expert.gate_proj.weight")] + ".mlp.shared_gate_up"
            target = params.get(target_name)
            if target is None:
                return None
            local = target.shape[0] // 2
            src = tensor[tp_rank * local : (tp_rank + 1) * local]
            with torch.no_grad():
                target[0:local].copy_(src.to(target.dtype))
            return target_name
        if name.endswith(".mlp.shared_expert.up_proj.weight"):
            target_name = name[: -len(".mlp.shared_expert.up_proj.weight")] + ".mlp.shared_gate_up"
            target = params.get(target_name)
            if target is None:
                return None
            local = target.shape[0] // 2
            src = tensor[tp_rank * local : (tp_rank + 1) * local]
            with torch.no_grad():
                target[local : 2 * local].copy_(src.to(target.dtype))
            return target_name
        if name.endswith(".mlp.shared_expert.down_proj.weight"):
            target_name = name[: -len(".mlp.shared_expert.down_proj.weight")] + ".mlp.shared_down"
            target = params.get(target_name)
            if target is None:
                return None
            local = target.shape[1]
            src = tensor[:, tp_rank * local : (tp_rank + 1) * local]
            with torch.no_grad():
                target.copy_(src.to(target.dtype))
            return target_name
        return None

    def _place_gdn_tensor(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> str | None:
        """Place one TP-sliced GDN attention tensor into its local param slot.

        The GDN conv dim is ``[q(key_dim), k(key_dim), v(value_dim)]`` and both
        the key and value heads divide evenly by TP, so the fused conv params are
        a non-uniform three-way slice (q/k local rows, v local rows). ``out_proj``
        is row-parallel (split value_dim).
        """
        full = _gdn_params_from_config(self.model.config)
        fkd = full.key_dim
        fvd = full.value_dim
        fnum_v = full.num_v_heads
        lkd = fkd // tp_size
        lvd = fvd // tp_size
        lv = fnum_v // tp_size

        base = name.replace(".linear_attn.", ".attention.")

        if base.endswith(".in_proj_qkv.weight") or base.endswith(".conv1d.weight"):
            if base.endswith(".in_proj_qkv.weight"):
                tname = base[: -len(".weight")]
            else:
                tname = base[: -len(".conv1d.weight")] + ".conv_weight"
            target = params.get(tname)
            if target is None:
                return None
            src = tensor
            if src.ndim == target.ndim + 1 and src.shape[1] == 1:
                src = src.squeeze(1)
            local_conv = target.shape[0]
            with torch.no_grad():
                target[0:lkd].copy_(src[tp_rank * lkd : (tp_rank + 1) * lkd].to(target.dtype))
                target[lkd : 2 * lkd].copy_(
                    src[fkd + tp_rank * lkd : fkd + (tp_rank + 1) * lkd].to(target.dtype)
                )
                target[2 * lkd : local_conv].copy_(
                    src[2 * fkd + tp_rank * lvd : 2 * fkd + (tp_rank + 1) * lvd].to(target.dtype)
                )
            return tname

        if base.endswith(".in_proj_a.weight") or base.endswith(".in_proj_b.weight"):
            if base.endswith(".in_proj_a.weight"):
                tname = base[: -len(".in_proj_a.weight")] + ".in_proj_ba"
            else:
                tname = base[: -len(".in_proj_b.weight")] + ".in_proj_ba"
            target = params.get(tname)
            if target is None:
                return None
            local_v = target.shape[0] // 2
            if base.endswith(".in_proj_a.weight"):
                src = tensor[tp_rank * local_v : (tp_rank + 1) * local_v]
                with torch.no_grad():
                    target[0:local_v].copy_(src.to(target.dtype))
            else:
                src = tensor[fnum_v + tp_rank * local_v : fnum_v + (tp_rank + 1) * local_v]
                with torch.no_grad():
                    target[local_v : 2 * local_v].copy_(src.to(target.dtype))
            return tname

        if base.endswith(".A_log") or base.endswith(".dt_bias"):
            target = params.get(base)
            if target is None:
                return None
            src = tensor[tp_rank * lv : (tp_rank + 1) * lv]
            with torch.no_grad():
                target.copy_(src.to(target.dtype))
            return base

        if base.endswith(".out_proj.weight"):
            tname = base[: -len(".out_proj.weight")] + ".out_proj"
            target = params.get(tname)
            if target is None:
                return None
            src = tensor[:, tp_rank * lvd : (tp_rank + 1) * lvd]
            with torch.no_grad():
                target.copy_(src.to(target.dtype))
            return tname

        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a real (or round-trip) checkpoint into the assembled model.

        Two tensor namespaces are handled in a single streaming pass. Per-rank
        memory stays at ~1/TP of the 512-expert bank: the fused params are
        expert-dimension sliced (T3.1b), so this rank only places its local
        expert range and no full bank is ever materialized (matching the T3.2
        loader):

        * **Per-expert W8A8 tensors** -- any name containing ``.mlp.experts.``
          owned by this rank is routed through the T3.1 mapper
          (:func:`map_expert_tensor`) and copied into its fused ``w13_*``/
          ``w2_*`` slot as it streams off the iterator. Names for peer ranks'
          experts are dropped without placement: the fork's loader-side filter
          skips their ``.weight`` payloads before disk I/O
          (``--enable-expert-parallel --enable-ep-weight-filter``); it still
          delivers the tiny peer scale/offset tensors, and without the filter
          even the weights, so the drop here is the correctness-critical half.
          After the pass, the provided expert index is validated against the
          frozen geometry restricted to this rank's local experts
          (:func:`validate_expert_weight_map`), rejecting missing, extra,
          wrong-dtype or wrong-shape expert tensors.
        * **Non-expert F16 tensors** (router / shared expert / attention / PLE /
          norms / lm_head / embeddings) load by name+shape after the
          ``hf_to_vllm_mapper`` prefix rewrite.

        A state-dict round-trip (fused params by name, no per-expert tensors) is
        also supported: with no ``.mlp.experts.`` names present the expert-set
        validation is skipped and the fused params load directly by name.
        """
        geometry = self._expert_geometry()
        has_experts = geometry["num_experts"] > 0
        tp_rank, tp_size = self.model.expert_sharding
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        # Metadata only (name -> {dtype, shape}); payloads are placed + released as
        # they stream, so this stays tiny even for the full 224 GB checkpoint.
        expert_index: dict[str, dict[str, object]] = {}

        for raw_name, weight in weights:
            if has_experts and is_expert_tensor_name(raw_name):
                if not expert_tensor_is_local(raw_name, geometry, tp_size, tp_rank):
                    continue  # peer-owned expert: placed (and read) by its owner
                mapping = map_expert_tensor(raw_name, geometry, tp_size, tp_rank)
                target = self._place_expert_tensor(params, mapping, weight)
                expert_index[raw_name] = {"dtype": weight.dtype, "shape": tuple(weight.shape)}
                loaded.add(target)
                continue
            # Non-expert tensor: rewrite prefix, then load by name + shape.
            for name, tensor in self.hf_to_vllm_mapper.apply([(raw_name, weight)]):
                param = params.get(name)
                # Vocab-parallel embedding / LM head are dim-0 sharded under a
                # real TP group: route through their weight_loader (which slices
                # the checkpoint's full [vocab, hidden] row to this rank's shard
                # and pads) instead of the strict full-shape copy, which would
                # otherwise silently skip them on TP>1.
                if name == "model.embed_tokens.weight" and param is not None:
                    self.model.embed_tokens.weight_loader(param, tensor)
                    loaded.add(name)
                    continue
                if name == "lm_head.weight" and param is not None:
                    self.lm_head.weight_loader(param, tensor)
                    loaded.add(name)
                    continue
                # Shared expert (dense MLP) is TP-sharded on its intermediate dim:
                # gate/up are column-parallel (split 2*shared_inter), down is
                # row-parallel (split shared_inter). Load this rank's local slice.
                if ".mlp.shared_expert." in name:
                    shared_target = self._place_shared_expert_tensor(params, name, tensor, tp_rank, tp_size)
                    if shared_target is not None:
                        loaded.add(shared_target)
                        continue
                # GDN linear attention is TP-sharded on its heads (q/k + v split
                # evenly, out_proj row-parallel). Load this rank's local slice.
                if ".linear_attn." in name:
                    gdn_target = self._place_gdn_tensor(params, name, tensor, tp_rank, tp_size)
                    if gdn_target is not None:
                        loaded.add(gdn_target)
                        continue
                # Round-trip / already-mapped names (a state-dict by fused param
                # name): strict full-shape copy before any checkpoint remap.
                if param is not None and tuple(param.shape) == tuple(tensor.shape):
                    with torch.no_grad():
                        param.copy_(tensor.to(param.dtype))
                    loaded.add(name)
                    continue
                # Eager-param remap: renames, fusions (shared expert / GDN a+b /
                # PLE key+value), the QSA index_qk split, and documented skips.
                placements = _remap_non_expert(name, self.model.config)
                if placements is None:
                    continue
                for target_name, src_slice, target_offset in placements:
                    target = params.get(target_name)
                    if target is None:
                        continue
                    source = tensor if src_slice is None else tensor[src_slice]
                    # Depthwise Conv1d checkpoints store [C, 1, K]; the eager
                    # params store the flattened [C, K].
                    if source.ndim == target.ndim + 1 and source.shape[1] == 1:
                        source = source.squeeze(1)
                    with torch.no_grad():
                        if target_offset is None:
                            if tuple(target.shape) != tuple(source.shape):
                                continue
                            target.copy_(source.to(target.dtype))
                        else:
                            if target_offset + source.shape[0] > target.shape[0]:
                                continue
                            if tuple(source.shape[1:]) != tuple(target.shape[1:]):
                                continue
                            target[target_offset : target_offset + source.shape[0]].copy_(source.to(target.dtype))
                    loaded.add(target_name)

        # Reject an incomplete / malformed expert set -- only when the checkpoint
        # actually carried per-expert tensors (a by-name round-trip carries none).
        if expert_index:
            validate_expert_weight_map(expert_index, geometry, tp_size=tp_size, tp_rank=tp_rank)
        return loaded

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del intermediate_tensors, kwargs  # single-stage PP; ngram kwargs unused here
        return self.model(input_ids, positions, inputs_embeds)


class AscendQwen4ExpForConditionalGeneration(AscendQwen4ExpForCausalLM):
    """Multimodal-rejecting alias.

    Registered for the ``Qwen4ExpForConditionalGeneration`` architecture so
    such checkpoints route here, then rejected at the first gate: the 310P path
    is text-only.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _reject_multimodal(vllm_config)  # first gate
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def get_multimodal_embeddings(self, *args: object, **kwargs: object):
        raise NotImplementedError(
            "AscendQwen4ExpForConditionalGeneration: multimodal embeddings are "
            "not supported on the Ascend 310P path (text-only)."
        )

    def embed_multimodal(self, *args: object, **kwargs: object):
        raise NotImplementedError(
            "AscendQwen4ExpForConditionalGeneration: multimodal inputs are not "
            "supported on the Ascend 310P path (text-only)."
        )


# Keep a reference so linters don't flag the authoritative singleton import as
# unused; downstream modules import it directly from dtype_policy.
_DEFAULT_DTYPE_POLICY = ASCEND_QWEN4EXP_DTYPE_POLICY

__all__ = [
    "AscendQwen4ExpDecoderLayer",
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
]
