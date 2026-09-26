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
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.offloader import get_offloader
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

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
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config
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
    AscendQSAFullAttentionSpec,
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
from .ops.qsa_batched_attention_310 import qsa_batched_prefill_310
from .ops.qsa_index_cache_310 import qsa_index_cache_update_310
from .ops.qsa_indexer import QSAGroupSelection, copy_group_selection_into, qsa_indexer_select_groups_310
from .ops.qsa_sparse_attention_310 import qsa_sparse_attention_310
from .ple_layer import AscendQwen4ExpPLELayer
from .qsa import (
    AscendQwen4ExpQSAAttention,
    QSADecoderProjections,
    apply_partial_rope,
    gemma_rmsnorm,
    partial_rope_cos_sin,
    run_qsa_decoder_attention,
)
from .qsa_head_sharding import qsa_head_shard
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
    local_expert_range,
    map_expert_tensor,
    validate_expert_weight_map,
)

_BATCHED_QSA_MIN_PREFILL_TOKENS = 16
_BATCHED_QSA_MAX_DECODE_TOKENS = 2
_BATCHED_QSA_MIN_DECODE_GROUPS = 256

# ``VllmConfig`` is only needed for typing; keep import light.
try:  # pragma: no cover - trivial import guard
    from vllm.config import VllmConfig
except Exception:  # pragma: no cover
    VllmConfig = object  # type: ignore[assignment, misc]

# Layer-type tags mirroring the HF Qwen4Exp ``layer_types`` vocabulary.
_LAYER_TYPE_LINEAR = "linear_attention"
_LAYER_TYPE_FULL = "full_attention"


def _mamba_runtime_spec_kwargs(vllm_config: object) -> dict[str, int | str]:
    """Keep the eager GDN specs aligned with vLLM's MambaBase contract."""
    cache_config = vllm_config.cache_config
    num_speculative_blocks = getattr(vllm_config, "num_speculative_tokens", 0)
    if getattr(cache_config, "use_kda_recoverssm", False):
        num_speculative_blocks = 0
    return {
        "mamba_cache_mode": cache_config.mamba_cache_mode,
        "num_speculative_blocks": num_speculative_blocks,
    }


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
    demand by the lazy transport), and QSA indexer layernorms that the eager
    indexer does not yet own.
    """
    hidden = int(config.hidden_size)
    hc_hidden = int(getattr(config, "hc_count", 2)) * hidden
    index_rows = int(getattr(config, "indexer_n_heads", 4)) * int(getattr(config, "indexer_head_dim", 128))

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
    if name.endswith(".mlp.shared_expert_gate.weight"):
        return plain(name[: -len(".weight")])

    # GDN linear-attention projections (checkpoint `linear_attn` -> model
    # `attention`) are TP-sharded and placed by the dedicated `_place_gdn_tensor`
    # handler in load_weights, so skip them here.
    if ".linear_attn." in name:
        return None

    # QSA sparse attention (checkpoint `self_attn` -> model `attention`).
    if ".self_attn." in name:
        base = name.replace(".self_attn.", ".attention.")
        if base.endswith(".q_proj.weight"):
            # The checkpoint interleaves query and output-gate rows inside each
            # head. ``load_weights`` handles the non-contiguous deinterleave.
            return None
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
        if base.endswith(".indexer.q_layernorm.weight"):
            return plain(base[: -len(".indexer.q_layernorm.weight")] + ".indexer.q_layernorm_weight")
        if base.endswith(".indexer.k_layernorm.weight"):
            return plain(base[: -len(".indexer.k_layernorm.weight")] + ".indexer.k_layernorm_weight")
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


class Qwen4ExpVLProcessingInfo(Qwen3VLProcessingInfo):
    """Use Qwen3-VL preprocessing with the compatible Qwen4Exp config."""

    def get_hf_config(self):
        # Qwen4Exp deliberately reuses Qwen3-VL's processor and vision config
        # schema, but it is not an instance of Qwen3VLConfig. Avoid the base
        # class's nominal type check while retaining all processor semantics.
        return self.ctx.get_hf_config()


class Qwen4ExpVLDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    pass


class Qwen4ExpVLMultiModalProcessor(Qwen3VLMultiModalProcessor):
    pass


# ===========================================================================
# Eager math helpers (Triton-free, deterministic)
# ===========================================================================
_NPU_GROUPED_RMS_NORM_MIN_TOKENS = 256


def _grouped_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    compute_dtype: torch.dtype,
    unit_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """GemmaRMSNorm applied per contiguous ``group_size`` lane (``*(1+w)``)."""
    num_tokens, channels = x.shape
    xc = x.to(compute_dtype)
    wc = weight.to(compute_dtype)
    grouped = xc.view(num_tokens, channels // group_size, group_size)
    if (
        x.device.type == "npu"
        and num_tokens >= _NPU_GROUPED_RMS_NORM_MIN_TOKENS
        and compute_dtype == torch.float32
        and unit_weight is not None
    ):
        # The native op fuses the square/mean/rsqrt/scale sequence.  Its
        # weight is shared across groups, so apply the Gemma per-channel
        # (1 + weight) affine afterwards in FP32 to retain exact semantics.
        import torch_npu

        normalized, _ = torch_npu.npu_rms_norm(grouped.reshape(-1, group_size), unit_weight, eps)
        return normalized.view(num_tokens, channels) * (1.0 + wc)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = (grouped * torch.rsqrt(variance + eps)).reshape(num_tokens, channels)
    return normalized * (1.0 + wc)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, compute_dtype: torch.dtype) -> torch.Tensor:
    """Plain RMSNorm (``* weight``) accumulated in ``compute_dtype``."""
    xc = x.to(compute_dtype)
    variance = xc.square().mean(dim=-1, keepdim=True)
    normalized = xc * torch.rsqrt(variance + eps)
    return normalized * weight.to(compute_dtype)


def _linear_operand_dtype(
    device_type: str,
    weight_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> torch.dtype:
    """Choose the eager GEMM operand dtype without copying NPU weights.

    Norms, routing, and attention reductions still use the policy's FP32
    accumulation dtype.  Casting every projection weight to FP32 at each call,
    however, materializes a full temporary weight matrix and prevents CANN from
    selecting its native FP16 cube path.  Keep the CPU reference path in the
    requested compute dtype, while NPU projections use their checkpoint storage
    dtype.
    """
    return weight_dtype if device_type == "npu" else compute_dtype


def _linear(x: torch.Tensor, weight: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Dtype-safe eager linear with a zero-copy NPU weight fast path."""
    operand_dtype = _linear_operand_dtype(x.device.type, weight.dtype, compute_dtype)
    linear_weight = weight if weight.dtype == operand_dtype else weight.to(operand_dtype)
    return F.linear(x.to(operand_dtype), linear_weight)


def _format_eager_linear_weights_npu(model: nn.Module) -> None:
    """Keep eager FP16 projection weights in the 310P cube's NZ layout.

    The custom Qwen4Exp layers use raw ``nn.Parameter`` weights rather than
    vLLM linear modules, so their load path does not run the usual Ascend
    post-load NZ conversion. Without it, every decode ``F.linear`` converts
    the same large ND weight to NZ again. Only known projection modules are
    visited; expert INT8 weights, embeddings, convolution filters, and scalar
    gates keep their existing formats.
    """
    projection_types = (
        _GatedResidual,
        _EagerDenseAttention,
        _GDNAttention,
        _QSAAttention,
        _EagerMLP,
        _EagerSparseMoE,
    )
    weights_to_format: list[nn.Parameter] = []
    for module in model.modules():
        if not isinstance(module, projection_types):
            continue
        for name, param in module.named_parameters(recurse=False):
            if (
                name != "conv_weight"
                and param.device.type == "npu"
                and param.dtype == ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype
                and param.ndim == 2
                and min(param.shape) >= 16
            ):
                weights_to_format.append(param)
    if not weights_to_format:
        return

    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    with torch.no_grad():
        for param in weights_to_format:
            param.data = torch_npu.npu_format_cast(param.data, ACL_FORMAT_FRACTAL_NZ)


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
        self.register_buffer("_rms_unit_weight", torch.ones(self.hidden_size, dtype=torch.float32), persistent=False)
        self.input_mix_weight_down = nn.Parameter(torch.zeros(lowrank, self.hyper_hidden, dtype=params_dtype))
        self.input_mix_weight_up = nn.Parameter(torch.zeros(self.hyper_hidden, lowrank, dtype=params_dtype))
        if use_combine:
            self.block_inject_weight = nn.Parameter(torch.zeros(hc_count, self.hyper_hidden, dtype=params_dtype))

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        return _grouped_rms_norm(
            hyper_input,
            self.hc_norm_weight,
            self.eps,
            self.hidden_size,
            self.compute_dtype,
            self._rms_unit_weight,
        )

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
    gated delta rule -> out_proj.

    Registered as a ``MambaBase`` so the v1 runner allocates and binds the GDN
    convolution + recurrent state. 310P uses the native causal-convolution
    prefill/decode and chunk/recurrent delta-rule kernels. The torch
    short-convolution path handles non-310P execution.
    """

    def __init__(
        self,
        *,
        config: object,
        dtype_policy: Qwen4ExpDtypePolicy,
        prefix: str = "",
        expert_sharding: tuple[int, int] = (0, 1),
        num_speculative_tokens: int = 0,
    ) -> None:
        super().__init__()
        _register_in_static_forward_context(prefix, self)
        self.prefix = prefix
        self.compute_dtype = dtype_policy.accumulation_dtype
        self.params_dtype = dtype_policy.main_dtype
        self.rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.mamba_conv_dtype = dtype_policy.mamba_conv_cache_dtype
        # The 310P recurrent GDN operator accepts FP16 state only.
        self.mamba_ssm_dtype = dtype_policy.main_dtype
        self.num_speculative_tokens = num_speculative_tokens
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
            raise ValueError(f"GDN heads (k={p.num_k_heads}, v={p.num_v_heads}) not divisible by TP {self.tp_size}")
        self.num_k_heads = p.num_k_heads // self.tp_size
        self.num_v_heads = p.num_v_heads // self.tp_size
        self.key_dim = p.head_k_dim * self.num_k_heads
        self.value_dim = p.head_v_dim * self.num_v_heads
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Parameter(torch.zeros(self.conv_dim, hidden, dtype=self.params_dtype))
        self.in_proj_z = nn.Parameter(torch.zeros(self.value_dim, hidden, dtype=self.params_dtype))
        self.conv_weight = nn.Parameter(torch.zeros(self.conv_dim, p.conv_kernel_size, dtype=self.params_dtype))
        self.in_proj_ba = nn.Parameter(torch.zeros(2 * self.num_v_heads, hidden, dtype=self.params_dtype))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=self.params_dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, dtype=self.params_dtype))
        self.norm_weight = nn.Parameter(torch.zeros(p.head_v_dim, dtype=self.params_dtype))
        self.out_proj = nn.Parameter(torch.zeros(hidden, self.value_dim, dtype=self.params_dtype))
        self._tp_reduce: object | None = None
        if self.tp_size > 1:
            try:
                from vllm.distributed import tensor_model_parallel_all_reduce

                self._tp_reduce = tensor_model_parallel_all_reduce
            except Exception:
                self._tp_reduce = None

    def _stateful_short_conv(
        self,
        mixed: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the 310P stateful op, or the torch fallback."""
        cache = self.kv_cache[0]
        metadata = get_forward_context().attn_metadata[self.prefix]
        spec_metadata = getattr(metadata, "spec_decode_metadata", None)
        if mixed.device.type == "npu" and spec_metadata is not None:
            conv_metadata = spec_metadata.spec_causal_conv1d
            return torch.ops._C_ascend.npu_causal_conv1d_310(
                mixed,
                self.conv_weight.transpose(0, 1),
                bias=None,
                conv_states=cache,
                query_start_loc=conv_metadata.query_start_loc,
                cache_indices=conv_metadata.cache_indices,
                initial_state_mode=None,
                num_accepted_tokens=conv_metadata.num_accepted_tokens,
                activation_mode=1,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=1,
            )
        if mixed.device.type == "npu":
            assert has_initial_state is not None
            if metadata.num_prefills > 0:
                return torch.ops._C_ascend.npu_causal_conv1d_310(
                    mixed,
                    self.conv_weight.transpose(0, 1),
                    bias=None,
                    conv_states=cache,
                    query_start_loc=query_start_loc,
                    cache_indices=state_indices,
                    initial_state_mode=has_initial_state,
                    num_accepted_tokens=None,
                    activation_mode=1,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=0,
                )
            if metadata.num_decodes > 0:
                num_decodes = metadata.num_decodes
                return torch.ops._C_ascend.npu_causal_conv1d_310(
                    mixed[:num_decodes],
                    self.conv_weight.transpose(0, 1),
                    bias=None,
                    conv_states=cache,
                    query_start_loc=None,
                    cache_indices=state_indices[:num_decodes],
                    initial_state_mode=has_initial_state[:num_decodes],
                    num_accepted_tokens=None,
                    activation_mode=1,
                    pad_slot_id=PAD_SLOT_ID,
                    run_mode=1,
                )

        assert has_initial_state is not None
        cache_key = "_qwen4exp_query_ranges"
        ranges = getattr(metadata, cache_key, None)
        if ranges is None:
            # The 310P metadata builder attaches this pinned host tensor. Read
            # it directly here so the CPU fallback does not import torch_npu.
            query_lens_cpu = getattr(metadata, "query_lens_cpu", None)
            if query_lens_cpu is not None and query_lens_cpu.device.type == "cpu":
                boundaries = [0]
                for query_length in query_lens_cpu.tolist():
                    boundaries.append(boundaries[-1] + query_length)
            else:
                boundaries = query_start_loc.to("cpu").tolist()
            ranges = tuple(zip(boundaries[:-1], boundaries[1:]))
            setattr(metadata, cache_key, ranges)

        result = torch.empty_like(mixed)
        weight = self.conv_weight.to(mixed.dtype)
        for request_idx, (start, stop) in enumerate(ranges):
            if stop <= start:
                continue
            cache_idx = state_indices[request_idx].long()
            prior = cache[cache_idx, : self.params.conv_kernel_size - 1].transpose(0, 1).to(mixed.dtype)
            keep_prior = has_initial_state[request_idx].to(mixed.dtype)
            prior = prior * keep_prior
            sequence = mixed[start:stop]
            history = torch.cat([prior, sequence.transpose(0, 1)], dim=-1)
            convolved = (
                F.conv1d(
                    history.unsqueeze(0),
                    weight.unsqueeze(1),
                    groups=self.conv_dim,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            result[start:stop] = F.silu(convolved)
            cache[cache_idx, : self.params.conv_kernel_size - 1].copy_(
                history[:, -(self.params.conv_kernel_size - 1) :].transpose(0, 1).to(cache.dtype)
            )
        return result

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_attn_backend(self) -> type[AttentionBackend]:
        # MambaBase otherwise selects the upstream GDN builder, which exposes
        # speculative masks but does not attach the 310P native convolution
        # metadata required by this model's GDN path during graph capture.
        from vllm_ascend._310p.ops.gdn_attn_builder_310 import AscendGDNAttentionBackend310

        return AscendGDNAttentionBackend310

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.params.conv_kernel_size - 1 + self.num_speculative_tokens, self.conv_dim),
            (self.num_v_heads, self.params.head_v_dim, self.params.head_k_dim),
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (self.mamba_conv_dtype, self.mamba_ssm_dtype)

    def get_kv_cache_spec(self, vllm_config: object) -> MambaSpec:
        # Match the model's per-layer spec: the GDN state is larger than the
        # generic ``MambaBase`` page-size padding, so build the spec without a
        # ``page_size_padded`` (letting it default to the raw state size).
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            **_mamba_runtime_spec_kwargs(vllm_config),
        )

    def _native_gating(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the launch-minimised 310P gate and cache weight-only constants."""
        from vllm_ascend._310p.ops.fla.fused_gdn_gating import (
            fused_gdn_gating_310,
            gdn_gating_constants,
            gdn_gating_tiled_constants,
        )

        cached = getattr(self, "_gdn_gating_cache", None)
        cache_key = (self.A_log.data_ptr(), self.dt_bias.data_ptr())
        if cached is None or cached[0] != cache_key:
            constants = gdn_gating_constants(self.A_log, self.dt_bias)
            tiled_constants = gdn_gating_tiled_constants(self.A_log, self.dt_bias)
            cached = (cache_key, constants, tiled_constants)
            self._gdn_gating_cache = cached
        g, beta = fused_gdn_gating_310(
            self.A_log,
            a,
            b,
            self.dt_bias,
            constants=cached[1],
            tiled_constants=cached[2],
        )
        return g, beta

    def _native_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        metadata: GDNAttentionMetadata,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        has_initial_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one batched native GDN operation instead of a Python token loop."""
        from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310
        from vllm_ascend._310p.ops.fla.gdn_310 import (
            _cached_chunk_plan,
            _cached_recurrent_step_meta,
            npu_recurrent_gated_delta_rule_310,
        )

        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        if metadata.spec_sequence_masks is not None:
            spec_metadata = metadata.spec_decode_metadata
            assert spec_metadata is not None
            return npu_recurrent_gated_delta_rule_310(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                state=self.kv_cache[1],
                cu_seqlens=query_start_loc,
                ssm_state_indices=state_indices,
                num_accepted_tokens=spec_metadata.spec_causal_conv1d.num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
                step_meta=_cached_recurrent_step_meta(
                    metadata,
                    "qwen4exp_spec",
                    query_start_loc,
                    state_indices,
                    v.shape[1],
                    uniform_state_indices=True,
                ),
            ).squeeze(0)
        if metadata.num_prefills > 0:
            assert has_initial_state is not None
            initial_state = self.kv_cache[1][state_indices].contiguous()
            initial_state = initial_state * has_initial_state[:, None, None, None].to(initial_state.dtype)
            out, final_state = chunk_gated_delta_rule_310(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                chunk_plan=_cached_chunk_plan(metadata, query_start_loc),
            )
            assert final_state is not None
            self.kv_cache[1][state_indices] = final_state.to(self.kv_cache[1].dtype)
            return out.squeeze(0)

        return npu_recurrent_gated_delta_rule_310(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state=self.kv_cache[1],
            cu_seqlens=query_start_loc,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            step_meta=_cached_recurrent_step_meta(
                metadata,
                "qwen4exp_decode",
                query_start_loc,
                state_indices,
                v.shape[1],
            ),
        ).squeeze(0)

    def forward(self, block_input: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions  # GDN applies no rotary
        seq_len = block_input.shape[0]
        p = self.params
        mixed = _linear(block_input, self.in_proj_qkv, self.compute_dtype).to(self.params_dtype)
        ba = _linear(block_input, self.in_proj_ba, self.compute_dtype).to(self.params_dtype)
        # Checkpoint packing is [b, a]. The fallback consumes vLLM's per-step
        # metadata and mutates the allocated convolution and recurrent caches
        # for both prefill and decode.
        b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        metadata_by_layer = get_forward_context().attn_metadata if is_forward_context_available() else None
        metadata = metadata_by_layer.get(self.prefix) if isinstance(metadata_by_layer, dict) else None
        if metadata is None:
            mixed = gdn_short_conv(
                mixed,
                self.conv_weight,
                activation="silu",
                compute_dtype=self.compute_dtype,
            )
            ranges = ((0, seq_len),)
            state_indices = None
            has_initial_state = None
        else:
            assert isinstance(metadata, GDNAttentionMetadata)
            seq_len = metadata.num_actual_tokens
            mixed = mixed[:seq_len]
            a = a[:seq_len]
            b = b[:seq_len]
            if metadata.spec_sequence_masks is not None:
                if metadata.num_prefills or metadata.num_decodes:
                    raise NotImplementedError("Qwen4Exp GDN mixed speculative/non-speculative batches")
                state_indices = metadata.spec_state_indices_tensor
                query_start_loc = metadata.spec_query_start_loc
                has_initial_state = None
            else:
                state_indices = metadata.non_spec_state_indices_tensor
                query_start_loc = metadata.non_spec_query_start_loc
                has_initial_state = metadata.has_initial_state
            assert state_indices is not None
            assert query_start_loc is not None
            if has_initial_state is None and metadata.spec_sequence_masks is None:
                default_has_state = metadata.num_prefills == 0
                has_initial_state = torch.full(
                    (query_start_loc.shape[0] - 1,),
                    default_has_state,
                    dtype=torch.bool,
                    device=query_start_loc.device,
                )
            mixed = self._stateful_short_conv(mixed, state_indices, query_start_loc, has_initial_state)
            ranges = getattr(metadata, "_qwen4exp_query_ranges", ())

        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(seq_len, self.num_k_heads, p.head_k_dim)
        k = k.reshape(seq_len, self.num_k_heads, p.head_k_dim)
        v = v.reshape(seq_len, self.num_v_heads, p.head_v_dim)
        if metadata is not None and block_input.device.type == "npu":
            g, beta = self._native_gating(a, b)
            out = self._native_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                metadata,
                state_indices,
                query_start_loc,
                has_initial_state,
            ).to(self.compute_dtype)
        else:
            g, beta = gdn_gating(
                self.A_log,
                a,
                b,
                self.dt_bias,
                compute_dtype=self.compute_dtype,
                backend="eager",
            )
            out = torch.empty(
                (seq_len, self.num_v_heads, p.head_v_dim),
                dtype=self.compute_dtype,
                device=block_input.device,
            )
            for request_idx, (start, stop) in enumerate(ranges):
                if stop <= start:
                    continue
                initial_state = None
                if state_indices is not None and has_initial_state is not None:
                    cache_idx = state_indices[request_idx].long()
                    initial_state = self.kv_cache[1][cache_idx].to(self.compute_dtype) * has_initial_state[
                        request_idx
                    ].to(self.compute_dtype)
                segment, final_state = gdn_delta_rule(
                    q[start:stop],
                    k[start:stop],
                    v[start:stop],
                    g[start:stop],
                    beta[start:stop],
                    initial_state=initial_state,
                    chunked=(stop - start) > 1,
                    chunk_size=QWEN4EXP_GDN_CHUNK_SIZE,
                    compute_dtype=self.compute_dtype,
                    backend="eager",
                )
                out[start:stop] = segment
                if state_indices is not None:
                    self.kv_cache[1][cache_idx].copy_(final_state.to(self.kv_cache[1].dtype))
        normed = _rms_norm(out, self.norm_weight, self.rms_norm_eps, self.compute_dtype)
        z = _linear(block_input, self.in_proj_z, self.compute_dtype).reshape(seq_len, self.num_v_heads, p.head_v_dim)
        out = (normed * torch.sigmoid(z)).reshape(seq_len, self.value_dim)
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
    """QSA layer backed by dedicated 310P index and sparse-attention kernels."""

    def __init__(
        self,
        *,
        config: object,
        layer_idx: int,
        dtype_policy: Qwen4ExpDtypePolicy,
        prefix: str = "",
        expert_sharding: tuple[int, int] = (0, 1),
    ) -> None:
        super().__init__()
        _register_in_static_forward_context(prefix, self)
        self.prefix = prefix
        self.compute_dtype = dtype_policy.accumulation_dtype
        self.params_dtype = dtype_policy.qsa_main_dtype
        hidden = int(config.hidden_size)
        self.head_dim = int(getattr(config, "head_dim", 256))
        self.tp_rank, self.tp_size = expert_sharding
        self.head_shard = qsa_head_shard(
            int(getattr(config, "num_attention_heads", 24)),
            int(getattr(config, "num_key_value_heads", 2)),
            self.head_dim,
            self.tp_rank,
            self.tp_size,
        )
        self.num_heads = self.head_shard.num_query_heads
        self.num_kv_heads = self.head_shard.num_kv_heads
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
        self.attn = AscendQwen4ExpQSAAttention(
            config=config,
            layer_idx=layer_idx,
            dtype_policy=dtype_policy,
            num_query_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self._tp_reduce: object | None = None
        if self.tp_size > 1:
            try:
                from vllm.distributed import tensor_model_parallel_all_reduce

                self._tp_reduce = tensor_model_parallel_all_reduce
            except ImportError:
                pass
        vllm_config = get_current_vllm_config_or_none()
        device_config = getattr(vllm_config, "device_config", None)
        device = getattr(device_config, "device", torch.device("cpu"))
        heads_per_kv_head = self.num_heads // self.num_kv_heads
        self.register_buffer(
            "_qsa_decode_group_list",
            torch.arange(
                1,
                _BATCHED_QSA_MAX_DECODE_TOKENS * self.num_kv_heads + 1,
                dtype=torch.int64,
                device=device,
            )
            * heads_per_kv_head,
            persistent=False,
        )

    def _reduce_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            if self._tp_reduce is None:
                raise RuntimeError("QSA TP needs tensor_model_parallel_all_reduce")
            output = self._tp_reduce(output)
        return output.to(self.params_dtype)

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self._attn_backend is None:
            self._attn_backend = _resolve_attn_backend(self.head_dim, self.params_dtype)
        return self._attn_backend

    def get_kv_cache_spec(self, vllm_config: object) -> AscendQSAFullAttentionSpec:
        del vllm_config
        return AscendQSAFullAttentionSpec(
            block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=self.params_dtype,
        )

    @staticmethod
    def _logical_query_positions(
        metadata: object,
        num_tokens: int,
        device: torch.device,
        rope_positions: torch.Tensor,
        layer_prefix: str | None = None,
    ) -> torch.Tensor:
        """Build causal KV positions without synchronizing an NPU tensor."""
        # The text runner's 1-D RoPE positions already are logical causal
        # positions, including packed requests and chunked-prefill offsets.
        # Reuse them instead of reconstructing them on the host and copying a
        # new tensor to the NPU once per QSA layer. MRoPE axes can differ from
        # causal positions, so that path still uses the request boundaries.
        if rope_positions.ndim == 1:
            return rope_positions[:num_tokens]

        def positions_from_host(current_metadata: object) -> torch.Tensor:
            seq_lens_cpu = getattr(current_metadata, "seq_lens_cpu", None)
            query_lens_cpu = getattr(current_metadata, "query_lens_cpu", None)
            if seq_lens_cpu is None or query_lens_cpu is None:
                raise RuntimeError("Qwen4Exp MRoPE requires host query boundaries for QSA causal selection")
            sequence_lengths = seq_lens_cpu.tolist()
            rows = []
            for sequence_length, query_length in zip(
                sequence_lengths,
                query_lens_cpu.tolist(),
                strict=True,
            ):
                rows.append(torch.arange(sequence_length - query_length, sequence_length, dtype=torch.int64))
            return torch.cat(rows)[:num_tokens]

        capture = BreakableCUDAGraphCapture.current()
        if capture is not None and capture._capturing:
            if layer_prefix is None:
                raise RuntimeError("Qwen4Exp QSA graph capture requires a layer prefix")
            from vllm_ascend.utils import weak_ref_tensor

            # MRoPE axes can differ from causal positions. The host metadata
            # must be read on every replay, but its pageable H2D copy cannot
            # occur inside an NPU graph segment.
            output = torch.empty(num_tokens, dtype=torch.int64, device=device)
            weak_output = weak_ref_tensor(output)

            def copy_current_positions() -> None:
                current_metadata = get_forward_context().attn_metadata[layer_prefix]
                weak_output.copy_(positions_from_host(current_metadata).to(device=device, non_blocking=True))

            capture.add_eager(copy_current_positions)
            return output

        return positions_from_host(metadata).to(device=device, non_blocking=True)

    @staticmethod
    def _dense_prefill_is_exact(metadata: object, token_budget: int) -> bool:
        """Whether QSA's selection contains the complete causal context.

        QSA chooses every visible token until the sequence exceeds its token
        budget.  In that region dense causal attention is mathematically
        identical and lets 310P use its vectorized FlashAttention/SplitFuse
        kernels instead of the scalar sparse kernel.  Consult host metadata
        only so this dispatch never synchronizes an NPU tensor.
        """
        if getattr(metadata, "num_prefills", 0) < 1 or getattr(metadata, "num_decodes", 0):
            return False
        state_name = getattr(getattr(metadata, "attn_state", None), "name", "")
        if state_name not in {"PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"}:
            return False
        seq_lens_cpu = getattr(metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            if seq_lens_cpu.device.type != "cpu" or seq_lens_cpu.numel() == 0:
                return False
            return bool(torch.all(seq_lens_cpu <= token_budget).item())
        seq_lens_list = getattr(metadata, "seq_lens_list", None)
        return bool(seq_lens_list) and max(seq_lens_list) <= token_budget

    @staticmethod
    def _dense_decode_is_exact(metadata: object, token_budget: int) -> bool:
        if getattr(getattr(metadata, "attn_state", None), "name", "") != "DecodeOnly":
            return False
        seq_lens_cpu = getattr(metadata, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and seq_lens_cpu.device.type == "cpu" and seq_lens_cpu.numel():
            return bool(torch.all(seq_lens_cpu <= token_budget).item())
        seq_lens_list = getattr(metadata, "seq_lens_list", None)
        return bool(seq_lens_list) and max(seq_lens_list) <= token_budget

    def _dense_decode_310(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: object,
    ) -> torch.Tensor:
        import torch_npu

        output = torch.empty_like(query)
        context_lens = metadata.seq_lens
        if context_lens.device != query.device:
            context_lens = context_lens.to(device=query.device, non_blocking=True)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.head_dim**-0.5,
            block_table=metadata.block_tables,
            context_lens=context_lens,
            out=output,
        )
        return output

    def _dense_prefill_310(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        metadata: object,
    ) -> torch.Tensor:
        """Run the exact dense-prefix equivalent on optimized 310P kernels."""
        import torch_npu

        from vllm_ascend._310p.attention.attention_mask import (
            AttentionMaskBuilder310,
            is_compressed_mask_supported,
        )
        from vllm_ascend._310p.attention.attention_v1 import (
            MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION,
            MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION,
        )
        from vllm_ascend._310p.attention.metadata_builder import get_query_lens_cpu

        output = torch.empty_like(query)
        state_name = metadata.attn_state.name
        if state_name == "PrefillNoCache":
            if is_compressed_mask_supported():
                torch_npu._npu_flash_attention_v3(
                    query=query,
                    key=key,
                    value=value,
                    mask=metadata.attn_mask,
                    seq_len=metadata.seq_lens,
                    scale_value=self.head_dim**-0.5,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    mask_type=MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION,
                    out=output,
                )
            else:
                torch_npu._npu_flash_attention(
                    query=query,
                    key=key,
                    value=value,
                    mask=metadata.attn_mask,
                    seq_len=metadata.seq_lens,
                    scale_value=self.head_dim**-0.5,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    out=output,
                )
            return output

        query_lens = get_query_lens_cpu(metadata)
        if query_lens is None:
            query_start_loc_cpu = metadata.query_start_loc.cpu()
            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        if metadata.seq_lens.device != query.device:
            metadata.seq_lens = metadata.seq_lens.to(device=query.device, non_blocking=True)
        if is_compressed_mask_supported():
            torch_npu._npu_paged_attention_splitfuse_v2(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                mask=AttentionMaskBuilder310.get_compressed_splitfuse_mask(query.device),
                block_table=metadata.block_tables,
                seq_len=query_lens,
                context_lens=metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.head_dim**-0.5,
                mask_type=MASK_TYPE_NORM_COMPRESS_PAGED_ATTENTION,
                out=output,
            )
        else:
            torch_npu._npu_paged_attention_splitfuse(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                mask=AttentionMaskBuilder310.get_splitfuse_mask(metadata, query.device),
                block_table=metadata.block_tables,
                seq_len=query_lens,
                context_lens=metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.head_dim**-0.5,
                out=output,
            )
        return output

    def forward(self, block_input: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        metadata_by_layer = get_forward_context().attn_metadata if is_forward_context_available() else None
        metadata = metadata_by_layer.get(self.prefix) if isinstance(metadata_by_layer, dict) else None
        if metadata is None or block_input.device.type != "npu":
            return self._reduce_output(
                run_qsa_decoder_attention(
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
            )

        seq_len = metadata.num_actual_tokens
        block_input = block_input[:seq_len]
        positions = positions[..., :seq_len]
        logical_positions = self._logical_query_positions(
            metadata,
            seq_len,
            block_input.device,
            positions,
            self.prefix,
        )
        q = _linear(block_input, self.q_proj, self.compute_dtype).view(seq_len, self.num_heads, self.head_dim)
        k = _linear(block_input, self.k_proj, self.compute_dtype).view(seq_len, self.num_kv_heads, self.head_dim)
        v = _linear(block_input, self.v_proj, self.compute_dtype).view(seq_len, self.num_kv_heads, self.head_dim)
        gate = _linear(block_input, self.gate_proj, self.compute_dtype).view(seq_len, self.num_heads, self.head_dim)
        index_q = _linear(block_input, self.iq_proj, self.compute_dtype).view(
            seq_len, self.index_n_heads, self.index_head_dim
        )
        index_k = _linear(block_input, self.ik_proj, self.compute_dtype)
        q, k = self.attn.project_qk(q, k, positions, positions, accum_dtype=self.compute_dtype)
        q = q.to(self.params_dtype)
        k = k.to(self.params_dtype)
        v = v.to(self.params_dtype)
        gate = gate.to(self.params_dtype)
        index_q = gemma_rmsnorm(
            index_q,
            self.indexer.q_layernorm_weight,
            self.indexer.rms_norm_eps,
            self.compute_dtype,
        )
        index_q = apply_partial_rope(
            index_q,
            positions,
            self.attn.rotary_dim,
            self.attn.rope_theta,
            self.compute_dtype,
            mrope_section=self.attn.mrope_section,
            mrope_interleaved=self.attn.mrope_interleaved,
        ).to(self.params_dtype)
        if positions.ndim == 1:
            index_key_positions = positions - torch.remainder(logical_positions, self.indexer.compress_ratio)
        else:
            # MRoPE axes do not advance monotonically through an image grid.
            # Preserve their exact per-token coordinates; causal group/tail
            # accounting independently uses logical_positions below.
            index_key_positions = positions
        index_key_cos, index_key_sin = partial_rope_cos_sin(
            index_key_positions,
            rotary_dim=self.attn.rotary_dim,
            base=self.attn.rope_theta,
            dtype=self.params_dtype,
            mrope_section=self.attn.mrope_section,
            mrope_interleaved=self.attn.mrope_interleaved,
        )
        from vllm_ascend.device.device_op import DeviceOperator

        key_cache, value_cache = self.kv_cache
        index_cache = getattr(self, "qsa_index_cache", None)
        if index_cache is None:
            raise RuntimeError(f"QSA index cache was not bound for {self.prefix}")
        slot_mapping = metadata.slot_mapping[:seq_len]
        DeviceOperator.reshape_and_cache(k, v, key_cache, value_cache, slot_mapping)
        cache_block_size = key_cache.shape[2]
        qsa_index_cache_update_310(
            index_cache,
            index_k.to(self.params_dtype),
            metadata.query_start_loc,
            slot_mapping,
            self.indexer.k_layernorm_weight,
            index_key_cos,
            index_key_sin,
            block_size=cache_block_size,
            rotary_dim=self.attn.rotary_dim,
            norm_eps=self.indexer.rms_norm_eps,
        )
        if self._dense_prefill_is_exact(metadata, self.indexer.token_topk):
            out = self._dense_prefill_310(q, k, v, key_cache, value_cache, metadata)
        elif self._dense_decode_is_exact(metadata, self.indexer.token_topk):
            out = self._dense_decode_310(q, key_cache, value_cache, metadata)
        else:
            seq_lens_cpu = getattr(metadata, "seq_lens_cpu", None)
            max_visible_groups = None
            max_visible_tokens = None
            if seq_lens_cpu is not None and seq_lens_cpu.device.type == "cpu" and seq_lens_cpu.numel():
                # Scheduler-owned host lengths avoid a device-to-host sync.
                max_visible_tokens = int(seq_lens_cpu.max().item())
                max_visible_groups = max_visible_tokens // self.indexer.compress_ratio
            capture = BreakableCUDAGraphCapture.current()
            if capture is not None and capture._capturing:
                # A graph captured for the first decode step cannot freeze
                # max_visible_groups: the visible QSA pages grow as the
                # request generates. Score/select outside the graph, copying
                # into fixed-width buffers consumed by the next segment.
                from vllm_ascend.utils import weak_ref_tensor

                num_groups = self.indexer.token_topk // self.indexer.compress_ratio
                selection = QSAGroupSelection(
                    group_indices=torch.empty((seq_len, num_groups), dtype=torch.int64, device=index_q.device),
                    group_counts=torch.empty(seq_len, dtype=torch.int64, device=index_q.device),
                    tail_starts=torch.empty(seq_len, dtype=torch.int64, device=index_q.device),
                    tail_counts=torch.empty(seq_len, dtype=torch.int64, device=index_q.device),
                )
                weak_index_q = weak_ref_tensor(index_q)
                weak_index_cache = weak_ref_tensor(index_cache)
                weak_positions = weak_ref_tensor(logical_positions)
                weak_selection = QSAGroupSelection(
                    weak_ref_tensor(selection.group_indices),
                    weak_ref_tensor(selection.group_counts),
                    weak_ref_tensor(selection.tail_starts),
                    weak_ref_tensor(selection.tail_counts),
                )

                def select_current_groups() -> None:
                    current_metadata = get_forward_context().attn_metadata[self.prefix]
                    current_seq_lens = getattr(current_metadata, "seq_lens_cpu", None)
                    current_max_groups = None
                    if (
                        current_seq_lens is not None
                        and current_seq_lens.device.type == "cpu"
                        and current_seq_lens.numel()
                    ):
                        current_max_groups = int(current_seq_lens.max().item()) // self.indexer.compress_ratio
                    current = qsa_indexer_select_groups_310(
                        weak_index_q,
                        weak_index_cache,
                        current_metadata.block_tables,
                        current_metadata.query_start_loc,
                        weak_positions,
                        compress_ratio=self.indexer.compress_ratio,
                        token_topk=self.indexer.token_topk,
                        max_visible_groups=current_max_groups,
                    )
                    copy_group_selection_into(weak_selection, current)

                capture.add_eager(select_current_groups)
            else:
                selection = qsa_indexer_select_groups_310(
                    index_q,
                    index_cache,
                    metadata.block_tables,
                    metadata.query_start_loc,
                    logical_positions,
                    compress_ratio=self.indexer.compress_ratio,
                    token_topk=self.indexer.token_topk,
                    max_visible_groups=max_visible_groups,
                )
            sparse_attention = qsa_sparse_attention_310
            use_batched_prefill = (
                metadata.num_prefills > 0 and metadata.num_decodes == 0 and seq_len >= _BATCHED_QSA_MIN_PREFILL_TOKENS
            )
            use_batched_decode = (
                metadata.num_decodes > 0
                and metadata.num_prefills == 0
                and seq_len <= _BATCHED_QSA_MAX_DECODE_TOKENS
                and selection.group_indices.shape[1] >= _BATCHED_QSA_MIN_DECODE_GROUPS
            )
            if metadata.block_tables.shape[0] == 1 and (use_batched_prefill or use_batched_decode):
                sparse_attention = qsa_batched_prefill_310
            sparse_kwargs = {}
            if sparse_attention is qsa_batched_prefill_310 and use_batched_decode:
                sparse_kwargs["decode_group_list"] = self._qsa_decode_group_list
            if sparse_attention is qsa_batched_prefill_310 and max_visible_tokens is not None:
                block_size = key_cache.shape[2]
                sparse_kwargs["visible_blocks"] = max(1, (max_visible_tokens + block_size - 1) // block_size)
            out = sparse_attention(
                q,
                key_cache,
                value_cache,
                selection,
                metadata.block_tables,
                metadata.query_start_loc,
                scale=self.head_dim**-0.5,
                compress_ratio=self.indexer.compress_ratio,
                **sparse_kwargs,
            )
        out = out * torch.sigmoid(gate)
        return self._reduce_output(_linear(out.reshape(seq_len, -1), self.o_proj, self.compute_dtype))


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


class _Qwen4ExpW8A8PostLoadMethod(QuantizeMethodBase):
    """Prepare the expert bank for 310P grouped and fallback matmuls."""

    # This method only reshapes scale tensors and clones per-expert views on
    # their existing device. Moving all expert parameters to NPU and back is
    # unnecessary, and can exhaust pinned host memory during TP4 loading.
    requires_device_loading = False

    def create_weights(self, layer: nn.Module, *weight_args, **extra_weight_attrs) -> None:
        raise RuntimeError("Qwen4Exp creates its fused expert weights directly")

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("Qwen4Exp routes expert execution through its model forward")

    @staticmethod
    def pack_expert_weight_bank(layer: nn.Module, name: str) -> None:
        """Pack one loaded [in, out] parameter list into NZ [expert, out, in].

        Preserve per-expert parameter names as views into the packed storage so
        state-dict round trips remain compatible without retaining two banks.
        The 310P grouped op consumes FRACTAL_NZ weights: converting once here
        avoids a full expert-bank transpose on every prefill/decode step.
        """
        weights = getattr(layer, name)
        packed = torch.stack([weight.t() for weight in weights], dim=0).contiguous()
        if packed.device.type == "npu":
            # Load-time worker-only import keeps the host reference importable.
            from vllm_ascend.utils import maybe_trans_nz

            packed = maybe_trans_nz(packed)
        setattr(layer, f"{name}_grouped", packed)
        setattr(
            layer,
            name,
            nn.ParameterList(
                nn.Parameter(packed[expert].t(), requires_grad=False) for expert in range(layer.num_local_experts)
            ),
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Weights stream into individual [in, out] parameters to avoid a full
        # checkpoint-bank allocation during loading. Once loaded, pack each
        # local layer into the [expert, out, in] layout consumed by 310P's
        # dynamic-quant grouped matmul. Do this one layer at a time: replacing
        # the original parameters with views releases their storage before the
        # next layer is packed, so steady-state NPU memory does not double.
        scale_dtype = getattr(layer, "params_dtype", ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.num_local_experts, -1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.num_local_experts, -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.num_local_experts, -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.num_local_experts, -1)
        # Materialize independent per-expert FP16 scale vectors once. Views of
        # the banked NPU parameter retain the full bank's physical storage on
        # 310P, which makes WeightQuantBatchMatmulV2 see E*N scales instead of
        # the selected expert's N scales.
        layer.w13_weight_scale_list = [
            scale.to(scale_dtype).clone() for scale in layer.w13_weight_scale.data.unbind(dim=0)
        ]
        layer.w2_weight_scale_list = [
            scale.to(scale_dtype).clone() for scale in layer.w2_weight_scale.data.unbind(dim=0)
        ]
        if hasattr(layer, "w13_weight") and layer.w13_weight[0].device.type == "npu":
            self.pack_expert_weight_bank(layer, "w13_weight")
            self.pack_expert_weight_bank(layer, "w2_weight")


class _EagerSparseMoE(nn.Module):
    """Routed-expert (W8A8) + shared-expert (F16) MoE block (T3.x, stub closed).

    Ports the upstream ``Qwen3NextSparseMoeBlock`` control flow onto the real
    Ascend 310P W8A8_DYNAMIC fused-expert path. The routed experts are held in
    the :class:`AscendW8A8DynamicFusedMoEMethod310` fused layout
    (``w13_*``/``w2_*``, gate+up column-fused) and evaluated with the
    T3.3-validated grouped QDQ math from :mod:`vllm_ascend.models.qwen4_exp.moe`
    (per-token INT8 activation quant, per-channel ``(q - offset) * scale`` weight
    dequant -- real experts are symmetric so ``offset == 0``). NPU execution
    uses the 310P dynamic-quant grouped matmul on packed expert banks, with the
    W8A16 weight-only path as a compatibility fallback. The pure-PyTorch QDQ
    path is retained as the host reference. The router runs in
    the policy ``router_dtype`` (fp32) with ``norm_topk_prob`` renormalization;
    the shared expert stays non-quantized F16 (per the T3.1 mapping contract) and
    is applied densely + unweighted.

    Expert-dimension TP slicing (the TP4 target): ``expert_sharding=(rank,
    size)`` gives this rank a contiguous, possibly uneven slice of experts
    (same linear placement the fork's ``ep_weight_filter`` loader skip uses).
    The router gate (``E_global`` rows)
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
        # The fork loader filter and weight mapper both use this same balanced,
        # contiguous ownership rule. In particular, 512 experts over six ranks
        # yields 86/86/85/85/85/85 without dummy experts or weight transfers.
        self.expert_tp_rank, self.expert_tp_size = (int(expert_sharding[0]), int(expert_sharding[1]))
        if self.expert_tp_size < 1 or not 0 <= self.expert_tp_rank < self.expert_tp_size:
            raise ValueError(f"expert_sharding={expert_sharding} out of range")
        if self.num_experts < self.expert_tp_size:
            raise ValueError(
                f"num_experts={self.num_experts} is smaller than expert TP size {self.expert_tp_size}; "
                "each rank must own at least one expert"
            )
        self.expert_offset, expert_stop = local_expert_range(self.num_experts, self.expert_tp_size, self.expert_tp_rank)
        self.num_local_experts = expert_stop - self.expert_offset
        self.num_global_experts = self.num_experts
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
        #   w13_weight        ParameterList[E_local] of int8 [hidden, 2*moe]
        #                     gate cols [0,moe), up [moe,2moe)
        #   w2_weight         ParameterList[E_local] of int8 [moe, hidden]
        #   w13_weight_scale  float32 [E_local, 2*moe, 1]  (offset likewise, symmetric == 0)
        #   w2_weight_scale   float32 [E_local, hidden, 1]
        # int8 params never require grad (only float/complex tensors may).
        self.w13_weight = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden, 2 * moe_inter, dtype=torch.int8), requires_grad=False)
            for _ in range(self.num_local_experts)
        )
        self.w2_weight = nn.ParameterList(
            nn.Parameter(torch.zeros(moe_inter, hidden, dtype=torch.int8), requires_grad=False)
            for _ in range(self.num_local_experts)
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
        self.quant_method: QuantizeMethodBase | None = None
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name != "torch_npu":
                raise
        else:
            self.quant_method = _Qwen4ExpW8A8PostLoadMethod()

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
            self.shared_expert_gate = nn.Parameter(torch.zeros(1, hidden, dtype=self.params_dtype))

    def forward(self, block_input: torch.Tensor) -> torch.Tensor:
        # Keep the 310P router GEMM on the native FP16 cube path; route_topk
        # promotes its comparatively small logits tensor for softmax/top-k.
        router_logits = _linear(block_input, self.gate, self.router_dtype)
        topk_weights, topk_ids = route_topk(
            router_logits,
            self.top_k,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        npu_expert_weights = block_input.device.type == "npu"
        use_packed_grouped = npu_expert_weights and hasattr(self, "w13_weight_grouped")
        w13_weight = (
            getattr(self, "w13_weight_grouped", self.w13_weight)
            if npu_expert_weights
            else [weight.t() for weight in self.w13_weight]
        )
        w2_weight = (
            getattr(self, "w2_weight_grouped", self.w2_weight)
            if npu_expert_weights
            else [weight.t() for weight in self.w2_weight]
        )
        out = w8a8_grouped_experts(
            block_input,
            topk_weights,
            topk_ids,
            w13_weight,
            self.w13_weight_scale
            if use_packed_grouped
            else getattr(self, "w13_weight_scale_list", self.w13_weight_scale),
            self.w13_weight_offset,
            w2_weight,
            self.w2_weight_scale if use_packed_grouped else getattr(self, "w2_weight_scale_list", self.w2_weight_scale),
            self.w2_weight_offset,
            expert_offset=self.expert_offset,
            num_global_experts=self.num_global_experts,
            use_packed_grouped=use_packed_grouped,
        )
        if self.has_shared_expert:
            shared_gate_up = _linear(block_input, self.shared_gate_up, self.compute_dtype)
            shared_partial = _linear(swiglu_gate_up(shared_gate_up), self.shared_down, self.compute_dtype)
            shared_gate = torch.sigmoid(_linear(block_input, self.shared_expert_gate, self.compute_dtype))
            out = out + shared_partial * shared_gate

        # Routed and shared expert projections are both sharded on their output
        # dimensions.  Their sum is linear with respect to the TP all-reduce, so
        # combine the local partials first and pay for one collective per layer.
        if self.expert_tp_size > 1:
            if self._tp_reduce is None:
                raise RuntimeError(
                    "Qwen4Exp expert-TP MoE needs an all-reduce: vllm.distributed."
                    "tensor_model_parallel_all_reduce was not importable at model init "
                    "and no override is installed on _EagerSparseMoE._tp_reduce."
                )
            out = self._tp_reduce(out)
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
        # The single-request fallback is used by the v1 runner during decode
        # graph capture. Construct these once instead of copying tiny host
        # tensors to the NPU on every token (or inside graph capture).
        self.register_buffer(
            "_single_request_start_locs",
            torch.tensor([0, 1], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_single_request_ngram_context",
            torch.full((1, self.ngram_size - 1), self.eos_token_id, dtype=torch.int64),
            persistent=False,
        )
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

    def _real_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        *,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Real SplitMix64 n-gram row ids for packed serving requests.

        Uses ``AscendQwen4ExpNGramEmbedding.compute_ngram_ids`` (T4.2-verified to
        match the checkpoint's ``layer_multipliers``). Serving supplies packed
        request boundaries and the preceding per-request token history. Eager
        host boots may omit both, in which case a single EOS-padded request is
        used. The returned ids index the full 320M-row padded vocab; when
        ``reduce`` is set (the stub-table boot path) they are reduced modulo the
        synthetic table.
        """
        if (query_start_loc is None) != (ngram_context is None):
            raise ValueError("query_start_loc and ngram_context must be provided together")
        if query_start_loc is None:
            seq_len = int(input_ids.shape[0])
            device = input_ids.device
            if self._single_request_start_locs.device != device:
                self._single_request_start_locs = self._single_request_start_locs.to(device=device, non_blocking=True)
                self._single_request_ngram_context = self._single_request_ngram_context.to(
                    device=device, non_blocking=True
                )
            query_start_loc = (
                self._single_request_start_locs if seq_len == 1 else self._single_request_start_locs * seq_len
            )
            ngram_context = self._single_request_ngram_context
        assert ngram_context is not None
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

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._ensure_ple_method(hidden_states.device)
        if self.ngram is not None:
            # Real hasher: full padded-vocab ids for the lazy-shard table, or
            # reduced to the synthetic stub table on the host dummy-boot path.
            ngram_ids = self._real_ngram_ids(
                input_ids,
                query_start_loc,
                ngram_context,
                reduce=self.checkpoint_dir is None,
            )
        else:
            ngram_ids = self._stub_ngram_ids(input_ids)
        return self.ple(hidden_states, ngram_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return self._forward_eager(hidden_states, input_ids, query_start_loc, ngram_context)

        # The checkpoint's n-gram table is demand-paged on the host. Hashing,
        # gathering, and the following host-to-device copy must run on every
        # replay, outside stream capture. Write into a stable graph-pool tensor
        # so subsequent captured layers always read the same address.
        from vllm_ascend.utils import weak_ref_tensor

        output = torch.empty_like(hidden_states)
        weak_output = weak_ref_tensor(output)
        weak_hidden = weak_ref_tensor(hidden_states)
        weak_input_ids = weak_ref_tensor(input_ids)
        weak_start_loc = weak_ref_tensor(query_start_loc)
        weak_context = weak_ref_tensor(ngram_context)

        def run_ple_eager() -> None:
            weak_output.copy_(self._forward_eager(weak_hidden, weak_input_ids, weak_start_loc, weak_context))

        capture.add_eager(run_ple_eager)
        return output


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
        num_speculative_tokens: int = 0,
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
                num_speculative_tokens=num_speculative_tokens,
            )
        else:
            if getattr(config, "indexer_n_heads", None) is not None:
                self.uses_qsa = True
                self.attention = _QSAAttention(
                    config=config,
                    layer_idx=layer_idx,
                    dtype_policy=dtype_policy,
                    prefix=attn_prefix,
                    expert_sharding=expert_sharding,
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
                config=config,
                dtype_policy=dtype_policy,
                expert_sharding=expert_sharding,
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
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PLE injects into the multi-stream state before the attention block.
        if self.ple is not None:
            hidden_states = self.ple(hidden_states, input_ids, query_start_loc, ngram_context)

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
            get_offloader().wrap_modules(
                (
                    AscendQwen4ExpDecoderLayer(
                        config=config,
                        layer_type=layer_types[idx],
                        layer_idx=idx,
                        dtype_policy=self.dtype_policy,
                        prefix=maybe_prefix(prefix, f"layers.{idx}"),
                        expert_sharding=self.expert_sharding,
                        checkpoint_dir=self.checkpoint_dir,
                        num_speculative_tokens=getattr(vllm_config, "num_speculative_tokens", 0),
                    )
                    for idx in range(config.num_hidden_layers)
                ),
            )
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
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if getattr(speculative_config, "method", None) == "mtp":
            # This buffer must outlive every ACL graph capture. A Python
            # assignment to an intermediate tensor in forward() only runs
            # while capturing; replaying an earlier graph shape would then
            # hand the drafter the last capture's stale HC state.
            device_config = getattr(vllm_config, "device_config", None)
            device = getattr(device_config, "device", torch.device("cpu"))
            max_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
            self.register_buffer(
                "_mtp_hidden_buffer",
                torch.empty(
                    (max_tokens, self.hc_count * self.hidden_size),
                    dtype=self.dtype_policy.main_dtype,
                    device=device,
                ),
                persistent=False,
            )
        else:
            self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hc_count * self.hidden_size
        )

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
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
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
            hidden_states = layer(
                hidden_states,
                positions,
                raw_input_ids,
                query_start_loc,
                ngram_context,
            )

        # Retain the multi-stream state for the MTP drafter (scheme A). The
        # graph records this copy into one stable, externally owned address;
        # replay therefore updates the same buffer regardless of capture size.
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            if num_tokens > self._mtp_hidden_buffer.shape[0]:
                raise ValueError("Qwen4Exp MTP hidden state exceeds the graph-stable buffer")
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states)
        # Final mixer collapses the streams to the sampled single-stream state.
        sample_hidden, _residual = self.hyper_connection_mixer.mix(hidden_states)
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
    # QSA applies the checkpoint's interleaved MRoPE directly from positions.
    # The 310P runner must not look for an external rotary-embedding module.
    uses_model_owned_mrope: ClassVar[Literal[True]] = True

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
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

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
        return (policy.mamba_conv_cache_dtype, policy.main_dtype)

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
        conv_state = (
            params.conv_kernel_size - 1 + getattr(vllm_config, "num_speculative_tokens", 0),
            conv_dim,
        )
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
            MambaAttentionBackendEnum.SHORT_CONV: (MambaStateCopyFuncCalculator.short_conv_state_copy_func()),
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
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                **_mamba_runtime_spec_kwargs(vllm_config),
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
                tp_replicated=True,
                mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
                **_mamba_runtime_spec_kwargs(vllm_config),
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
        if vllm_config is None:
            vllm_config = self.vllm_config
        config = self.config
        policy = self.dtype_policy
        head_dim = int(getattr(config, "head_dim", 0)) or (
            int(config.hidden_size) // int(getattr(config, "num_attention_heads", 1))
        )
        num_kv_heads = int(getattr(config, "num_key_value_heads", 1))
        qsa_num_kv_heads = (
            qsa_head_shard(
                int(getattr(config, "num_attention_heads", 1)),
                num_kv_heads,
                head_dim,
                *self.model.expert_sharding,
            ).num_kv_heads
            if getattr(config, "indexer_n_heads", None) is not None
            else num_kv_heads
        )

        try:
            gdn_params = _gdn_params_from_config(config)
        except Exception:  # pragma: no cover - GDN geometry may be absent
            gdn_params = None

        spec: dict[str, KVCacheSpec] = {}
        gdn_shapes = self.get_gdn_mamba_state_shape_from_config(vllm_config) if gdn_params is not None else ()
        mamba_runtime_kwargs = _mamba_runtime_spec_kwargs(vllm_config)
        for idx, layer_type in enumerate(self.model.layer_types):
            name = f"model.layers.{idx}.attention"
            if layer_type == _LAYER_TYPE_LINEAR and gdn_params is not None:
                spec[name] = MambaSpec(
                    shapes=gdn_shapes,
                    dtypes=(policy.mamba_conv_cache_dtype, policy.main_dtype),
                    block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    **mamba_runtime_kwargs,
                )
            else:
                spec_cls = (
                    AscendQSAFullAttentionSpec
                    if layer_type == _LAYER_TYPE_FULL and getattr(config, "indexer_n_heads", None) is not None
                    else FullAttentionSpec
                )
                spec[name] = spec_cls(
                    block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
                    num_kv_heads=qsa_num_kv_heads if spec_cls is AscendQSAFullAttentionSpec else num_kv_heads,
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
        gdn_shapes = self.get_gdn_mamba_state_shape_from_config(self.vllm_config) if gdn_params is not None else ()
        mamba_runtime_kwargs = _mamba_runtime_spec_kwargs(self.vllm_config)

        for idx, layer_type in enumerate(self.model.layer_types):
            name = f"model.layers.{idx}"
            if layer_type == _LAYER_TYPE_LINEAR and gdn_params is not None:
                mamba_layers[f"{name}.linear_attn"] = MambaSpec(
                    shapes=gdn_shapes,
                    dtypes=(policy.mamba_conv_cache_dtype, policy.main_dtype),
                    block_size=DEFAULT_ATTENTION_BLOCK_SIZE,
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    **mamba_runtime_kwargs,
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

        ``mapping`` is the T3.1 :class:`ExpertTensorMapping`. Quantized weights
        land transposed in the expert-specific ``.{expert_index}`` parameter at
        ``[:, row_start:row_stop]``; compact scale/offset banks retain the
        leading expert dimension. Rejects a dtype/shape mismatch with the
        mapper's own error taxonomy before any copy.
        """
        target_name = f"model.layers.{mapping.layer}.mlp.{mapping.target_param}"
        # Quantized weights are registered as individual expert parameters so
        # their post-load NZ conversion never duplicates a full layer bank.
        # Scale/offset tensors remain compact fused banks.
        param_name = f"{target_name}.{mapping.expert_index}" if mapping.kind == "weight" else target_name
        param = params.get(param_name)
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
            if mapping.kind == "weight":
                param[:, mapping.row_start : mapping.row_stop].copy_(weight.t().to(param.dtype))
            else:
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
                target[lkd : 2 * lkd].copy_(src[fkd + tp_rank * lkd : fkd + (tp_rank + 1) * lkd].to(target.dtype))
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
            # in_proj_a and in_proj_b are separate [num_v_heads, hidden]
            # tensors. Match vLLM's packed_modules_mapping order [b, a].
            src = tensor[tp_rank * local_v : (tp_rank + 1) * local_v]
            with torch.no_grad():
                if base.endswith(".in_proj_b.weight"):
                    target[0:local_v].copy_(src.to(target.dtype))
                else:
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

        if base.endswith(".in_proj_z.weight"):
            tname = base[: -len(".weight")]
            target = params.get(tname)
            if target is None:
                return None
            src = tensor[tp_rank * lvd : (tp_rank + 1) * lvd]
            with torch.no_grad():
                target.copy_(src.to(target.dtype))
            return tname

        if base.endswith(".norm.weight"):
            tname = base[: -len(".weight")] + "_weight"
            target = params.get(tname)
            if target is None:
                return None
            # GDN RMSNorm is over head_v_dim and is shared by all value heads,
            # so every TP rank receives the complete vector.
            with torch.no_grad():
                target.copy_(tensor.to(target.dtype))
            return tname

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

    def _place_qsa_q_gate_tensor(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        tensor: torch.Tensor,
    ) -> tuple[str, str] | None:
        """Deinterleave a QSA checkpoint's per-head ``[query, gate]`` rows."""
        if not name.endswith(".self_attn.q_proj.weight"):
            return None
        stem = name[: -len(".self_attn.q_proj.weight")] + ".attention"
        q_name = stem + ".q_proj"
        gate_name = stem + ".gate_proj"
        q_target = params.get(q_name)
        gate_target = params.get(gate_name)
        if q_target is None or gate_target is None:
            return None
        num_heads = int(self.model.config.num_attention_heads)
        head_dim = int(self.model.config.head_dim)
        hidden_size = int(self.model.config.hidden_size)
        expected_shape = (num_heads * 2 * head_dim, hidden_size)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name}: expected interleaved QSA q/gate shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        tp_rank, tp_size = getattr(self.model, "expert_sharding", (0, 1))
        shard = qsa_head_shard(
            num_heads,
            int(getattr(self.model.config, "num_key_value_heads", 1)),
            head_dim,
            tp_rank,
            tp_size,
        )
        per_head = tensor.reshape(num_heads, 2, head_dim, hidden_size)
        local_heads = per_head[shard.query_start : shard.query_start + shard.num_query_heads]
        with torch.no_grad():
            q_target.copy_(local_heads[:, 0].reshape_as(q_target).to(q_target.dtype))
            gate_target.copy_(local_heads[:, 1].reshape_as(gate_target).to(gate_target.dtype))
        return q_name, gate_name

    def _place_qsa_head_tensor(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> str | None:
        """Slice checkpoint K/V rows or output columns by their GQA group."""
        suffixes = ("k_proj", "v_proj", "o_proj")
        suffix = next((item for item in suffixes if name.endswith(f".self_attn.{item}.weight")), None)
        if suffix is None:
            return None
        target_name = name.replace(".self_attn.", ".attention.")[: -len(".weight")]
        target = params.get(target_name)
        if target is None:
            return None
        config = self.model.config
        if getattr(config, "indexer_n_heads", None) is None:
            return None
        shard = qsa_head_shard(
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(config.head_dim),
            tp_rank,
            tp_size,
        )
        hidden_size = int(config.hidden_size)
        expected_shape = (
            (hidden_size, int(config.num_attention_heads) * shard.head_dim)
            if suffix == "o_proj"
            else (int(config.num_key_value_heads) * shard.head_dim, hidden_size)
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name}: expected QSA shape {expected_shape}, got {tuple(tensor.shape)}")
        local_tensor = tensor[:, shard.query_rows] if suffix == "o_proj" else tensor[shard.kv_rows]
        if tuple(local_tensor.shape) != tuple(target.shape):
            raise ValueError(f"{name}: local shape {tuple(local_tensor.shape)} != {tuple(target.shape)}")
        with torch.no_grad():
            target.copy_(local_tensor.to(target.dtype))
        return target_name

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
                qsa_targets = self._place_qsa_q_gate_tensor(params, name, tensor)
                if qsa_targets is not None:
                    loaded.update(qsa_targets)
                    continue
                qsa_target = self._place_qsa_head_tensor(params, name, tensor, tp_rank, tp_size)
                if qsa_target is not None:
                    loaded.add(qsa_target)
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
        _format_eager_linear_weights_npu(self.model)
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
        del intermediate_tensors  # single-stage PP
        return self.model(
            input_ids,
            positions,
            inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen4ExpVLMultiModalProcessor,
    info=Qwen4ExpVLProcessingInfo,
    dummy_inputs=Qwen4ExpVLDummyInputsBuilder,
)
class AscendQwen4ExpForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
):
    """Qwen3-VL vision frontend backed by the Ascend Qwen4Exp language model."""

    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True
    uses_model_owned_mrope: ClassVar[Literal[True]] = True
    requires_raw_input_tokens = True

    @classmethod
    def get_model_state_cls(cls):
        return AscendQwen4ExpForCausalLM.get_model_state_cls()

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return AscendQwen4ExpForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return AscendQwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return AscendQwen4ExpForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return AscendQwen4ExpForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        return AscendQwen4ExpForCausalLM.get_mamba_specs_from_config(vllm_config)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Avoid building a Qwen3 text decoder merely to replace it with the much
        # larger Qwen4Exp hybrid decoder. The frontend is shared; the language
        # model is constructed exactly once below.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError("Qwen4Exp conditional generation requires multimodal_config")

        self.config = config
        self.model_config = vllm_config.model_config
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        pruning_spec = multimodal_config.get_video_pruning_spec()
        if pruning_spec is None:
            self.video_pruning_method = None
            self.video_pruning_rate = multimodal_config.video_pruning_rate
        else:
            self.video_pruning_method, self.video_pruning_rate = pruning_spec
        self.is_multimodal_pruning_enabled = multimodal_config.is_multimodal_pruning_enabled()

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        visual_indexes = getattr(config.vision_config, "deepstack_visual_indexes", [])
        self.use_deepstack = bool(visual_indexes)
        self.deepstack_num_level = len(visual_indexes)
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = AscendQwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream vision tensors to the ViT and all other tensors to Qwen4Exp."""
        visual_loaded: set[str] = set()

        def language_weights():
            for name, weight in weights:
                if name.startswith("model.visual."):
                    visual_name = name.removeprefix("model.visual.")
                    loaded = self.visual.load_weights([(visual_name, weight)])
                    visual_loaded.update(f"visual.{item}" for item in loaded)
                else:
                    yield name, weight

        language_loaded = self.language_model.load_weights(language_weights())
        return visual_loaded | {f"language_model.{name}" for name in language_loaded}


# Keep a reference so linters don't flag the authoritative singleton import as
# unused; downstream modules import it directly from dtype_policy.
_DEFAULT_DTYPE_POLICY = ASCEND_QWEN4EXP_DTYPE_POLICY

__all__ = [
    "AscendQwen4ExpDecoderLayer",
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
]
