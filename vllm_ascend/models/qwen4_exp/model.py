# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P Qwen4Exp (Qwen3.8-Flash-Next) model skeleton (plan T1.2).

This module owns the importable, registerable package skeleton and wires every
component to the authoritative :mod:`dtype_policy`. The heavy native layer
logic (dense/QSA attention, GDN linear attention, sparse MoE, PLE, n-gram
embedding) is filled by later tasks -- their entry points are present as
clearly-marked ``NotImplementedError`` / ``TODO(Tx.y)`` stubs so the model
constructs on meta device today without dragging any Triton/CUDA import onto
the 310P path.

Heavy upstream layer classes (``Qwen3NextSparseMoeBlock``,
``QwenGatedDeltaNetAttention``, QSA backends) are imported lazily inside the
stubbed builders so importing this package never pulls a kernel module.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
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

from .dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    Qwen4ExpDtypePolicy,
)

# ``VllmConfig`` is only needed for typing; keep import light.
try:  # pragma: no cover - trivial import guard
    from vllm.config import VllmConfig
except Exception:  # pragma: no cover
    VllmConfig = object  # type: ignore[assignment, misc]


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


class AscendQwen4ExpDecoderLayer(nn.Module):
    """Lightweight decoder-layer placeholder.

    Records its layer type and the dtype policy so later tasks can attach the
    real attention / MoE / PLE submodules. Constructs cheaply on meta device;
    calling ``forward`` before those tasks land raises ``NotImplementedError``.
    """

    def __init__(
        self,
        *,
        config: object,
        layer_type: str,
        layer_idx: int,
        dtype_policy: Qwen4ExpDtypePolicy,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.dtype_policy = dtype_policy
        self.prefix = prefix
        # TODO(T4.x/T5.x): build QSA/dense attention, GDN linear attention, the
        # Qwen3NextSparseMoeBlock, PLE injection, and hyperconnection residual
        # here, all reading dtypes from ``dtype_policy``.

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "TODO(T4.x/T5.x): AscendQwen4ExpDecoderLayer.forward "
            f"(layer_type={self.layer_type!r}) is implemented in a later task."
        )


class AscendQwen4ExpModel(nn.Module):
    """Backbone: embeddings + decoder-layer stack + final norm."""

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = getattr(vllm_config, "quant_config", None)
        self.dtype_policy = Qwen4ExpDtypePolicy.from_vllm_config(vllm_config)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=self.dtype_policy.embedding_dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        layer_types = self._resolve_layer_types(config)
        self.layers = nn.ModuleList(
            AscendQwen4ExpDecoderLayer(
                config=config,
                layer_type=layer_types[idx],
                layer_idx=idx,
                dtype_policy=self.dtype_policy,
                prefix=maybe_prefix(prefix, f"layers.{idx}"),
            )
            for idx in range(config.num_hidden_layers)
        )

        # The final norm is a vLLM CustomOp that requires an active
        # ``set_current_vllm_config`` context; build it lazily so the skeleton
        # constructs on meta device outside a config context.
        self.norm: RMSNorm | None = None
        # PP is single-stage in the skeleton; downstream PP splitting is T-later.
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self._mtp_hidden_buffer: torch.Tensor | None = None

    def _build_final_norm(self) -> RMSNorm:
        """Lazily build the final RMSNorm (needs an active vLLM config context).

        Called by later tasks (T4.x/T5.x) from within the runner's
        ``set_current_vllm_config`` scope.
        """
        if self.norm is None:
            self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        return self.norm

    @staticmethod
    def _resolve_layer_types(config: object) -> list[str]:
        num_layers = config.num_hidden_layers
        layer_types = getattr(config, "layer_types", None)
        if not layer_types:
            return ["full_attention"] * num_layers
        # Pad/trim defensively so a tiny random config still constructs.
        resolved = list(layer_types)
        if len(resolved) < num_layers:
            resolved += ["full_attention"] * (num_layers - len(resolved))
        return resolved[:num_layers]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # Alias used by several vLLM code paths.
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_input_ids(input_ids)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError("TODO(T4.x/T5.x/T6.1): AscendQwen4ExpModel.forward is implemented in later tasks.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError(
            "TODO(T3.1): AscendQwen4ExpModel.load_weights with fused-expert mapping is implemented in a later task."
        )


class AscendQwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    MixtureOfExperts,
    IsHybrid,
):
    """Text-only Ascend Qwen4Exp causal LM (skeleton)."""

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
        """Return the hybrid model-state class.

        TODO(T-later): return the concrete Ascend Qwen4Exp model-state class
        (mirrors fork ``nvidia/model_state.py``). The hook surface is stable so
        the runner can bind to it now.
        """
        raise NotImplementedError("TODO(T-later): AscendQwen4ExpModelState is provided by a later task.")

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
        # TODO(T-later): mirror fork MambaStateShapeCalculator wiring; shapes are
        # config-driven and verified against pinned CANN on hardware.
        raise NotImplementedError("TODO(T-later): GDN mamba state shape is provided by a later task.")

    @classmethod
    def get_ple_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        raise NotImplementedError("TODO(T-later): PLE mamba state shape is provided by a later task.")

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    # -- MoE / weight-load hooks ------------------------------------------

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Fused-expert weight-mapping hook.

        TODO(T3.1): return the fused-expert param mapping so ``load_weights``
        can remap serialized per-expert tensors onto the fused MoE weights.
        """
        return []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Per-component weight loader (fused-expert mapping is T3.1).

        Kept as a clean hook: applies ``hf_to_vllm_mapper`` and delegates to the
        (later) fused-expert mapping via :meth:`get_expert_mapping`.
        """
        raise NotImplementedError(
            "TODO(T3.1): AscendQwen4ExpForCausalLM.load_weights fused-expert "
            "mapping is implemented in a later task; get_expert_mapping() is "
            "the wiring hook."
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError(
            "TODO(T4.x/T5.x/T6.1): AscendQwen4ExpForCausalLM.forward is implemented in later tasks."
        )


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
