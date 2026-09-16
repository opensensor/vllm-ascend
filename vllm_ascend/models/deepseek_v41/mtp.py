# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P DeepSeek V4.1 MTP-3 drafter + DSpark wiring (plan E4.2).

This is an *adaptation* of the shipped, ``torch_npu``-native
``deepseek_v4`` speculative-decode path -- the serial multi-token predictor
(``vllm_ascend.models.deepseek_v4.mtp:DeepSeekV4MTP`` /
``DeepSeekMultiTokenPredictorLayer``) and the block drafter
(``vllm_ascend.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM``) -- NOT a
green-field port. Both shipped modules top-level import heavy vLLM machinery
(``ReplicatedLinear``, ``support_torch_compile``, ``FusedMoE`` ...) and, through
``DeepseekV4DecoderLayer``, the shipped ``muls_add_triton`` op, so neither is
importable on the CPU host (no NPU, no Triton on the 310P dev lane).

So -- exactly like the E4.1 assembly (``assembly.py``) -- this module provides a
**host-constructible, Triton-free** MTP-3 drafter that reuses the E4.1 *eager*
decoder-layer component (:class:`AscendDeepseekV41EagerDecoderLayer`) as the
per-step transformer block, and carries the V4.1 MTP / DSpark config so the
assembly / runner can wire the on-device path at a follow-on S-task.

What is real here (E4.2)
------------------------
* ``num_nextn_predict_layers = 3`` MTP layers, each with the
  ``eh_proj`` / ``enorm`` / ``hnorm`` / ``shared_head`` plumbing read off the
  config, adapting the shipped ``DeepSeekMultiTokenPredictorLayer``.
* Each MTP layer's transformer block reuses the E4.1 eager
  ``AscendDeepseekV41EagerDecoderLayer`` (the same MLA / indexer / W2-MoE /
  Engram-free eager stack the host boot already exercises).
* A deterministic, host-runnable single-step draft ``forward`` +
  ``compute_logits`` (no runner needed).
* The DSpark config (``dspark_target_layer_ids=[37,38,39]``,
  ``dspark_n_routed_experts=128``, top-3, ``block_size=5``,
  ``markov_rank=256``) is recorded on the model via :class:`DSparkDraftSpec`
  so the assembly / runner can apply the block drafter.

What is deferred to a follow-on S-task
--------------------------------------
* The multi-step speculative-decode orchestration (runner-driven
  ``spec_step_idx`` loop, target-layer hidden-state feeding, verification /
  rejection sampling).
* The on-device DSpark *block* drafter (Markov head + confidence head +
  ``block_size``-wide draft) -- the shipped
  ``DSparkDeepseekV4ForCausalLM`` is resolved lazily via
  :func:`shipped_dspark_base` and applied there.
* Real checkpoint ``load_weights`` (shipped ``mtp.*`` -> V4.1 remap).
* Repointing the ``DeepSeekV41MTPModel`` registry row from the E2.1 fail-fast
  ``model:DeepSeekV41MTP`` stub to this real class (a one-line change in
  ``models/__init__.py``, out of scope for E4.2 which must not touch the
  registration).

Triton hygiene (grep-gate)
--------------------------
This module contains ZERO ``triton`` references in code. The E4.1 eager
components and the shipped device drafters are resolved **lazily** (inside
construction / via the ``shipped_*_base`` helpers), never at module import, so
``import vllm_ascend.models.deepseek_v41.mtp`` pulls neither Triton nor
``torch_npu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from .dtype_policy import ASCEND_DEEPSEEKV41_DTYPE_POLICY, DeepseekV41DtypePolicy

if TYPE_CHECKING:  # keep runtime imports light (no heavy vLLM / Triton pull)
    from vllm.config import VllmConfig


# ===========================================================================
# V4.1 MTP / DSpark config contract (authoritative defaults; read off config)
# ===========================================================================

# MTP-3: three serial next-token-prediction layers (DeepSeek V4.1 Flash).
DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS = 3
# DSpark block drafter attaches to the last three decoder layers.
DEEPSEEKV41_DSPARK_TARGET_LAYER_IDS = (37, 38, 39)
# DSpark routed-expert geometry (decoupled from the 384-expert backbone).
DEEPSEEKV41_DSPARK_N_ROUTED_EXPERTS = 128
DEEPSEEKV41_DSPARK_NUM_EXPERTS_PER_TOK = 3
DEEPSEEKV41_DSPARK_BLOCK_SIZE = 5
DEEPSEEKV41_DSPARK_MARKOV_RANK = 256


# ===========================================================================
# Eager math helpers (Triton-free, deterministic; dtype-safe like E4.1)
# ===========================================================================
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, compute_dtype: torch.dtype) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(compute_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.to(compute_dtype)).to(orig_dtype)


def _linear(x: torch.Tensor, weight: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """Dtype-safe eager matmul: ``x @ weight.T`` in ``compute_dtype``."""
    out = torch.matmul(x.to(compute_dtype), weight.to(compute_dtype).transpose(-1, -2))
    return out.to(x.dtype)


def _seeded(shape: tuple[int, ...], seed: int, scale: float, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * scale).to(dtype)


# ===========================================================================
# Lazy component / shipped-base resolution (keeps the import Triton-free)
# ===========================================================================
def _eager_decoder_components() -> tuple[type, Any]:
    """Lazily import the E4.1 eager decoder-layer component + expert-bank builder.

    Imported here (not at module top) so that merely importing this module
    stays Triton-free and free of the assembly's ``tests``/``tools`` reference
    coupling. The assembly import itself pulls neither Triton nor ``torch_npu``
    (see ``tests/ut/deepseek_w2/test_deepseekv41_assembly.py``).
    """
    from .assembly import AscendDeepseekV41EagerDecoderLayer, build_dummy_w2_expert_bank

    return AscendDeepseekV41EagerDecoderLayer, build_dummy_w2_expert_bank


def shipped_mtp_base() -> type:
    """Lazily import the shipped ``DeepSeekV4MTP`` serial multi-token predictor.

    TODO(S-task): the on-device MTP-3 path reuses this class (which inherits the
    E3.3 W2 / eager-combine fixes automatically). Importing it transitively
    pulls ``torch_npu`` + the shipped ``muls_add_triton`` op, so it is resolved
    only when the runner wires the device path -- never at module import.
    """
    from vllm_ascend.models.deepseek_v4.mtp import DeepSeekV4MTP

    return DeepSeekV4MTP


def shipped_dspark_base() -> type:
    """Lazily import the shipped ``DSparkDeepseekV4ForCausalLM`` block drafter.

    TODO(S-task): the DSpark block-draft path (Markov head + confidence head +
    ``block_size``-wide draft over ``dspark_target_layer_ids``) reuses this
    class. Guarded/lazy so the import path stays Triton-free / ``torch_npu``-free
    on the 310P host.
    """
    from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM

    return DSparkDeepseekV4ForCausalLM


def _get_dspark_num_mtp_layers(config: Any) -> int:
    """Mirror the shipped ``dspark._get_dspark_num_mtp_layers`` resolution."""
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "num_nextn_predict_layers", DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS)
    return int(num_layers or DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS)


# ===========================================================================
# DSpark config record (reuses the shipped dspark.py config plumbing)
# ===========================================================================
@dataclass(frozen=True)
class DSparkDraftSpec:
    """Frozen DSpark block-drafter config, recorded for the assembly / runner.

    Adapts the shipped ``DeepseekV4DSparkModel`` config reads
    (``dspark_target_layer_ids`` / ``dspark_block_size`` /
    ``dspark_markov_rank`` / ``dspark_num_mtp_layers``) into a device-free
    record so the on-device block drafter (``shipped_dspark_base()``) can be
    parametrised without importing ``torch_npu`` on the host.
    """

    target_layer_ids: tuple[int, ...]
    n_routed_experts: int
    num_experts_per_tok: int
    block_size: int
    markov_rank: int
    num_mtp_layers: int

    @classmethod
    def from_config(cls, config: Any) -> DSparkDraftSpec:
        target = getattr(config, "dspark_target_layer_ids", DEEPSEEKV41_DSPARK_TARGET_LAYER_IDS)
        n_routed = getattr(
            config,
            "dspark_n_routed_experts",
            getattr(config, "n_routed_experts", DEEPSEEKV41_DSPARK_N_ROUTED_EXPERTS),
        )
        return cls(
            target_layer_ids=tuple(int(i) for i in target),
            n_routed_experts=int(n_routed),
            num_experts_per_tok=int(
                getattr(config, "dspark_num_experts_per_tok", DEEPSEEKV41_DSPARK_NUM_EXPERTS_PER_TOK)
            ),
            block_size=int(getattr(config, "dspark_block_size", DEEPSEEKV41_DSPARK_BLOCK_SIZE)),
            markov_rank=int(getattr(config, "dspark_markov_rank", DEEPSEEKV41_DSPARK_MARKOV_RANK)),
            num_mtp_layers=_get_dspark_num_mtp_layers(config),
        )


def _resolve_text_config(config: Any, vllm_config: Any) -> Any:
    """Return the effective V4.1 text config from an explicit config or a VllmConfig.

    Config plumbing only (mirrors ``model._resolve_v41_text_config``): a draft
    ``VllmConfig`` nests the language-model fields under
    ``speculative_config.draft_model_config.hf_config`` (possibly wrapped in a
    multimodal ``text_config``); a plain config is returned as-is.
    """
    if config is not None:
        return config
    if vllm_config is None:
        raise ValueError("DeepSeekV41MTP requires either config= or vllm_config=.")

    spec_config = getattr(vllm_config, "speculative_config", None)
    draft_config = getattr(spec_config, "draft_model_config", None)
    hf_config = getattr(draft_config, "hf_config", None)
    if hf_config is not None:
        return getattr(hf_config, "text_config", None) or hf_config

    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is not None:
        return text_config
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "text_config", None) or hf_config


# ===========================================================================
# MTP-3 drafter (host-constructible; reuses the E4.1 eager decoder layer)
# ===========================================================================
class _SharedHead(nn.Module):
    """Adapts the shipped ``SharedHead``: an RMSNorm + tied LM head.

    Host-constructible eager stand-in (plain ``nn.Parameter``s) for the shipped
    ``RMSNorm`` + ``ParallelLMHead`` so the class builds on CPU.
    """

    def __init__(self, *, hidden_size: int, vocab_size: int, eps: float, dtype_policy: DeepseekV41DtypePolicy,
                 seed: int) -> None:
        super().__init__()
        self.eps = eps
        self.accum_dtype = dtype_policy.accumulation_dtype
        self.norm = nn.Parameter(torch.ones(hidden_size, dtype=dtype_policy.main_dtype))
        self.head = nn.Parameter(_seeded((vocab_size, hidden_size), seed, 0.05, dtype_policy.lm_head_dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.norm, self.eps, self.accum_dtype)


class DeepSeekV41MTPLayer(nn.Module):
    """One MTP next-token-prediction layer (adapts ``DeepSeekMultiTokenPredictorLayer``).

    Carries the ``enorm`` / ``hnorm`` / ``eh_proj`` / ``shared_head`` plumbing and
    reuses the E4.1 eager decoder-layer component as the ``mtp_block``.
    """

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        dtype_policy: DeepseekV41DtypePolicy,
        eager_decoder_layer_cls: type,
        w2_experts: list,
        shared_expert: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype_policy = dtype_policy
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.main_dtype = dtype_policy.main_dtype
        self.accum_dtype = dtype_policy.accumulation_dtype

        hidden = int(config.hidden_size)
        self.hidden_size = hidden
        self.hc_mult = int(getattr(config, "hc_mult", 2))

        # -- MTP projection plumbing (adapts the shipped e_proj/h_proj+norms) --
        self.enorm = nn.Parameter(torch.ones(hidden, dtype=self.main_dtype))
        self.hnorm = nn.Parameter(torch.ones(hidden, dtype=self.main_dtype))
        # Canonical DeepSeek MTP fuses [enorm(embed) ; hnorm(prev_hidden)] with a
        # single eh_proj (2*hidden -> hidden), matching the shipped
        # ``_rewrite_spec_layer_name`` weight key ``eh_proj``.
        self.eh_proj = nn.Parameter(_seeded((hidden, 2 * hidden), 4100 + layer_idx, 0.05, self.main_dtype))
        self.shared_head = _SharedHead(
            hidden_size=hidden,
            vocab_size=int(config.vocab_size),
            eps=self.eps,
            dtype_policy=dtype_policy,
            seed=4200 + layer_idx,
        )

        # -- transformer block: reuse the E4.1 eager decoder layer -------------
        # Ratio 0 (dense MLA, no indexer, no Engram) keeps the draft block light
        # and deterministic; the DSpark sparse-expert override is recorded
        # separately (DSparkDraftSpec) and applied on-device by the runner.
        self.mtp_block = eager_decoder_layer_cls(
            config=config,
            layer_idx=layer_idx,
            compress_ratio=0,
            dtype_policy=dtype_policy,
            engram=None,
            engram_hash_index=None,
            w2_experts=w2_experts,
            shared_expert=shared_expert,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic host single-step draft. ``[T, hidden]`` in, ``[T, hidden]`` out."""
        normed_embeds = _rms_norm(inputs_embeds, self.enorm, self.eps, self.accum_dtype)
        normed_hidden = _rms_norm(previous_hidden_states, self.hnorm, self.eps, self.accum_dtype)
        fused = torch.cat([normed_embeds, normed_hidden], dim=-1)
        hidden = _linear(fused, self.eh_proj, self.main_dtype).to(self.main_dtype)

        # Lift into the hc_mult multi-stream residual the eager decoder expects.
        stream = hidden.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        stream = self.mtp_block(stream, positions, None)
        return stream.to(self.accum_dtype).mean(dim=1).to(self.main_dtype)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.shared_head(hidden_states)
        return _linear(normed, self.shared_head.head, self.accum_dtype)


class DeepSeekV41MTP(nn.Module):
    """Host-constructible MTP-3 drafter for DeepSeek V4.1 (adapts ``DeepSeekV4MTP``).

    Constructs on CPU from a V4.1 text config (or a draft ``VllmConfig``),
    exposes ``num_nextn_predict_layers == 3`` and
    ``dspark_target_layer_ids == (37, 38, 39)``, records the DSpark block-drafter
    config, and runs a deterministic single-step host draft. The device MTP-3 /
    DSpark orchestration is deferred to a follow-on S-task (see module docstring).
    """

    # Class-level contract so the registered class reports the V4.1 geometry
    # even before an instance is built (mirrors the E2.1 stub attributes).
    num_nextn_predict_layers = DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS
    dspark_target_layer_ids = DEEPSEEKV41_DSPARK_TARGET_LAYER_IDS

    def __init__(
        self,
        *,
        config: Any = None,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cfg = _resolve_text_config(config, vllm_config)
        self.config = cfg
        self.prefix = prefix

        dtype_policy = (
            DeepseekV41DtypePolicy.from_vllm_config(vllm_config)
            if vllm_config is not None
            else ASCEND_DEEPSEEKV41_DTYPE_POLICY
        )
        self.dtype_policy = dtype_policy
        self.main_dtype = dtype_policy.main_dtype

        # Instance geometry (config-driven; falls back to the V4.1 defaults).
        self.num_nextn_predict_layers = int(
            getattr(cfg, "num_nextn_predict_layers", DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS)
        )
        self.mtp_start_layer_idx = int(getattr(cfg, "num_hidden_layers", 40))
        self.hidden_size = int(cfg.hidden_size)
        self.vocab_size = int(cfg.vocab_size)

        # DSpark block-drafter config recorded for the assembly / runner (E4.2).
        self.dspark_spec = DSparkDraftSpec.from_config(cfg)
        self.dspark_target_layer_ids = self.dspark_spec.target_layer_ids

        # Shared token embedding (adapts the shipped MTP ``embed_tokens``).
        self.embed_tokens = nn.Parameter(
            _seeded((self.vocab_size, self.hidden_size), 4000, 0.05, dtype_policy.embedding_dtype)
        )

        # Reuse the E4.1 eager decoder layer as each MTP step's transformer
        # block, sharing one dummy W2 expert bank across the three layers.
        eager_decoder_layer_cls, build_bank = _eager_decoder_components()
        num_experts = int(cfg.n_routed_experts)
        moe_inter = int(cfg.moe_intermediate_size)
        w2_experts = build_bank(num_experts, self.hidden_size, moe_inter, seed=4300)
        shared_expert = build_bank(1, self.hidden_size, moe_inter, seed=4355)[0]

        self.layers = nn.ModuleList(
            [
                DeepSeekV41MTPLayer(
                    config=cfg,
                    layer_idx=self.mtp_start_layer_idx + step,
                    dtype_policy=dtype_policy,
                    eager_decoder_layer_cls=eager_decoder_layer_cls,
                    w2_experts=w2_experts,
                    shared_expert=shared_expert,
                )
                for step in range(self.num_nextn_predict_layers)
            ]
        )

    # -- reporting -----------------------------------------------------------
    def mtp_layer_count(self) -> int:
        return len(self.layers)

    def dspark_config(self) -> DSparkDraftSpec:
        return self.dspark_spec

    def report(self) -> dict[str, Any]:
        """Config summary the assembly / runner reads to wire the device path."""
        return {
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
            "mtp_start_layer_idx": self.mtp_start_layer_idx,
            "dspark_target_layer_ids": self.dspark_spec.target_layer_ids,
            "dspark_n_routed_experts": self.dspark_spec.n_routed_experts,
            "dspark_num_experts_per_tok": self.dspark_spec.num_experts_per_tok,
            "dspark_block_size": self.dspark_spec.block_size,
            "dspark_markov_rank": self.dspark_spec.markov_rank,
        }

    # -- host draft ----------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.long(), self.embed_tokens.to(self.main_dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Deterministic single-step draft for MTP step ``spec_step_idx``.

        The multi-step verify/rollback loop over ``spec_step_idx`` is the
        runner's job (deferred S-task); a single step is fully host-runnable.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        step = spec_step_idx % self.num_nextn_predict_layers
        return self.layers[step](inputs_embeds, previous_hidden_states, positions)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
        step = spec_step_idx % self.num_nextn_predict_layers
        return self.layers[step].compute_logits(hidden_states)


__all__ = [
    "DeepSeekV41MTP",
    "DeepSeekV41MTPLayer",
    "DSparkDraftSpec",
    "shipped_mtp_base",
    "shipped_dspark_base",
    "DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS",
    "DEEPSEEKV41_DSPARK_TARGET_LAYER_IDS",
    "DEEPSEEKV41_DSPARK_N_ROUTED_EXPERTS",
    "DEEPSEEKV41_DSPARK_NUM_EXPERTS_PER_TOK",
    "DEEPSEEKV41_DSPARK_BLOCK_SIZE",
    "DEEPSEEKV41_DSPARK_MARKOV_RANK",
]
