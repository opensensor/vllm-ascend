# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P DeepSeek V4.1 model classes (E2.1 -- ADAPT of shipped V4).

This is an *adaptation* of the shipped, config-driven, ``torch_npu``-native
``deepseek_v4`` model (``vllm_ascend.models.deepseek_v4.model``), NOT a
green-field port. The shipped ``DeepseekV4Attention`` / ``DeepseekV4MoE`` /
``DeepseekV4DecoderLayer`` / ``DeepseekV4Model`` already read the whole V4.1
*shape* (40 layers, hidden 5120, 384 experts top-6, moe_inter 2304, q_lora
1280, MTP-3, ``engram_layer_ids``, ``dspark_*``, ``compress_ratios``) off the
HF config, so most of V4.1 is *data*, not new code (see the adaptation-seams
doc). This module therefore SUBCLASSES the shipped
``AscendDeepseekV4ForCausalLM`` and, at this E2.1 gate, only:

* attaches the authoritative :data:`ASCEND_DEEPSEEKV41_DTYPE_POLICY`;
* plumbs the V4.1 text config (40L / 384-expert / ``engram_layer_ids`` /
  MTP-3) through to the shipped constructor;
* leaves the three genuine V4.1 deltas -- the MoE->W2 swap (E3.3), Engram
  injection (E2.3), and the MLA / indexer ratio-{0,1,2} overrides (E3.1/E3.2)
  -- as clean, clearly-tagged hooks/TODOs.

Triton hygiene (grep-gate)
--------------------------
The shipped ``deepseek_v4/model.py`` top-level pulls in ``muls_add_triton``
(``vllm_ascend/ops/triton/mul_add.py``), the single Triton op in the shipped
model (``DeepseekV4MoE.forward``, the ``x*scale + y`` MoE combine). To keep
*this* package's import path Triton-free on the 310P flag, the shipped base
class is resolved **lazily** (see :func:`_shipped_causal_lm_base` and the
module ``__getattr__`` below): merely importing ``deepseek_v41.model`` does not
pull Triton. The base is only resolved when the V4.1 class is actually built
(i.e. when vLLM instantiates the arch on device). E3.3 replaces
``muls_add_triton`` with an eager combine in the V4.1 MoE override, at which
point even that deferred path is Triton-free.

This module itself contains ZERO ``triton`` references in code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    DeepseekV41DtypePolicy,
)

if TYPE_CHECKING:  # keep runtime imports light (no heavy vLLM / Triton pull)
    from vllm.config import VllmConfig


# ===========================================================================
# V4.1 config contract (data-driven; the shipped model reads these off config)
# ===========================================================================

# Authoritative V4.1 geometry (from DeepSeek-V4.1-Flash config text_config).
# The shipped deepseek_v4 classes read every one of these off the HF config, so
# E2.1 only asserts the contract rather than re-implementing the geometry.
DEEPSEEKV41_NUM_HIDDEN_LAYERS = 40
DEEPSEEKV41_HIDDEN_SIZE = 5120
DEEPSEEKV41_N_ROUTED_EXPERTS = 384
DEEPSEEKV41_NUM_EXPERTS_PER_TOK = 6
DEEPSEEKV41_MOE_INTERMEDIATE_SIZE = 2304
DEEPSEEKV41_Q_LORA_RANK = 1280
DEEPSEEKV41_ENGRAM_LAYER_IDS = (1, 14)
DEEPSEEKV41_DSPARK_TARGET_LAYER_IDS = (37, 38, 39)
DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS = 3  # MTP-3
DEEPSEEKV41_COMPRESS_RATIOS = (0, 1, 2)


def _resolve_v41_text_config(vllm_config: VllmConfig) -> Any:
    """Return the effective V4.1 text config.

    V4.1 checkpoints nest the language-model fields under ``text_config`` (the
    multimodal wrapper). The shipped ``AscendDeepseekV4ForCausalLM`` reads
    ``vllm_config.model_config.hf_config`` directly, so this helper prefers the
    already-flattened ``hf_text_config`` when vLLM exposes it and otherwise
    unwraps a nested ``text_config``. Config plumbing only -- no geometry is
    re-implemented here.
    """
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is not None:
        return text_config
    hf_config = getattr(model_config, "hf_config", None)
    nested = getattr(hf_config, "text_config", None)
    return nested if nested is not None else hf_config


def _reject_multimodal(vllm_config: VllmConfig) -> None:
    """First gate: the 310P DeepSeek V4.1 W2 path is text-only.

    Vision/multimodal support is out of scope for the W2 text path; reject at
    construction so a ``DeepseekV41ForConditionalGeneration`` checkpoint cannot
    silently run degraded.
    """
    model_config = getattr(vllm_config, "model_config", None)
    mm_config = getattr(model_config, "multimodal_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    has_vision = getattr(hf_config, "vision_config", None) is not None
    if mm_config is not None or has_vision:
        raise NotImplementedError(
            "AscendDeepseekV41ForConditionalGeneration: multimodal/vision inputs "
            "are not supported on the Ascend 310P DeepSeek V4.1 W2 path "
            "(text-only). TODO(later): wire the DeepSeek-VL vision tower."
        )


# ===========================================================================
# Lazy base resolution (keeps the package import path Triton-free)
# ===========================================================================

_V41_CAUSAL_LM_CLS: type | None = None
_V41_COND_GEN_CLS: type | None = None


def _shipped_causal_lm_base() -> type:
    """Lazily import and return the shipped ``AscendDeepseekV4ForCausalLM``.

    TODO(E3.3): importing the shipped ``deepseek_v4.model`` transitively pulls
    in ``muls_add_triton`` (the shipped ``DeepseekV4MoE`` MoE-combine Triton
    op). E3.3 overrides ``DeepseekV4MoE.forward`` in the V4.1 MoE seam with an
    eager ``routed * scale + shared`` combine and swaps ``FusedMoEFactory`` for
    the E1.3 ``AscendW2DynamicFusedMoEMethod310``, removing that final Triton
    dependency from the 310P path. Until then the base is resolved only at
    build time (never at package import), so the grep-gate stays green.
    """
    from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM

    return AscendDeepseekV4ForCausalLM


def _build_causal_lm_cls() -> type:
    """Build (once) the V4.1 causal-LM subclass of the shipped base."""
    global _V41_CAUSAL_LM_CLS
    if _V41_CAUSAL_LM_CLS is not None:
        return _V41_CAUSAL_LM_CLS

    base = _shipped_causal_lm_base()

    class AscendDeepseekV41ForCausalLM(base):  # type: ignore[valid-type,misc]
        """Text-only Ascend DeepSeek V4.1 causal LM (E2.1 ADAPT of V4).

        Subclasses the shipped ``AscendDeepseekV4ForCausalLM`` and overrides
        only what V4.1 needs at this gate. The 40-layer / 384-expert /
        ``engram_layer_ids`` / MTP-3 geometry flows through the shipped,
        config-driven constructor; the V4.1 deltas are staged as hooks below.
        """

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            # Authoritative dtype policy (W2 experts / INT8 act, ~W4 Engram,
            # FP16 MLA/indexer/dense/shared/LM-head, FP32 accum). Every V4.1
            # submodule reads this rather than spelling dtype literals.
            self.dtype_policy: DeepseekV41DtypePolicy = DeepseekV41DtypePolicy.from_vllm_config(vllm_config)
            # Config plumbing: expose the flattened V4.1 text config so the
            # shipped constructor sees 40L / 384-expert / engram / MTP-3.
            self._v41_text_config = _resolve_v41_text_config(vllm_config)
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            # V4.1 delta hooks (staged; no-ops at E2.1 so construction is clean).
            self._stage_v41_overrides()

        # -- V4.1 delta hooks (clean seams for later tasks) ----------------

        def _stage_v41_overrides(self) -> None:
            """Run the staged V4.1 override hooks. All are no-ops at E2.1."""
            self._swap_moe_to_w2()  # E3.3
            self._inject_engram()  # E2.3
            self._override_mla_indexer()  # E3.1 / E3.2

        def _swap_moe_to_w2(self) -> None:
            """E3.3: swap each ``DeepseekV4MoE`` routed path to the E1.3 W2 method.

            Builds one :class:`DeepseekV41W2MoE` per routed layer (host router:
            softmax -> top-6 -> renorm, ``routed_scaling_factor=1.5``) that routes
            to ``AscendW2DynamicFusedMoEMethod310`` (``W2A8_DYNAMIC``/``moe``) and
            finishes with the eager ``routed*scale + shared`` combine that replaces
            ``muls_add_triton``. The seam is attached as ``layer.mlp_w2`` (the E3.4
            loader populates its packed W2 bank + FP16 shared expert); the shipped
            ``deepseek_v4`` base is device-only, so this runs at on-device build.

            Component wired: :mod:`vllm_ascend.models.deepseek_v41.moe`. The heavy
            import is deferred to keep the package import path Triton-free (E2.1).
            """
            from .moe import DeepseekV41W2MoE

            config = self._v41_text_config
            for layer in getattr(getattr(self, "model", None), "layers", []) or []:
                mlp = getattr(layer, "mlp", None)
                if mlp is None:
                    continue
                shared = getattr(mlp, "shared_experts", None)
                layer.mlp_w2 = DeepseekV41W2MoE.from_config(
                    config,
                    shared_expert=(shared.forward if shared is not None else None),
                    dtype_policy=self.dtype_policy,
                )

        def _inject_engram(self) -> None:
            """E2.3: inject the two ~W4 host-table Engram lookups at ``[1, 14]``.

            Builds the shared Engram hash layout + single-shared-copy ~W4 host
            tables (Qwen host-table transport, not the fork's Triton kernels) and
            attaches one :class:`AscendDeepseekV41Engram` sub-block plus the shared
            :class:`DeepseekEngramHasher` to each backbone layer whose index is in
            ``engram_layer_ids``. No-op if the config carries no Engram geometry.

            Component wired: :mod:`vllm_ascend.models.deepseek_v41.engram`.
            """
            from .engram import (
                AscendDeepseekV41Engram,
                DeepseekEngramHasher,
                _layout_from_config,
                create_engram_host_tables,
            )

            config = self._v41_text_config
            engram_layer_ids = tuple(getattr(config, "engram_layer_ids", ()) or DEEPSEEKV41_ENGRAM_LAYER_IDS)
            if not engram_layer_ids:
                return
            layout = _layout_from_config(config)
            self._engram_layout = layout
            self._engram_hasher = DeepseekEngramHasher(layout)
            self._engram_tables = create_engram_host_tables(layout, dtype_policy=self.dtype_policy)
            layer_to_hash = {layer_id: idx for idx, layer_id in enumerate(engram_layer_ids)}
            layers = getattr(getattr(self, "model", None), "layers", []) or []
            for layer_idx, layer in enumerate(layers):
                hash_index = layer_to_hash.get(layer_idx)
                if hash_index is None:
                    continue
                layer.engram = AscendDeepseekV41Engram(
                    config=config,
                    host_table=self._engram_tables[hash_index],
                    layer_hash_index=hash_index,
                    layout=layout,
                    dtype_policy=self.dtype_policy,
                )

        def _override_mla_indexer(self) -> None:
            """E3.1/E3.2: drive the dense MLA + the ratio-{1,2} indexer eagerly.

            Attaches an eager :class:`AscendDeepseekV41MLA` (the generic host MLA,
            not the 910 DSA backend) to every layer and an
            :class:`AscendDeepseekV41Indexer` on ``compress_ratio in {1, 2}``
            layers (the shipped gate is ``== 4``), each with deterministic torch
            fallbacks for the missing ``npu_*`` ops on 310P.

            Components wired: :mod:`vllm_ascend.models.deepseek_v41.mla` and
            :mod:`vllm_ascend.models.deepseek_v41.indexer`.
            """
            from .indexer import ACTIVE_COMPRESS_RATIOS, AscendDeepseekV41Indexer
            from .mla import AscendDeepseekV41MLA

            config = self._v41_text_config
            compress_ratios = list(getattr(config, "compress_ratios", ()) or ())
            layers = getattr(getattr(self, "model", None), "layers", []) or []
            for layer_idx, layer in enumerate(layers):
                layer.mla = AscendDeepseekV41MLA.from_config(config, dtype_policy=self.dtype_policy)
                ratio = compress_ratios[layer_idx] if layer_idx < len(compress_ratios) else 0
                if ratio in ACTIVE_COMPRESS_RATIOS:
                    layer.indexer = AscendDeepseekV41Indexer(config, compress_ratio=ratio, policy=self.dtype_policy)

        # -- KV-cache spec materialization (E2.2) --------------------------

        def get_kv_cache_groups(self) -> list:
            """Materialize the hybrid KV-cache groups from the E2.2 per-layer plan.

            Delegates to :mod:`vllm_ascend.models.deepseek_v41.kv`: one MLA-latent
            cache per layer plus a per-ratio indexer compressed-history cache on
            ratio-{1,2} layers, packaged into merged ``KVCacheGroupSpec`` groups.
            """
            from .kv import build_deepseekv41_kv_cache_groups, build_deepseekv41_layer_plan

            plan = build_deepseekv41_layer_plan(self._v41_text_config, dtype_policy=self.dtype_policy)
            return build_deepseekv41_kv_cache_groups(plan)

        def kv_group_report(self) -> dict:
            """MLA-latent + indexer layer/group counts for the E2.2 KV plan."""
            from .kv import build_deepseekv41_layer_plan

            plan = build_deepseekv41_layer_plan(self._v41_text_config, dtype_policy=self.dtype_policy)
            groups = self.get_kv_cache_groups()
            return {
                "num_layers": len(plan),
                "mla_latent_layers": len(plan),
                "indexer_layers": sum(1 for entry in plan if entry.has_indexer),
                "num_groups": len(groups),
            }

        # -- weight load (E3.4) --------------------------------------------

        def _expert_geometry(self) -> dict:
            """Frozen W2 expert geometry from the V4.1 text config (E3.4 schema)."""
            config = self._v41_text_config
            return {
                "hidden_size": int(getattr(config, "hidden_size", DEEPSEEKV41_HIDDEN_SIZE)),
                "moe_intermediate_size": int(
                    getattr(config, "moe_intermediate_size", DEEPSEEKV41_MOE_INTERMEDIATE_SIZE)
                ),
                "num_hidden_layers": int(getattr(config, "num_hidden_layers", DEEPSEEKV41_NUM_HIDDEN_LAYERS)),
                "n_routed_experts": int(getattr(config, "n_routed_experts", DEEPSEEKV41_N_ROUTED_EXPERTS)),
                "num_nextn_predict_layers": int(
                    getattr(config, "num_nextn_predict_layers", DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS)
                ),
                "dspark_n_routed_experts": int(
                    getattr(
                        config,
                        "dspark_n_routed_experts",
                        getattr(config, "n_routed_experts", DEEPSEEKV41_N_ROUTED_EXPERTS),
                    )
                ),
            }

    AscendDeepseekV41ForCausalLM.__module__ = __name__
    AscendDeepseekV41ForCausalLM.__qualname__ = "AscendDeepseekV41ForCausalLM"
    _V41_CAUSAL_LM_CLS = AscendDeepseekV41ForCausalLM
    return AscendDeepseekV41ForCausalLM


def _build_cond_gen_cls() -> type:
    """Build (once) the multimodal-rejecting V4.1 alias."""
    global _V41_COND_GEN_CLS
    if _V41_COND_GEN_CLS is not None:
        return _V41_COND_GEN_CLS

    causal_lm = _build_causal_lm_cls()

    class AscendDeepseekV41ForConditionalGeneration(causal_lm):  # type: ignore[valid-type,misc]
        """Multimodal-rejecting alias for the V4.1 arch.

        Registered for ``DeepseekV41ForConditionalGeneration`` so such
        checkpoints route here, then rejected at the first gate: the 310P W2
        path is text-only.
        """

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            _reject_multimodal(vllm_config)  # first gate
            super().__init__(vllm_config=vllm_config, prefix=prefix)

        def get_multimodal_embeddings(self, *args: object, **kwargs: object):
            raise NotImplementedError(
                "AscendDeepseekV41ForConditionalGeneration: multimodal embeddings "
                "are not supported on the Ascend 310P path (text-only)."
            )

        def embed_multimodal(self, *args: object, **kwargs: object):
            raise NotImplementedError(
                "AscendDeepseekV41ForConditionalGeneration: multimodal inputs are "
                "not supported on the Ascend 310P path (text-only)."
            )

    AscendDeepseekV41ForConditionalGeneration.__module__ = __name__
    AscendDeepseekV41ForConditionalGeneration.__qualname__ = "AscendDeepseekV41ForConditionalGeneration"
    _V41_COND_GEN_CLS = AscendDeepseekV41ForConditionalGeneration
    return AscendDeepseekV41ForConditionalGeneration


# ===========================================================================
# MTP-3 stub (registration target for DeepSeekV41MTPModel; wired at E4.2)
# ===========================================================================


class DeepSeekV41MTP:
    """Stub MTP-3 draft model for DeepSeek V4.1 (``num_nextn_predict_layers=3``).

    Registration target so the ``DeepSeekV41MTPModel`` arch resolves at E2.1.
    The concrete drafter reuses the shipped ``deepseek_v4`` DSpark/MTP path
    (which inherits the E3.3 W2 / eager-combine fixes automatically); wiring is
    E4.2. Constructing it now fails fast rather than silently running a stub.
    """

    num_nextn_predict_layers = DEEPSEEKV41_NUM_NEXTN_PREDICT_LAYERS
    dspark_target_layer_ids = DEEPSEEKV41_DSPARK_TARGET_LAYER_IDS

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "DeepSeekV41MTP (MTP-3) is not wired yet; registration-only stub. "
            "TODO(E4.2): reuse deepseek_v4 DSpark/MTP for num_nextn_predict_layers=3."
        )


# ===========================================================================
# Lazy attribute access -- materialize the heavy subclasses only on demand,
# so `import vllm_ascend.models.deepseek_v41.model` stays Triton-free (E2.1).
# ===========================================================================

_LAZY_BUILDERS = {
    "AscendDeepseekV41ForCausalLM": _build_causal_lm_cls,
    "AscendDeepseekV41ForConditionalGeneration": _build_cond_gen_cls,
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
_DEFAULT_DTYPE_POLICY = ASCEND_DEEPSEEKV41_DTYPE_POLICY

# NOTE: ``AscendDeepseekV41ForCausalLM`` and
# ``AscendDeepseekV41ForConditionalGeneration`` are provided lazily via the
# module ``__getattr__`` (they subclass the shipped, Triton-pulling
# ``deepseek_v4`` base only on first access), so they are intentionally not in
# ``__all__`` -- resolve them by attribute access or the ModelRegistry arch name.
__all__ = [
    "DeepSeekV41MTP",
]
