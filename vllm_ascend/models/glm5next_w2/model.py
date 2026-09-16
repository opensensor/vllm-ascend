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

from typing import TYPE_CHECKING, Any

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
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            # 310P W2 delta hooks (staged; no-ops at G3 so construction is clean).
            self._stage_w2_overrides()

        # -- 310P W2 delta hooks (clean seams for later tasks) -------------

        def _stage_w2_overrides(self) -> None:
            """Run the staged 310P W2 override hooks. All are no-ops at G3."""
            self._swap_kda_to_eager()  # G4
            self._override_dsa_indexer()  # G5
            self._swap_moe_to_w2()  # G6

        def _swap_kda_to_eager(self) -> None:
            """G4: swap each KDA layer's Triton gated-delta recurrence for the
            310P Triton-free path (eager or the ``_310p/ops/fla`` GDN kernels).

            No-op at G3; the KDA layers (``KDA_LAYERS``, 34 of the 45) are left
            on the shipped path. Component wired: ``glm5next_w2.kda`` (G4).
            """

        def _override_dsa_indexer(self) -> None:
            """G5: drive the DSA (deepseek-sparse-attention) ``full_attn_layers``
            top-k indexer eagerly on 310P, reusing the deepseek_v41 indexer path.

            No-op at G3. Component wired: reuse of
            ``vllm_ascend.models.deepseek_v41.indexer`` (G5).
            """

        def _swap_moe_to_w2(self) -> None:
            """G6: swap each ``Glm5NextMoE`` routed path to the E1.3 W2 method.

            Builds the host router (softmax -> top-8 -> renorm,
            ``routed_scaling_factor=2.5``) routing to
            ``AscendW2DynamicFusedMoEMethod310`` (``W2A8_DYNAMIC``/``moe``) plus
            the eager ``routed*scale + shared`` combine that replaces
            ``FusedMoEFactory``. No-op at G3. Component wired:
            ``glm5next_w2.moe`` (G6).
            """

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
