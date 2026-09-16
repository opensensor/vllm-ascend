# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 routed-expert MoE on the 2-bit (W2) 310P path (E3.3).

This is the V4.1 / 310P *adaptation* of the shipped ``DeepseekV4MoE`` routed
path (``vllm_ascend/models/deepseek_v4/model.py``). The shipped module builds a
``FusedMoEFactory`` and finishes with the Triton MoE-combine op
``muls_add_triton`` (``x * scale + y``, ``vllm_ascend/ops/triton/mul_add.py``).
Neither is available on the Triton-free 310P lane, so this module keeps the
exact same *math* while swapping both pieces for eager, host-parity-verified
primitives:

1. **Router (host-side).** ``softmax`` over the ``384`` routed-expert logits ->
   top-``6`` -> renormalize (``norm_topk_prob=True``). This is the E1.2
   :func:`~vllm_ascend.models.deepseek_v41.w2_unpack.route_topk_w2` (identical,
   bit-for-bit, to the E0.4 reference ``route_topk`` in the renormalized path).
   The router runs *here* and produces ``topk_ids`` / ``topk_weights`` -- it does
   **not** run inside a ``FusedMoEFactory`` internal router.

2. **Routed experts -> E1.3 W2 method.** The ``topk_ids`` / ``topk_weights`` are
   handed to :class:`AscendW2DynamicFusedMoEMethod310`
   (``_310p/quantization/methods/w2_dynamic.py``, resolved through the 310P
   registry ``get_scheme_class("W2A8_DYNAMIC", "moe")``), **not** to a
   ``FusedMoEFactory``. On CPU the method takes its import-guarded host path,
   which re-expresses the device INT8 grouped matmul via the E1.2 active-expert
   unpack (:func:`~vllm_ascend.models.deepseek_v41.w2_unpack.w2_active_moe_forward`
   / ``w2_group_qdq_linear``). The shared expert is deliberately kept **out** of
   the method call (passed ``w2_shared_expert=None``) so the routed output comes
   back un-scaled and the eager combine below owns the ``routed_scaling_factor``.

3. **Eager MoE combine (``muls_add_triton`` -> ``torch.addcmul``).** The shipped
   ``muls_add_triton(routed, shared, routed_scaling_factor)`` (``routed * scale +
   shared``) is replaced by :func:`eager_moe_combine`, a plain
   ``torch.addcmul`` (``shared + routed * scale``; falls back to ``routed *
   scale`` when there is no shared expert). ``routed_scaling_factor = 1.5`` for
   V4.1. The **shared expert stays FP16** (``policy.shared_expert_dtype``); the
   combine accumulates in ``policy.accumulation_dtype`` (fp32), promoted to the
   operand precision so a float64 host-parity run stays exact.

All dtypes are read from the authoritative E2.1 policy
(:data:`~vllm_ascend.models.deepseek_v41.dtype_policy.ASCEND_DEEPSEEKV41_DTYPE_POLICY`),
never spelled as literals here.

Import hygiene (Triton-free, E2.1 contract)
-------------------------------------------
Importing this module pulls in **only** ``torch``, the host-clean E1.2
``w2_unpack`` helpers and the E2.1 dtype policy -- never ``muls_add_triton`` and
never the heavy 310P quantization-methods stack (which imports ``torch_npu``).
The E1.3 method is resolved **lazily** in :func:`resolve_w2_moe_method`, at wire
time only, exactly as the E2.1 model defers the shipped base-class import.

Wiring surface for E4.1 (do not edit ``model.py`` here)
------------------------------------------------------
E4.1's ``AscendDeepseekV41ForCausalLM._swap_moe_to_w2`` hook builds one
:class:`DeepseekV41W2MoE` per ``DeepseekV4MoE`` (via :meth:`from_config`),
populates ``w2_experts`` (the packed W2 bank) and the FP16 ``shared_expert`` from
the E3.4 loader, computes ``router_logits`` with the shipped gate, and calls
:meth:`DeepseekV41W2MoE.forward` -- replacing both the ``FusedMoEFactory`` and
the ``muls_add_triton`` combine in the shipped ``DeepseekV4MoE.forward``.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    DeepseekV41DtypePolicy,
)
from .w2_unpack import route_topk_w2

if TYPE_CHECKING:  # keep runtime imports light (no heavy vLLM / Triton pull)
    from vllm_ascend.quantization.methods.base import AscendMoEScheme

# ---------------------------------------------------------------------------
# V4.1 routed-MoE geometry. The authoritative values live on the HF text config
# (and mirror the shape contract in ``deepseek_v41/model.py``); these named
# constants are the defaults used when a config is not supplied, so no bare
# magic numbers reach the routing/combine call sites (AGENTS.md).
# ---------------------------------------------------------------------------
DEEPSEEKV41_N_ROUTED_EXPERTS = 384
DEEPSEEKV41_NUM_EXPERTS_PER_TOK = 6
# DeepSeek V4.1 scales the routed contribution by 1.5 *outside* the router path
# (normalize top-k weights, then scale the routed output), matching the shipped
# ``DeepseekV4MoE.routed_scaling_factor`` default.
DEEPSEEKV41_ROUTED_SCALING_FACTOR = 1.5
# ``norm_topk_prob``: renormalize the kept top-k router weights to sum to 1.
DEEPSEEKV41_NORM_TOPK_PROB = True

# 310P-local registry coordinates for the E1.3 W2 fused-MoE method.
W2_MOE_QUANT_TYPE = "W2A8_DYNAMIC"
W2_MOE_LAYER_TYPE = "moe"

__all__ = [
    "DEEPSEEKV41_N_ROUTED_EXPERTS",
    "DEEPSEEKV41_NUM_EXPERTS_PER_TOK",
    "DEEPSEEKV41_ROUTED_SCALING_FACTOR",
    "DEEPSEEKV41_NORM_TOPK_PROB",
    "W2_MOE_QUANT_TYPE",
    "W2_MOE_LAYER_TYPE",
    "resolve_w2_moe_method",
    "eager_moe_combine",
    "DeepseekV41W2MoE",
]


def resolve_w2_moe_method() -> AscendMoEScheme:
    """Resolve the E1.3 :class:`AscendW2DynamicFusedMoEMethod310` (lazy import).

    The 310P quantization-methods package imports ``torch_npu`` and the device
    ops stack, so it is imported *here* (at wire time) rather than at module
    import -- this keeps ``import ...deepseek_v41.moe`` host-clean and Triton-free
    (the same deferral the E2.1 model uses for the shipped base class). Importing
    the package registers every 310P scheme; the class is then looked up through
    the 310P-local registry.

    Returns:
        A freshly constructed W2 fused-MoE method instance.
    """
    # Importing the package runs its __init__, which imports ``w2_dynamic`` and
    # registers ``("W2A8_DYNAMIC", "moe")`` in the 310P registry.
    import vllm_ascend._310p.quantization.methods as _methods_pkg  # noqa: F401
    from vllm_ascend._310p.quantization.methods.registry import get_scheme_class

    scheme_cls = get_scheme_class(W2_MOE_QUANT_TYPE, W2_MOE_LAYER_TYPE)
    if scheme_cls is None:  # pragma: no cover - registration is unconditional
        raise RuntimeError(
            f"310P registry has no scheme for ({W2_MOE_QUANT_TYPE!r}, {W2_MOE_LAYER_TYPE!r}); "
            "the DeepSeek V4.1 W2 fused-MoE method (E1.3) failed to register."
        )
    return scheme_cls()


def eager_moe_combine(
    routed: torch.Tensor,
    shared_output: torch.Tensor | None,
    routed_scaling_factor: float,
    *,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    """Eager replacement for ``muls_add_triton`` (``routed * scale + shared``).

    The shipped ``DeepseekV4MoE`` finishes with
    ``muls_add_triton(routed, shared, routed_scaling_factor)`` -- a Triton
    ``x * scale + y`` kernel. This computes the identical value with a plain
    :func:`torch.addcmul` (``shared + routed * scale``), and with just
    ``routed * scale`` when the layer has no shared expert (mirroring the shipped
    ``final_hidden_states *= routed_scaling_factor`` branch).

    The add accumulates in ``accumulation_dtype`` (fp32 per the E2.1 policy),
    promoted against the operand dtype so a float64 host-parity run stays exact
    while an fp16 device run accumulates in fp32. The result is returned in that
    promoted accumulation precision; the caller casts to ``main_dtype``.
    """
    acc_dtype = torch.promote_types(routed.dtype, accumulation_dtype)
    routed_acc = routed.to(acc_dtype)
    if shared_output is None:
        return routed_acc * routed_scaling_factor
    shared_acc = shared_output.to(acc_dtype)
    scale = torch.as_tensor(routed_scaling_factor, dtype=acc_dtype, device=routed_acc.device)
    # muls_add_triton(routed, shared, scale) == routed * scale + shared.
    return torch.addcmul(shared_acc, routed_acc, scale)


class DeepseekV41W2MoE(nn.Module):
    """V4.1 routed-expert MoE: host router -> E1.3 W2 method -> eager combine.

    This is the Triton-free 310P adaptation of the shipped ``DeepseekV4MoE``
    routed path. It owns three things the shipped module delegates to Triton /
    ``FusedMoEFactory``:

    * the **router** (softmax -> top-``top_k`` -> renormalize), run host-side and
      producing ``topk_ids`` / ``topk_weights``;
    * the **routed experts**, dispatched through the E1.3
      :class:`AscendW2DynamicFusedMoEMethod310` (W2 packed weights + INT8
      dynamic activations), *not* a ``FusedMoEFactory``;
    * the **combine**, an eager ``routed * routed_scaling_factor + shared`` in
      place of ``muls_add_triton``.

    The packed W2 expert bank (``w2_experts``) and the FP16 ``shared_expert`` are
    populated by the E3.4 loader; the E1.3 method is resolved lazily on first use
    (or injected for tests).
    """

    def __init__(
        self,
        *,
        num_experts: int = DEEPSEEKV41_N_ROUTED_EXPERTS,
        top_k: int = DEEPSEEKV41_NUM_EXPERTS_PER_TOK,
        renormalize: bool = DEEPSEEKV41_NORM_TOPK_PROB,
        routed_scaling_factor: float = DEEPSEEKV41_ROUTED_SCALING_FACTOR,
        method: AscendMoEScheme | None = None,
        w2_experts: list[Any] | None = None,
        shared_expert: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dtype_policy: DeepseekV41DtypePolicy | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.renormalize = bool(renormalize)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.dtype_policy = dtype_policy or ASCEND_DEEPSEEKV41_DTYPE_POLICY
        # The packed W2 bank (E3.4 loader) and always-on FP16 shared expert.
        self.w2_experts = w2_experts
        # ``shared_expert`` may be an nn.Module (the shipped DeepseekV2MLP, FP16)
        # or any callable; nn.Module stores a non-module callable as a plain attr.
        self.shared_expert = shared_expert
        # Resolved lazily (see :attr:`method`) so import stays Triton-free.
        self._method = method

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        method: AscendMoEScheme | None = None,
        w2_experts: list[Any] | None = None,
        shared_expert: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dtype_policy: DeepseekV41DtypePolicy | None = None,
    ) -> DeepseekV41W2MoE:
        """Build from an HF config, reading the V4.1 routed-MoE geometry.

        Reads ``n_routed_experts`` / ``num_experts_per_tok`` / ``norm_topk_prob``
        / ``routed_scaling_factor`` off ``config`` (falling back to the V4.1
        defaults), so E4.1 can wire straight from the shipped ``DeepseekV4MoE``'s
        config without re-spelling the geometry.
        """
        return cls(
            num_experts=getattr(config, "n_routed_experts", DEEPSEEKV41_N_ROUTED_EXPERTS),
            top_k=getattr(config, "num_experts_per_tok", DEEPSEEKV41_NUM_EXPERTS_PER_TOK),
            renormalize=getattr(config, "norm_topk_prob", DEEPSEEKV41_NORM_TOPK_PROB),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", DEEPSEEKV41_ROUTED_SCALING_FACTOR),
            method=method,
            w2_experts=w2_experts,
            shared_expert=shared_expert,
            dtype_policy=dtype_policy,
        )

    @property
    def method(self) -> AscendMoEScheme:
        """The E1.3 W2 fused-MoE method (resolved lazily on first access)."""
        if self._method is None:
            self._method = resolve_w2_moe_method()
        return self._method

    # -- router -------------------------------------------------------------

    def route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Host router: ``softmax`` -> top-``top_k`` -> renormalize.

        Thin wrapper over the E1.2
        :func:`~vllm_ascend.models.deepseek_v41.w2_unpack.route_topk_w2`, which is
        bit-identical to the E0.4 reference router in the renormalized path.

        Returns:
            ``(topk_ids[int64, [T, top_k]], topk_weights[float64, [T, top_k]])``.
        """
        return route_topk_w2(router_logits, self.top_k, renormalize=self.renormalize)

    # -- routed experts (E1.3 method) --------------------------------------

    def routed_experts_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        experts: list[Any] | None = None,
    ) -> torch.Tensor:
        """Dispatch the routed experts through the E1.3 W2 method.

        Hands the pre-selected ``topk_ids`` / ``topk_weights`` to
        :meth:`AscendW2DynamicFusedMoEMethod310.apply`. The shared expert is held
        back (``w2_shared_expert=None``) so the routed output is *un-scaled* and
        the eager combine owns ``routed_scaling_factor``. The method takes its
        host path on CPU (device INT8 grouped matmul re-expressed via the E1.2
        active-expert unpack) and its device path on a real 310P wave.
        """
        experts = experts if experts is not None else self.w2_experts
        if experts is None:
            raise ValueError(
                "DeepseekV41W2MoE.routed_experts_forward requires the packed W2 expert bank; "
                "set `w2_experts` (populated by the E3.4 loader) or pass `experts=`."
            )
        # The method reads the bank off the layer object; shared expert is
        # deliberately None here (the eager combine applies routed_scaling_factor).
        method_layer = types.SimpleNamespace(w2_experts=experts, w2_shared_expert=None)
        return self.method.apply(method_layer, hidden_states, topk_weights, topk_ids, None, None)

    # -- eager combine (muls_add_triton replacement) -----------------------

    def combine(self, routed: torch.Tensor, shared_output: torch.Tensor | None) -> torch.Tensor:
        """Eager ``routed * routed_scaling_factor + shared`` (see :func:`eager_moe_combine`)."""
        return eager_moe_combine(
            routed,
            shared_output,
            self.routed_scaling_factor,
            accumulation_dtype=self.dtype_policy.accumulation_dtype,
        )

    # -- full forward -------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_output: torch.Tensor | None = None,
        *,
        experts: list[Any] | None = None,
    ) -> torch.Tensor:
        """Routed-expert MoE forward: route -> W2 experts -> eager combine.

        Args:
            hidden_states: ``[T, hidden]`` activations.
            router_logits: ``[T, num_experts]`` gate logits (from the shipped
                gate linear, upstream).
            shared_output: optional pre-computed FP16 shared-expert output. When
                ``None`` and :attr:`shared_expert` is set, it is computed here
                (FP16, ``policy.shared_expert_dtype``).
            experts: optional explicit packed W2 bank (else :attr:`w2_experts`).

        Returns:
            ``[T, hidden]`` combined output in the promoted accumulation
            precision (the caller casts to ``main_dtype``).
        """
        topk_ids, topk_weights = self.route(router_logits)
        routed = self.routed_experts_forward(hidden_states, topk_ids, topk_weights, experts=experts)
        if shared_output is None and self.shared_expert is not None:
            # Shared expert stays FP16 and rides the residual un-scaled.
            shared_output = self.shared_expert(hidden_states)
        return self.combine(routed, shared_output)
