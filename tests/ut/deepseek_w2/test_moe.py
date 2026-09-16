# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the 310P DeepSeek V4.1 W2 routed-expert MoE seam (E3.3).

Covers :mod:`vllm_ascend.models.deepseek_v41.moe` -- the Triton-free adaptation
of the shipped ``DeepseekV4MoE`` routed path:

  * ``import ...deepseek_v41.moe`` is Triton-free: it does not pull the shipped
    ``muls_add_triton`` op nor the heavy 310P quantization-methods stack.
  * The router runs host-side (softmax -> top-6 -> renormalize) and its
    ``topk_ids`` / ``topk_weights`` are handed to the E1.3
    ``AscendW2DynamicFusedMoEMethod310`` (resolved via the 310P registry), not a
    ``FusedMoEFactory``.
  * The eager combine replaces ``muls_add_triton`` with ``routed * 1.5 +
    shared`` and the shared expert stays out of the routed method call.
  * Full-forward parity: ``routed * routed_scaling_factor + shared`` equals the
    E0.4 / E1.2 W2 reference (``w2_moe_forward`` / ``w2_active_moe_forward``)
    within the declared ``W2_MOE`` tolerances; router renorm + scaling 1.5 are
    correct and routing skew holds.

Like ``test_w2_method.py``, the shared UT conftest cannot import in this
environment (it eagerly patches the Triton-dependent ops stack), so this file
installs the minimal ``torch_npu`` / device stubs the E1.3 method needs and MUST
be run with ``--noconftest``:

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_moe.py
"""

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Triton-free import assertion FIRST: importing the moe seam must not pull the
# shipped ``muls_add_triton`` op nor the heavy 310P methods stack. Capture this
# before the torch_npu bootstrap (which is only needed to *resolve* the E1.3
# method) so the check reflects a clean host import.
# ---------------------------------------------------------------------------
import vllm_ascend.models.deepseek_v41.moe as moe  # noqa: E402

_MUL_ADD_MODULE = "vllm_ascend.ops.triton.mul_add"
_MUL_ADD_NOT_LOADED_BY_IMPORT = _MUL_ADD_MODULE not in sys.modules
_W2_DYNAMIC = "vllm_ascend._310p.quantization.methods.w2_dynamic"
_W2_DYNAMIC_NOT_LOADED_BY_IMPORT = _W2_DYNAMIC not in sys.modules


# ---------------------------------------------------------------------------
# Bootstrap the torch_npu / device stubs the E1.3 method import needs (mirrors
# tests/ut/deepseek_w2/test_w2_method.py).
# ---------------------------------------------------------------------------


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    for key, value in attrs.items():
        setattr(m, key, value)
    sys.modules[name] = m
    return m


def _pkg(name, path=None, **attrs):
    m = _mod(name, **attrs)
    m.__path__ = [path] if path else []
    return m


def _install_npu_stubs():
    if "torch_npu" not in sys.modules:
        tn = _pkg("torch_npu")
        tn.npu = MagicMock()
        tn._C = MagicMock()
        sys.modules["torch_npu._C"] = tn._C
    sys.modules.setdefault("triton.runtime", _mod("triton.runtime"))
    sys.modules.setdefault("vllm_ascend._build_info", _mod("vllm_ascend._build_info", __device_type__="A2"))

    import vllm.distributed.utils as _vllm_dist_utils

    if not hasattr(_vllm_dist_utils, "is_weak_contiguous"):
        _vllm_dist_utils.is_weak_contiguous = lambda *a, **k: True  # type: ignore[attr-defined]

    _pkg("vllm_ascend.ops")
    _pkg("vllm_ascend.ops.fused_moe")
    _pkg("vllm_ascend.ops.fused_moe.dataclass")
    _mod(
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts",
        MoEWeights=type("MoEWeights", (), {}),
        build_fused_experts_input=lambda **k: MagicMock(),
    )
    _mod("vllm_ascend.ops.fused_moe.dataclass.moe_mlp", MoEMlpComputeInput=type("MoEMlpComputeInput", (), {}))
    _mod("vllm_ascend.ops.fused_moe.routed_experts", AscendRoutedExperts=type("AscendRoutedExperts", (), {}))
    _mod(
        "vllm_ascend.ops.fused_moe.moe_utils",
        maybe_normalize_mxfp_scale_layout=lambda x: x,
        cumsum_group_list=lambda *a, **k: None,
    )
    _mod(
        "vllm_ascend.ops.linear",
        AscendRowParallelLinear=type("AscendRowParallelLinear", (), {}),
        AscendUnquantizedLinearMethod=type("AscendUnquantizedLinearMethod", (), {}),
    )
    _mod("vllm_ascend.ascend_forward_context", _EXTRA_CTX=MagicMock())

    import vllm_ascend

    va_dir = os.path.dirname(vllm_ascend.__file__)
    _pkg("vllm_ascend.quantization.methods", path=os.path.join(va_dir, "quantization", "methods"))
    _pkg("vllm_ascend._310p", path=os.path.join(va_dir, "_310p"))
    _pkg("vllm_ascend._310p.quantization", path=os.path.join(va_dir, "_310p", "quantization"))


_install_npu_stubs()

import pytest  # noqa: E402
import torch  # noqa: E402

from tests.ut.deepseek_w2.reference.tolerances import (  # noqa: E402
    W2_MOE_ATOL,
    W2_MOE_RTOL,
)
from tests.ut.deepseek_w2.reference.w2_moe_reference import (  # noqa: E402
    W2Expert,
    route_topk,
    w2_moe_forward,
)
from vllm_ascend._310p.quantization.methods.registry import get_scheme_class  # noqa: E402
from vllm_ascend._310p.quantization.methods.w2_dynamic import (  # noqa: E402
    AscendW2DynamicFusedMoEMethod310,
)
from vllm_ascend.models.deepseek_v41.dtype_policy import (  # noqa: E402
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
)
from vllm_ascend.models.deepseek_v41.w2_unpack import w2_active_moe_forward  # noqa: E402

# --- reduced-scale DeepSeek geometry (top-6 + shared), dims multiples of 32 ---
_HIDDEN = 64
_INTER = 96
_NUM_EXPERTS = 12
_TOP_K = 6
_SCALE = 1.5


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


def _make_experts(num_experts, seed, hidden=_HIDDEN, inter=_INTER):
    experts = []
    for e in range(num_experts):
        gate = _rand((inter, hidden), seed + 10 * e + 1, scale=0.4)
        up = _rand((inter, hidden), seed + 10 * e + 2, scale=0.4)
        down = _rand((hidden, inter), seed + 10 * e + 3, scale=0.4)
        experts.append(W2Expert(gate, up, down))
    return experts


def _moe(experts=None, shared=None, method=None):
    return moe.DeepseekV41W2MoE(
        num_experts=_NUM_EXPERTS,
        top_k=_TOP_K,
        routed_scaling_factor=_SCALE,
        method=method,
        w2_experts=experts,
        shared_expert=(shared.forward if shared is not None else None),
    )


# ===========================================================================
# Triton-free import contract
# ===========================================================================


def test_import_is_triton_free():
    # Importing the moe seam pulls neither the shipped ``muls_add_triton`` op
    # (the swap target) nor the heavy 310P methods stack.
    assert _MUL_ADD_NOT_LOADED_BY_IMPORT
    assert _W2_DYNAMIC_NOT_LOADED_BY_IMPORT
    assert not hasattr(moe, "muls_add_triton")


def _moe_source() -> str:
    return Path(moe.__file__).read_text(encoding="utf-8")


def _import_lines(src: str) -> str:
    return "\n".join(line for line in src.splitlines() if line.strip().startswith(("import ", "from ")))


def test_source_swaps_triton_for_eager_addcmul():
    src = _moe_source()
    # The Triton op is never imported nor referenced outside prose; the eager
    # ``torch.addcmul`` replacement is what actually runs.
    assert "muls_add_triton" not in _import_lines(src)
    assert "vllm_ascend.ops.triton" not in _import_lines(src)
    assert "torch.addcmul(" in src


def test_no_hot_path_item():
    # AGENTS.md: no per-element ``.item()`` in the MoE hot path.
    assert ".item()" not in _moe_source()


# ===========================================================================
# Registry resolution: routing hands off to the E1.3 method
# ===========================================================================


def test_resolve_returns_registry_w2_method():
    method = moe.resolve_w2_moe_method()
    assert isinstance(method, AscendW2DynamicFusedMoEMethod310)
    assert type(method) is get_scheme_class("W2A8_DYNAMIC", "moe")


def test_lazy_method_property_resolves_w2_method():
    layer = _moe()
    assert layer._method is None  # not resolved at construction (import stays clean)
    assert isinstance(layer.method, AscendW2DynamicFusedMoEMethod310)


def test_from_config_reads_geometry():
    config = types.SimpleNamespace(
        n_routed_experts=384,
        num_experts_per_tok=6,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
    )
    layer = moe.DeepseekV41W2MoE.from_config(config)
    assert layer.num_experts == 384
    assert layer.top_k == 6
    assert layer.renormalize is True
    assert layer.routed_scaling_factor == 1.5


# ===========================================================================
# Router: softmax -> top-6 -> renormalize, scaling 1.5
# ===========================================================================


def test_router_topk_shape_and_renorm():
    layer = _moe()
    router_logits = _rand((13, _NUM_EXPERTS), 55)
    topk_ids, topk_weights = layer.route(router_logits)
    assert topk_ids.shape == (13, _TOP_K)
    assert topk_weights.shape == (13, _TOP_K)
    # Renormalized weights sum to 1 per token.
    torch.testing.assert_close(topk_weights.sum(dim=-1), torch.ones(13, dtype=torch.float64), rtol=1e-12, atol=1e-12)
    # Bit-identical to the E0.4 reference router.
    ref_ids, ref_w = route_topk(router_logits, _TOP_K)
    assert torch.equal(topk_ids, ref_ids)
    torch.testing.assert_close(topk_weights, ref_w, rtol=1e-12, atol=1e-12)


def test_router_renorm_has_teeth():
    # Without renormalization the kept top-k probabilities sum to < 1; the V4.1
    # router renormalizes, so the two differ -- the renorm is not a no-op.
    layer = _moe()
    router_logits = _rand((7, _NUM_EXPERTS), 71)
    _, renorm_w = layer.route(router_logits)
    from vllm_ascend.models.deepseek_v41.w2_unpack import route_topk_w2

    _, raw_w = route_topk_w2(router_logits, _TOP_K, renormalize=False)
    assert torch.all(raw_w.sum(dim=-1) < 1.0)
    torch.testing.assert_close(renorm_w.sum(dim=-1), torch.ones(7, dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_routing_skew_holds():
    # Bias the logits so a small subset of experts dominates: the active-expert
    # set is then far smaller than the bank and the per-expert load is skewed.
    gen = torch.Generator().manual_seed(88)
    router_logits = torch.randn(64, _NUM_EXPERTS, generator=gen, dtype=torch.float64) * 0.1
    favored = [1, 4, 7]
    router_logits[:, favored] += 12.0  # dominate the softmax
    layer = _moe()
    topk_ids, _ = layer.route(router_logits)
    active = torch.unique(topk_ids)
    # Every token routes its top slots to the favored experts first.
    assert set(favored).issubset(set(active.tolist()))
    counts = torch.bincount(topk_ids.reshape(-1), minlength=_NUM_EXPERTS)
    favored_counts = counts[favored]
    non_favored = counts[[e for e in range(_NUM_EXPERTS) if e not in favored]]
    # Skew: every favored expert takes all tokens; every tail expert takes fewer.
    assert int(favored_counts.min()) == router_logits.shape[0]
    assert int(non_favored.max()) < int(favored_counts.min())
    assert int(favored_counts.min()) > int(counts.float().mean())


# ===========================================================================
# Eager combine (muls_add_triton replacement)
# ===========================================================================


def test_combine_is_eager_routed_scale_plus_shared():
    layer = _moe()
    routed = _rand((5, _HIDDEN), 90)
    shared = _rand((5, _HIDDEN), 91)
    got = layer.combine(routed, shared)
    expected = routed * _SCALE + shared
    torch.testing.assert_close(got, expected, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_combine_without_shared_scales_routed():
    layer = _moe()
    routed = _rand((5, _HIDDEN), 92)
    got = layer.combine(routed, None)
    torch.testing.assert_close(got, routed * _SCALE, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_eager_combine_accumulates_in_fp32_for_fp16_inputs():
    routed = torch.randn(4, _HIDDEN, dtype=torch.float16)
    shared = torch.randn(4, _HIDDEN, dtype=torch.float16)
    out = moe.eager_moe_combine(
        routed, shared, _SCALE, accumulation_dtype=ASCEND_DEEPSEEKV41_DTYPE_POLICY.accumulation_dtype
    )
    # fp16 operands + fp32 accumulation dtype -> fp32 result (policy-driven).
    assert out.dtype == torch.float32


# ===========================================================================
# Full-forward parity with the E0.4 / E1.2 W2 reference
# ===========================================================================


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_forward_matches_reference_with_shared(seed):
    x = _rand((11, _HIDDEN), seed, scale=1.0)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 9999)[0]
    router_logits = _rand((11, _NUM_EXPERTS), seed + 7)

    layer = _moe(experts=experts, shared=shared)
    got = layer.forward(x, router_logits)

    # Reference: routed (E1.2 active-expert path, no shared) scaled by 1.5, plus
    # the un-scaled FP16 shared expert -- exactly the eager combine's semantics.
    routed_ref = w2_active_moe_forward(x, experts, router_logits, _TOP_K)
    expected = _SCALE * routed_ref + shared.forward(x)
    torch.testing.assert_close(got, expected, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)

    # Cross-check the routed piece against the E0.4 grouped reference too.
    routed_ref_e04 = w2_moe_forward(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(routed_ref, routed_ref_e04, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


@pytest.mark.parametrize("seed", [110, 111])
def test_forward_matches_reference_without_shared(seed):
    x = _rand((9, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = _rand((9, _NUM_EXPERTS), seed + 3)

    layer = _moe(experts=experts, shared=None)
    got = layer.forward(x, router_logits)

    routed_ref = w2_active_moe_forward(x, experts, router_logits, _TOP_K)
    expected = _SCALE * routed_ref
    torch.testing.assert_close(got, expected, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_forward_accepts_precomputed_shared_output():
    seed = 120
    x = _rand((8, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 1)[0]
    router_logits = _rand((8, _NUM_EXPERTS), seed + 2)

    # E4.1 may compute the shared output upstream and pass it in.
    layer = _moe(experts=experts, shared=None)
    shared_output = shared.forward(x)
    got = layer.forward(x, router_logits, shared_output=shared_output)

    routed_ref = w2_active_moe_forward(x, experts, router_logits, _TOP_K)
    expected = _SCALE * routed_ref + shared_output
    torch.testing.assert_close(got, expected, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_routed_requires_expert_bank():
    layer = _moe(experts=None)
    x = _rand((3, _HIDDEN), 400)
    router_logits = _rand((3, _NUM_EXPERTS), 401)
    with pytest.raises(ValueError):
        layer.forward(x, router_logits)


# ===========================================================================
# The router output is what is handed to the E1.3 method
# ===========================================================================


def test_forward_hands_router_topk_to_method():
    seed = 500
    x = _rand((6, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = _rand((6, _NUM_EXPERTS), seed + 1)

    captured = {}

    class _SpyMethod:
        def apply(self, layer, hs, topk_weights, topk_ids, shared_experts, shared_experts_input):
            captured["topk_ids"] = topk_ids
            captured["topk_weights"] = topk_weights
            captured["shared_on_layer"] = getattr(layer, "w2_shared_expert", "MISSING")
            captured["experts_on_layer"] = layer.w2_experts
            return torch.zeros_like(hs)

    layer = _moe(experts=experts, shared=None, method=_SpyMethod())
    layer.forward(x, router_logits)

    exp_ids, exp_w = route_topk(router_logits, _TOP_K)
    assert torch.equal(captured["topk_ids"], exp_ids)
    torch.testing.assert_close(captured["topk_weights"], exp_w, rtol=1e-12, atol=1e-12)
    # Shared expert is held back from the routed method call (combine owns 1.5x).
    assert captured["shared_on_layer"] is None
    assert captured["experts_on_layer"] is experts
