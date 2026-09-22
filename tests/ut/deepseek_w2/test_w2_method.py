# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the 310P DeepSeek V4.1 W2 fused-MoE method (E1.3).

Covers :class:`AscendW2DynamicFusedMoEMethod310`
(``vllm_ascend/_310p/quantization/methods/w2_dynamic.py``), the device-facing
wrapper around the validated E1.2 active-expert unpack path:

  * Param creation from a synthetic W2 index: packed codes ``uint8`` and per
    ``[32, 32]`` block scales ``float32`` in the E1.1 pack shapes.
  * ``apply`` host path (and the :meth:`moe_forward` E1.2 wrapper) equals the
    E0.4 / E1.2 W2 reference within the declared ``W2_MOE`` tolerances.
  * The 310P registry resolves the method (``W2A8_DYNAMIC`` / ``moe``) while the
    existing W8 schemes still load -- no regression.

The shared UT conftest cannot import in this environment (it eagerly patches the
Triton-dependent ops stack), so this file installs the minimal ``torch_npu`` /
device stubs it needs and MUST be run with ``--noconftest``:

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_w2_method.py
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Bootstrap: the 310P quantization-methods package pulls in torch_npu and the
# whole ops/device stack (Triton), neither of which is importable host-side.
# Install just enough stubs (mirroring the shared conftest) plus fake, real-path
# package shells for the heavy parents so their eager __init__ side effects are
# skipped while the real ``methods`` submodules still load and register.
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

    # Installed vLLM in this env predates ``is_weak_contiguous``; the routed
    # experts module imports it at module scope.
    import vllm.distributed.utils as _vllm_dist_utils

    if not hasattr(_vllm_dist_utils, "is_weak_contiguous"):
        _vllm_dist_utils.is_weak_contiguous = lambda *a, **k: True  # type: ignore[attr-defined]

    # Fake the heavy ops leaves the W8 method modules import (they only need the
    # symbol names to define classes; the CPU tests never call them).
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

    # Import the real top-level package (host-clean) to locate the source dirs,
    # then shell the heavy parents with real __path__ so only their eager
    # __init__ modules (modelslim_config / the base method registry) are skipped.
    import vllm_ascend

    va_dir = os.path.dirname(vllm_ascend.__file__)
    _pkg("vllm_ascend.quantization.methods", path=os.path.join(va_dir, "quantization", "methods"))
    _pkg("vllm_ascend._310p", path=os.path.join(va_dir, "_310p"))
    _pkg("vllm_ascend._310p.quantization", path=os.path.join(va_dir, "_310p", "quantization"))


_install_npu_stubs()

import pytest  # noqa: E402
import torch  # noqa: E402

# The package import registers every 310P scheme (W2 + the W8 family).
import vllm_ascend._310p.quantization.methods as _methods_pkg  # noqa: E402,F401
from tests.ut.deepseek_w2.reference.tolerances import (  # noqa: E402
    W2_MOE_ATOL,
    W2_MOE_RTOL,
)
from tests.ut.deepseek_w2.reference.w2_moe_reference import (  # noqa: E402
    W2Expert,
    w2_moe_forward,
)
from tools.deepseek_w2.w2_format import (  # noqa: E402
    W2_BLOCK_COLS,
    W2_BLOCK_ROWS,
    W2_CODES_PER_BYTE,
)
from vllm_ascend._310p.quantization.methods.registry import get_scheme_class  # noqa: E402
from vllm_ascend._310p.quantization.methods.w2_dynamic import (  # noqa: E402
    W2_CUBE_INPUT_TILE,
    W2_CUBE_MAX_TOKENS,
    W2_CUBE_MIN_INPUT_DIM,
    AscendW2DynamicFusedMoEMethod310,
    _can_use_w2_cube,
    _device_kernel_available,
    _is_nvfp4,
    _nvfp4_dequant_fp32,
)
from vllm_ascend.models.deepseek_v41.w2_unpack import route_topk_w2  # noqa: E402

# --- reduced-scale DeepSeek geometry (top-6 + shared), dims multiples of 32 ---
_HIDDEN = 64
_INTER = 96
_NUM_EXPERTS = 12
_TOP_K = 6


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


class _FakeLayer:
    """Minimal stand-in for the routed-experts layer the E3.4 loader populates."""

    def __init__(self, experts, shared_expert=None):
        self.w2_experts = experts
        self.w2_shared_expert = shared_expert


def _method():
    return AscendW2DynamicFusedMoEMethod310()


# --- registry resolution + no W8 regression ---------------------------------


def test_registry_resolves_w2_method():
    cls = get_scheme_class("W2A8_DYNAMIC", "moe")
    assert cls is AscendW2DynamicFusedMoEMethod310


def test_w8_methods_still_registered():
    # The W2 registration is purely additive: every pre-existing W8 scheme still
    # resolves to a class.
    for quant_type, layer_type in [
        ("W8A8_DYNAMIC", "moe"),
        ("W8A8_DYNAMIC", "linear"),
        ("W8A8", "linear"),
        ("W8A8S", "linear"),
        ("W8A8SC", "linear"),
    ]:
        assert get_scheme_class(quant_type, layer_type) is not None
    # The W8 moe scheme is a distinct class from the new W2 one.
    assert get_scheme_class("W8A8_DYNAMIC", "moe") is not AscendW2DynamicFusedMoEMethod310


def test_device_kernel_guard_is_host_under_stub():
    # The stubbed torch_npu lacks the fused INT8 grouped-matmul symbol, so apply
    # deterministically takes the host math path in the CPU UT.
    assert _device_kernel_available() is False


def test_cube_kernel_accepts_validated_l0c_rows_and_rejects_larger_groups():
    packed_w2 = torch.zeros(128, W2_CUBE_MIN_INPUT_DIM // W2_CODES_PER_BYTE, dtype=torch.uint8)
    op = object()

    assert W2_CUBE_MAX_TOKENS == 128
    assert _can_use_w2_cube(op, packed_w2, W2_CUBE_MIN_INPUT_DIM, W2_CUBE_MAX_TOKENS, False)
    assert not _can_use_w2_cube(op, packed_w2, W2_CUBE_MIN_INPUT_DIM, W2_CUBE_MAX_TOKENS + 1, False)
    assert not _can_use_w2_cube(None, packed_w2, W2_CUBE_MIN_INPUT_DIM, 1, False)
    too_small_w2 = torch.zeros(128, W2_CUBE_INPUT_TILE // W2_CODES_PER_BYTE, dtype=torch.uint8)
    assert not _can_use_w2_cube(op, too_small_w2, W2_CUBE_INPUT_TILE, 1, False)
    misaligned_k = W2_CUBE_INPUT_TILE - W2_BLOCK_COLS
    misaligned_w2 = torch.zeros(128, misaligned_k // W2_CODES_PER_BYTE, dtype=torch.uint8)
    assert not _can_use_w2_cube(op, misaligned_w2, misaligned_k, 1, False)


def test_cube_kernel_accepts_w4_but_rejects_nvfp4():
    packed_w4 = torch.zeros(128, W2_CUBE_MIN_INPUT_DIM // 2, dtype=torch.uint8)
    packed_w2 = torch.zeros(128, W2_CUBE_MIN_INPUT_DIM // W2_CODES_PER_BYTE, dtype=torch.uint8)
    partial_tile_w4 = torch.zeros(64, W2_CUBE_MIN_INPUT_DIM // 2, dtype=torch.uint8)

    assert _can_use_w2_cube(object(), packed_w4, W2_CUBE_MIN_INPUT_DIM, 1, False)
    assert not _can_use_w2_cube(object(), packed_w2, W2_CUBE_MIN_INPUT_DIM, 1, True)
    assert not _can_use_w2_cube(object(), partial_tile_w4, W2_CUBE_MIN_INPUT_DIM, 1, False)


# --- param creation from a synthetic W2 index -------------------------------


def test_get_weight_shapes_and_dtypes():
    params = _method().get_weight(_NUM_EXPERTS, _INTER, _HIDDEN, torch.float16)
    w13 = params["w13_codes"]
    w2 = params["w2_codes"]
    assert w13.dtype == torch.uint8
    assert w2.dtype == torch.uint8
    assert w13.shape == (_NUM_EXPERTS, 2 * _INTER, _HIDDEN // W2_CODES_PER_BYTE)
    assert w2.shape == (_NUM_EXPERTS, _HIDDEN, _INTER // W2_CODES_PER_BYTE)


def test_get_dynamic_quant_param_shapes_and_dtypes():
    params = _method().get_dynamic_quant_param(_NUM_EXPERTS, _INTER, _HIDDEN, torch.float16)
    w13s = params["w13_scale"]
    w2s = params["w2_scale"]
    assert w13s.dtype == torch.float32
    assert w2s.dtype == torch.float32
    assert w13s.shape == (_NUM_EXPERTS, (2 * _INTER) // W2_BLOCK_ROWS, _HIDDEN // W2_BLOCK_COLS)
    assert w2s.shape == (_NUM_EXPERTS, _HIDDEN // W2_BLOCK_ROWS, _INTER // W2_BLOCK_COLS)


def test_shared_expert_param_shapes_and_dtypes():
    method = _method()
    codes = method.get_shared_expert_weight(_INTER, _HIDDEN, torch.float16)
    scales = method.get_shared_expert_dynamic_quant_param(_INTER, _HIDDEN, torch.float16)
    assert codes["shared_w13_codes"].dtype == torch.uint8
    assert codes["shared_w2_codes"].dtype == torch.uint8
    assert codes["shared_w13_codes"].shape == (2 * _INTER, _HIDDEN // W2_CODES_PER_BYTE)
    assert codes["shared_w2_codes"].shape == (_HIDDEN, _INTER // W2_CODES_PER_BYTE)
    assert scales["shared_w13_scale"].dtype == torch.float32
    assert scales["shared_w13_scale"].shape == ((2 * _INTER) // W2_BLOCK_ROWS, _HIDDEN // W2_BLOCK_COLS)
    assert scales["shared_w2_scale"].shape == (_HIDDEN // W2_BLOCK_ROWS, _INTER // W2_BLOCK_COLS)


# --- host math parity with the E0.4 / E1.2 W2 reference ----------------------


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_moe_forward_matches_reference_with_shared(seed):
    x = _rand((11, _HIDDEN), seed, scale=1.0)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 9999)[0]
    router_logits = _rand((11, _NUM_EXPERTS), seed + 7)

    got = _method().moe_forward(x, experts, router_logits, _TOP_K, shared)
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K, shared)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


@pytest.mark.parametrize("seed", [110, 111])
def test_moe_forward_matches_reference_without_shared(seed):
    x = _rand((9, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = _rand((9, _NUM_EXPERTS), seed + 3)

    got = _method().moe_forward(x, experts, router_logits, _TOP_K)
    ref = w2_moe_forward(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


@pytest.mark.parametrize("seed", [200, 201])
def test_apply_host_path_matches_reference(seed):
    x = _rand((10, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    shared = _make_experts(1, seed + 5)[0]
    router_logits = _rand((10, _NUM_EXPERTS), seed + 2)

    # Routing happens upstream; apply receives the selected ids/weights.
    topk_ids, topk_weights = route_topk_w2(router_logits, _TOP_K, renormalize=True)
    layer = _FakeLayer(experts, shared)
    got = _method().apply(layer, x, topk_weights, topk_ids, None, None)

    ref = w2_moe_forward(x, experts, router_logits, _TOP_K, shared)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_apply_host_path_matches_reference_no_shared():
    seed = 300
    x = _rand((7, _HIDDEN), seed)
    experts = _make_experts(_NUM_EXPERTS, seed)
    router_logits = _rand((7, _NUM_EXPERTS), seed + 1)

    topk_ids, topk_weights = route_topk_w2(router_logits, _TOP_K, renormalize=True)
    layer = _FakeLayer(experts, None)
    got = _method().apply(layer, x, topk_weights, topk_ids, None, None)

    ref = w2_moe_forward(x, experts, router_logits, _TOP_K)
    torch.testing.assert_close(got, ref, rtol=W2_MOE_RTOL, atol=W2_MOE_ATOL)


def test_apply_requires_expert_bank():
    method = _method()
    layer = types.SimpleNamespace()  # no w2_experts attribute
    x = _rand((3, _HIDDEN), 400)
    router_logits = _rand((3, _NUM_EXPERTS), 401)
    topk_ids, topk_weights = route_topk_w2(router_logits, _TOP_K)
    with pytest.raises(ValueError):
        method.apply(layer, x, topk_weights, topk_ids, None, None)


def test_nvfp4_dequant_matches_reference_and_detector():
    from tools.deepseek_w2.w2_format import NVFP4_BLOCK_COLS, dequantize_nvfp4

    torch.manual_seed(7)
    out_f, in_f = 64, 128
    nibbles = torch.randint(0, 16, (out_f, in_f), dtype=torch.uint8)
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    packed = (low | (high << 4)).to(torch.uint8)  # [out, in//2]
    block_scale = torch.rand(out_f, in_f // NVFP4_BLOCK_COLS) * 0.01 + 0.01

    assert _is_nvfp4(block_scale, out_f, in_f)
    assert not _is_nvfp4(torch.rand(out_f // 32, in_f // 32), out_f, in_f)

    got = _nvfp4_dequant_fp32(packed, block_scale, out_f, in_f)
    ref = dequantize_nvfp4(packed, block_scale, out_f, in_f).float()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
