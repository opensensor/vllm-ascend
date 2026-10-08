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
import weakref
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
    pack_codes,
    unpack_codes,
)
from vllm_ascend._310p.quantization.methods.registry import get_scheme_class  # noqa: E402
from vllm_ascend._310p.quantization.methods.w2_dynamic import (  # noqa: E402
    W2_CUBE_INPUT_TILE,
    W2_CUBE_MAX_TOKENS,
    W2_CUBE_MIN_INPUT_DIM,
    W2_GROUPED_MAX_ROUTES,
    W3_CUBE_MAX_INPUT_DIM,
    AscendW2DynamicFusedMoEMethod310,
    _can_use_w2_cube,
    _can_use_w2_grouped_cube,
    _device_kernel_available,
    _infer_bits,
    _is_nvfp4,
    _nvfp4_dequant_fp32,
    _stage_packed_expert,
    _w2_dequant_fp32,
)
from vllm_ascend.models.deepseek_v41.w2_unpack import route_topk_w2  # noqa: E402

# --- reduced-scale DeepSeek geometry (top-6 + shared), dims multiples of 32 ---
_HIDDEN = 64
_INTER = 96
_NUM_EXPERTS = 12
_TOP_K = 6


def test_w3_exact_eager_moe_and_cube_exclusion():
    hidden = inter = 64
    torch.manual_seed(103)
    expert = types.SimpleNamespace(hidden=hidden, inter=inter)
    for projection in ("gate", "up", "down"):
        codes = torch.randint(-4, 4, (inter, hidden), dtype=torch.int8)
        packed = pack_codes(codes, 3)
        scale = torch.rand(inter // 32, hidden // 32) * 0.05 + 0.01
        setattr(expert, f"{projection}_packed", packed)
        setattr(expert, f"{projection}_scale", scale)
        assert _infer_bits(packed, hidden) == 3
        torch.testing.assert_close(unpack_codes(packed, hidden, 3), codes)
    with pytest.raises(ValueError, match="invalid"):
        _infer_bits(torch.empty(64, 31, dtype=torch.uint8), hidden)

    x = torch.randn(2, hidden) * 0.1
    ids = torch.zeros((2, 1), dtype=torch.int64)
    weights = torch.tensor([[0.7], [0.3]])
    method = AscendW2DynamicFusedMoEMethod310()
    actual = method._apply_device([expert], x, weights, ids, None)
    layer = types.SimpleNamespace(w2_experts=[expert], w2_shared_expert=None)
    via_apply = method.apply(layer, x, weights, ids, None, None)
    torch.testing.assert_close(via_apply, actual, atol=0, rtol=0)
    gate_weight = _w2_dequant_fp32(expert.gate_packed, expert.gate_scale, inter, hidden)
    up_weight = _w2_dequant_fp32(expert.up_packed, expert.up_scale, inter, hidden)
    down_weight = _w2_dequant_fp32(expert.down_packed, expert.down_scale, hidden, inter)
    expected = (torch.nn.functional.silu(x @ gate_weight.T) * (x @ up_weight.T)) @ down_weight.T * weights
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


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


def test_resident_packed_expert_staging_is_zero_copy():
    expert = _make_experts(1, seed=21)[0]

    assert _stage_packed_expert(expert, expert.gate_packed.device) is expert


def test_device_path_skips_zero_weight_ep_pairs_before_expert_access():
    method = _method()
    x = torch.randn(2, _HIDDEN)
    topk_ids = torch.zeros(2, 3, dtype=torch.int64)
    topk_weights = torch.zeros(2, 3)

    output = method._apply_device(
        experts=[],
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert=None,
    )

    assert torch.equal(output, torch.zeros_like(x, dtype=torch.float32))


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
    packed_w3 = torch.zeros(128, W2_CUBE_MIN_INPUT_DIM * 3 // 8, dtype=torch.uint8)
    packed_w2 = torch.zeros(128, W2_CUBE_MIN_INPUT_DIM // W2_CODES_PER_BYTE, dtype=torch.uint8)
    partial_tile_w4 = torch.zeros(64, W2_CUBE_MIN_INPUT_DIM // 2, dtype=torch.uint8)

    assert _can_use_w2_cube(object(), packed_w4, W2_CUBE_MIN_INPUT_DIM, 1, False)
    assert _can_use_w2_cube(object(), packed_w3, W2_CUBE_MIN_INPUT_DIM, 1, False)
    oversized_w3 = torch.zeros(128, (W3_CUBE_MAX_INPUT_DIM + W2_CUBE_INPUT_TILE) * 3 // 8, dtype=torch.uint8)
    assert not _can_use_w2_cube(object(), oversized_w3, W3_CUBE_MAX_INPUT_DIM + W2_CUBE_INPUT_TILE, 1, False)
    assert not _can_use_w2_cube(object(), packed_w2, W2_CUBE_MIN_INPUT_DIM, 1, True)
    assert not _can_use_w2_cube(object(), partial_tile_w4, W2_CUBE_MIN_INPUT_DIM, 1, False)


def test_grouped_cube_accepts_canonical_w3_bank():
    hidden = inter = W2_CUBE_MIN_INPUT_DIM

    class _Bank:
        grouped_ready = True
        local_expert_offset = 0
        gate_packed_bank = torch.zeros(2, inter, hidden * 3 // 8, dtype=torch.uint8)
        up_packed_bank = torch.zeros(2, inter, hidden * 3 // 8, dtype=torch.uint8)
        down_packed_bank = torch.zeros(2, hidden, inter * 3 // 8, dtype=torch.uint8)
        gate_scale_bank = torch.ones(2, inter // 32, hidden // 32)
        up_scale_bank = torch.ones(2, inter // 32, hidden // 32)
        down_scale_bank = torch.ones(2, hidden // 32, inter // 32)

        def __getitem__(self, index):
            return types.SimpleNamespace(hidden=hidden, inter=inter)

    assert _can_use_w2_grouped_cube(object(), _Bank(), 8)


def test_grouped_cube_routes_without_host_tensor_lists(monkeypatch):
    hidden = inter = W2_CUBE_MIN_INPUT_DIM
    local_experts = 2

    class _GroupedBank(list):
        grouped_ready = True
        local_expert_offset = 2
        num_local_experts = local_experts

    bank = _GroupedBank([types.SimpleNamespace(hidden=hidden, inter=inter) for _ in range(4)])
    bank.gate_packed_bank = torch.zeros(local_experts, inter, hidden // 2, dtype=torch.uint8)
    bank.up_packed_bank = torch.zeros_like(bank.gate_packed_bank)
    bank.down_packed_bank = torch.zeros(local_experts, hidden, inter // 2, dtype=torch.uint8)
    bank.gate_scale_bank = torch.ones(local_experts, inter // 32, hidden // 32)
    bank.up_scale_bank = torch.ones_like(bank.gate_scale_bank)
    bank.down_scale_bank = torch.ones(local_experts, hidden // 32, inter // 32)

    observed_group_ends = []

    def fake_grouped_op(inputs, codes, scales, group_ends):
        del inputs, scales
        observed_group_ends.append(group_ends.detach().cpu().clone())
        output = torch.zeros(topk_ids.numel(), codes.shape[1], dtype=torch.float16)
        start = 0
        for expert, end in enumerate(group_ends.detach().cpu().numpy()):
            output[start:end] = expert + 1
            start = int(end)
        return output

    # EP remaps peer routes to its first resident id but masks their weights.
    # The grouped path must recover the peer rows and avoid projecting them.
    topk_ids = torch.tensor([[2, 2], [3, 2]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25, 0.0], [0.4, 0.0]])
    x = torch.randn(2, hidden)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: (_ for _ in ()).throw(AssertionError("host sync")))

    assert topk_ids.numel() <= W2_GROUPED_MAX_ROUTES
    assert _can_use_w2_grouped_cube(fake_grouped_op, bank, topk_ids.numel())
    output = _method()._apply_device_grouped(fake_grouped_op, bank, x, topk_weights, topk_ids, None)

    expected = torch.tensor([0.25, 0.8]).view(2, 1).expand_as(output)
    torch.testing.assert_close(output, expected)
    assert all(torch.equal(group_ends, torch.tensor([1, 2])) for group_ends in observed_group_ends)


@pytest.mark.parametrize("empty_peer_rows", [False, True])
def test_grouped_prefill_fuses_route_combine_and_skips_peer_rows(monkeypatch, empty_peer_rows):
    from vllm_ascend._310p.quantization.methods import w2_dynamic

    hidden = inter = W2_CUBE_MIN_INPUT_DIM
    num_tokens = 9
    top_k = 2
    bank = types.SimpleNamespace(
        local_expert_offset=0,
        num_local_experts=1,
        fused_route_combine=True,
        empty_peer_rows=empty_peer_rows,
        gate_packed_bank=torch.zeros(1, inter, hidden // 2, dtype=torch.uint8),
        up_packed_bank=torch.zeros(1, inter, hidden // 2, dtype=torch.uint8),
        down_packed_bank=torch.zeros(1, hidden, inter // 2, dtype=torch.uint8),
        gate_scale_bank=torch.ones(1, inter // 32, hidden // 32),
        up_scale_bank=torch.ones(1, inter // 32, hidden // 32),
        down_scale_bank=torch.ones(1, hidden // 32, inter // 32),
    )
    topk_ids = torch.tensor([[0, 1]] * num_tokens)
    topk_weights = torch.tensor([[1.0, 0.0]] * num_tokens)
    x = torch.ones(num_tokens, hidden)
    calls = []
    initialize_calls = []

    def fake_grouped_op(inputs, codes, scales, group_ends, initialize_output=True):
        del inputs, scales
        initialize_calls.append(initialize_output)
        output = torch.full(
            (num_tokens * top_k, codes.shape[1]),
            0 if initialize_output else torch.nan,
            dtype=torch.float16,
        )
        output[: int(group_ends[-1])] = 1
        return output

    def fake_unpermute(rows, inverse, *, probs):
        calls.append((inverse.dtype, probs.dtype))
        unsorted = rows.index_select(0, inverse.long()).reshape(num_tokens, top_k, hidden).float()
        active = probs.unsqueeze(-1) != 0
        return (torch.where(active, unsorted, 0) * probs.unsqueeze(-1)).sum(1).half()

    monkeypatch.setattr(w2_dynamic, "torch_npu", types.SimpleNamespace(npu_moe_token_unpermute=fake_unpermute))
    output = _method()._apply_device_grouped(fake_grouped_op, bank, x, topk_weights, topk_ids, None)
    torch.testing.assert_close(output, torch.ones_like(output))
    assert calls == [(torch.int32, torch.float32)]
    assert initialize_calls == [not empty_peer_rows] * 3


# --- param creation from a synthetic W2 index -------------------------------


@pytest.mark.parametrize("num_tokens", [1, 4, 9])
def test_fp32_combine_dispatch_preserves_precision_and_ignores_poisoned_peers(monkeypatch, num_tokens):
    hidden = inter = W2_CUBE_MIN_INPUT_DIM
    bank = types.SimpleNamespace(
        local_expert_offset=0,
        num_local_experts=1,
        fp32_route_combine=True,
        gate_packed_bank=torch.zeros(1, inter, hidden // 2, dtype=torch.uint8),
        up_packed_bank=torch.zeros(1, inter, hidden // 2, dtype=torch.uint8),
        down_packed_bank=torch.zeros(1, hidden, inter // 2, dtype=torch.uint8),
        gate_scale_bank=torch.ones(1, inter // 32, hidden // 32),
        up_scale_bank=torch.ones(1, inter // 32, hidden // 32),
        down_scale_bank=torch.ones(1, hidden // 32, inter // 32),
    )
    # A nonzero peer weight must also be excluded by the local boundary.
    ids = torch.tensor([[0, 1]] * num_tokens)
    weights = torch.tensor([[0.1234567, 0.8765433]] * num_tokens)
    calls = []

    def projection(x, codes, scales, ends, initialize_output=True):
        assert not initialize_output
        output = torch.full((num_tokens * 2, codes.shape[1]), torch.nan, dtype=torch.float16)
        output[:num_tokens] = 1
        return output

    def combine(rows, inverse, route_weights, ends):
        calls.append((rows.dtype, inverse.dtype, route_weights.dtype, ends.dtype))
        selected = rows[inverse].reshape(num_tokens, 2, hidden).float()
        live = (inverse < ends[-1]).reshape(num_tokens, 2, 1)
        return (torch.where(live, selected, 0) * route_weights.unsqueeze(-1)).sum(1)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_w2_route_combine_310", combine, raising=False)
    shared = types.SimpleNamespace(forward=lambda x: torch.full_like(x, 0.25))
    output = _method()._apply_device_grouped(projection, bank, torch.ones(num_tokens, hidden), weights, ids, shared)
    assert output.dtype == torch.float32
    expected = (weights[:, :1] + 0.25).expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert calls == [(torch.float16, torch.int64, torch.float32, torch.int64)]


@pytest.mark.parametrize("fused_gate_up", [False, True])
@pytest.mark.parametrize("prefill_swiglu", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 4, 9])
@pytest.mark.parametrize("combine_mode", ["torch", "cann", "fp32"])
def test_grouped_intermediates_released_before_later_stages(
    monkeypatch, fused_gate_up, combine_mode, prefill_swiglu, num_tokens
):
    from vllm_ascend._310p.quantization.methods import w2_dynamic

    hidden, inter, top_k = 32, 64, 2
    bank = types.SimpleNamespace(
        local_expert_offset=0,
        num_local_experts=2,
        fused_route_combine=combine_mode == "cann",
        fp32_route_combine=combine_mode == "fp32",
        prefill_swiglu=prefill_swiglu,
        gate_packed_bank=torch.zeros(2, inter, hidden // 2, dtype=torch.uint8),
        up_packed_bank=torch.zeros(2, inter, hidden // 2, dtype=torch.uint8),
        down_packed_bank=torch.zeros(2, hidden, inter // 2, dtype=torch.uint8),
        gate_scale_bank=torch.ones(2, inter // 32, hidden // 32),
        up_scale_bank=torch.ones(2, inter // 32, hidden // 32),
        down_scale_bank=torch.ones(2, hidden // 32, inter // 32),
    )
    if fused_gate_up:
        bank.gate_up_packed_bank = torch.cat((bank.gate_packed_bank, bank.up_packed_bank), dim=1)
        bank.gate_up_scale_bank = torch.cat((bank.gate_scale_bank, bank.up_scale_bank), dim=1)
    refs = {}
    projection_outputs = []
    shared_calls = []

    def projection(inputs, codes, scales, ends, *args):
        if codes is bank.down_packed_bank:
            assert refs["gather"]() is None
            # Includes the base tensor retained by chunk views of fused gate/up.
            assert all(ref() is None for ref in projection_outputs)
            refs["activation"] = weakref.ref(inputs)
            result = inputs[:, :hidden].clone()
            refs["routed"] = weakref.ref(result)
        else:
            refs["gather"] = weakref.ref(inputs)
            result = inputs[:, :1].expand(-1, codes.shape[1]).clone()
            projection_outputs.append(weakref.ref(result))
        return result

    swiglu_calls = []

    def swiglu(gate_up):
        assert refs["gather"]() is None
        assert gate_up.is_contiguous()
        swiglu_calls.append(True)
        gate, up = gate_up.chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate.float()) * up.float()).half()

    monkeypatch.setattr(torch.ops._C_ascend, "npu_w2_swiglu_310", swiglu, raising=False)

    def combine(rows, inverse, weights, ends=None):
        assert refs["activation"]() is None
        return (rows[inverse.long()].reshape(num_tokens, top_k, hidden).float() * weights.unsqueeze(-1)).sum(1)

    def shared_forward(inputs):
        assert all(ref() is None for ref in refs.values())
        shared_calls.append(True)
        return torch.full_like(inputs, 0.25)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_w2_route_combine_310", combine, raising=False)
    monkeypatch.setattr(
        w2_dynamic,
        "torch_npu",
        types.SimpleNamespace(npu_moe_token_unpermute=lambda rows, inverse, *, probs: combine(rows, inverse, probs)),
    )
    x = torch.arange(1, num_tokens + 1, dtype=torch.float16).unsqueeze(1).expand(-1, hidden)
    weights = torch.tensor([[0.25, 0.5]] * num_tokens)
    ids = torch.tensor([[1, 0]] * num_tokens)
    with torch.inference_mode():
        output = _method()._apply_device_grouped(
            projection, bank, x, weights, ids, types.SimpleNamespace(forward=shared_forward)
        )
    activation = (torch.nn.functional.silu(x.float()) * x.float()).half().float()
    torch.testing.assert_close(output, activation * 0.75 + 0.25, rtol=0, atol=0)
    assert shared_calls == [True]
    assert swiglu_calls == ([True] if fused_gate_up and prefill_swiglu and num_tokens >= 9 else [])


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 9, 640])
@pytest.mark.parametrize("has_extension", [False, True])
def test_prefill_fp32_combine_preserves_decode_and_checks_extension_at_boundary(monkeypatch, num_tokens, has_extension):
    hidden = inter = 32
    bank = types.SimpleNamespace(
        local_expert_offset=0,
        num_local_experts=1,
        prefill_fp32_route_combine=False,
        gate_packed_bank=torch.zeros(1, inter, hidden // 2, dtype=torch.uint8),
        up_packed_bank=torch.zeros(1, inter, hidden // 2, dtype=torch.uint8),
        down_packed_bank=torch.zeros(1, hidden, inter // 2, dtype=torch.uint8),
        gate_scale_bank=torch.ones(1, 1, 1),
        up_scale_bank=torch.ones(1, 1, 1),
        down_scale_bank=torch.ones(1, 1, 1),
    )
    ids = torch.tensor([[0, 1]] * num_tokens)
    weights = torch.tensor([[0.1234567, 0.8765433]] * num_tokens)
    initialize_calls = []
    combine_calls = []

    def projection(inputs, codes, scales, ends, initialize_output=True):
        initialize_calls.append(initialize_output)
        result = torch.full(
            (2 * num_tokens, codes.shape[1]), 0 if initialize_output else torch.nan, dtype=torch.float16
        )
        result[:num_tokens] = 1
        return result

    def combine(rows, inverse, route_weights, ends):
        combine_calls.append(True)
        selected = rows[inverse].reshape(num_tokens, 2, hidden).float()
        live = (inverse < ends[-1]).reshape(num_tokens, 2, 1)
        return (torch.where(live, selected, 0) * route_weights.unsqueeze(-1)).sum(1)

    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_w2_route_combine_310", combine if has_extension else None, raising=False
    )
    x = torch.ones(num_tokens, hidden)
    shared = types.SimpleNamespace(forward=lambda inputs: torch.full_like(inputs, 0.25))
    baseline = _method()._apply_device_grouped(projection, bank, x, weights, ids, shared)
    assert initialize_calls == [True] * 3
    initialize_calls.clear()
    bank.prefill_fp32_route_combine = True
    if num_tokens >= 9 and not has_extension:
        with pytest.raises(RuntimeError, match="requires.*extension and OPP"):
            _method()._apply_device_grouped(projection, bank, x, weights, ids, shared)
        assert initialize_calls == []
        return
    actual = _method()._apply_device_grouped(projection, bank, x, weights, ids, shared)
    assert actual.dtype == baseline.dtype == torch.float32
    assert torch.equal(actual, baseline)
    assert initialize_calls == [num_tokens < 9] * 3
    assert combine_calls == ([True] if num_tokens >= 9 else [])


def test_prefill_swiglu_missing_extension_fails_before_projection(monkeypatch):
    bank = types.SimpleNamespace(
        local_expert_offset=0,
        num_local_experts=1,
        prefill_swiglu=True,
        gate_packed_bank=torch.zeros(1, 32, 16, dtype=torch.uint8),
        gate_scale_bank=torch.ones(1, 1, 1),
        gate_up_packed_bank=torch.zeros(1, 64, 16, dtype=torch.uint8),
        gate_up_scale_bank=torch.ones(1, 2, 1),
    )
    monkeypatch.setattr(torch.ops._C_ascend, "npu_w2_swiglu_310", None, raising=False)
    with pytest.raises(RuntimeError, match="prefill SwiGLU requires.*extension and OPP"):
        _method()._apply_device_grouped(
            None, bank, torch.ones(9, 32), torch.ones(9, 1), torch.zeros(9, 1, dtype=torch.int64), None
        )


def test_fp32_combine_missing_extension_fails_before_projection(monkeypatch):
    bank = types.SimpleNamespace(local_expert_offset=0, num_local_experts=1, fp32_route_combine=True)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_w2_route_combine_310", None, raising=False)
    with pytest.raises(RuntimeError, match="requires.*extension and OPP"):
        _method()._apply_device_grouped(
            None, bank, torch.ones(1, 256), torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int64), None
        )


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
