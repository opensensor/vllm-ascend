# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Qwen4Exp GDN wiring on Ascend 310P (task T5.1).

Verifies that :mod:`vllm_ascend.models.qwen4_exp.qwen4exp_gdn` wires the
Qwen4Exp Gated DeltaNet layers (short conv + gating + chunked/unchunked delta
rule) correctly against:

* the T0.6 float64 golden reference (``tests/ut/qwen38_1m/reference``) at the
  declared GDN tolerances, and
* the real Triton-free 310P ``fla`` PyTorch kernels (float32-internal), at a
  documented float32 tolerance -- exercising the actual kernel wiring.

Also asserts the config-derived GDN state shapes and the T1.2 policy dtypes.

Run with ``pytest --noconftest`` (the shared ut conftest fails to import here).
"""

import importlib.machinery
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.ut.qwen38_1m.reference.gdn_reference import (
    causal_depthwise_conv1d,
    gdn_delta_rule_chunked,
    gdn_delta_rule_recurrent,
    gdn_gating,
    preprocess_qk,
)
from tests.ut.qwen38_1m.reference.tolerances import (
    GDN_CHUNK_ATOL,
    GDN_CHUNK_RTOL,
    GDN_CONV_ATOL,
    GDN_CONV_RTOL,
)
from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.qwen4exp_gdn import (
    Qwen4ExpGDNParams,
    gdn_conv_state_shape,
    gdn_delta_rule,
    gdn_recurrent_state_shape,
    gdn_short_conv,
    gdn_state_dtypes,
)
from vllm_ascend.models.qwen4_exp.qwen4exp_gdn import (
    gdn_gating as adapter_gdn_gating,
)

# The fla PyTorch kernels import ``l2norm_310p``, which pulls in ``torch_npu``
# and ``vllm.third_party.flash_linear_attention`` -- neither exists on this
# CPU-only host. Only the ``backend="fla_pytorch"`` path needs them, so stub
# them *transiently* (never at import) and always remove ``torch_npu`` again, so
# sibling tests that assert ``"torch_npu" not in sys.modules`` are unaffected.
_FLA_STUB_PACKAGES = (
    "vllm.third_party.flash_linear_attention",
    "vllm.third_party.flash_linear_attention.ops",
    "vllm.third_party.flash_linear_attention.ops.utils",
)


@contextmanager
def _fla_host_stubs():
    saved = {name: sys.modules.get(name) for name in ("torch_npu", *_FLA_STUB_PACKAGES)}
    try:
        if sys.modules.get("torch_npu") is None:
            stub = types.ModuleType("torch_npu")
            # transformers' is_torch_npu_available() calls find_spec on it.
            stub.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
            sys.modules["torch_npu"] = stub
        for name in _FLA_STUB_PACKAGES:
            if sys.modules.get(name) is None:
                mod = types.ModuleType(name)
                mod.__path__ = []  # mark as a package
                mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                sys.modules[name] = mod
        utils = sys.modules[_FLA_STUB_PACKAGES[-1]]
        if not hasattr(utils, "tensor_cache"):
            utils.tensor_cache = lambda fn: fn
        yield
    finally:
        # Restore the pre-test state; crucially, drop the ``torch_npu`` stub so
        # the "host never imports torch_npu" invariant other tests check holds.
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


# Compact but representative GDN head geometry for a CPU reference (matches the
# T0.6 self-consistency test: hidden-2560 uses small head dims).
_NUM_HEADS = 4
_K_DIM = 16
_V_DIM = 16

# --- float32 wiring tolerances (declared before any comparison) -------------
# The stock fla kernels compute the recurrence/gating in float32, so their
# agreement with the adapter's float32 eager twin (identical math, same dtype)
# is at float32 rounding level over a few hundred tokens of reassociated
# products. 1e-4 is safe (a real wiring bug -- wrong scale, transposed state,
# swapped q/k -- shifts outputs by O(1)) and achievable.
GDN_FLA_FP32_RTOL = 1e-3
GDN_FLA_FP32_ATOL = 1e-4

# Authoritative Qwen4Exp (Qwen3-Next family) GDN config for the hidden-2560
# model: the linear_* GDN keys plus the dense-attention rotary params.
_QWEN4EXP_GDN_CONFIG = SimpleNamespace(
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    head_dim=256,
    partial_rotary_factor=0.25,
)


def _make_inputs(seq_len, seed, with_state, num_k_heads=_NUM_HEADS, num_v_heads=_NUM_HEADS):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(seq_len, num_k_heads, _K_DIM, generator=gen, dtype=torch.float64)
    k = torch.randn(seq_len, num_k_heads, _K_DIM, generator=gen, dtype=torch.float64)
    v = torch.randn(seq_len, num_v_heads, _V_DIM, generator=gen, dtype=torch.float64)
    a = torch.randn(seq_len, num_v_heads, generator=gen, dtype=torch.float64)
    b = torch.randn(seq_len, num_v_heads, generator=gen, dtype=torch.float64)
    A_log = torch.randn(num_v_heads, generator=gen, dtype=torch.float64)
    dt_bias = torch.randn(num_v_heads, generator=gen, dtype=torch.float64)
    state = None
    if with_state:
        state = torch.randn(num_v_heads, _V_DIM, _K_DIM, generator=gen, dtype=torch.float64)
    return q, k, v, a, b, A_log, dt_bias, state


def _reference_outputs(q, k, v, a, b, A_log, dt_bias, state, chunked, chunk_size=64):
    """Golden T0.6 output: gate -> preprocess -> (chunked|recurrent) delta rule."""
    g, beta = gdn_gating(a, b, A_log, dt_bias)
    qn, kn = preprocess_qk(q, k)
    if chunked:
        return gdn_delta_rule_chunked(qn, kn, v, g, beta, state, chunk_size=chunk_size)
    return gdn_delta_rule_recurrent(qn, kn, v, g, beta, state)


# ---------------------------------------------------------------------------
# Config-derived state shapes / dtypes.
# ---------------------------------------------------------------------------
def test_params_from_hf_config():
    params = Qwen4ExpGDNParams.from_hf_config(_QWEN4EXP_GDN_CONFIG)
    assert params.num_k_heads == 16
    assert params.num_v_heads == 32
    assert params.head_k_dim == 128
    assert params.head_v_dim == 128
    assert params.conv_kernel_size == 4
    assert params.key_dim == 2048
    assert params.value_dim == 4096
    # mixed_qkv = q(key_dim) + k(key_dim) + v(value_dim).
    assert params.conv_dim == 8192


def test_partial_rotary_and_short_conv_params():
    """Verify the partial-rotary / short-conv params from the HF config."""
    params = Qwen4ExpGDNParams.from_hf_config(_QWEN4EXP_GDN_CONFIG)
    # head_dim 256 * partial_rotary_factor 0.25 -> rotary dim 64.
    assert params.partial_rotary_factor == 0.25
    assert params.rotary_dim == 64
    # short conv kernel is the GDN causal depthwise kernel width.
    assert params.conv_kernel_size == 4
    params.validate()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_recurrent_state_shape(tp_size):
    params = Qwen4ExpGDNParams.from_hf_config(_QWEN4EXP_GDN_CONFIG)
    shape = gdn_recurrent_state_shape(params, tp_size=tp_size)
    assert shape == (params.num_v_heads // tp_size, params.head_v_dim, params.head_k_dim)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("num_spec", [0, 3])
def test_conv_state_shape(tp_size, num_spec):
    params = Qwen4ExpGDNParams.from_hf_config(_QWEN4EXP_GDN_CONFIG)
    shape = gdn_conv_state_shape(params, tp_size=tp_size, num_spec=num_spec, layout="DS")
    assert shape == (params.conv_dim // tp_size, params.conv_kernel_size - 1 + num_spec)
    # SD layout transposes the pair.
    shape_sd = gdn_conv_state_shape(params, tp_size=tp_size, num_spec=num_spec, layout="SD")
    assert shape_sd == (shape[1], shape[0])


def test_state_dtypes_from_policy():
    conv_dtype, ssm_dtype = gdn_state_dtypes()
    # Per T1.2 policy: conv rides fp16 main dtype, SSM state stays fp32.
    assert conv_dtype is ASCEND_QWEN4EXP_DTYPE_POLICY.mamba_conv_cache_dtype
    assert ssm_dtype is ASCEND_QWEN4EXP_DTYPE_POLICY.mamba_ssm_cache_dtype
    assert conv_dtype == torch.float16
    assert ssm_dtype == torch.float32


def test_model_short_conv_fallback_carries_paged_state():
    from vllm_ascend.models.qwen4_exp.model import _GDNAttention

    channels, kernel = 4, 4
    weight = torch.randn(channels, kernel)
    first = torch.randn(5, channels)
    second = torch.randn(1, channels)
    cache = torch.zeros(2, channels, kernel - 1)
    owner = SimpleNamespace(
        kv_cache=(cache,),
        prefix="model.layers.0.attention",
        conv_weight=weight,
        conv_dim=channels,
    )
    metadata = SimpleNamespace()
    context = SimpleNamespace(attn_metadata={"model.layers.0.attention": metadata})

    with patch(
        "vllm_ascend.models.qwen4_exp.model.get_forward_context",
        return_value=context,
    ):
        out_first = _GDNAttention._stateful_short_conv(
            owner,
            first,
            torch.tensor([1]),
            torch.tensor([0, len(first)]),
            torch.tensor([False]),
        )
        delattr(metadata, "_qwen4exp_query_ranges")
        out_second = _GDNAttention._stateful_short_conv(
            owner,
            second,
            torch.tensor([1]),
            torch.tensor([0, len(second)]),
            torch.tensor([True]),
        )

    expected = gdn_short_conv(
        torch.cat([first, second]),
        weight,
        activation="silu",
        compute_dtype=first.dtype,
    )
    torch.testing.assert_close(out_first, expected[:-1])
    torch.testing.assert_close(out_second, expected[-1:])
    torch.testing.assert_close(cache[1], torch.cat([first, second])[-(kernel - 1) :].T)


@pytest.mark.parametrize("is_prefill", [True, False])
def test_model_native_delta_rule_uses_batched_310p_kernel(is_prefill):
    """The production path must not fall back to the per-request Python loop."""
    from vllm_ascend.models.qwen4_exp.model import _GDNAttention

    tokens, num_heads, head_dim = 3, 2, 4
    q = torch.randn(tokens, num_heads, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn(1, tokens, num_heads)
    beta = torch.rand_like(g)
    recurrent_cache = torch.zeros(4, num_heads, head_dim, head_dim)
    owner = SimpleNamespace(kv_cache=(torch.empty(0), recurrent_cache))
    metadata = SimpleNamespace(num_prefills=int(is_prefill))
    state_indices = torch.tensor([2])
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32)
    has_initial_state = torch.tensor([False])

    chunk_output = v.unsqueeze(0) + 1
    final_state = torch.full((1, num_heads, head_dim, head_dim), 7.0)
    chunk_kernel = Mock(return_value=(chunk_output, final_state))
    recurrent_output = v.unsqueeze(0) + 2
    recurrent_kernel = Mock(return_value=recurrent_output)

    chunk_module = types.ModuleType("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
    chunk_module.chunk_gated_delta_rule_310 = chunk_kernel
    gdn_module = types.ModuleType("vllm_ascend._310p.ops.fla.gdn_310")
    gdn_module._cached_chunk_plan = Mock(return_value="chunk-plan")
    gdn_module._cached_recurrent_step_meta = Mock(return_value="step-meta")
    gdn_module.npu_recurrent_gated_delta_rule_310 = recurrent_kernel

    with patch.dict(
        sys.modules,
        {
            chunk_module.__name__: chunk_module,
            gdn_module.__name__: gdn_module,
        },
    ):
        output = _GDNAttention._native_delta_rule(
            owner,
            q,
            k,
            v,
            g,
            beta,
            metadata,
            state_indices,
            query_start_loc,
            has_initial_state,
        )

    if is_prefill:
        chunk_kernel.assert_called_once()
        recurrent_kernel.assert_not_called()
        torch.testing.assert_close(output, chunk_output.squeeze(0))
        torch.testing.assert_close(recurrent_cache[2], final_state[0])
    else:
        recurrent_kernel.assert_called_once()
        chunk_kernel.assert_not_called()
        torch.testing.assert_close(output, recurrent_output.squeeze(0))


# ---------------------------------------------------------------------------
# Gating parity vs T0.6.
# ---------------------------------------------------------------------------
def test_gating_parity_high_precision():
    _, _, _, a, b, A_log, dt_bias, _ = _make_inputs(50, seed=3, with_state=False)
    g_ref, beta_ref = gdn_gating(a, b, A_log, dt_bias)
    g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, compute_dtype=torch.float64)
    torch.testing.assert_close(g, g_ref, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
    torch.testing.assert_close(beta, beta_ref, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
    assert torch.all(g <= 0)
    assert torch.all((beta > 0) & (beta < 1))


# ---------------------------------------------------------------------------
# Short-conv parity vs T0.6.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("activation", [None, "silu"])
def test_short_conv_parity(activation):
    gen = torch.Generator().manual_seed(21)
    seq_len, channels, kernel = 24, 8, 4
    x = torch.randn(seq_len, channels, generator=gen, dtype=torch.float64)
    weight = torch.randn(channels, kernel, generator=gen, dtype=torch.float64)
    bias = torch.randn(channels, generator=gen, dtype=torch.float64)
    ref = causal_depthwise_conv1d(x, weight, bias=bias, activation=activation)
    out = gdn_short_conv(x, weight, bias=bias, activation=activation, compute_dtype=torch.float64)
    torch.testing.assert_close(out, ref, rtol=GDN_CONV_RTOL, atol=GDN_CONV_ATOL)


# ---------------------------------------------------------------------------
# Delta-rule parity vs T0.6 (chunked + unchunked), high precision (float64).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 64, 65, 130])
@pytest.mark.parametrize("with_state", [False, True])
def test_recurrent_parity_high_precision(seq_len, with_state):
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(seq_len, seed=500 + seq_len, with_state=with_state)
    o_ref, s_ref = _reference_outputs(q, k, v, a, b, A_log, dt_bias, state, chunked=False)
    g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, compute_dtype=torch.float64)
    o, s = gdn_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        chunked=False,
        compute_dtype=torch.float64,
        backend="eager",
    )
    torch.testing.assert_close(o, o_ref, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
    torch.testing.assert_close(s, s_ref, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)


@pytest.mark.parametrize("seq_len", [1, 7, 33, 64, 65, 130])
@pytest.mark.parametrize("with_state", [False, True])
def test_chunked_parity_high_precision(seq_len, with_state):
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(seq_len, seed=700 + seq_len, with_state=with_state)
    # Compare the chunked adapter output against the *recurrent* golden form
    # (the two are algebraically identical; this is the strongest check).
    o_ref, s_ref = _reference_outputs(q, k, v, a, b, A_log, dt_bias, state, chunked=False)
    g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, compute_dtype=torch.float64)
    o, s = gdn_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        chunked=True,
        compute_dtype=torch.float64,
        backend="eager",
    )
    torch.testing.assert_close(o, o_ref, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
    torch.testing.assert_close(s, s_ref, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)


def test_chunked_equals_recurrent_adapter():
    """Adapter's own chunked and unchunked forms must agree (float64)."""
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(97, seed=7, with_state=True)
    g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, compute_dtype=torch.float64)
    o_rec, s_rec = gdn_delta_rule(q, k, v, g, beta, initial_state=state, chunked=False, compute_dtype=torch.float64)
    o_chunk, s_chunk = gdn_delta_rule(q, k, v, g, beta, initial_state=state, chunked=True, compute_dtype=torch.float64)
    torch.testing.assert_close(o_chunk, o_rec, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
    torch.testing.assert_close(s_chunk, s_rec, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)


def test_grouped_value_heads_chunked_equals_recurrent():
    """Grouped-value attention (Hv = 2 * Hk) is consistent across both forms."""
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(48, seed=13, with_state=True, num_k_heads=2, num_v_heads=4)
    g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, compute_dtype=torch.float64)
    o_rec, s_rec = gdn_delta_rule(q, k, v, g, beta, initial_state=state, chunked=False, compute_dtype=torch.float64)
    o_chunk, s_chunk = gdn_delta_rule(q, k, v, g, beta, initial_state=state, chunked=True, compute_dtype=torch.float64)
    assert o_rec.shape == (48, 4, _V_DIM)
    torch.testing.assert_close(o_chunk, o_rec, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
    torch.testing.assert_close(s_chunk, s_rec, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)


# ---------------------------------------------------------------------------
# Real fla-kernel wiring (float32-internal): exercised + bounded vs reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
def test_fla_pytorch_kernel_wiring(chunked, with_state):
    seq_len = 96
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(seq_len, seed=900, with_state=with_state)
    o_ref, s_ref = _reference_outputs(q, k, v, a, b, A_log, dt_bias, state, chunked=chunked)
    with _fla_host_stubs():
        # Gate via the real fla gating kernel; run delta rule via the real fla path.
        g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, backend="fla_pytorch")
        o, s = gdn_delta_rule(
            q,
            k,
            v,
            g.double(),
            beta.double(),
            initial_state=state,
            chunked=chunked,
            backend="fla_pytorch",
        )
    torch.testing.assert_close(o.double(), o_ref, rtol=GDN_FLA_FP32_RTOL, atol=GDN_FLA_FP32_ATOL)
    torch.testing.assert_close(s.double(), s_ref, rtol=GDN_FLA_FP32_RTOL, atol=GDN_FLA_FP32_ATOL)


@pytest.mark.parametrize("chunked", [False, True])
def test_eager_fp32_matches_fla_kernels(chunked):
    """The adapter's float32 eager twin reproduces the stock fla kernels."""
    seq_len = 80
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(seq_len, seed=111, with_state=True)
    with _fla_host_stubs():
        g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, backend="fla_pytorch")
        o_fla, s_fla = gdn_delta_rule(
            q, k, v, g.double(), beta.double(), initial_state=state, chunked=chunked, backend="fla_pytorch"
        )
    o_eager, s_eager = gdn_delta_rule(
        q,
        k,
        v,
        g.double(),
        beta.double(),
        initial_state=state,
        chunked=chunked,
        backend="eager",
        compute_dtype=torch.float32,
    )
    torch.testing.assert_close(o_eager.double(), o_fla.double(), rtol=GDN_FLA_FP32_RTOL, atol=GDN_FLA_FP32_ATOL)
    torch.testing.assert_close(s_eager.double(), s_fla.double(), rtol=GDN_FLA_FP32_RTOL, atol=GDN_FLA_FP32_ATOL)


def test_stock_fla_kernels_miss_float64_tolerance():
    """Document why a dtype-honoring eager path exists (RED for the stock path).

    The stock fla kernels are float32-internal, so they cannot meet the T0.6
    float64 delta-rule tolerance; the adapter's eager float64 path is what
    reaches it. This test pins that limitation.
    """
    seq_len = 96
    q, k, v, a, b, A_log, dt_bias, state = _make_inputs(seq_len, seed=42, with_state=False)
    o_ref, _ = _reference_outputs(q, k, v, a, b, A_log, dt_bias, state, chunked=False)
    with _fla_host_stubs():
        g, beta = adapter_gdn_gating(A_log, a, b, dt_bias, backend="fla_pytorch")
        o_fla, _ = gdn_delta_rule(
            q, k, v, g.double(), beta.double(), initial_state=state, chunked=False, backend="fla_pytorch"
        )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(o_fla.double(), o_ref, rtol=GDN_CHUNK_RTOL, atol=GDN_CHUNK_ATOL)
