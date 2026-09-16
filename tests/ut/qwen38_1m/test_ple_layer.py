# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity tests for the Ascend 310P Qwen4Exp PLE injection layer (plan T4.3).

The Triton-free PLE layer (:class:`AscendQwen4ExpPLELayer`) does n-gram row
gather (through the T4.1 host PLE table method), the merged key/value
projection, the sigmoid gate, and the dilated causal short convolution combined
with the outer residual. These tests assert:

* **Parity vs the T0.6 eager reference** (``ple_reference.ple_gate`` /
  ``ple_short_conv``): the whole layer output matches an independent reference
  composition of the same math within the declared PLE tolerances. The layer,
  the ops module, and the PLE table are driven in float64 so the parity isolates
  the *formulas* from fp16 rounding; the reduction ops re-derive the float64
  precision through their dtype-promotion policy.
* **PLE row identity** (feeds T4.2): the gathered ``[T, ple_embed_dim]``
  embedding is exactly the per-head concatenation of the requested table rows,
  through both host transports -- (b) ``/dev/shm`` shared-mmap and (a) pinned-UVA
  (device calls mocked).
* The **device dtype path**: with the authoritative float16 policy the layer's
  parameters and output ride the policy dtypes.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_ple_layer.py
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.reference.ple_reference import (
    ple_gate as ref_ple_gate,
)
from tests.ut.qwen38_1m.reference.ple_reference import (
    ple_short_conv as ref_ple_short_conv,
)
from tests.ut.qwen38_1m.reference.tolerances import (
    PLE_CONV_ATOL,
    PLE_CONV_RTOL,
    PLE_GATE_ATOL,
    PLE_GATE_RTOL,
)
from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    Qwen4ExpDtypePolicy,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    AscendPLEPinnedHostEmbeddingMethod,
    AscendPLETransport,
    create_ple_embedding_method,
)
from vllm_ascend.models.qwen4_exp.ple_layer import AscendQwen4ExpPLELayer

# Host RAM the byte-budget check is told to assume, so tiny synthetic tables
# never trip the 48 GiB OS/transfer reserve on small-memory CI hosts.
_HUGE_HOST_BYTES = 10**15
_EPS = 1e-6

# Small synthetic geometry: hidden_size H, hc_count HC -> hc_hidden = H*HC.
_HIDDEN_SIZE = 4
_HC_COUNT = 3
_NGRAM_SIZE = 3  # -> dilation 3, (ngram_size - 1) = 2 predecessor orders
_HEADS_PER_NGRAM = 2  # -> num_ngram_heads = 2 * 2 = 4
_PER_HEAD_DIM = 5  # -> ple_embed_dim = 4 * 5 = 20
_CONV_KERNEL = 4
_NUM_NGRAM_HEADS = (_NGRAM_SIZE - 1) * _HEADS_PER_NGRAM
_PLE_EMBED_DIM = _NUM_NGRAM_HEADS * _PER_HEAD_DIM
_TABLE_ROWS = 64


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=_HIDDEN_SIZE,
        hc_count=_HC_COUNT,
        ple_embed_dim=_PLE_EMBED_DIM,
        ple_conv_kernel_size=_CONV_KERNEL,
        ngram_size=_NGRAM_SIZE,
        heads_per_ngram=_HEADS_PER_NGRAM,
        rms_norm_eps=_EPS,
    )


def _f64_policy() -> Qwen4ExpDtypePolicy:
    """Policy whose PLE table + projection ride float64 for exact-math parity."""
    base = ASCEND_QWEN4EXP_DTYPE_POLICY.as_dict()
    base["ngram_embedding_dtype"] = torch.float64
    base["ple_projection_dtype"] = torch.float64
    return Qwen4ExpDtypePolicy(**base)


def _rand(shape, seed, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


def _known_table(rows: int, dim: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Deterministic table with a distinct, non-degenerate value per row/col."""
    r = torch.arange(rows, dtype=torch.float64).unsqueeze(1)
    c = torch.arange(dim, dtype=torch.float64).unsqueeze(0)
    return (torch.sin(0.5 * r + 0.25 * c) + 0.01 * r).to(dtype)


def _ngram_ids(num_tokens: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, _TABLE_ROWS, (num_tokens, _NUM_NGRAM_HEADS), generator=gen, dtype=torch.long)


def _mmap_method(table: torch.Tensor, policy: Qwen4ExpDtypePolicy):
    shm_path = os.path.join("/dev/shm", f"vllm_ascend_ple_test_{uuid.uuid4().hex}.bin")
    method = create_ple_embedding_method(
        num_embeddings=table.shape[0],
        embedding_dim=table.shape[1],
        transport=AscendPLETransport.SHARED_MMAP,
        dtype_policy=policy,
        shm_path=shm_path,
        table_source=table,
        host_total_bytes=_HUGE_HOST_BYTES,
    )
    return method, shm_path


def _pinned_method(table: torch.Tensor, policy: Qwen4ExpDtypePolicy):
    """Transport (a) with the device calls mocked (host-importable)."""
    return AscendPLEPinnedHostEmbeddingMethod(
        table.shape[0],
        table.shape[1],
        table_source=table,
        uva_probe=lambda: True,
        # Plain CPU allocation stands in for the pinned+UVA allocation on host.
        pinned_allocator=lambda n, d, dt: torch.empty(n, d, dtype=dt),
        accelerator_view_fn=None,
        dtype_policy=policy,
        host_total_bytes=_HUGE_HOST_BYTES,
    )


def _build_layer(method, policy: Qwen4ExpDtypePolicy, params_dtype: torch.dtype | None) -> AscendQwen4ExpPLELayer:
    layer = AscendQwen4ExpPLELayer(
        config=_config(),
        layer_idx=0,
        ple_method=method,
        dtype_policy=policy,
        params_dtype=params_dtype,
        prefix="ple.test",
    )
    return layer


def _load_known_weights(layer: AscendQwen4ExpPLELayer, dtype: torch.dtype) -> None:
    hc_hidden = layer.hc_hidden_size
    out = sum(layer.output_sizes)
    layer.kv_proj_weight.data = _rand((out, layer.ple_embed_dim), 71).to(dtype)
    layer.norm_key_weight.data = (0.1 * _rand((hc_hidden,), 72)).to(dtype)
    layer.norm_query_weight.data = (0.1 * _rand((hc_hidden,), 73)).to(dtype)
    layer.norm_conv_weight.data = (0.1 * _rand((hc_hidden,), 74)).to(dtype)
    layer.conv_weight.data = _rand((hc_hidden, layer.conv_kernel_size), 75).to(dtype)


def _reference_output(
    layer: AscendQwen4ExpPLELayer,
    table: torch.Tensor,
    hidden_states: torch.Tensor,
    ngram_ids: torch.Tensor,
) -> torch.Tensor:
    """Independent T0.6-reference composition of the same PLE math (float64)."""
    # Row gather: per-head concatenation of the requested table rows.
    gathered = table.to(torch.float64)[ngram_ids]  # [T, heads, per_head_dim]
    embeddings = gathered.reshape(ngram_ids.shape[0], -1)  # [T, ple_embed_dim]
    weight = layer.kv_proj_weight.detach().to(torch.float64)
    kv = embeddings @ weight.t()
    key, value = kv.split(layer.output_sizes, dim=-1)
    gated, conv_input = ref_ple_gate(
        key,
        value,
        hidden_states.to(torch.float64),
        layer.norm_key_weight.detach(),
        layer.norm_query_weight.detach(),
        layer.norm_conv_weight.detach(),
        _EPS,
    )
    return ref_ple_short_conv(
        conv_input,
        gated,
        hidden_states.to(torch.float64),
        layer.conv_weight.detach(),
        layer.short_conv_dilation,
        activation="silu",
    )


# ---------------------------------------------------------------------------
# Parity vs the T0.6 eager reference (both host transports).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("transport", ["mmap", "pinned"])
@pytest.mark.parametrize("num_tokens", [1, 3, 8])
def test_layer_matches_reference(transport, num_tokens):
    policy = _f64_policy()
    table = _known_table(_TABLE_ROWS, _PER_HEAD_DIM, torch.float64)
    shm_path = None
    if transport == "mmap":
        method, shm_path = _mmap_method(table, policy)
    else:
        method = _pinned_method(table, policy)
    try:
        layer = _build_layer(method, policy, params_dtype=torch.float64)
        _load_known_weights(layer, torch.float64)
        hidden_states = _rand((num_tokens, layer.hc_hidden_size), 90 + num_tokens)
        ngram_ids = _ngram_ids(num_tokens, 300 + num_tokens)

        out = layer.forward(hidden_states, ngram_ids)
        expected = _reference_output(layer, table, hidden_states, ngram_ids)

        assert out.dtype == torch.float64
        assert out.shape == (num_tokens, layer.hc_hidden_size)
        # Gate + conv share the tighter conv tolerance; use the looser gate one
        # to bound the composed output (both are rounding-level in float64).
        max_err = (out - expected).abs().max().item()
        assert max_err < PLE_GATE_ATOL, f"max abs error {max_err} exceeds {PLE_GATE_ATOL}"
        torch.testing.assert_close(out, expected, rtol=PLE_GATE_RTOL, atol=PLE_GATE_ATOL)
        torch.testing.assert_close(out, expected, rtol=PLE_CONV_RTOL, atol=PLE_CONV_ATOL)
    finally:
        method.close()
        if shm_path is not None and os.path.exists(shm_path):
            os.unlink(shm_path)


# ---------------------------------------------------------------------------
# PLE row identity (feeds T4.2).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("transport", ["mmap", "pinned"])
def test_row_identity_through_layer(transport):
    policy = _f64_policy()
    table = _known_table(_TABLE_ROWS, _PER_HEAD_DIM, torch.float64)
    shm_path = None
    if transport == "mmap":
        method, shm_path = _mmap_method(table, policy)
    else:
        method = _pinned_method(table, policy)
    try:
        layer = _build_layer(method, policy, params_dtype=torch.float64)
        num_tokens = 6
        ngram_ids = _ngram_ids(num_tokens, 12345)

        embeddings = layer.gather_embeddings(ngram_ids, torch.float64)
        assert embeddings.shape == (num_tokens, _PLE_EMBED_DIM)
        # Each head slice must equal the exact requested table row.
        for t in range(num_tokens):
            for head in range(_NUM_NGRAM_HEADS):
                sl = slice(head * _PER_HEAD_DIM, (head + 1) * _PER_HEAD_DIM)
                expected_row = table[int(ngram_ids[t, head])].to(torch.float64)
                assert torch.equal(embeddings[t, sl], expected_row)
        # Direct method-level identity too (batched gather, one call).
        rows = method.gather_rows(ngram_ids.reshape(-1))
        assert torch.equal(rows, table.to(torch.float64)[ngram_ids.reshape(-1)])
    finally:
        method.close()
        if shm_path is not None and os.path.exists(shm_path):
            os.unlink(shm_path)


# ---------------------------------------------------------------------------
# Device dtype path: the authoritative float16 policy governs params + output.
# ---------------------------------------------------------------------------
def test_device_dtype_policy_path():
    policy = ASCEND_QWEN4EXP_DTYPE_POLICY  # float16 main + fp32 accumulation
    table = _known_table(_TABLE_ROWS, _PER_HEAD_DIM, policy.cast_site("ngram_embedding"))
    method, shm_path = _mmap_method(table, policy)
    try:
        layer = _build_layer(method, policy, params_dtype=None)
        # Params default to the projection dtype (float16), accumulation is fp32.
        assert layer.kv_proj_weight.dtype == policy.cast_site("ple_projection")
        assert layer.norm_accumulation_dtype == policy.cast_site("ple_norm_accumulation")
        _load_known_weights(layer, layer.params_dtype)
        num_tokens = 5
        hidden_states = _rand((num_tokens, layer.hc_hidden_size), 51).to(policy.cast_site("ple_projection"))
        ngram_ids = _ngram_ids(num_tokens, 52)
        out = layer.forward(hidden_states, ngram_ids)
        assert out.dtype == policy.cast_site("ple_projection")
        assert out.shape == (num_tokens, layer.hc_hidden_size)
        assert torch.isfinite(out.float()).all()
    finally:
        method.close()
        if os.path.exists(shm_path):
            os.unlink(shm_path)


# ---------------------------------------------------------------------------
# Guardrails.
# ---------------------------------------------------------------------------
def test_forward_requires_ple_method():
    layer = AscendQwen4ExpPLELayer(config=_config(), layer_idx=0, ple_method=None, params_dtype=torch.float64)
    with pytest.raises(RuntimeError, match="PLE embedding method"):
        layer.forward(_rand((2, layer.hc_hidden_size), 1), _ngram_ids(2, 2))


def test_forward_rejects_shape_mismatch():
    policy = _f64_policy()
    table = _known_table(_TABLE_ROWS, _PER_HEAD_DIM, torch.float64)
    method, shm_path = _mmap_method(table, policy)
    try:
        layer = _build_layer(method, policy, params_dtype=torch.float64)
        with pytest.raises(ValueError, match="token"):
            layer.forward(_rand((3, layer.hc_hidden_size), 1), _ngram_ids(2, 2))
        with pytest.raises(ValueError, match="hc_hidden"):
            layer.forward(_rand((2, layer.hc_hidden_size + 1), 1), _ngram_ids(2, 2))
    finally:
        method.close()
        if os.path.exists(shm_path):
            os.unlink(shm_path)
