# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the eager 310P DeepSeek V4.1 MLA (plan E3.1).

Everything runs host-side with NO NPU / NO Triton. The eager module
(:mod:`vllm_ascend.models.deepseek_v41.mla`) is validated against the E0.4
golden reference (``tests/ut/deepseek_w2/reference/mla_reference``): its
absorbed-latent and dense forms must both equal the float64 dense/absorbed
oracle at ``MLA_RTOL``/``MLA_ATOL``. Latent-KV cache shape/dtype come from the
E2.1 dtype policy, and an incremental (cache) decode must reproduce prefill.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_mla.py
"""

import pytest
import torch

from tests.ut.deepseek_w2.reference.mla_reference import (
    QK_ROPE_HEAD_DIM,
    MLAConfig,
    MLAWeights,
    apply_rope,
    make_random_weights,
    mla_absorbed_reference,
    mla_dense_reference,
)
from tests.ut.deepseek_w2.reference.tolerances import (
    MLA_ATOL,
    MLA_ROPE_ATOL,
    MLA_ROPE_RTOL,
    MLA_RTOL,
)
from vllm_ascend.models.deepseek_v41.dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
)
from vllm_ascend.models.deepseek_v41.mla import (
    AscendDeepseekV41MLA,
    LatentKVCache,
    apply_decoupled_rope,
)

# V4.1 latent geometry the task pins (kv_lora 512, decoupled rope tail 64).
V41_KV_LORA_RANK = 512
V41_QK_ROPE_HEAD_DIM = 64


def _cfg() -> MLAConfig:
    # Small but structurally faithful (mirrors the E0.4 self-consistency cfg).
    return MLAConfig(
        hidden_size=48,
        num_heads=3,
        q_lora_rank=40,
        kv_lora_rank=24,
        qk_nope_head_dim=16,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=20,
        eps=1e-6,
    )


def _build_mla(cfg: MLAConfig, w: MLAWeights, dtype: torch.dtype = torch.float64) -> AscendDeepseekV41MLA:
    """Build the eager module and load the reference weights verbatim (same layout)."""
    mla = AscendDeepseekV41MLA(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        rms_norm_eps=cfg.eps,
        rope_base=cfg.rope_base,
        dtype=dtype,
    )
    with torch.no_grad():
        mla.w_dq.copy_(w.w_dq)
        mla.q_a_norm.copy_(w.q_a_norm)
        mla.w_uq.copy_(w.w_uq)
        mla.w_dkv.copy_(w.w_dkv)
        mla.kv_a_norm.copy_(w.kv_a_norm)
        mla.w_uk.copy_(w.w_uk)
        mla.w_uv.copy_(w.w_uv)
        mla.w_o.copy_(w.w_o)
    return mla


def _hidden(seq_len: int, seed: int, hidden_size: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(seq_len, hidden_size, generator=gen, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Parity vs the E0.4 reference (the acceptance gate)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("seq_len", [1, 4, 9, 16])
def test_absorbed_matches_reference(seed, seq_len):
    cfg = _cfg()
    w = make_random_weights(cfg, seed)
    mla = _build_mla(cfg, w)
    hidden = _hidden(seq_len, seed + 100, cfg.hidden_size)
    positions = torch.arange(seq_len)

    ref_dense = mla_dense_reference(hidden, positions, cfg, w)
    ref_absorbed = mla_absorbed_reference(hidden, positions, cfg, w)
    out = mla(hidden, positions, absorbed=True)

    assert out.dtype == torch.float64
    assert out.shape == (seq_len, cfg.hidden_size)
    torch.testing.assert_close(out, ref_dense, rtol=MLA_RTOL, atol=MLA_ATOL)
    torch.testing.assert_close(out, ref_absorbed, rtol=MLA_RTOL, atol=MLA_ATOL)


@pytest.mark.parametrize("seq_len", [1, 5, 12])
def test_dense_mode_matches_reference(seq_len):
    cfg = _cfg()
    w = make_random_weights(cfg, 3)
    mla = _build_mla(cfg, w)
    hidden = _hidden(seq_len, seq_len + 7, cfg.hidden_size)
    positions = torch.arange(seq_len)

    out_dense = mla(hidden, positions, absorbed=False)
    ref_dense = mla_dense_reference(hidden, positions, cfg, w)
    torch.testing.assert_close(out_dense, ref_dense, rtol=MLA_RTOL, atol=MLA_ATOL)


@pytest.mark.parametrize("seq_len", [1, 6, 13])
def test_absorbed_equals_dense_within_module(seq_len):
    """The module's two forms are the same math (absorption identity)."""
    cfg = _cfg()
    w = make_random_weights(cfg, 9)
    mla = _build_mla(cfg, w)
    hidden = _hidden(seq_len, seq_len + 20, cfg.hidden_size)
    positions = torch.arange(seq_len)
    absorbed = mla(hidden, positions, absorbed=True)
    dense = mla(hidden, positions, absorbed=False)
    torch.testing.assert_close(absorbed, dense, rtol=MLA_RTOL, atol=MLA_ATOL)


def test_default_positions_are_arange():
    cfg = _cfg()
    w = make_random_weights(cfg, 4)
    mla = _build_mla(cfg, w)
    hidden = _hidden(8, 44, cfg.hidden_size)
    out_default = mla(hidden, None)
    out_explicit = mla(hidden, torch.arange(8))
    torch.testing.assert_close(out_default, out_explicit, rtol=MLA_RTOL, atol=MLA_ATOL)


def test_causality_first_token_independent_of_future():
    cfg = _cfg()
    w = make_random_weights(cfg, 7)
    mla = _build_mla(cfg, w)
    hidden = _hidden(6, 8, cfg.hidden_size)
    positions = torch.arange(6)
    out_a = mla(hidden, positions)
    hidden2 = hidden.clone()
    hidden2[3:] += 5.0
    out_b = mla(hidden2, positions)
    torch.testing.assert_close(out_a[0], out_b[0], rtol=MLA_RTOL, atol=MLA_ATOL)
    assert not torch.allclose(out_a[4], out_b[4])


# ---------------------------------------------------------------------------
# Decoupled RoPE helper parity with the reference
# ---------------------------------------------------------------------------
def test_rope_helper_matches_reference():
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(5, 3, QK_ROPE_HEAD_DIM, generator=gen, dtype=torch.float64)
    positions = torch.arange(5)
    ours = apply_decoupled_rope(x, positions)
    ref = apply_rope(x, positions)
    torch.testing.assert_close(ours, ref, rtol=MLA_ROPE_RTOL, atol=MLA_ROPE_ATOL)


# ---------------------------------------------------------------------------
# Latent-KV cache: shape/dtype from config + policy, and write/read parity
# ---------------------------------------------------------------------------
def test_latent_kv_cache_shape_and_dtype_from_policy():
    # Realistic V4.1 latent geometry; default dtype => policy (FP16).
    mla = AscendDeepseekV41MLA(
        hidden_size=5120,
        num_attention_heads=8,
        q_lora_rank=1536,
        kv_lora_rank=V41_KV_LORA_RANK,
        qk_nope_head_dim=128,
        qk_rope_head_dim=V41_QK_ROPE_HEAD_DIM,
    )
    policy = ASCEND_DEEPSEEKV41_DTYPE_POLICY
    # FP16 main dtype for the projections (no dtype literals in the module).
    assert mla.w_dq.dtype == policy.mla_dtype
    assert mla.param_dtype == policy.mla_dtype

    cache = mla.new_latent_kv_cache(max_seq_len=32)
    assert isinstance(cache, LatentKVCache)
    assert cache.c.shape == (32, V41_KV_LORA_RANK)
    assert cache.k_rope.shape == (32, V41_QK_ROPE_HEAD_DIM)
    assert cache.c.dtype == policy.kv_cache_dtype
    assert cache.k_rope.dtype == policy.kv_cache_dtype
    assert cache.kv_lora_rank == V41_KV_LORA_RANK
    assert cache.qk_rope_head_dim == V41_QK_ROPE_HEAD_DIM


def test_project_latent_kv_shapes():
    cfg = _cfg()
    w = make_random_weights(cfg, 2)
    mla = _build_mla(cfg, w)
    hidden = _hidden(7, 21, cfg.hidden_size)
    c, k_rope = mla.project_latent_kv(hidden)
    assert c.shape == (7, cfg.kv_lora_rank)
    assert k_rope.shape == (7, cfg.qk_rope_head_dim)


@pytest.mark.parametrize("seq_len", [1, 4, 10])
def test_incremental_decode_matches_prefill(seq_len):
    """Token-by-token decode through the latent-KV cache == full prefill."""
    cfg = _cfg()
    w = make_random_weights(cfg, 1)
    mla = _build_mla(cfg, w)
    hidden = _hidden(seq_len, seq_len + 55, cfg.hidden_size)
    positions = torch.arange(seq_len)

    prefill = mla(hidden, positions, absorbed=True)

    cache = mla.new_latent_kv_cache(max_seq_len=seq_len, dtype=torch.float64)
    rows = []
    for t in range(seq_len):
        row = mla(hidden[t : t + 1], positions[t : t + 1], kv_cache=cache, absorbed=True)
        rows.append(row)
    decode = torch.cat(rows, dim=0)

    assert cache.length == seq_len
    torch.testing.assert_close(decode, prefill, rtol=MLA_RTOL, atol=MLA_ATOL)


def test_cache_overflow_raises():
    cfg = _cfg()
    w = make_random_weights(cfg, 0)
    mla = _build_mla(cfg, w)
    cache = mla.new_latent_kv_cache(max_seq_len=1, dtype=torch.float64)
    hidden = _hidden(2, 3, cfg.hidden_size)
    with pytest.raises(ValueError, match="overflow"):
        mla(hidden, torch.arange(2), kv_cache=cache)


# ---------------------------------------------------------------------------
# from_config entry point (what E4.1 wires) + import hygiene
# ---------------------------------------------------------------------------
def test_from_config_reads_geometry():
    class _Cfg:
        hidden_size = 5120
        num_attention_heads = 8
        q_lora_rank = 1536
        kv_lora_rank = V41_KV_LORA_RANK
        qk_nope_head_dim = 128
        qk_rope_head_dim = V41_QK_ROPE_HEAD_DIM
        v_head_dim = 128
        rms_norm_eps = 1e-5
        rope_theta = 10000.0

    mla = AscendDeepseekV41MLA.from_config(_Cfg())
    assert mla.hidden_size == 5120
    assert mla.num_heads == 8
    assert mla.q_lora_rank == 1536
    assert mla.kv_lora_rank == V41_KV_LORA_RANK
    assert mla.qk_rope_head_dim == V41_QK_ROPE_HEAD_DIM
    assert mla.v_head_dim == 128
    assert abs(mla.eps - 1e-5) < 1e-12
    # softmax scale over the full (nope+rope) query dim.
    assert abs(mla.softmax_scale - (128 + 64) ** -0.5) < 1e-12


def test_mla_source_has_no_triton_or_torch_npu_import():
    """The eager MLA module must not import triton or torch_npu (host lane)."""
    import pathlib

    src = pathlib.Path(__file__).parents[3] / "vllm_ascend" / "models" / "deepseek_v41" / "mla.py"
    hits = []
    for lineno, line in enumerate(src.read_text().splitlines(), start=1):
        code = line.split("#", 1)[0]  # comments are documentation, not code
        if "import" in code and ("triton" in code or "torch_npu" in code):
            hits.append((lineno, line))
    assert not hits, f"triton/torch_npu import reachable in mla.py: {hits}"


def test_mla_import_does_not_pull_shipped_device_mla():
    """Fresh interpreter: importing the eager MLA must not pull the shipped
    device MLA / DSA / V4 base (all torch_npu + Triton), so it stays importable
    on the CPU host. ``import vllm_ascend`` loads Triton globally, so a bare
    ``'triton' in sys.modules`` check is not a valid signal (mirrors the
    package test); the honest gate is that *this* module pulls neither the
    shipped device MLA nor the shipped V4 base nor torch_npu."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import vllm_ascend.models.deepseek_v41.mla as m\n"
        "assert hasattr(m, 'AscendDeepseekV41MLA')\n"
        "assert 'torch_npu' not in sys.modules, 'torch_npu pulled into eager MLA import'\n"
        "assert 'vllm_ascend.ops.mla' not in sys.modules, 'shipped device MLA pulled in'\n"
        "assert 'vllm_ascend.attention.mla_v1' not in sys.modules, 'shipped AscendMLAImpl pulled in'\n"
        "assert 'vllm_ascend.models.deepseek_v4.model' not in sys.modules, 'shipped V4 base pulled in'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout
