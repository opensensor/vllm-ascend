# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Qwen4Exp GDN state lifecycle on Ascend 310P (T5.2).

Covers copy / slot-remap semantics across prefill / decode / preemption / reuse
for the Gated DeltaNet conv + recurrent (SSM) state, host-side with NO NPU:

* **chunked-vs-unchunked**: a sequence processed in one shot equals the same
  sequence split across block boundaries with recurrent state carried between
  segments (both the chunk-parallel and token-recurrent engines);
* **preemption == unpreempted**: an interleaved 2-request decode whose second
  request is force-preempted mid-stream and resumed (fail-closed: state dropped,
  recomputed on resume) produces bit-identical per-request outputs to a run with
  no preemption;
* **no aliasing**: two live requests never map to the same state block, and a
  freed/preempted block is zeroed before reuse (no stale carryover).

Also exercises the per-rank replication (4 TP ranks) fan-out on the 310P model
state class and the fork-mirroring copy-func pair.

Run with ``--noconftest`` (the shared ut conftest fails to import on this host):

    python3 -m pytest -q --noconftest \
        tests/ut/qwen38_1m/test_gdn_lifecycle.py
"""

import importlib.util
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.models.qwen4_exp.qwen4exp_gdn import (
    GDNStateCopySpec,
    Qwen4ExpGDNParams,
    Qwen4ExpGDNStateLayout,
    Qwen4ExpGDNStatePool,
    gdn_conv_copy_spec,
    gdn_delta_rule,
    gdn_gating,
    gdn_prefill_in_chunks,
    gdn_state_copy_funcs,
    gdn_temporal_copy_spec,
)

_CPU = torch.device("cpu")
_CHUNK = 16

# Lifecycle equality must be exact-to-rounding: carry across a block boundary and
# recompute-on-resume are the *same* float64 recurrence reassociated, so they
# agree far tighter than the T0.6 GDN chunk bound (rtol 1e-8 / atol 1e-9).
_LIFECYCLE_RTOL = 1e-10
_LIFECYCLE_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Test fixtures / helpers (all float64 for exact lifecycle equality)
# ---------------------------------------------------------------------------
def _params() -> Qwen4ExpGDNParams:
    return Qwen4ExpGDNParams(
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel_size=4,
        head_dim=16,
        partial_rotary_factor=0.5,
    )


def _float64_layout() -> Qwen4ExpGDNStateLayout:
    """A layout whose recurrent state is float64, for exact pool round-trips."""
    params = _params()
    base = Qwen4ExpGDNStateLayout.from_params(params, tp_size=1)
    return Qwen4ExpGDNStateLayout(
        conv_shape=base.conv_shape,
        recurrent_shape=base.recurrent_shape,
        conv_dtype=torch.float64,
        recurrent_dtype=torch.float64,
        tp_size=1,
    )


def _make_sequence(seq_len: int, *, seed: int):
    """Return (q, k, v, g, beta) for one request over ``seq_len`` tokens (f64)."""
    params = _params()
    gen = torch.Generator().manual_seed(seed)
    hk, hv = params.num_k_heads, params.num_v_heads
    kd, vd = params.head_k_dim, params.head_v_dim
    q = torch.randn(seq_len, hk, kd, generator=gen, dtype=torch.float64)
    k = torch.randn(seq_len, hk, kd, generator=gen, dtype=torch.float64)
    v = torch.randn(seq_len, hv, vd, generator=gen, dtype=torch.float64)
    a = torch.randn(seq_len, hv, generator=gen, dtype=torch.float64)
    b = torch.randn(seq_len, hv, generator=gen, dtype=torch.float64)
    A_log = torch.randn(hv, generator=gen, dtype=torch.float64)
    dt_bias = torch.randn(hv, generator=gen, dtype=torch.float64)
    g, beta = gdn_gating(A_log, a, b, dt_bias, compute_dtype=torch.float64)
    return q, k, v, g, beta


def _slice(seq, start, stop):
    return tuple(t[start:stop] for t in seq)


# ---------------------------------------------------------------------------
# Copy-func / layout (mirror the fork MambaAttentionBackendEnum.GDN_ATTN pair)
# ---------------------------------------------------------------------------
def test_state_copy_funcs_pair_matches_fork_shape():
    conv_fn, temporal_fn = gdn_state_copy_funcs()
    assert conv_fn is gdn_conv_copy_spec
    assert temporal_fn is gdn_temporal_copy_spec


def test_temporal_copy_spec_selects_accepted_block():
    state = torch.arange(5 * 6, dtype=torch.float32).reshape(5, 6)
    block_ids = [3, 1, 4]
    spec = gdn_temporal_copy_spec(state, block_ids, cur_block_idx=0, num_accepted_tokens=2)
    assert isinstance(spec, GDNStateCopySpec)
    # cur_block_idx + num_accepted - 1 = 1 -> block_ids[1] = 1.
    assert spec.start_addr == state[1].data_ptr()
    assert spec.num_elements == state[1].numel()


def test_conv_copy_spec_ds_rejects_offset():
    state = torch.zeros(4, 3, 2)
    with pytest.raises(ValueError, match="fused postprocess"):
        gdn_conv_copy_spec(state, [0, 1], cur_block_idx=0, num_accepted_tokens=2)


def test_layout_from_params_uses_policy_dtypes():
    layout = Qwen4ExpGDNStateLayout.from_params(_params(), tp_size=1)
    # recurrent [num_v_heads, head_v_dim, head_k_dim]; ssm state is fp32.
    assert layout.recurrent_shape == (4, 8, 8)
    assert layout.recurrent_dtype == torch.float32
    assert layout.conv_dtype == torch.float16


# ---------------------------------------------------------------------------
# No-alias slot pool: allocate / free / reuse / preemption
# ---------------------------------------------------------------------------
def test_pool_allocates_distinct_blocks_no_alias():
    pool = Qwen4ExpGDNStatePool(4, _float64_layout(), device=_CPU)
    block_a = pool.allocate(request_id=10)
    block_b = pool.allocate(request_id=11)
    assert block_a != block_b
    # Reverse map has no two requests on one block.
    active = pool.active_blocks()
    assert len(set(active.values())) == len(active) == 2


def test_pool_double_allocate_is_fail_closed():
    pool = Qwen4ExpGDNStatePool(2, _float64_layout(), device=_CPU)
    pool.allocate(1)
    with pytest.raises(ValueError, match="already resident"):
        pool.allocate(1)


def test_pool_exhaustion_is_fail_closed():
    pool = Qwen4ExpGDNStatePool(1, _float64_layout(), device=_CPU)
    pool.allocate(1)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.allocate(2)


def test_pool_reuse_zeroes_stale_state():
    pool = Qwen4ExpGDNStatePool(1, _float64_layout(), device=_CPU)
    pool.allocate(1)
    pool.recurrent_state(1).fill_(3.0)
    pool.free(1)
    # Same physical block handed to a new request must be zero (no carryover).
    block = pool.allocate(2)
    assert block == 0
    assert torch.count_nonzero(pool.recurrent_state(2)) == 0


def test_pool_preempt_drops_state_fail_closed_on_access():
    pool = Qwen4ExpGDNStatePool(2, _float64_layout(), device=_CPU)
    pool.allocate(1)
    pool.recurrent_state(1).fill_(2.0)
    pool.preempt(1)
    assert not pool.is_resident(1)
    with pytest.raises(KeyError, match="not resident"):
        pool.recurrent_state(1)


def test_pool_free_of_absent_request_raises():
    pool = Qwen4ExpGDNStatePool(2, _float64_layout(), device=_CPU)
    with pytest.raises(KeyError):
        pool.free(99)


# ---------------------------------------------------------------------------
# Acceptance 1: chunked-vs-unchunked produce identical outputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunked", [True, False])
def test_split_prefill_equals_single_shot(chunked):
    """Carrying state across block boundaries == a single-shot run."""
    seq = _make_sequence(40, seed=1)
    o_single, s_single = gdn_delta_rule(
        *seq, initial_state=None, chunked=chunked, chunk_size=_CHUNK, compute_dtype=torch.float64
    )
    o_split, s_split = gdn_prefill_in_chunks(
        *seq,
        split_points=[16, 28],
        initial_state=None,
        chunked=chunked,
        chunk_size=_CHUNK,
        compute_dtype=torch.float64,
    )
    torch.testing.assert_close(o_split, o_single, rtol=_LIFECYCLE_RTOL, atol=_LIFECYCLE_ATOL)
    torch.testing.assert_close(s_split, s_single, rtol=_LIFECYCLE_RTOL, atol=_LIFECYCLE_ATOL)


def test_chunked_and_unchunked_lifecycle_agree():
    """The chunk-parallel and token-recurrent engines agree across a split."""
    seq = _make_sequence(40, seed=2)
    o_chunk, s_chunk = gdn_prefill_in_chunks(
        *seq, split_points=[16, 28], chunked=True, chunk_size=_CHUNK, compute_dtype=torch.float64
    )
    o_rec, s_rec = gdn_prefill_in_chunks(
        *seq, split_points=[16, 28], chunked=False, chunk_size=_CHUNK, compute_dtype=torch.float64
    )
    torch.testing.assert_close(o_chunk, o_rec, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(s_chunk, s_rec, rtol=1e-8, atol=1e-9)


def test_prefill_in_chunks_rejects_bad_split():
    seq = _make_sequence(10, seed=3)
    with pytest.raises(ValueError, match="ascending"):
        gdn_prefill_in_chunks(*seq, split_points=[8, 4], compute_dtype=torch.float64)


# ---------------------------------------------------------------------------
# Acceptance 2: interleaved 2-request decode + forced preemption == unpreempted
# ---------------------------------------------------------------------------
def _decode_via_pool(pool, request_id, seq, start, stop):
    """Run tokens [start:stop) one at a time through the resident state block."""
    outs = []
    for t in range(start, stop):
        step = _slice(seq, t, t + 1)
        initial = pool.recurrent_state(request_id)
        out, new_state = gdn_delta_rule(*step, initial_state=initial, chunked=False, compute_dtype=torch.float64)
        pool.write_recurrent(request_id, new_state)
        outs.append(out)
    return torch.cat(outs, dim=0)


def test_interleaved_preemption_matches_unpreempted():
    len_a, len_b = 7, 9
    seq_a = _make_sequence(len_a, seed=11)
    seq_b = _make_sequence(len_b, seed=22)

    # --- reference: interleaved decode, no preemption -------------------
    pool = Qwen4ExpGDNStatePool(4, _float64_layout(), device=_CPU)
    block_a = pool.allocate("A")
    block_b = pool.allocate("B")
    assert block_a != block_b  # no aliasing between the two live requests
    out_a_ref = _decode_via_pool(pool, "A", seq_a, 0, len_a)
    out_b_ref = _decode_via_pool(pool, "B", seq_b, 0, len_b)

    # --- preempted run: B is evicted mid-stream, then resumed ----------
    pool2 = Qwen4ExpGDNStatePool(4, _float64_layout(), device=_CPU)
    pool2.allocate("A")
    pool2.allocate("B")
    preempt_at = 5
    # A runs to completion; B runs up to the preemption point, interleaved.
    out_a = _decode_via_pool(pool2, "A", seq_a, 0, len_a)
    out_b_head = _decode_via_pool(pool2, "B", seq_b, 0, preempt_at)

    # Force-preempt B: its state block is dropped (fail-closed).
    pool2.preempt("B")
    assert not pool2.is_resident("B")
    # A must be untouched by B's preemption.
    torch.testing.assert_close(pool2.recurrent_state("A"), pool.recurrent_state("A"))

    # Resume B: re-seed a fresh block and recompute state from the tokens
    # already consumed (the explicit fail-closed recovery path).
    resumed_block = pool2.allocate("B")
    # Reuse must not alias A's live block.
    assert resumed_block != pool2.block_of("A")
    _, recomputed = gdn_delta_rule(
        *_slice(seq_b, 0, preempt_at), initial_state=None, chunked=True, compute_dtype=torch.float64
    )
    pool2.write_recurrent("B", recomputed)
    out_b_tail = _decode_via_pool(pool2, "B", seq_b, preempt_at, len_b)
    out_b = torch.cat([out_b_head, out_b_tail], dim=0)

    torch.testing.assert_close(out_a, out_a_ref, rtol=_LIFECYCLE_RTOL, atol=_LIFECYCLE_ATOL)
    torch.testing.assert_close(out_b, out_b_ref, rtol=_LIFECYCLE_RTOL, atol=_LIFECYCLE_ATOL)


# ---------------------------------------------------------------------------
# Acceptance 3: aliasing -- two requests never point to the same block
# ---------------------------------------------------------------------------
def test_no_alias_across_churn():
    """Through a churn of allocate/free/preempt, live requests stay disjoint."""
    pool = Qwen4ExpGDNStatePool(3, _float64_layout(), device=_CPU)
    pool.allocate("r0")
    pool.allocate("r1")
    pool.free("r0")
    pool.allocate("r2")  # reuses r0's block
    pool.preempt("r1")
    pool.allocate("r3")  # reuses r1's block
    active = pool.active_blocks()
    assert set(active) == {"r2", "r3"}
    # Distinct requests -> distinct blocks (no aliasing) at every point.
    assert len(set(active.values())) == len(active)


# ---------------------------------------------------------------------------
# Model-state class: per-rank (4 TP ranks) lifecycle fan-out
# ---------------------------------------------------------------------------
class _StubBase:
    def prepare_inputs(self, input_batch, req_states):  # pragma: no cover
        return {}

    def prepare_dummy_inputs(self, num_reqs, num_tokens):  # pragma: no cover
        return {}


def _register(name, module):
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    sys.modules.setdefault(name, module)


def _install_host_stubs():
    if getattr(sys.modules.get("torch_npu"), "__gdn_lifecycle_stub__", False):
        return
    triton_runtime = MagicMock()
    triton_runtime.driver.active.utils.get_device_properties.return_value = {
        "num_aic": 8,
        "num_vectorcore": 8,
    }
    sys.modules.setdefault("triton.runtime", triton_runtime)

    torch_npu = types.ModuleType("torch_npu")
    torch_npu.__path__ = []  # type: ignore[attr-defined]
    torch_npu.__gdn_lifecycle_stub__ = True  # type: ignore[attr-defined]
    _register("torch_npu", torch_npu)

    build_info = types.ModuleType("vllm_ascend._build_info")
    build_info.__device_type__ = "_310P"  # type: ignore[attr-defined]
    _register("vllm_ascend._build_info", build_info)

    try:  # noqa: SIM105
        torch.utils.rename_privateuse1_backend("npu")
    except RuntimeError:
        pass
    torch.npu = MagicMock()  # type: ignore[attr-defined]
    torch.npu.is_available = MagicMock(return_value=False)
    torch.npu.Event = MagicMock

    rope_mod = types.ModuleType("vllm_ascend._310p.ops.rotary_embedding")
    rope_mod.prepare_mrope_cos_sin_slices_from_runner = MagicMock()
    _register("vllm_ascend._310p.ops.rotary_embedding", rope_mod)

    device_op_mod = types.ModuleType("vllm_ascend.device.device_op")
    device_op_mod.DeviceOperator = MagicMock()
    _register("vllm_ascend.device.device_op", device_op_mod)

    input_batch_mod = types.ModuleType("vllm_ascend.worker.v2.input_batch")
    input_batch_mod.AscendInputBatch = type("AscendInputBatch", (), {})
    _register("vllm_ascend.worker.v2.input_batch", input_batch_mod)

    default_mod = types.ModuleType("vllm_ascend.worker.v2.model_states.default")
    default_mod.AscendModelState = type("AscendModelState", (_StubBase,), {})
    _register("vllm_ascend.worker.v2.model_states.default", default_mod)

    hybrid_mod = types.ModuleType("vllm_ascend.worker.v2.model_states.mamba_hybrid")
    hybrid_mod.AscendMambaHybridModelState = type("AscendMambaHybridModelState", (_StubBase,), {})
    _register("vllm_ascend.worker.v2.model_states.mamba_hybrid", hybrid_mod)

    sampler_mod = types.ModuleType("vllm_ascend._310p.worker.v2.sampler")
    sampler_mod.Ascend310PSampler = type("Ascend310PSampler", (), {})
    _register("vllm_ascend._310p.worker.v2.sampler", sampler_mod)


def _cleanup_host_stubs():
    for name in list(sys.modules):
        if name == "torch_npu" or name.startswith("torch_npu."):
            del sys.modules[name]


def _make_model_state():
    _install_host_stubs()
    from vllm_ascend._310p.worker.v2.model_state import Ascend310PQwen4ExpModelState

    _cleanup_host_stubs()
    state = Ascend310PQwen4ExpModelState.__new__(Ascend310PQwen4ExpModelState)
    state.device = _CPU
    return state


def test_model_state_per_rank_replication_no_alias():
    state = _make_model_state()
    state._init_gdn_state_pools(_params(), num_blocks=4, tp_size=1, num_ranks=4)
    assert len(state._gdn_state_pools) == 4

    blocks_r0 = state.gdn_begin_request(1)
    blocks_r1 = state.gdn_begin_request(2)
    assert len(blocks_r0) == len(blocks_r1) == 4
    assert state.gdn_is_resident(1) and state.gdn_is_resident(2)

    # Within each rank the two requests hold distinct blocks (no aliasing).
    for rank in range(4):
        active = state.gdn_active_blocks(rank)
        assert active[1] != active[2]
        assert len(set(active.values())) == 2


def test_model_state_preempt_fans_across_ranks_fail_closed():
    state = _make_model_state()
    state._init_gdn_state_pools(_params(), num_blocks=2, tp_size=1, num_ranks=4)
    state.gdn_begin_request(7)
    state.gdn_preempt_request(7)
    assert not state.gdn_is_resident(7)
    for rank in range(4):
        with pytest.raises(KeyError):
            state.gdn_recurrent_state(7, rank=rank)


def test_model_state_gdn_step_advances_and_persists():
    state = _make_model_state()
    layout = _float64_layout()
    # Use a float64 pool so the step round-trips exactly against a direct call.
    state.gdn_state_layout = layout
    state.gdn_num_ranks = 1
    state._gdn_state_pools = [Qwen4ExpGDNStatePool(2, layout, device=_CPU)]
    state.gdn_begin_request(0)

    seq = _make_sequence(3, seed=5)
    # Two decode steps through the model-state helper.
    out0 = state.gdn_step(0, *_slice(seq, 0, 1), chunked=False, compute_dtype=torch.float64)
    out1 = state.gdn_step(0, *_slice(seq, 1, 2), chunked=False, compute_dtype=torch.float64)
    stepped = torch.cat([out0, out1], dim=0)

    ref, _ = gdn_delta_rule(*_slice(seq, 0, 2), initial_state=None, chunked=False, compute_dtype=torch.float64)
    torch.testing.assert_close(stepped, ref, rtol=_LIFECYCLE_RTOL, atol=_LIFECYCLE_ATOL)


def test_model_state_complete_frees_all_ranks():
    state = _make_model_state()
    state._init_gdn_state_pools(_params(), num_blocks=2, tp_size=1, num_ranks=4)
    state.gdn_begin_request(3)
    state.gdn_complete_request(3)
    assert not state.gdn_is_resident(3)
    # Blocks are back in every rank's free pool (re-allocatable).
    state.gdn_begin_request(4)
    assert state.gdn_is_resident(4)
