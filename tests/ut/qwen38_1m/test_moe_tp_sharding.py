# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-dimension TP slicing for the Qwen4Exp W8A8 fused bank (T3.1b).

The 512-expert bank must never be materialized per rank: on TP4 each rank keeps
one contiguous quarter of the *global* expert ids (the same linear placement the
fork's ``ep_weight_filter`` loader skip uses), the router gate and shared expert
stay replicated, and the routed output is the all-reduced sum of the per-rank
partials. CPU-only (NO NPU, NO Triton, NO 224 GB load).

Covered, CPU-side:

1. ``local_expert_range`` / ``expert_tensor_is_local`` -- ownership, including
   non-divisible counts mirroring the fork ``compute_local_expert_ids`` linear
   placement (skip-tested against the fork module when its checkout exists).
2. ``map_expert_tensor`` local-slot resolution + peer/non-geometry rejection;
   ``expected_expert_tensor_names`` + ``validate_expert_weight_map`` per-rank
   semantics (peer scale/offset tolerated and recorded, missing local rejected,
   non-geometry expert ids still rejected).
3. ``w8a8_grouped_experts`` sharded forward: sum of per-rank partials == the
   full-bank forward; skew; unsharded default unchanged.
4. ``_EagerSparseMoE`` with ``expert_sharding=(r, 4)``: bank sizes, router
   identity across ranks, partial -> all-reduced sum + shared expert ONCE, and
   the loud failure when no all-reduce is installed.
5. Model-level ``load_weights`` on TP4-built models: rank-local placement from a
   full streamed checkpoint AND from a loader-filtered stream (peer weights
   absent, peer scale/offset delivered); missing local expert -> Missing error.

Declared tolerance (PRD §8.1, fixed before the asserts): TP-parity uses the
frozen ``W8A8_GEMM_ATOL``/``W8A8_GEMM_RTOL`` of the T3.3 W8A8 harness -- the
sharded path re-adds the *identical* float32 expert contributions in a
different associative order (observed 1.5e-5 at this geometry). The block-level
comparison additionally allows the MoE block's final ``float16`` output
rounding (half an ulp at ~1e3 magnitudes ~ 0.25) because the TP1 reference is
compared through that cast.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_moe_tp_sharding.py
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from tests.ut.qwen38_1m.reference.tolerances import (
    W8A8_GEMM_ATOL,
    W8A8_GEMM_RTOL,
)
from tests.ut.qwen38_1m.test_model_load_and_moe import (
    _build,
    _expert_meta_index,
    _non_expert_payloads,
    _single_rank_tp,
    _synth_expert_payloads,
    _tiny_moe_config,
)
from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.moe import route_topk, w8a8_grouped_experts
from vllm_ascend.models.qwen4_exp.weight_mapping import (
    MissingTensorError,
    WeightMappingError,
    expected_expert_tensor_names,
    expert_tensor_is_local,
    local_expert_range,
    map_expert_tensor,
    validate_expert_weight_map,
)

_GEOMETRY = {
    "num_hidden_layers": 1,
    "num_experts": 8,
    "moe_intermediate_size": 16,
    "hidden_size": 32,
}
_TP = 4


def _expert_name(layer: int, expert: int, proj: str, kind: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"


# ---------------------------------------------------------------------------
# 1. ownership
# ---------------------------------------------------------------------------
def test_local_expert_range_single_rank_is_full():
    assert local_expert_range(8) == (0, 8)
    assert local_expert_range(8, 1, 0) == (0, 8)


def test_local_expert_range_divisible_partitions_are_contiguous_quarters():
    ranges = [local_expert_range(8, _TP, r) for r in range(_TP)]
    assert ranges == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_local_expert_range_non_divisible_mirrors_fork_linear_placement():
    # base=2, remainder=2 -> counts 3,3,2,2 (the fork compute_local_expert_ids
    # linear placement), not naive floor slices.
    ranges = [local_expert_range(10, 4, r) for r in range(4)]
    assert ranges == [(0, 3), (3, 6), (6, 8), (8, 10)]
    all_ids = set()
    for start, stop in ranges:
        all_ids.update(range(start, stop))
    assert all_ids == set(range(10))  # exact cover


@pytest.mark.parametrize("tp_rank,tp_size", [(-1, 4), (4, 4), (0, 0)])
def test_local_expert_range_invalid_raises(tp_rank, tp_size):
    with pytest.raises(WeightMappingError):
        local_expert_range(8, tp_size, tp_rank)


def test_fork_ep_weight_filter_agreement():
    """The fork loader filter and this mapper must agree on ownership."""
    import importlib.util
    from pathlib import Path

    fork_file = Path("/run/media/matteius/20TB-drive/vllm/vllm/model_executor/model_loader/ep_weight_filter.py")
    if not fork_file.exists():  # CI-safe: the fork checkout is not always present
        pytest.skip("vLLM fork checkout not available on this host")
    spec = importlib.util.spec_from_file_location("_fork_ep_weight_filter", fork_file)
    fork_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fork_mod)
    for num_experts, ep in [(8, 4), (10, 4), (512, 4)]:
        for rank in range(ep):
            fork_local = fork_mod.compute_local_expert_ids(num_experts, ep, rank, placement="linear")
            start, stop = local_expert_range(num_experts, ep, rank)
            assert fork_local == set(range(start, stop)), (num_experts, rank)


def test_expert_tensor_is_local_gates_and_tolerates():
    # tp=1 (host default): everything is local, nothing to gate.
    assert expert_tensor_is_local(_expert_name(0, 7, "gate_proj", "weight"), _GEOMETRY)
    # non-expert names are never gated.
    assert expert_tensor_is_local("model.language_model.layers.0.input_layernorm.weight", _GEOMETRY, _TP, 2)
    # ownership by contiguous quarter.
    for rank in range(_TP):
        for expert in range(8):
            owned = 2 * rank <= expert < 2 * (rank + 1)
            name = _expert_name(0, expert, "up_proj", "weight_scale")
            assert expert_tensor_is_local(name, _GEOMETRY, _TP, rank) is owned


def test_expert_tensor_is_local_non_geometry_id_stays_reachable():
    # An id outside the frozen geometry is NOT a peer tensor: it must stay
    # "local" so map_expert_tensor raises the actionable rejection.
    assert expert_tensor_is_local(_expert_name(0, 99, "gate_proj", "weight"), _GEOMETRY, _TP, 0)


# ---------------------------------------------------------------------------
# 2. mapper + validation per rank
# ---------------------------------------------------------------------------
def test_map_expert_tensor_resolves_local_slot():
    for rank in range(_TP):
        start, _ = local_expert_range(8, _TP, rank)
        for expert in range(start, start + 2):
            mapping = map_expert_tensor(_expert_name(0, expert, "gate_proj", "weight"), _GEOMETRY, _TP, rank)
            assert mapping.expert == expert  # source (global) id retained
            assert mapping.expert_index == expert - start  # local bank slot
            assert (mapping.row_start, mapping.row_stop) == (0, 16)
            up = map_expert_tensor(_expert_name(0, expert, "up_proj", "weight"), _GEOMETRY, _TP, rank)
            assert (up.row_start, up.row_stop) == (16, 32)
            assert mapping.expected_dtype == torch.int8
    # tp=1 regression: local == global.
    m0 = map_expert_tensor(_expert_name(0, 5, "down_proj", "weight"), _GEOMETRY)
    assert m0.expert_index == 5


def test_map_expert_tensor_rejects_peer():
    with pytest.raises(WeightMappingError, match="not owned by tp_rank 1"):
        map_expert_tensor(_expert_name(0, 5, "gate_proj", "weight"), _GEOMETRY, _TP, 1)  # owned by rank 2


def test_expected_names_partition_exactly_across_ranks():
    per_rank = [expected_expert_tensor_names(_GEOMETRY, _TP, r) for r in range(_TP)]
    assert all(len(s) == 18 for s in per_rank)  # 1 layer x 2 experts x 3 proj x (w,s,o)
    total = len(per_rank[0]) + len(per_rank[1]) + len(per_rank[2]) + len(per_rank[3])
    assert total == len(set().union(*per_rank))  # pairwise disjoint
    assert set().union(*per_rank) == expected_expert_tensor_names(_GEOMETRY)


def _full_meta():
    return _expert_meta_index(_GEOMETRY)


def test_validate_per_rank_ok_with_local_only():
    meta = _full_meta()
    for rank in range(_TP):
        start, stop = local_expert_range(8, _TP, rank)
        provided = {n: m for n, m in meta.items() if any(f"experts.{e}." in n for e in range(start, stop))}
        result = validate_expert_weight_map(provided, _GEOMETRY, tp_size=_TP, tp_rank=rank)
        assert result.peer_expert_tensors == []
        assert {e.expert_index for e in result.entries} == {0, 1}


def test_validate_tolerates_peer_scale_offset_delivered_by_loader():
    # The fork loader filter skips peer ``.weight`` payloads but still delivers
    # peer scale/offset tensors -- they must be recorded, never an error.
    meta = _full_meta()
    start, stop = local_expert_range(8, _TP, 1)
    keep = {n: m for n, m in meta.items() if any(f"experts.{e}." in n for e in range(start, stop))}
    peer_extra = {n: m for n, m in meta.items() if "weight_scale" in n and n not in keep}
    assert peer_extra  # some peer scales exist
    result = validate_expert_weight_map(keep | peer_extra, _GEOMETRY, tp_size=_TP, tp_rank=1)
    assert sorted(result.peer_expert_tensors) == sorted(peer_extra)


def test_validate_missing_local_and_non_geometry_rejected():
    meta = _full_meta()
    start, stop = local_expert_range(8, _TP, 2)
    provided = {n: m for n, m in meta.items() if any(f"experts.{e}." in n for e in range(start, stop))}
    drop_key = f"model.language_model.layers.0.mlp.experts.{start}.gate_proj.weight_scale"
    dropped = provided.pop(drop_key)
    with pytest.raises(MissingTensorError):
        validate_expert_weight_map(provided, _GEOMETRY, tp_size=_TP, tp_rank=2)
    # Expert id outside the frozen geometry: an error for every rank (never a
    # silent peer-skip hole).
    bogus = {**provided, _expert_name(0, 40, "gate_proj", "weight"): dropped}
    with pytest.raises(WeightMappingError):
        validate_expert_weight_map(bogus, _GEOMETRY, tp_size=_TP, tp_rank=2)


# ---------------------------------------------------------------------------
# 3. grouped forward under expert slicing
# ---------------------------------------------------------------------------
def _sharded_banks(seed: int = 7):
    """Full-bank tensors + the TP per-rank payload slices for one layer."""
    gen = torch.Generator().manual_seed(seed)
    hidden, moe, experts = _GEOMETRY["hidden_size"], _GEOMETRY["moe_intermediate_size"], 8
    w13 = torch.randint(-127, 128, (experts, 2 * moe, hidden), generator=gen, dtype=torch.int8)
    w13_s = torch.rand(experts, 2 * moe, 1, generator=gen) * 0.02 + 0.01
    w13_o = torch.zeros(experts, 2 * moe, 1, dtype=torch.float32)
    w2 = torch.randint(-127, 128, (experts, hidden, moe), generator=gen, dtype=torch.int8)
    w2_s = torch.rand(experts, hidden, 1, generator=gen) * 0.02 + 0.01
    w2_o = torch.zeros(experts, hidden, 1, dtype=torch.float32)
    banks = []
    per = experts // _TP
    for rank in range(_TP):
        lo = rank * per
        banks.append(
            dict(
                offset=lo,
                w13=w13[lo : lo + per].clone(),
                s13=w13_s[lo : lo + per].clone(),
                o13=w13_o[lo : lo + per].clone(),
                w2=w2[lo : lo + per].clone(),
                s2=w2_s[lo : lo + per].clone(),
                o2=w2_o[lo : lo + per].clone(),
            )
        )
    return (w13, w13_s, w13_o, w2, w2_s, w2_o), banks


def test_sharded_partial_sum_equals_full_forward():
    (w13, s13, o13, w2, s2, o2), banks = _sharded_banks()
    torch.manual_seed(0)
    x = torch.randn(6, 32)
    logits = torch.randn(6, 8)
    tw, tid = route_topk(logits, 3, renormalize=True)

    full = w8a8_grouped_experts(x, tw, tid, w13, s13, o13, w2, s2, o2)
    # unsharded default path must be bit-identical to the pre-slicing forward
    assert torch.equal(full, w8a8_grouped_experts(x, tw, tid, w13, s13, o13, w2, s2, o2, expert_offset=0))

    partials = [
        w8a8_grouped_experts(
            x,
            tw,
            tid,
            b["w13"],
            b["s13"],
            b["o13"],
            b["w2"],
            b["s2"],
            b["o2"],
            expert_offset=b["offset"],
            num_global_experts=8,
        )
        for b in banks
    ]
    summed = partials[0]
    for p in partials[1:]:
        summed = summed + p
    err = (summed - full).abs().max().item()
    # Declared tolerance (module docstring): identical float32 contributions in
    # TP-reassociated order; the frozen W8A8 tolerance dominates it.
    assert math.isfinite(err) and err <= W8A8_GEMM_ATOL + W8A8_GEMM_RTOL * full.abs().max().item()


def test_sharded_skew_concentrates_on_one_rank():
    (w13, s13, o13, w2, s2, o2), banks = _sharded_banks(seed=11)
    x = torch.randn(4, 32)
    tw = torch.full((4, 3), 1 / 3, dtype=torch.float32)
    tid = torch.full((4, 3), 7, dtype=torch.int64)  # every token -> expert 7 (rank 3)

    partials = [
        w8a8_grouped_experts(
            x,
            tw,
            tid,
            b["w13"],
            b["s13"],
            b["o13"],
            b["w2"],
            b["s2"],
            b["o2"],
            expert_offset=b["offset"],
            num_global_experts=8,
        )
        for b in banks
    ]
    for r, p in enumerate(partials):
        if r == _TP - 1:
            assert p.abs().sum() > 0
        else:
            assert torch.equal(p, torch.zeros_like(p))
    full = w8a8_grouped_experts(x, tw, tid, w13, s13, o13, w2, s2, o2)
    # single-rank contribution == full (within the same declared tolerance)
    assert (partials[-1] - full).abs().max().item() <= W8A8_GEMM_ATOL + W8A8_GEMM_RTOL * full.abs().max().item()


# ---------------------------------------------------------------------------
# 4. block-level TP forward
# ---------------------------------------------------------------------------
def _block(rank: int, *, shared_inter: int = 16):
    from vllm_ascend.models.qwen4_exp.model import _EagerSparseMoE

    cfg = _tiny_moe_config(num_layers=1, num_experts=8, top_k=3, shared_inter=shared_inter)
    return _EagerSparseMoE(config=cfg, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY, expert_sharding=(rank, _TP))


def test_block_allocates_local_banks_and_replicates_router():
    blocks = [_block(r) for r in range(_TP)]
    for r, b in enumerate(blocks):
        assert len(b.w13_weight) == 2  # local experts only
        assert all(weight.shape == (32, 32) for weight in b.w13_weight)
        assert b.w13_weight_scale.shape[0] == 2
        assert b.gate.shape == (8, 32)  # replicated router
        assert b.expert_offset == 2 * r
        assert b.num_global_experts == 8


def test_block_balances_non_divisible_experts():
    from vllm_ascend.models.qwen4_exp.model import _EagerSparseMoE

    cfg = _tiny_moe_config(num_layers=1, num_experts=6, top_k=3, shared_inter=0)
    blocks = [
        _EagerSparseMoE(config=cfg, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY, expert_sharding=(rank, 4))
        for rank in range(4)
    ]
    assert [(block.expert_offset, block.num_local_experts) for block in blocks] == [(0, 2), (2, 2), (4, 1), (5, 1)]
    assert [len(block.w13_weight) for block in blocks] == [2, 2, 1, 1]
    with pytest.raises(ValueError, match="each rank must own at least one expert"):
        _EagerSparseMoE(config=cfg, dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY, expert_sharding=(0, 8))


def test_block_tp_reduce_partial_plus_shared_once():
    (w13, s13, o13, w2, s2, o2), banks = _sharded_banks(seed=23)
    blocks = [_block(r) for r in range(_TP)]
    # ONE replicated router (the TP premise): identical gate across all ranks.
    gen = torch.Generator().manual_seed(99)
    router = torch.randn(8, 32, generator=gen).to(ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype) * 0.05
    shared_up = torch.randn(32, 32, generator=gen) * 0.05
    shared_dn = torch.randn(32, 16, generator=gen) * 0.05
    for b, src in zip(blocks, banks):
        local_inter = b.local_shared_inter
        shared_start = b.expert_tp_rank * local_inter
        local_shared_up = torch.cat(
            (
                shared_up[shared_start : shared_start + local_inter],
                shared_up[16 + shared_start : 16 + shared_start + local_inter],
            )
        )
        local_shared_dn = shared_dn[:, shared_start : shared_start + local_inter]
        with torch.no_grad():
            b.gate.copy_(router)
            b.shared_gate_up.copy_(local_shared_up.to(b.params_dtype))
            b.shared_down.copy_(local_shared_dn.to(b.params_dtype))
            for target, source in zip(b.w13_weight, src["w13"]):
                target.copy_(source.t())
            b.w13_weight_scale.copy_(src["s13"])
            b.w13_weight_offset.copy_(src["o13"])
            for target, source in zip(b.w2_weight, src["w2"]):
                target.copy_(source.t())
            b.w2_weight_scale.copy_(src["s2"])
            b.w2_weight_offset.copy_(src["o2"])
    # TP1 reference block: default unsharded bank, same router/shared weights.
    from vllm_ascend.models.qwen4_exp.model import _EagerSparseMoE

    ref = _EagerSparseMoE(
        config=_tiny_moe_config(num_layers=1, num_experts=8, top_k=3, shared_inter=16),
        dtype_policy=ASCEND_QWEN4EXP_DTYPE_POLICY,
    )
    with torch.no_grad():
        ref.gate.copy_(router)
        ref.shared_gate_up.copy_(shared_up.to(ref.params_dtype))
        ref.shared_down.copy_(shared_dn.to(ref.params_dtype))
        for target, source in zip(ref.w13_weight, w13):
            target.copy_(source.t())
        ref.w13_weight_scale.copy_(s13)
        ref.w13_weight_offset.copy_(o13)
        for target, source in zip(ref.w2_weight, w2):
            target.copy_(source.t())
        ref.w2_weight_scale.copy_(s2)
        ref.w2_weight_offset.copy_(o2)

    torch.manual_seed(5)
    x = torch.randn(5, 32).to(ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype)

    captured: list[torch.Tensor] = []
    outs = []
    for r, b in enumerate(blocks):

        def _all_reduce(t, r=r):
            captured.append(t)
            return t

        b._tp_reduce = _all_reduce
        outs.append(b(x).to(torch.float32))
        assert len(captured) == r + 1  # reduce installed and called once

    summed = captured[0]
    for t in captured[1:]:
        summed = summed + t
    ref_out = ref(x).to(torch.float32)
    # Declared tolerance (module docstring): identical float32 contributions in
    # TP-reassociated order, plus the reference block's final fp16 output
    # rounding (half an ulp at this magnitude ~ 0.25, observed 0.22).
    fp16_output_round = 0.51 * torch.finfo(torch.float16).eps * 2 * ref_out.abs().max().item()
    budget = W8A8_GEMM_ATOL + W8A8_GEMM_RTOL * ref_out.abs().max().item() + fp16_output_round
    # Each captured tensor already contains that rank's routed and shared
    # projection partial. Their emulated all-reduce equals the TP1 block.
    err = (summed - ref_out).abs().max().item()
    assert err <= budget, (err, budget)
    # The fake reduction returns its input, so each local output is exactly its
    # combined partial rounded to the block's main dtype.
    expected_out0 = captured[0].to(ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype)
    assert torch.equal(outs[0].to(torch.float32), expected_out0.to(torch.float32))


def test_block_missing_reduce_fails_loudly():
    b = _block(1)
    b._tp_reduce = None
    with pytest.raises(RuntimeError, match="all-reduce"):
        b(torch.zeros(1, 32, dtype=ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype))


# ---------------------------------------------------------------------------
# 5. model-level TP4 load_weights
# ---------------------------------------------------------------------------
class _TpEnv:
    """Build TP-size-N Qwen4Exp models on one CPU process (shimmed comm)."""

    def __init__(self, tp_size: int):
        self.tp_size = tp_size

    def _vllm_config(self, cfg):
        from tests.ut.qwen38_1m.test_model_load_and_moe import _vllm_config

        vc = _vllm_config(cfg)
        vc.parallel_config = type(vc.parallel_config)(tensor_parallel_size=self.tp_size)
        return vc

    def build(self, rank: int, cfg):
        from vllm.config import set_current_vllm_config

        from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM

        vmod = "vllm.model_executor.layers.vocab_parallel_embedding"
        vllm_config = self._vllm_config(cfg)
        with (
            _single_rank_tp(),  # keep embed/lm_head fully replicated on host
            patch(f"{vmod}.get_tensor_model_parallel_rank", return_value=0),
            patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=rank),
            set_current_vllm_config(vllm_config),
        ):
            model = AscendQwen4ExpForCausalLM(vllm_config=vllm_config)
        assert model.model.expert_sharding == (rank, self.tp_size)
        return model


def _load(model, geometry):
    stream = _synth_expert_payloads(geometry, seed=42) + _non_expert_payloads(model)
    loaded = model.load_weights(iter(stream))
    return loaded


def test_tp_model_places_only_its_experts_and_validates():
    """Each TP4 rank places exactly its local expert payloads (full unfiltered
    stream delivered to every rank) and validation passes per-rank."""
    geometry = dict(_GEOMETRY)
    cfg = _tiny_moe_config(
        num_layers=1,
        num_experts=8,
        top_k=3,
        hidden=geometry["hidden_size"],
        moe_inter=geometry["moe_intermediate_size"],
        shared_inter=0,
    )
    env = _TpEnv(_TP)
    truth = {name: t for name, t in _synth_expert_payloads(geometry, seed=42)}
    for rank in range(_TP):
        model = env.build(rank, cfg)
        loaded = _load(model, geometry)
        layer0 = model.model.layers[0].mlp
        assert len(layer0.w13_weight) == 2
        for local, expert in enumerate(range(2 * rank, 2 * rank + 2)):
            gate = truth[_expert_name(0, expert, "gate_proj", "weight")]
            up = truth[_expert_name(0, expert, "up_proj", "weight")]
            dn = truth[_expert_name(0, expert, "down_proj", "weight")]
            assert torch.equal(layer0.w13_weight[local][:, :16], gate.view(16, 32).t()), (rank, expert)
            assert torch.equal(layer0.w13_weight[local][:, 16:], up.view(16, 32).t()), (rank, expert)
            assert torch.equal(
                layer0.w2_weight[local],
                dn.view(32, 16).t(),
            ), (rank, expert)
            assert "model.layers.0.mlp.w13_weight" in loaded


def test_tp_model_survives_loader_filter_and_missing_local():
    geometry = dict(_GEOMETRY)
    cfg = _tiny_moe_config(
        num_layers=1,
        num_experts=8,
        top_k=3,
        hidden=geometry["hidden_size"],
        moe_inter=geometry["moe_intermediate_size"],
        shared_inter=0,
    )
    env = _TpEnv(_TP)
    all_payloads = _synth_expert_payloads(geometry, seed=42)

    # Loader-filtered shape for rank 1: peer .weight skipped, scale/offset kept.
    def _keep(name: str) -> bool:
        expert = name.split(".experts.")[1].split(".")[0]
        owned = 2 <= int(expert) < 4
        return owned or not name.endswith("weight")

    filtered = [(n, t) for n, t in all_payloads if _keep(n)]
    rank1 = env.build(1, cfg)
    rank1.load_weights(iter(filtered + _non_expert_payloads(rank1)))

    # Full stream to the same rank must land IDENTICAL bank contents.
    rank1_full = env.build(1, cfg)
    rank1_full.load_weights(iter(all_payloads + _non_expert_payloads(rank1_full)))
    m1 = rank1.model.layers[0].mlp
    m2 = rank1_full.model.layers[0].mlp
    for name in ("w13_weight", "w2_weight"):
        first = getattr(m1, name)
        second = getattr(m2, name)
        assert len(first) == len(second)
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(first, second)), name
    fused_params = (
        "w13_weight_scale",
        "w13_weight_offset",
        "w2_weight_scale",
        "w2_weight_offset",
    )
    for p in fused_params:
        assert torch.equal(getattr(m1, p), getattr(m2, p)), p

    # Dropping one of rank 1's OWNED expert tensors is a hard failure.
    owned_gate = _expert_name(0, 2, "gate_proj", "weight")
    partial = [(n, t) for n, t in all_payloads if n != owned_gate]
    rank1_missing = env.build(1, cfg)
    with pytest.raises(MissingTensorError):
        rank1_missing.load_weights(iter(list(partial) + _non_expert_payloads(rank1_missing)))


def test_tp1_full_stream_regression():
    """tp_size=1 still loads every expert (host bring-up path untouched)."""
    geometry = dict(_GEOMETRY)
    cfg = _tiny_moe_config(num_layers=1, num_experts=8, top_k=3, shared_inter=0)
    model = _build(cfg)
    assert model.model.expert_sharding == (0, 1)
    loaded = _load(model, geometry)
    assert len(model.model.layers[0].mlp.w13_weight) == 8
    assert "model.layers.0.mlp.w13_weight" in loaded
