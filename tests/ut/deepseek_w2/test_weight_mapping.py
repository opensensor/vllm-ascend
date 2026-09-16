# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepSeek V4.1 W2 weight mapping + streamed placement (E3.4).

Runs host-side only. The placement/classification/rejection tests are fully
hermetic (synthetic metadata streams). One evidence test drives the simulator
from the *real* W2 artifact headers when the model volume is mounted; it skips
otherwise. Execute with ``--noconftest`` because the shared
``tests/ut/conftest.py`` is broken:

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_weight_mapping.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_ascend.models.deepseek_v41.weight_mapping import (
    DEFAULT_TP_SIZE,
    PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES,
    DeepSeekV41W2PlacementSimulator,
    DtypeMismatchError,
    DuplicateTensorError,
    ExtraTensorError,
    MissingTensorError,
    ShapeMismatchError,
    TensorMeta,
    WeightClass,
    classify_tensor,
    expected_expert_blocks,
    expected_expert_dtype,
    expected_expert_shape,
    iter_artifact_tensor_metas,
    map_expert_tensor,
    validate_expert_tensor,
    validate_weight_map,
)
from vllm_ascend.observability.deepseek_w2_mem_accounting import (
    DeepSeekW2MemComponent,
    MemComponent,
)

# --- Real artifact locations (used only by the mounted-volume evidence test) --

_ARTIFACT_DIR = Path("/run/media/matteius/20TB-drive/models/DeepSeek-V4.1-W2-310p")
_MANIFEST_PATH = Path("artifacts/deepseek-v41-w2/manifest.json")

# --- Tiny synthetic geometry (multiples of the [32, 32] block) ----------------

SYNTH_GEOMETRY = {
    "hidden_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "n_routed_experts": 4,
    "num_nextn_predict_layers": 1,
    "dspark_n_routed_experts": 2,
}


def _expert_metas(geometry: dict) -> list[TensorMeta]:
    """Every routed-expert tensor for ``geometry`` with correct shapes/dtypes."""
    metas: list[TensorMeta] = []
    for block, num_experts in expected_expert_blocks(geometry).items():
        for expert in range(num_experts):
            for slot in ("w1", "w2", "w3"):
                for kind in ("codes", "scale"):
                    name = f"{block}.ffn.experts.{expert}.{slot}_{kind}"
                    metas.append(
                        TensorMeta(
                            name=name,
                            dtype=expected_expert_dtype(kind),
                            shape=expected_expert_shape(slot, kind, geometry),
                        )
                    )
    return metas


def _synthetic_stream(geometry: dict) -> list[TensorMeta]:
    """Experts + an Engram host table + FP16 weights + an excluded vision tensor."""
    metas = _expert_metas(geometry)
    # Engram ~W4 host table (one big embedding-code partition + a wkv pair).
    metas.append(TensorMeta("layers.1.engram.embed_codes.p0", "U8", (4096, 128)))
    metas.append(TensorMeta("layers.1.engram.embed_scale.p0", "F32", (128, 8)))
    metas.append(TensorMeta("layers.1.engram.wkv_codes", "U8", (256, 96)))
    # FP16 device weights: MLA, shared expert, embed, lm_head, a norm, engram q.
    metas.append(TensorMeta("layers.0.attn.wq_a.weight", "F16", (128, 64)))
    metas.append(TensorMeta("layers.0.ffn.shared_experts.w1.weight", "F16", (32, 64)))
    metas.append(TensorMeta("embed.weight", "F16", (256, 64)))
    metas.append(TensorMeta("head.weight", "F16", (256, 64)))
    metas.append(TensorMeta("norm.weight", "F16", (64,)))
    metas.append(TensorMeta("layers.1.engram.q_weight", "F16", (4, 64)))
    # Excluded vision tower (text-only deployment).
    metas.append(TensorMeta("vision.blocks.0.attn.wo.weight", "F16", (64, 64)))
    metas.append(TensorMeta("vision.aligner.0.weight", "F16", (64, 64)))
    return metas


# =============================================================================
# Classification
# =============================================================================


def test_classify_routes_every_lane():
    assert classify_tensor("layers.0.ffn.experts.5.w1_codes") is WeightClass.W2_EXPERT
    assert classify_tensor("mtp.1.ffn.experts.7.w2_scale") is WeightClass.W2_EXPERT
    assert classify_tensor("layers.1.engram.embed_codes.p3") is WeightClass.ENGRAM_HOST
    assert classify_tensor("layers.14.engram.wkv_scale") is WeightClass.ENGRAM_HOST
    # Engram q/k projections are tiny FP16, NOT the ~W4 host table.
    assert classify_tensor("layers.1.engram.q_weight") is WeightClass.FP16
    assert classify_tensor("layers.1.engram.k_weight") is WeightClass.FP16
    assert classify_tensor("layers.0.attn.wq_a.weight") is WeightClass.FP16
    assert classify_tensor("layers.14.attn.indexer.wk.weight") is WeightClass.FP16
    assert classify_tensor("layers.0.ffn.shared_experts.w1.weight") is WeightClass.FP16
    assert classify_tensor("head.weight") is WeightClass.FP16
    assert classify_tensor("norm.weight") is WeightClass.FP16
    assert classify_tensor("vision.blocks.0.attn.wo.weight") is WeightClass.EXCLUDE
    assert classify_tensor("model.aligner.proj.weight") is WeightClass.EXCLUDE


# =============================================================================
# Expert fusion mapping (w1/w3 -> w13, w2 -> down)
# =============================================================================


def test_map_expert_fuses_gate_up_into_w13():
    g = SYNTH_GEOMETRY
    inter = g["moe_intermediate_size"]
    w1 = map_expert_tensor("layers.0.ffn.experts.3.w1_codes", g)
    w3 = map_expert_tensor("layers.0.ffn.experts.3.w3_codes", g)
    assert w1.target_param == "w13_codes" and w1.row_offset == 0 and w1.fuses_into_w13
    assert w3.target_param == "w13_codes" and w3.row_offset == inter and w3.fuses_into_w13
    # Scales fuse along the block-row axis (inter // 32).
    w1s = map_expert_tensor("layers.0.ffn.experts.3.w1_scale", g)
    w3s = map_expert_tensor("layers.0.ffn.experts.3.w3_scale", g)
    assert w1s.target_param == "w13_scale" and w1s.row_offset == 0
    assert w3s.target_param == "w13_scale" and w3s.row_offset == inter // 32


def test_map_expert_down_projection_not_fused():
    g = SYNTH_GEOMETRY
    w2c = map_expert_tensor("mtp.0.ffn.experts.1.w2_codes", g)
    w2s = map_expert_tensor("mtp.0.ffn.experts.1.w2_scale", g)
    assert w2c.target_param == "w2_codes" and w2c.row_offset == 0 and not w2c.fuses_into_w13
    assert w2s.target_param == "w2_scale" and w2s.row_offset == 0
    assert w2c.block == "mtp.0" and w2c.expert_id == 1


def test_map_expert_rejects_non_expert():
    with pytest.raises(ValueError):
        map_expert_tensor("layers.0.attn.wq_a.weight", SYNTH_GEOMETRY)


def test_expected_expert_shape_matches_pack_contract():
    g = SYNTH_GEOMETRY
    # codes packed along input axis; scales one per [32, 32] block.
    assert expected_expert_shape("w1", "codes", g) == (32, 16)
    assert expected_expert_shape("w2", "codes", g) == (64, 8)
    assert expected_expert_shape("w1", "scale", g) == (1, 2)
    assert expected_expert_shape("w2", "scale", g) == (2, 1)


# =============================================================================
# Rejection classes: missing / extra / duplicate / wrong-shape / wrong-dtype
# =============================================================================


def test_validate_expert_tensor_accepts_valid():
    meta = TensorMeta("layers.0.ffn.experts.0.w1_codes", "U8", (32, 16))
    mapping = validate_expert_tensor(meta, SYNTH_GEOMETRY)
    assert mapping.target_param == "w13_codes"


def test_reject_wrong_shape():
    bad = TensorMeta("layers.0.ffn.experts.0.w1_codes", "U8", (32, 15))
    with pytest.raises(ShapeMismatchError):
        validate_expert_tensor(bad, SYNTH_GEOMETRY)


def test_reject_wrong_dtype():
    bad = TensorMeta("layers.0.ffn.experts.0.w1_codes", "F16", (32, 16))
    with pytest.raises(DtypeMismatchError):
        validate_expert_tensor(bad, SYNTH_GEOMETRY)
    bad_scale = TensorMeta("layers.0.ffn.experts.0.w1_scale", "F16", (1, 2))
    with pytest.raises(DtypeMismatchError):
        validate_expert_tensor(bad_scale, SYNTH_GEOMETRY)


def test_reject_missing_expert_tensor():
    names = [m.name for m in _expert_metas(SYNTH_GEOMETRY)]
    names.remove("layers.0.ffn.experts.0.w2_codes")
    with pytest.raises(MissingTensorError):
        validate_weight_map(names, SYNTH_GEOMETRY)


def test_reject_extra_expert_tensor():
    names = [m.name for m in _expert_metas(SYNTH_GEOMETRY)]
    # Expert id beyond n_routed_experts for a dense layer.
    names.append("layers.0.ffn.experts.99.w1_codes")
    with pytest.raises(ExtraTensorError):
        validate_weight_map(names, SYNTH_GEOMETRY)


def test_reject_duplicate_tensor():
    names = [m.name for m in _expert_metas(SYNTH_GEOMETRY)]
    names.append("layers.0.ffn.experts.0.w1_codes")  # duplicate
    with pytest.raises(DuplicateTensorError):
        validate_weight_map(names, SYNTH_GEOMETRY)


def test_validate_weight_map_accepts_full_schema():
    names = [m.name for m in _synthetic_stream(SYNTH_GEOMETRY)]
    summary = validate_weight_map(names, SYNTH_GEOMETRY)
    # 2 dense layers x 4 experts + 1 mtp x 2 experts = 10 experts x 6 tensors.
    assert summary["expert_blocks"] == 3
    assert summary["expert_tensors"] == 10 * 6


# =============================================================================
# Placement simulation (TP4 / EP4), accounting, no double-instantiation
# =============================================================================


def test_tp_placement_balanced_and_no_double_instantiation():
    metas = _synthetic_stream(SYNTH_GEOMETRY)
    sim = DeepSeekV41W2PlacementSimulator(SYNTH_GEOMETRY, parallel_mode="tp")
    acc = sim.simulate(metas)
    assert acc.world_size == DEFAULT_TP_SIZE
    # Even byte split -> perfectly balanced, imbalance 0.
    device_totals = [acc.rank_report(r).device_bytes() for r in range(4)]
    assert len(set(device_totals)) == 1
    assert acc.imbalance() == 0.0
    # Every sharded byte placed exactly once (ledger self-check inside simulate).
    assert sim.ledger.sharded_placed_bytes == sim.ledger.sharded_total_bytes
    # Aggregate device bytes == sum of per-rank device bytes (nothing lost/dup'd).
    assert sum(device_totals) == sim.ledger.device_logical_bytes


def test_ep_placement_keeps_whole_experts_and_bounds_bank():
    metas = _synthetic_stream(SYNTH_GEOMETRY)
    sim = DeepSeekV41W2PlacementSimulator(SYNTH_GEOMETRY, parallel_mode="ep")
    acc = sim.simulate(metas)
    # No rank holds the full expert bank: each rank's W2 bytes are a strict
    # fraction of the aggregate routed-expert bytes.
    w2_per_rank = [acc.rank_report(r).components.get(DeepSeekW2MemComponent.W2_EXPERT, 0) for r in range(4)]
    w2_total = sum(w2_per_rank)
    assert w2_total > 0
    assert all(0 < b < w2_total for b in w2_per_rank)
    # 10 experts over 4 ranks is balanced within the 5% guard.
    acc.validate_balance()


def test_engram_host_counted_once_not_times_world_size():
    metas = _synthetic_stream(SYNTH_GEOMETRY)
    sim = DeepSeekV41W2PlacementSimulator(SYNTH_GEOMETRY, parallel_mode="tp")
    acc = sim.simulate(metas)
    engram_bytes = 4096 * 128 * 1 + 128 * 8 * 4 + 256 * 96 * 1
    # Recorded identically on every rank, but deduped to a single shared copy.
    assert acc.host_table_bytes() == engram_bytes
    # Host bytes are excluded from every rank's device total.
    for r in range(4):
        rep = acc.rank_report(r)
        assert rep.components[DeepSeekW2MemComponent.ENGRAM_HOST] == engram_bytes
        assert DeepSeekW2MemComponent.ENGRAM_HOST not in _device_components(rep)


def _device_components(report):
    from vllm_ascend.observability.deepseek_w2_mem_accounting import HOST_COMPONENTS

    return {c for c in report.components if c not in HOST_COMPONENTS}


def test_excluded_vision_not_placed():
    metas = _synthetic_stream(SYNTH_GEOMETRY)
    sim = DeepSeekV41W2PlacementSimulator(SYNTH_GEOMETRY, parallel_mode="tp")
    sim.simulate(metas)
    assert sim.ledger.excluded_count == 2
    assert sim.ledger.excluded_bytes == 2 * (64 * 64 * 2)


def test_bounded_working_set_on_large_synthetic_bank():
    # 200 experts x 6 tensors streamed, but the working set stays O(1).
    big = dict(SYNTH_GEOMETRY, n_routed_experts=200, num_nextn_predict_layers=0, num_hidden_layers=1)
    sim = DeepSeekV41W2PlacementSimulator(big, parallel_mode="tp")
    sim.simulate(_expert_metas(big))
    assert sim.ledger.peak_working_set_objects == 1


def test_embed_gets_embedding_component():
    metas = _synthetic_stream(SYNTH_GEOMETRY)
    sim = DeepSeekV41W2PlacementSimulator(SYNTH_GEOMETRY, parallel_mode="tp")
    acc = sim.simulate(metas)
    total_embed = sum(acc.rank_report(r).components.get(MemComponent.EMBEDDING, 0) for r in range(4))
    assert total_embed == 256 * 64 * 2


# =============================================================================
# Real-artifact evidence (skipped unless the model volume is mounted)
# =============================================================================


@pytest.mark.skipif(
    not (_ARTIFACT_DIR / "model.safetensors.index.json").is_file(),
    reason="DeepSeek V4.1 W2 artifact volume not mounted",
)
@pytest.mark.parametrize("parallel_mode", ["tp", "ep"])
def test_real_artifact_per_chip_under_budget(parallel_mode):
    manifest = json.loads(_MANIFEST_PATH.read_text())
    metas = list(iter_artifact_tensor_metas(_ARTIFACT_DIR))
    # Coverage: no missing / extra / duplicate routed-expert tensors.
    summary = validate_weight_map((m.name for m in metas), manifest["geometry"])
    assert summary["expert_blocks"] == 43
    assert summary["expert_tensors"] == 94464

    sim = DeepSeekV41W2PlacementSimulator.from_manifest(manifest, parallel_mode=parallel_mode)
    report = sim.predicted_report(metas)
    # Per-rank device bytes under the per-chip weight budget, and balanced.
    assert report["under_budget"]
    assert report["max_per_chip_device_bytes"] <= PER_CHIP_DEVICE_WEIGHT_BUDGET_BYTES
    assert report["imbalance"] <= 0.05
    # Engram host table counted once (~92 GiB), never x world_size.
    assert report["engram_host_bytes"] > 80 * (1024**3)
    # No double-instantiation across the real 94k-tensor expert bank.
    assert sim.ledger.sharded_placed_bytes == sim.ledger.sharded_total_bytes
