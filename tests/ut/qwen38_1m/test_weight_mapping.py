# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Qwen4Exp W8A8 expert weight mapping + load-time rejection
(plan T3.1).

These run host-side (CPU, no NPU, no Triton) and NEVER load the 224 GB
checkpoint. The expected-tensor set is driven from the frozen checkpoint
manifest geometry; the "provided" side is a *synthetic* safetensors-style index
(tensor name -> {"dtype", "shape"}) built to match that geometry.

Run:
    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_weight_mapping.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm_ascend.models.qwen4_exp.dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY
from vllm_ascend.models.qwen4_exp.weight_mapping import (
    DuplicateTensorError,
    ExtraTensorError,
    MissingTensorError,
    TensorDtypeError,
    TensorShapeError,
    WeightMappingError,
    expected_expert_tensor_names,
    geometry_from_manifest,
    map_expert_tensor,
    validate_expert_weight_map,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPO_ROOT / "artifacts" / "qwen38-1m" / "checkpoint-manifest.json"

# Ground-truth W8A8 geometry (verified against real shard headers).
_EXPERTS = 512
_LAYERS = 48
_MOE_INTERMEDIATE = 640
_HIDDEN = 2560
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_KINDS = ("weight", "weight_scale", "weight_offset")
_EXPECTED_PROJECTIONS = _LAYERS * _EXPERTS * len(("gate_proj", "up_proj", "down_proj"))  # 73,728


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def geometry(manifest) -> dict:
    return geometry_from_manifest(manifest)


def _expert_name(layer: int, expert: int, proj: str, kind: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"


def _weight_shape(proj: str) -> list[int]:
    if proj == "down_proj":
        return [_HIDDEN, _MOE_INTERMEDIATE]
    return [_MOE_INTERMEDIATE, _HIDDEN]


def _row_dim(proj: str) -> int:
    return _HIDDEN if proj == "down_proj" else _MOE_INTERMEDIATE


def _build_synthetic_index() -> dict[str, dict]:
    """A safetensors-style index that exactly matches the manifest geometry."""
    index: dict[str, dict] = {}
    for layer in range(_LAYERS):
        for expert in range(_EXPERTS):
            for proj in _PROJECTIONS:
                index[_expert_name(layer, expert, proj, "weight")] = {
                    "dtype": "I8",
                    "shape": _weight_shape(proj),
                }
                for kind in ("weight_scale", "weight_offset"):
                    index[_expert_name(layer, expert, proj, kind)] = {
                        "dtype": "F32",
                        "shape": [_row_dim(proj), 1],
                    }
    # A few representative non-expert F16 tensors that must NOT be mapped.
    index["model.language_model.embed_tokens.weight"] = {"dtype": "F16", "shape": [248320, 2560]}
    index["lm_head.weight"] = {"dtype": "F16", "shape": [248320, 2560]}
    index["model.language_model.layers.0.mlp.gate.weight"] = {"dtype": "F16", "shape": [512, 2560]}
    index["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"] = {
        "dtype": "F16",
        "shape": [640, 2560],
    }
    index["model.language_model.layers.0.self_attn.q_norm.weight"] = {"dtype": "F16", "shape": [256]}
    index["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight"] = {
        "dtype": "F16",
        "shape": [2500012, 160],
    }
    # An integer index tensor (PLE) - non-float, must be tolerated, not F16-checked.
    index["model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"] = {
        "dtype": "I64",
        "shape": [16],
    }
    return index


# --------------------------------------------------------------------------- #
# Expected set is driven from the manifest
# --------------------------------------------------------------------------- #
def test_geometry_from_manifest_matches_ground_truth(geometry):
    assert geometry["num_hidden_layers"] == _LAYERS
    assert geometry["num_experts"] == _EXPERTS
    assert geometry["moe_intermediate_size"] == _MOE_INTERMEDIATE
    assert geometry["hidden_size"] == _HIDDEN


def test_expected_set_counts_match_manifest(manifest, geometry):
    names = expected_expert_tensor_names(geometry)
    # 73,728 weights + 73,728 scales + 73,728 offsets.
    assert len(names) == 3 * _EXPECTED_PROJECTIONS
    weights = {n for n in names if n.endswith(".weight")}
    scales = {n for n in names if n.endswith(".weight_scale")}
    offsets = {n for n in names if n.endswith(".weight_offset")}
    assert len(weights) == _EXPECTED_PROJECTIONS == manifest["component_counts"]["expert_weight"]
    assert len(scales) == _EXPECTED_PROJECTIONS == manifest["component_counts"]["expert_weight_scale"]
    assert len(offsets) == _EXPECTED_PROJECTIONS == manifest["component_counts"]["expert_weight_offset"]


# --------------------------------------------------------------------------- #
# Happy path: all 73,728 projections mapped into the fused MoE layout
# --------------------------------------------------------------------------- #
def test_happy_path_covers_all_expert_projections(geometry):
    index = _build_synthetic_index()
    mapping = validate_expert_weight_map(index, geometry)

    assert len(mapping.weight_entries) == _EXPECTED_PROJECTIONS
    assert len(mapping.scale_entries) == _EXPECTED_PROJECTIONS
    assert len(mapping.offset_entries) == _EXPECTED_PROJECTIONS

    # Every (layer, expert, proj) weight projection is covered exactly once.
    covered = {(e.layer, e.expert, e.proj) for e in mapping.weight_entries}
    assert len(covered) == _EXPECTED_PROJECTIONS
    assert {e.layer for e in mapping.weight_entries} == set(range(_LAYERS))
    assert {e.expert for e in mapping.weight_entries} == set(range(_EXPERTS))

    # Non-expert tensors are recorded but never mapped into the MoE layout.
    assert "lm_head.weight" in mapping.non_expert_tensors
    assert not any("mlp.experts." not in e.source_name for e in mapping.entries)


def test_gate_up_down_target_slots(geometry):
    index = _build_synthetic_index()
    mapping = validate_expert_weight_map(index, geometry)
    by_name = {e.source_name: e for e in mapping.weight_entries}

    gate = by_name[_expert_name(5, 17, "gate_proj", "weight")]
    up = by_name[_expert_name(5, 17, "up_proj", "weight")]
    down = by_name[_expert_name(5, 17, "down_proj", "weight")]

    # gate + up fuse into w13_weight; gate occupies rows [0, moe), up [moe, 2*moe).
    assert gate.target_param == "w13_weight"
    assert (gate.expert_index, gate.row_start, gate.row_stop) == (17, 0, _MOE_INTERMEDIATE)
    assert up.target_param == "w13_weight"
    assert (up.expert_index, up.row_start, up.row_stop) == (17, _MOE_INTERMEDIATE, 2 * _MOE_INTERMEDIATE)
    # down_proj is w2_weight, full [hidden, moe] slot.
    assert down.target_param == "w2_weight"
    assert (down.expert_index, down.row_start, down.row_stop) == (17, 0, _HIDDEN)


def test_scale_offset_target_params(geometry):
    index = _build_synthetic_index()
    mapping = validate_expert_weight_map(index, geometry)
    by_name = {e.source_name: e for e in mapping.entries}

    assert by_name[_expert_name(0, 0, "gate_proj", "weight_scale")].target_param == "w13_weight_scale"
    assert by_name[_expert_name(0, 0, "up_proj", "weight_offset")].target_param == "w13_weight_offset"
    assert by_name[_expert_name(0, 0, "down_proj", "weight_scale")].target_param == "w2_weight_scale"
    assert by_name[_expert_name(0, 0, "down_proj", "weight_offset")].target_param == "w2_weight_offset"


# --------------------------------------------------------------------------- #
# Each rejection class fires exactly once
# --------------------------------------------------------------------------- #
def test_missing_tensor_rejected(geometry):
    index = _build_synthetic_index()
    victim = _expert_name(3, 200, "up_proj", "weight")
    del index[victim]
    with pytest.raises(MissingTensorError) as exc:
        validate_expert_weight_map(index, geometry)
    assert victim in str(exc.value)
    assert exc.value.category == "missing"


def test_extra_tensor_rejected(geometry):
    index = _build_synthetic_index()
    bogus = _expert_name(0, 9999, "gate_proj", "weight")  # expert id out of range
    index[bogus] = {"dtype": "I8", "shape": [_MOE_INTERMEDIATE, _HIDDEN]}
    with pytest.raises(ExtraTensorError) as exc:
        validate_expert_weight_map(index, geometry)
    assert bogus in str(exc.value)
    assert exc.value.category == "extra"


def test_duplicate_tensor_rejected(geometry):
    # A safetensors dict cannot repeat a key, so a duplicate is expressed as a
    # sequence of (name, meta) pairs with a repeated source tensor.
    items = list(_build_synthetic_index().items())
    dup = _expert_name(0, 0, "gate_proj", "weight")
    items.append((dup, {"dtype": "I8", "shape": [_MOE_INTERMEDIATE, _HIDDEN]}))
    with pytest.raises(DuplicateTensorError) as exc:
        validate_expert_weight_map(items, geometry)
    assert dup in str(exc.value)
    assert exc.value.category == "duplicate"


def test_wrong_shape_rejected(geometry):
    index = _build_synthetic_index()
    victim = _expert_name(1, 1, "down_proj", "weight")
    index[victim] = {"dtype": "I8", "shape": [_HIDDEN, _MOE_INTERMEDIATE - 1]}  # bad in-dim
    with pytest.raises(TensorShapeError) as exc:
        validate_expert_weight_map(index, geometry)
    assert victim in str(exc.value)
    assert exc.value.category == "wrong_shape"


def test_wrong_dtype_expert_rejected(geometry):
    index = _build_synthetic_index()
    victim = _expert_name(2, 2, "gate_proj", "weight")
    index[victim] = {"dtype": "F16", "shape": [_MOE_INTERMEDIATE, _HIDDEN]}  # should be I8
    with pytest.raises(TensorDtypeError) as exc:
        validate_expert_weight_map(index, geometry)
    assert victim in str(exc.value)
    assert exc.value.category == "wrong_dtype"


def test_wrong_dtype_scale_rejected(geometry):
    index = _build_synthetic_index()
    victim = _expert_name(2, 2, "gate_proj", "weight_scale")
    index[victim] = {"dtype": "I8", "shape": [_MOE_INTERMEDIATE, 1]}  # should be F32
    with pytest.raises(TensorDtypeError) as exc:
        validate_expert_weight_map(index, geometry)
    assert victim in str(exc.value)


# --------------------------------------------------------------------------- #
# Every tensor is checked against the frozen dtype policy before mapping
# --------------------------------------------------------------------------- #
def test_non_expert_must_be_fp16(geometry):
    index = _build_synthetic_index()
    # Router weight forced to F32 - violates the F16 non-expert policy.
    index["model.language_model.layers.0.mlp.gate.weight"] = {"dtype": "F32", "shape": [512, 2560]}
    with pytest.raises(TensorDtypeError) as exc:
        validate_expert_weight_map(index, geometry)
    assert exc.value.category == "wrong_dtype"
    assert "gate.weight" in str(exc.value)


def test_dtype_policy_is_the_frozen_source(geometry):
    # The mapping enforces the frozen policy: F16 main, F32 accumulation/scale.
    assert ASCEND_QWEN4EXP_DTYPE_POLICY.main_dtype == torch.float16
    assert ASCEND_QWEN4EXP_DTYPE_POLICY.accumulation_dtype == torch.float32
    index = _build_synthetic_index()
    # Happy path passes under the frozen policy.
    validate_expert_weight_map(index, geometry)


def test_map_expert_tensor_rejects_out_of_range(geometry):
    with pytest.raises(WeightMappingError):
        map_expert_tensor(_expert_name(0, _EXPERTS, "gate_proj", "weight"), geometry)
    with pytest.raises(WeightMappingError):
        map_expert_tensor("model.language_model.embed_tokens.weight", geometry)
