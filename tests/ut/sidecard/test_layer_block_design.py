# SPDX-License-Identifier: Apache-2.0
"""Capacity gates and complete deferred-state contracts for the whole-layer design."""

import copy
import json
import socket
import struct
from pathlib import Path

import pytest

from tools.sidecard.layer_block_design import (
    GIB,
    Budget,
    aggregate_inventory,
    boundary_bytes,
    candidate,
    design,
    validate_expert_descriptor,
)
from tools.sidecard.stage_probe import fixture
from tools.sidecard.stage_wire import MAGIC, PREFIX, descriptors, metadata, receive_into, send, validate_metadata

SHA = "a" * 64
INVENTORY = (
    Path(__file__).resolve().parents[3] / "artifacts/sidecard-layer-block-design-20261011/checkpoint-inventory.json"
)


def test_checkpoint_capacity_cannot_be_pooled_between_architectures():
    inventory = json.loads(INVENTORY.read_text())
    plan = design(inventory)
    assert inventory["main_weights_bytes"] / GIB == pytest.approx(131.92037737)
    chosen = plan["preferred"]
    assert chosen["gpu"]["weights"] / GIB == pytest.approx(76.58663318)
    assert all(row["weights"] / GIB == pytest.approx(29.92236723) for row in chosen["npus"])
    assert chosen["fits"] and chosen["cross_architecture_boundaries"] == 2
    assert chosen["gpu_layers"] == list(range(18)) + list(range(33, 45))
    assert chosen["npu_layers"] == list(range(18, 33))
    huge = Budget(200 * GIB, 0, 0, 0, 0)
    small = Budget(40 * GIB, 0, 0, 0, 0)
    assert not candidate(inventory, 18, 33, small, (huge, huge))["fits"]
    assert not candidate(inventory, 18, 33, huge, (small, Budget(10 * GIB, 0, 0, 0, 0)))["fits"]
    assert not plan["runtime_implemented"]


def test_startup_and_runtime_workspace_are_disjoint_lifetimes():
    budget = Budget(10 * GIB, 2 * GIB, GIB, GIB, 4 * GIB)
    ledger = budget.ledger(4 * GIB, 0)
    assert ledger["peak_budgeted_bytes"] == 9 * GIB
    assert ledger["headroom_bytes"] == GIB
    assert ledger["fits"]
    with pytest.raises(ValueError):
        Budget(10 * GIB, -1, 0, 0, 0)


@pytest.mark.parametrize("rows", [1, 4, 128, 512])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_all_deferred_fields_and_positions_survive_round_trip(rows, dtype):
    value = metadata(rows, 7, SHA, residual_dtype=dtype)
    fields, size = descriptors(rows, dtype)
    assert size == boundary_bytes(rows, 2 if dtype == "fp16" else 4)
    assert len(fixture(rows, dtype, 0)) == size
    assert fixture(rows, dtype, 0) != fixture(rows, dtype, 1)
    assert [row["name"] for row in fields] == ["hidden_states", "residual", "post", "comb", "positions"]
    # Socketpair exercises the codec without any accelerator contexts.
    # Keep the synchronous socket test small; descriptor bounds cover bulk cases.
    if rows == 1:
        body = fixture(rows, dtype, 0)
        target = bytearray(size)
        parent, child = socket.socketpair()
        try:
            send(parent, value, body)
            assert receive_into(child, memoryview(target), checkpoint_sha256=SHA, sequence=7) == value
            assert bytes(target) == body
            residual = fields[1]
            if dtype == "fp32":
                assert struct.unpack_from("<f", target, residual["offset"])[0] == 70000.0
        finally:
            parent.close()
            child.close()


@pytest.mark.parametrize("field", ["hidden_states", "residual", "post", "comb", "positions"])
def test_dropping_any_state_field_fails(field):
    value = metadata(1, 0, SHA)
    value["fields"] = [row for row in value["fields"] if row["name"] != field]
    with pytest.raises(ValueError):
        validate_metadata(value)


def test_rejects_wrong_identity_sequence_and_oversized_frame():
    for bad in ("identity", "sequence", "prefix"):
        parent, child = socket.socketpair()
        try:
            value = metadata(1, 2, SHA)
            body = fixture(1, "fp32", 0)
            if bad == "prefix":
                parent.sendall(PREFIX.pack(MAGIC, 1, 1024**3))
            else:
                send(parent, value, body)
            with pytest.raises(ValueError):
                receive_into(
                    child,
                    memoryview(bytearray(len(body))),
                    checkpoint_sha256="b" * 64 if bad == "identity" else SHA,
                    sequence=0 if bad == "sequence" else 2,
                )
        finally:
            parent.close()
            child.close()
    with pytest.raises(ValueError):
        validate_metadata({"rows": 1})


def test_geometry_and_missing_backbone_rejected():
    with pytest.raises(ValueError):
        descriptors(513)
    with pytest.raises(ValueError):
        descriptors(True)
    with pytest.raises(ValueError):
        descriptors(1, "bf16")
    with pytest.raises(ValueError):
        metadata(1, -1, SHA)
    inventory = json.loads(INVENTORY.read_text())
    bad = copy.deepcopy(inventory)
    bad["layers"].pop()
    budget = Budget(200 * GIB, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        candidate(bad, 18, 33, budget, (budget, budget))
    with pytest.raises(ValueError):
        aggregate_inventory([], inventory["geometry"], {}, ["linear_attention"] * 45, {})


def test_expert_ids_shapes_and_scale_types_are_canonical():
    name = "model.language_model.layers.3.mlp.experts.287.gate_proj_codes"
    validate_expert_descriptor(name, dict(shape=[2048, 2048], dtype="I8"), {"3": 4})
    for altered, descriptor in (
        (name.replace("287", "288"), dict(shape=[2048, 2048], dtype="I8")),
        (name, dict(shape=[2048, 1024], dtype="I8")),
        (name.replace("codes", "scale"), dict(shape=[64, 128], dtype="F32")),
        (name.replace("gate_proj", "other_proj"), dict(shape=[2048, 2048], dtype="I8")),
    ):
        with pytest.raises(ValueError):
            validate_expert_descriptor(altered, descriptor, {"3": 4})
