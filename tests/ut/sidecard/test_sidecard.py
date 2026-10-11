# SPDX-License-Identifier: Apache-2.0
"""CPU contract gates for joined resident experts and malformed/stale replies."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from tools.sidecard.__main__ import inventory
from tools.sidecard.placement import ExpertPlacement, glm_placement, storage_lower_bound, validate_glm_config
from tools.sidecard.reference import ExpertReply, Route, combine, dispatch, fp32


@pytest.fixture
def batch():
    placement = ExpertPlacement(("cuda:0", "npu:0", "npu:1"), (("cuda:0", "npu:0", "npu:1", "npu:1"),))
    return dict(
        placement=placement,
        execution_id="request-A:step-7",
        layer_id=3,
        hidden=((2.0, -3.0), (4.0, 5.0)),
        expert_ids=((2, 0, 1), (3, 2, 0)),
        weights=((0.2, 0.5, 0.3), (0.6, 0.1, 0.3)),
    )


def expert_output(expert, hidden):
    return tuple(fp32(value * (expert + 1)) for value in hidden)


def replies_for(batch):
    replies = []
    for request in dispatch(**batch):
        hidden = dict(zip(request.token_ids, request.hidden))
        outputs = tuple(expert_output(route.expert, hidden[route.token]) for route in request.routes)
        replies.append(ExpertReply(request.execution_id, request.layer_id, request.owner, request.routes, outputs))
    return replies


def test_partitioned_execution_matches_serial_experts_with_shared_once(batch):
    shared = ((7.0, 11.0), (13.0, 17.0))
    actual = combine(**batch, replies=replies_for(batch)[::-1], shared=shared, routed_scale=2.5)
    expected = []
    for token, ids in enumerate(batch["expert_ids"]):
        summed = [0.0, 0.0]
        for slot, expert in enumerate(ids):
            output = expert_output(expert, batch["hidden"][token])
            weight = fp32(batch["weights"][token][slot])
            for column, value in enumerate(output):
                summed[column] = fp32(summed[column] + fp32(weight * value))
        expected.append(tuple(fp32(fp32(value * 2.5) + shared[token][column]) for column, value in enumerate(summed)))
    assert actual == tuple(expected)


def test_owner_packets_deduplicate_token_rows_and_keep_slot_identity(batch):
    packet = next(packet for packet in dispatch(**batch) if packet.owner == "npu:1")
    assert packet.token_ids == (0, 1)
    assert len(packet.routes) == 3
    assert packet.routes == (Route(0, 0, 2), Route(1, 0, 3), Route(1, 1, 2))


def test_cancellation_sensitive_combine_uses_route_order_not_reply_order():
    placement = ExpertPlacement(("cuda:0", "npu:0"), (("npu:0", "cuda:0", "npu:0"),))
    batch = dict(
        placement=placement,
        execution_id="step",
        layer_id=3,
        hidden=((0.0,),),
        expert_ids=((0, 1, 2),),
        weights=((1.0, 1.0, 1.0),),
    )
    replies = (
        ExpertReply("step", 3, "cuda:0", (Route(0, 1, 1),), ((1.0,),)),
        ExpertReply("step", 3, "npu:0", (Route(0, 2, 2), Route(0, 0, 0)), ((-1e8,), (1e8,))),
    )
    assert combine(**batch, replies=replies, shared=((0.0,),), routed_scale=1.0) == ((0.0,),)


def test_empty_owner_is_not_required_and_zero_weight_route_remains_accounted(batch):
    batch["expert_ids"] = ((0,), (0,))
    batch["weights"] = ((0.0,), (1.0,))
    assert len(dispatch(**batch)) == 1
    assert len(replies_for(batch)[0].routes) == 2


@pytest.mark.parametrize(
    "failure", ["stale", "layer", "duplicate_owner", "missing_owner", "misowned", "duplicate_route", "shape", "nan"]
)
def test_bad_reply_is_rejected(batch, failure):
    replies = replies_for(batch)
    first = replies[0]
    if failure == "stale":
        replies[0] = replace(first, execution_id="previous-step")
    elif failure == "layer":
        replies[0] = replace(first, layer_id=7)
    elif failure == "duplicate_owner":
        replies.append(first)
    elif failure == "missing_owner":
        replies.pop()
    elif failure == "misowned":
        replies[0] = replace(first, routes=(Route(0, 0, 2), *first.routes[1:]))
    elif failure == "duplicate_route":
        replies[0] = replace(first, routes=(first.routes[0], first.routes[0]))
    elif failure == "shape":
        replies[0] = replace(first, outputs=((1.0,),) * len(first.routes))
    else:
        replies[0] = replace(first, outputs=((float("nan"), 1.0),) * len(first.routes))
    with pytest.raises(ValueError):
        combine(**batch, replies=replies, shared=((0.0, 0.0),) * 2, routed_scale=2.5)


@pytest.mark.parametrize(
    "failure", ["negative_id", "large_id", "duplicate_id", "negative_weight", "nan_weight", "rows", "dense_layer"]
)
def test_bad_dispatch_is_rejected(batch, failure):
    if failure in ("negative_id", "large_id", "duplicate_id"):
        batch["expert_ids"] = (
            (-1, 0, 1) if failure == "negative_id" else (4, 0, 1) if failure == "large_id" else (0, 0, 1),
            (3, 2, 0),
        )
    elif failure in ("negative_weight", "nan_weight"):
        batch["weights"] = ((-0.1 if failure == "negative_weight" else float("nan"), 0.5, 0.3), (0.6, 0.1, 0.3))
    elif failure == "rows":
        batch["hidden"] = ((2.0,), (4.0, 5.0))
    else:
        batch["layer_id"] = 2
    with pytest.raises(ValueError):
        dispatch(**batch)


@pytest.mark.parametrize("cuda_experts", [0, 1, 72, 287, 288])
def test_glm_placement_covers_each_expert_and_balances_overflow(cuda_experts):
    placement = glm_placement(cuda_experts)
    assert len(placement.layers) == 42
    assert all(len(layer) == 288 for layer in placement.layers)
    layer = placement.layer(3)
    assert layer.count("cuda:0") == cuda_experts
    counts = [layer.count(owner) for owner in placement.owners[1:]]
    assert max(counts) - min(counts) <= 1
    assert ExpertPlacement.from_dict(json.loads(json.dumps(placement.to_dict()))) == placement
    assert sum(storage_lower_bound(placement).values()) == 42 * 288 * 3 * 4096 * 2048 // 2


@pytest.mark.parametrize("count", [-1, 289, True, 1.5])
def test_invalid_cuda_count(count):
    with pytest.raises(ValueError):
        glm_placement(count)


def test_incomplete_or_unknown_manifest_is_rejected():
    with pytest.raises(ValueError):
        ExpertPlacement(("cuda:0",), (("cuda:0", "npu:0"),))
    with pytest.raises(ValueError):
        ExpertPlacement(("cuda:0",), (("cuda:0",), ("cuda:0", "cuda:0")))
    with pytest.raises(ValueError):
        ExpertPlacement.from_dict({"schema_version": 99})


def test_mutable_placement_and_wrong_storage_geometry_are_rejected(batch):
    with pytest.raises(ValueError, match="immutable"):
        ExpertPlacement(("cuda:0",), (["cuda:0"],))
    with pytest.raises(ValueError, match="GLM placement"):
        storage_lower_bound(batch["placement"])


def glm_config():
    return dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        first_k_dense_replace=3,
        n_routed_experts=288,
        num_experts_per_tok=8,
        hidden_size=4096,
        moe_intermediate_size=2048,
        routed_scaling_factor=2.5,
    )


def test_config_and_cli_do_not_admit_capacity(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": glm_config()}))
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard", "plan", "--model-config", str(config)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["capacity_admitted"] is False
    assert report["hardware_validated"] is False
    assert report["mtp_enabled"] is False
    assert report["placement"]["layers"][0].count("cuda:0") == 72
    assert len(report["model_config_sha256"]) == 64


def test_wrong_model_config_is_rejected():
    config = glm_config()
    validate_glm_config(config)
    config["num_hidden_layers"] = 46
    with pytest.raises(ValueError, match="num_hidden_layers"):
        validate_glm_config(config)


def test_inventory_records_absent_drivers_without_claiming_validation(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("driver command unavailable")

    monkeypatch.setattr(subprocess, "run", absent)
    report = inventory()
    assert report["inventory_only"] is True
    assert report["hardware_validated"] is False
    assert all("error" in receipt for receipt in report["receipts"].values())
