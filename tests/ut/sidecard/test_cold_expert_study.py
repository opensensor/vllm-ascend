# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.cold_expert_probe import receive, split_route_weights, validate_header
from tools.sidecard.cold_expert_study import dependencies, select_hot, study, validate_packets


def packet(sequence, position, offset):
    return dict(
        sequence=sequence,
        position=position,
        rows=1,
        rank=0,
        layers=[
            dict(
                layer=index,
                expert_ids=[list(range(offset, offset + 8))],
                route_weights=[[2.5 / 8] * 8],
                components_ms={"attention": 1.0},
            )
            for index in range(18, 33)
        ],
    )


def test_prefill_policy_cannot_see_future_decode_routes():
    prefill, decode = packet(0, 0, 0), packet(1, 1, 8)
    causal = dependencies([decode], select_hot([prefill], 8))
    oracle = dependencies([decode], select_hot([decode], 8))
    assert causal["cold_route_fraction"] == 1
    assert causal["mean_waiting_layers"] == 15
    assert causal["mean_chip_requests"] == 15
    assert oracle["tokens_with_any_cold_expert"] == 0


def test_dependency_counts_both_chips_and_preserves_small_route_weights():
    observed = packet(0, 0, 0)
    for layer in observed["layers"]:
        layer["expert_ids"][0][-1] = 144
        layer["route_weights"][0][-1] = 1e-9
    result = dependencies([observed], select_hot([], 0))
    assert result["mean_chip_requests"] == 30
    assert result["mean_waiting_layers"] == 15
    assert result["cold_route_fraction"] == 1


def test_weight_budget_is_accounted_before_resident_expert_selection():
    layers = [
        dict(layer=index, dense_bytes=100, expert_bytes=2880 if index >= 3 else 0, bits=4 if index >= 3 else None)
        for index in range(45)
    ]
    inventory = dict(
        layers=layers, main_weights_bytes=1000 + sum(row["dense_bytes"] + row["expert_bytes"] for row in layers)
    )
    backbone = inventory["main_weights_bytes"] - 15 * 2880
    report = study(
        [packet(0, 0, 0), packet(1, 1, 8)],
        {"workloads": [dict(name="test", first_sequence=0, end_sequence=2, prompt_tokens=1)]},
        inventory,
        backbone + 15 * 8 * 10,
    )
    assert report["ledger"]["uniform_hot_experts_per_layer"] == 8
    assert report["ledger"]["complete_gpu_weight_bytes"] == backbone + 15 * 8 * 10
    policy = next(row for row in report["workloads"][0]["policies"] if row["hot_experts_per_layer"] == 8)
    assert policy["fits_provisional_weight_budget"]
    assert policy["prefill_trained"]["mean_waiting_layers"] == 15
    assert policy["decode_hindsight_bound"]["mean_waiting_layers"] == 0
    assert not report["production_speed_validated"]


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "missing_layer",
        "foreign_id",
        "repeated_id",
        "nonfinite_weight",
        "negative_timing",
        "invalid_position",
    ],
)
def test_reject_corrupted_routing_evidence(corruption):
    row = packet(0, 0, 0)
    rows = [row]
    if corruption == "duplicate":
        rows.append(copy.deepcopy(row))
    elif corruption == "missing_layer":
        row["layers"].pop()
    elif corruption == "foreign_id":
        row["layers"][0]["expert_ids"][0][0] = 288
    elif corruption == "repeated_id":
        row["layers"][0]["expert_ids"][0][0] = 1
    elif corruption == "nonfinite_weight":
        row["layers"][0]["route_weights"][0][0] = float("nan")
    elif corruption == "invalid_position":
        row["position"] = -1
    else:
        row["layers"][0]["components_ms"]["attention"] = -1
    with pytest.raises(ValueError):
        validate_packets(rows)


@pytest.mark.parametrize("cold", [0, 1, 2, 4, 8])
def test_split_preserves_original_normalization_and_slots(cold):
    weights = torch.arange(1, 9).float()[None] / 36 * 2.5
    cold_weights, hot_weights = split_route_weights(weights, cold)
    torch.testing.assert_close(cold_weights + hot_weights, weights, rtol=0, atol=0)
    assert torch.count_nonzero(cold_weights) == cold
    assert torch.count_nonzero(hot_weights) == 8 - cold
    assert float(weights.sum()) == pytest.approx(2.5, abs=1e-6)


@pytest.mark.parametrize("rows,cold", [(0, 1), (33, 1), (1, 0), (1, 9), (True, 1), (1, True)])
def test_private_probe_rejects_invalid_geometry(rows, cold):
    with pytest.raises(ValueError):
        validate_header(rows, cold)


def test_fragmented_private_packet_and_eof():
    chunks = iter([b"ab", b"c", b"def"])

    def recv_into(target):
        body = next(chunks, b"")
        target[: len(body)] = body
        return len(body)

    connection = SimpleNamespace(recv_into=recv_into)
    target = bytearray(6)
    receive(connection, memoryview(target))
    assert target == b"abcdef"
    with pytest.raises(EOFError):
        receive(connection, memoryview(bytearray(1)))
