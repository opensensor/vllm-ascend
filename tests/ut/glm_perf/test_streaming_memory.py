# SPDX-License-Identifier: Apache-2.0
"""Compile real contract geometry and reject unsafe ownership/capacity changes."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.glm_perf.streaming_memory import (
    PHASES,
    SlotLedger,
    aliases,
    memory_contract,
    projection_allocations,
    validate_aliases,
)
from tools.glm_perf.streaming_protocol import file_digest

V1_HEADER = """namespace GlmStreamingContract {
constexpr uint32_t VERSION = 1;
constexpr uint32_t ROWS = 32, MAX_ROWS = 31, COLUMNS = 128, GROUP = 32;
constexpr uint32_t PAIR_K = 64, WEIGHT_TILE_K = 256, MAX_K = 4096;
constexpr uint32_t SLOTS = 2, EVENT_FIRST = 0;
constexpr uint32_t ELEMENTS = ROWS * COLUMNS;
constexpr uint32_t OPERAND_A_SLOT_BYTES = 2 * ROWS * PAIR_K / 2;
constexpr uint32_t OPERAND_B_SLOT_BYTES = COLUMNS * PAIR_K / 2;
constexpr uint32_t PRODUCT_SLOT_BYTES = 2 * ELEMENTS * sizeof(int32_t);
constexpr uint32_t PRODUCT_STORAGE_BYTES = SLOTS * PRODUCT_SLOT_BYTES;
constexpr uint32_t CONVERTED_BYTES = ELEMENTS * sizeof(float);
constexpr uint32_t FACTORS_OFFSET = 0;
constexpr uint32_t BROADCAST_OFFSET = ELEMENTS * sizeof(float);
constexpr uint32_t RETAINED_INDEX_OFFSET = COLUMNS * WEIGHT_TILE_K * 3 / 4;
constexpr uint32_t EPILOGUE_TILE_BYTES = 2 * OPERAND_A_SLOT_BYTES;
constexpr uint32_t SDK_UB_RESERVE = 8192;
}
"""
V1_PROOF_WITHOUT_HEADER_SHA256 = "c7b24c5e65da56795167970b55dc7222295ef6e10c712b09d0125ed53c92813a"


def test_generic_projection_fits_with_two_slots_and_sdk_reserve():
    proof = memory_contract()
    assert proof["projections"]["gate_generic"]["ub_total"] == 237440
    assert proof["projections"]["down_generic"]["ub_total"] == 249664
    assert proof["projections"]["down_generic"]["ub_spare_after_sdk_reserve"] == 4288
    assert proof["projections"]["gate_w4"]["ub_total"] == 198528
    assert proof["global_memory_delta"] == 0
    assert proof["partial_products_if_materialized"] == 2031616
    assert proof["hardware_validated"] is False


def test_actual_header_geometry_matches_python_contract(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "geometry.cpp"
    source.write_text("""#include "glm_streaming_contract.h"
#include <iostream>
int main() {
 using namespace GlmStreamingContract;
 std::cout << PRODUCT_STORAGE_BYTES << ' ' << CONVERTED_BYTES << ' '
           << OPERAND_A_SLOT_BYTES << ' ' << OPERAND_B_SLOT_BYTES << ' '
           << EPILOGUE_TILE_BYTES << ' ' << SDK_UB_RESERVE << ' ' << VERSION << ' '
           << MASK_BYTES << ' ' << RETAINED_EXPERT_END_OFFSET << ' ' << RETAINED_EXPERT_END_BYTES << ' '
           << SCALE_GATHER_INDEX_OFFSET << ' ' << SCALE_GATHER_INDEX_BYTES << ' '
           << GROUP_SCALE_SOURCE_OFFSET << ' ' << GROUP_SCALE_SOURCE_BYTES;
}
""")
    binary = tmp_path / "geometry"
    subprocess.run(
        ["c++", "-std=c++17", "-I" + str(root / "tools/glm_perf"), str(source), "-o", str(binary)], check=True
    )
    assert subprocess.check_output([str(binary)], text=True).split() == [
        "65536",
        "16384",
        "2048",
        "4096",
        "4096",
        "8192",
        "2",
        "32768",
        "24704",
        "2304",
        "27008",
        "1024",
        "28032",
        "128",
    ]


def test_reconstruction_cannot_overlap_live_product_slots():
    values = aliases()
    values[0]["phases"].append("projection")
    with pytest.raises(ValueError, match="overlapping live"):
        validate_aliases(values, projection_allocations(gate=True, compact_w4=False))


def test_factor_scratch_cannot_overlap_live_quantizer():
    values = aliases()
    next(row for row in values if row["name"] == "quant_scratch")["phases"].append("projection")
    with pytest.raises(ValueError, match="overlapping live"):
        validate_aliases(values, projection_allocations(gate=True, compact_w4=False))


def test_bounds_reject_alias_overrun():
    values = aliases()
    values[1]["bytes"] = 100000
    with pytest.raises(ValueError, match="exceeds"):
        validate_aliases(values, projection_allocations(gate=True, compact_w4=False))


@pytest.mark.parametrize("operation", ["issue", "readback", "consume"])
def test_uninitialized_slot_rejected(operation):
    with pytest.raises(ValueError, match="unsafe"):
        SlotLedger().transition(0, operation)


def test_product_slot_reuse_requires_consumer_completion():
    ledger = SlotLedger()
    ledger.transition(0, "load")
    ledger.transition(0, "issue")
    ledger.transition(0, "readback")
    with pytest.raises(ValueError, match="unsafe"):
        ledger.transition(0, "load")
    with pytest.raises(ValueError, match="unsafe"):
        ledger.transition(0, "drain")
    ledger.transition(0, "consume")
    ledger.transition(0, "load")


def test_long_stream_and_final_slot_drain():
    ledger = SlotLedger()
    for pair in range(65):
        slot = pair % 2
        for operation in ("load", "issue", "readback", "consume"):
            ledger.transition(slot, operation)
    for slot in (0, 1):
        ledger.transition(slot, "drain")
    assert ledger.states == ("free", "free")


def test_unknown_sdk_limits_cannot_claim_proven_feasibility():
    with pytest.raises(ValueError, match="capacities"):
        memory_contract({"architecture": 2201, "capacities": {}})


def test_wrong_nominal_capacity_rejected():
    proof = memory_contract()
    caps = dict(proof["capacities"], ub=131072)
    with pytest.raises(ValueError, match="capacities"):
        memory_contract({"architecture": 2002, "capacities": caps})


def test_cli_does_not_claim_device_validation(tmp_path):
    output = tmp_path / "contract.json"
    subprocess.run(["python", "-m", "tools.glm_perf.streaming_memory", "--output", str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["sdk_verified"] is False
    assert result["hardware_validated"] is False


def test_explicit_frozen_header_binding_preserves_every_memory_bound(tmp_path):
    header = tmp_path / "frozen-contract.h"
    header.write_bytes(
        Path(__file__).resolve().parents[3].joinpath("tools/glm_perf/glm_streaming_contract.h").read_bytes()
    )
    current = memory_contract()
    frozen = memory_contract(header_path=header)
    assert frozen["header_sha256"] == file_digest(header)
    assert {key: value for key, value in frozen.items() if key != "header_sha256"} == {
        key: value for key, value in current.items() if key != "header_sha256"
    }


def test_explicit_missing_header_cannot_produce_a_proof(tmp_path):
    with pytest.raises(FileNotFoundError):
        memory_contract(header_path=tmp_path / "missing-contract.h")


def test_explicit_v1_header_preserves_exact_historical_proof(tmp_path):
    header = tmp_path / "v1-contract.h"
    header.write_text(V1_HEADER)
    frozen = memory_contract(header_path=header)
    assert frozen.pop("header_sha256") == file_digest(header)
    assert "contract_version" not in frozen
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == V1_PROOF_WITHOUT_HEADER_SHA256


def test_v2_retained_ranges_live_in_every_phase_without_new_allocation():
    proof = memory_contract()
    assert proof["contract_version"] == 2
    expected = {
        "retained_indices": (24576, 24704),
        "retained_expert_ends": (24704, 27008),
        "scale_gather_indices": (27008, 28032),
        "group_scale_source": (28032, 28160),
    }
    for value in proof["aliases"]:
        if value["name"] in expected:
            assert (value["offset"], value["offset"] + value["bytes"]) == expected[value["name"]]
            assert value["buffer"] == "mask"
            assert value["phases"] == list(PHASES)
    assert expected.keys() <= {row["name"] for row in proof["aliases"]}
    for projection in proof["projections"].values():
        assert next(row for row in projection["ub_allocations"] if row["name"] == "mask")["bytes"] == 32768


@pytest.mark.parametrize("phase", PHASES)
def test_v2_gather_indices_cannot_overlap_live_expert_ends(phase):
    values = aliases()
    gather = next(row for row in values if row["name"] == "scale_gather_indices")
    gather.update(offset=26976, phases=[phase])
    with pytest.raises(ValueError, match="overlapping live"):
        validate_aliases(values, projection_allocations(gate=True, compact_w4=False))


def test_v2_immutable_source_cannot_overlap_expanded_indices():
    values = aliases()
    next(row for row in values if row["name"] == "group_scale_source")["offset"] = 28000
    with pytest.raises(ValueError, match="overlapping live"):
        validate_aliases(values, projection_allocations(gate=False, compact_w4=True))


@pytest.mark.parametrize("version", [0, 3, 2002])
def test_unsupported_explicit_header_version_is_rejected(tmp_path, version):
    header = tmp_path / "unsupported.h"
    header.write_text(V1_HEADER.replace("VERSION = 1", f"VERSION = {version}"))
    with pytest.raises(ValueError, match="unsupported streaming contract version"):
        memory_contract(header_path=header)


@pytest.mark.parametrize(
    "before,after",
    [
        ("ROWS = 32", "ROWS = 16"),
        ("SDK_UB_RESERVE = 8192", "SDK_UB_RESERVE = 4096"),
        ("MASK_BYTES = 32768", "MASK_BYTES = 28000"),
        ("GROUP_SCALE_SOURCE_BYTES = ROWS * sizeof(uint32_t)", "GROUP_SCALE_SOURCE_BYTES = 256"),
        (
            "SCALE_GATHER_INDEX_OFFSET = RETAINED_EXPERT_END_OFFSET + RETAINED_EXPERT_END_BYTES",
            "SCALE_GATHER_INDEX_OFFSET = 26000",
        ),
    ],
)
def test_explicit_header_mismatched_constants_are_rejected(tmp_path, before, after):
    current = Path(__file__).resolve().parents[3] / "tools/glm_perf/glm_streaming_contract.h"
    header = tmp_path / "mismatched.h"
    header.write_text(current.read_text().replace(before, after))
    with pytest.raises(ValueError, match="constants do not match"):
        memory_contract(header_path=header)


def test_header_without_version_is_not_an_admissible_contract(tmp_path):
    header = tmp_path / "missing-version.h"
    header.write_text("// explicit bytes are not a frozen contract\n")
    with pytest.raises(ValueError, match="unsupported streaming contract version"):
        memory_contract(header_path=header)
