# SPDX-License-Identifier: Apache-2.0
"""Host-only byte, ABI, event-generation and complete-rank envelope checks."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tools.qwen4exp import streaming_memory as memory

ROOT = Path(__file__).resolve().parents[3]


def test_exact_capacities_and_unused_scratch_are_counted():
    value = memory.contract()
    assert value["usage_bytes"] == {"UB": 140032, "L1": 167936, "L0A": 4096, "L0B": 16384, "L0C": 16384}
    by_name = {r.name: r for r in memory.regions()}
    assert by_name["quantizer_scratch"].nbytes == 32768
    assert by_name["persistent_gate"].nbytes == 4096
    assert by_name["cube_product"].nbytes == 16384
    assert by_name["raw_product"].nbytes == 32768
    assert by_name["packed_activation"].layout == "[slot,limb,GROUP/K0,M,K0/2]"
    assert by_name["activation_l1"].layout == "[slot,limb,GROUP/K0,M,K0/2]"
    assert by_name["activation_l0"].layout == "[slot,limb,M/16,GROUP/K0,16,K0/2]"
    assert value["capacities_bytes"]["UB"] + value["ub_sdk_reserve_bytes"] == 262144
    assert value["hardware_validated"] is False
    assert value["whole_rank"]["available_bytes"] is None


def test_contract_header_json_are_exact_and_hash_sealed():
    value = memory.contract()
    sealed = value.pop("contract_sha256")
    assert (
        hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        == sealed
    )
    saved = json.loads((ROOT / "artifacts/qwen38-streaming-upgrade/T2/contract.json").read_text())
    assert saved == memory.contract()
    assert (ROOT / "tools/qwen4exp/qwen_streaming_contract.h").read_text() == memory.header_text()


def test_sdk_provenance_hashes_match():
    folder = ROOT / "artifacts/qwen38-streaming-upgrade/T2"
    value = json.loads((folder / "sdk-provenance.json").read_text())
    assert value["npu_opened"] is False
    assert value["server_contacted"] is False
    for filename, digest in value["files"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("groups", range(1, 21))
@pytest.mark.parametrize("rows", [1, 8, 15, 16])
def test_startup_steady_state_drain(groups, rows):
    mode = "bulk" if rows == 16 else "sparse"
    proof = memory.prove_schedule(groups, rows, mode)
    assert proof["additions"] == list(range(groups))
    assert proof["permitted_cube_consumer_pairs"] == [(i, i - 1) for i in range(1, groups)]
    assert proof["hardware_overlap_proven"] is False
    signals = [entry[1:] for entry in proof["trace"] if entry[0] == "signal"]
    waits = [entry[1:] for entry in proof["trace"] if entry[0] == "wait"]
    assert signals == waits


@pytest.mark.parametrize(
    "groups,rows,mode",
    [(0, 16, "bulk"), (21, 16, "bulk"), (2, 0, "sparse"), (2, 17, "sparse"), (2, 8, "bulk"), (2, 16, "unknown")],
)
def test_unsupported_projection_shapes(groups, rows, mode):
    with pytest.raises(ValueError):
        memory.prove_schedule(groups, rows, mode)


@pytest.mark.parametrize("changed", ["size", "offset", "alias", "overcommit", "duplicate", "space"])
def test_invalid_memory_layouts_are_rejected(changed):
    regions = list(memory.regions())
    if changed == "size":
        regions[0] = replace(regions[0], nbytes=1)
    elif changed == "offset":
        regions[0] = replace(regions[0], offset=1)
    elif changed == "alias":
        regions[1] = replace(regions[1], offset=regions[0].offset)
    elif changed == "overcommit":
        regions[0] = replace(regions[0], nbytes=262144)
    elif changed == "duplicate":
        regions[1] = replace(regions[1], name=regions[0].name)
    elif changed == "space":
        regions[0] = replace(regions[0], space="made_up")
    with pytest.raises(ValueError):
        memory.validate_regions(tuple(regions))


def make_ownership(alias=False):
    return memory.Ownership(
        (
            memory.Region("a", "UB", 0, 64, "byte", "test", "test"),
            memory.Region("b", "UB", 32 if alias else 64, 64, "byte", "test", "test"),
        )
    )


def test_live_alias_rejected_by_dynamic_lifetime_checker():
    proof = make_ownership(alias=True)
    proof.acquire("a", 0)
    with pytest.raises(ValueError, match="live alias"):
        proof.acquire("b", 1)


def test_premature_reuse_and_free_are_rejected():
    proof = make_ownership()
    proof.acquire("a", 0)
    with pytest.raises(ValueError, match="reuse"):
        proof.acquire("a", 1)
    with pytest.raises(ValueError, match="premature release"):
        proof.release("a", 0, ("V_M", 0, 0))
    proof.signal("V_M", 0, 0)
    with pytest.raises(ValueError, match="premature release"):
        proof.release("a", 0, ("V_M", 0, 0))
    token = proof.wait("V_M", 0, 0)
    proof.release("a", 0, token)
    proof.drain()


def test_pre_acquisition_completion_does_not_authorize_free():
    proof = make_ownership()
    proof.signal("V_M", 0, 0)
    token = proof.wait("V_M", 0, 0)
    proof.acquire("a", 0)
    with pytest.raises(ValueError, match="premature release"):
        proof.release("a", 0, token)


def test_event_reuse_unmatched_wait_and_stale_generation_rejected():
    proof = make_ownership()
    proof.signal("M_V", 0, 3)
    with pytest.raises(ValueError, match="reused before wait"):
        proof.signal("M_V", 0, 4)
    with pytest.raises(ValueError, match="unmatched or stale"):
        proof.wait("M_V", 0, 2)
    proof.wait("M_V", 0, 3)
    with pytest.raises(ValueError, match="generation reused"):
        proof.signal("M_V", 0, 3)
    with pytest.raises(ValueError, match="unmatched or stale"):
        proof.wait("M_V", 0, 3)


@pytest.mark.parametrize("direction,event_id", [("invented", 0), ("M_V", 8), ("M_V", -1)])
def test_target_event_limits(direction, event_id):
    with pytest.raises(ValueError, match="invalid hardware event"):
        make_ownership().signal(direction, event_id, 0)


def test_direction_has_independent_event_pool():
    proof = make_ownership()
    proof.signal("M_V", 0, 0)
    proof.signal("V_M", 0, 1)
    proof.wait("V_M", 0, 1)
    proof.wait("M_V", 0, 0)
    proof.drain()


@pytest.mark.parametrize("live", [True, False])
def test_incomplete_drain_rejected(live):
    proof = make_ownership()
    if live:
        proof.acquire("a", 0)
    else:
        proof.signal("M_V", 0, 0)
    with pytest.raises(ValueError, match="drain"):
        proof.drain()


def test_double_co1_cannot_be_inferred_from_pair_products():
    proof = make_ownership()
    proof.acquire("a", 0)
    with pytest.raises(ValueError, match="reuse"):
        proof.acquire("a", 1)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "none", "bool", "negative", "overcommit"])
def test_rank_envelope_fails_closed(mutation):
    values = dict.fromkeys(memory.RANK_COMPONENTS, 10)
    if mutation == "missing":
        del values["graphs"]
    elif mutation == "unknown":
        values["unaccounted"] = 0
    elif mutation == "none":
        values["hccl"] = None
    elif mutation == "bool":
        values["graphs"] = True
    elif mutation == "negative":
        values["mamba_archive"] = -1
    elif mutation == "overcommit":
        values["new_native_resources"] = 1000
    with pytest.raises(ValueError):
        memory.rank_envelope(1000, values, 32)


def test_rank_envelope_includes_simultaneous_old_new_and_shadow():
    values = dict.fromkeys(memory.RANK_COMPONENTS, 0)
    values.update(
        old_native_resources=200, new_native_resources=200, shadow_comparison=300, verification_scratch=100, hccl=50
    )
    assert memory.rank_envelope(1000, values, 100)["headroom_bytes"] == 50
    with pytest.raises(ValueError):
        memory.rank_envelope(None, values)


def test_route_workspace_is_only_moe_slice_and_keeps_gm_boundary():
    value = memory.route_workspace(2560, 10)
    assert value["rows"] == 25600
    assert value["components"]["projected_gate_up_fp16"] == 25600 * 1280 * 2
    assert value["components"]["routed_output_fp16"] == 25600 * 2560 * 2
    assert value["total_bytes"] == sum(value["components"].values())
    assert value["measured_bus_bytes"] is False
    allocated = memory.route_regions(2560, 10)
    usage = memory.validate_regions(allocated, (("GM", value["total_bytes"]),))
    assert usage == {"GM": value["total_bytes"]}
    assert "swiglu_hidden_fp16" in value["components"]
    for region in allocated:
        assert region.lifetime == "entire routed MoE call"


def test_small_gm_arena_alignment_includes_padding():
    allocated = memory.route_regions(1, 1)
    logical_bytes = memory.route_workspace(1, 1)["total_bytes"]
    assert allocated[-1].end >= logical_bytes
    assert all(region.offset % 32 == 0 and region.nbytes % 32 == 0 for region in allocated)
    with pytest.raises(ValueError, match="overcommit"):
        memory.validate_regions(allocated, (("GM", allocated[-1].end - 1),))


@pytest.mark.parametrize("tokens,top_k", [(0, 1), (2561, 1), (1, 11), (1, 0), (True, 1)])
def test_route_workspace_rejects_unsupported_bounds(tokens, top_k):
    with pytest.raises(ValueError):
        memory.route_workspace(tokens, top_k)


def test_rejected_all_group_materialization_exceeds_ub():
    assert dict(memory.CAPACITIES)["UB"] < memory.MAX_GROUPS * 2 * memory.M * memory.N * 4
    assert dict(memory.CAPACITIES)["UB"] < 2 * 2 * 32 * 320 * 4 * 2
    assert "surviving projected GM boundary" in memory.contract()["epilogue"]["qualified_reference"]


def test_generated_header_compiles_and_indexes_both_limbs(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host g++ unavailable; C++ ABI compile deferred")
    source = tmp_path / "check.cpp"
    source.write_text(
        '#include "qwen_streaming_contract.h"\n#include <cassert>\n'
        "int main() {\nusing namespace qwen_streaming;\n"
        "static_assert(M==16 && N==128 && GROUP==128 && K0==64 && SLOTS==2);\n"
        "bool seen[2*M*N] = {};\n"
        "for(unsigned s=0;s<N/BLOCK;++s)for(unsigned l=0;l<2;++l)\n"
        "for(unsigned r=0;r<M;++r)for(unsigned c=0;c<BLOCK;++c){\n"
        "auto i=ProductIndex(s,l,r,c);assert(i<2*M*N && !seen[i]);seen[i]=true;}\n"
        "for(bool item:seen)assert(item);\n"
        "for(unsigned g=0;g<20;++g)assert(EventId(Slot(g))==g%2);\n}\n"
    )
    result = subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT / "tools/qwen4exp"), str(source), "-o", str(tmp_path / "check")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(tmp_path / "check")], check=True)
