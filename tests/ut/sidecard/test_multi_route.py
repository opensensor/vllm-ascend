# SPDX-License-Identifier: Apache-2.0
"""Portable ownership, arithmetic-order, source-patch, and admission gates."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.multi_route_probe import raw_reference
from tools.sidecard.native_multi_route import adapter, build
from tools.sidecard.native_multi_route.contract import ordered_reduce, route_cases, route_order, workspace_bytes


@pytest.mark.parametrize("rank", (0, 1))
def test_all_owned_counts_and_every_route_position(rank):
    cases = route_cases(rank)
    counts = {len(route_order(case["ids"], case["weights"], rank)) for case in cases}
    assert counts == set(range(9))
    for slot in range(8):
        case = next(case for case in cases if case["name"] == f"single-slot-{slot}")
        assert route_order(case["ids"], case["weights"], rank) == (slot,)
    case = next(case for case in cases if case["name"] == "duplicate-eight")
    assert route_order(case["ids"], case["weights"], rank) == (0, 1, 3, 4, 5, 6, 7)


@pytest.mark.parametrize("rank", (0, 1))
def test_numerical_reduction_retains_duplicate_weighting_and_skips_poison(rank):
    owner, peer = rank * 144, (1 - rank) * 144
    ids = [owner + 2, peer, owner + 2, owner, peer + 143, owner + 2, owner + 143, owner]
    weights = [0.5, 0.75, 0.0, 1.25, 0.25, 0.125, 0.0, 0.875]
    values = torch.full((8, 3), float("nan"))
    values[0], values[3], values[5], values[7] = (torch.tensor([v, v * 2, -v]) for v in (2.0, 4.0, 8.0, 16.0))
    assert torch.equal(ordered_reduce(values, ids, weights, rank), torch.tensor([21.0, 42.0, -21.0]))
    assert torch.equal(ordered_reduce(values, ids, [0.0] * 8, rank), torch.zeros(3))


def test_expert_major_order_is_numerically_observable():
    ids, weights = [2, 0, 1, 144, 144, 144, 144, 144], [1.0] * 8
    rows = torch.full((8, 1), float("nan"))
    rows[:3, 0] = torch.tensor([1.0, 65504.0, -65504.0])
    # Half rows, independent FP32 scale multiply; expert-major adds cancellation
    # before the tiny weighted contribution, unlike original top-k order.
    weights[1] = weights[2] = 1024.0
    assert ordered_reduce(rows, ids, weights, 0).item() == 1.0
    assert ((rows[0].float() + rows[1].float() * 1024) + rows[2].float() * 1024).item() == 0.0


@pytest.mark.parametrize(
    "ids,weights,rank",
    (
        ([0] * 7, [1.0] * 8, 0),
        ([288] * 8, [1.0] * 8, 0),
        ([True] * 8, [1.0] * 8, 0),
        ([0] * 8, [float("nan")] * 8, 0),
        ([0] * 8, [float("inf")] * 8, 0),
        ([0] * 8, [True] * 8, 0),
        ([0] * 8, [1.0] * 8, True),
        ([0] * 8, [1.0] * 8, 2),
    ),
)
def test_route_contract_rejects_invalid_input(ids, weights, rank):
    with pytest.raises(ValueError):
        route_order(ids, weights, rank)


@pytest.mark.parametrize("relative", tuple(build.PATCHES))
def test_threshold_patch_is_exact_and_rejects_drift(relative):
    before, after = build.PATCHES[relative]
    source = "prefix\n" + before + "\nsuffix\n"
    assert build.patch_text(relative, source) == source.replace(before, after)
    for bad in (source.replace(before, after), source + before):
        with pytest.raises(ValueError, match="patch anchor"):
            build.patch_text(relative, bad)


def test_build_abi_defines_and_workspace_budget(tmp_path):
    commands = build.commands(tmp_path / "build", Path("/pinned/cann"))
    assert len(commands) == 8  # host compiler plus seven real AI-Core binaries
    down = [command for command in commands if "glm_fused_down" in command[2]]
    assert len(down) == 2
    for command in down:
        assert "-DGLM_FP16_ROUTE_WORKSPACE" in command
        assert "-DGLM_NATIVE_ROUTE_COLUMNS" in command
        assert "-DGLM_BULK_ROUTE_STORE" in command
        assert "-DGLM_FUSED_GATE_UP" not in command
    assert any("-DGLM_STATIC_WEIGHT_BITS=4" in command for command in commands)
    sizes = workspace_bytes()
    assert sizes["route_values"] == 65536
    assert sizes["input_low"] == 4096
    assert sizes["route_order"] == 64
    assert sizes["route_ends"] == 1152
    assert sizes["routed_input"] == 145 * 64 * 2048


def test_original_authentication_fails_closed_before_import_or_compilation(tmp_path):
    (tmp_path / "provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="original native provenance"):
        build.authenticate(tmp_path, tmp_path / "untrusted.cpp")


def candidate_fixture(root):
    package = root / build.HELPER_PACKAGE
    package.mkdir()
    (package / "__init__.py").write_text("")
    binaries = {}
    for name in (
        "pack",
        "route_input",
        "reduce",
        "gate_up",
        "gate_up_w4",
        "down",
        "down_w4",
    ):
        path = root / f"glm_fused_{name}.bin"
        path.write_bytes(b"test fixture binary, never executed")
        binaries[path.name] = build.digest(path)
    for name in ("candidate.patch", "glm_reconstruction_bridge_v1140.so"):
        (root / name).write_bytes(b"test receipt")
    provenance = {
        "_build": {"helper_package": build.HELPER_PACKAGE, "namespace": "glm_reconstruction_v1140", "version": 1140},
        "_helpers": {"__init__.py": build.digest(package / "__init__.py")},
    }
    (root / "provenance.json").write_text(json.dumps(provenance))
    manifest = dict(
        schema_version=1,
        candidate="multi-route-r1-fixed-slots-v1",
        original_provenance_sha256=build.ORIGINAL_PROVENANCE,
        clipped_source_sha256=build.CLIPPED_SOURCE,
        build_complete=True,
        bridge_namespace="glm_reconstruction_v1140",
        swiglu_limit=10.0,
        activation_bits=4,
        rows=1,
        top_k=8,
        binaries=binaries,
        source_sha256={build.HELPER_PACKAGE + "/__init__.py": build.digest(package / "__init__.py")},
        patch_sha256=build.digest(root / "candidate.patch"),
        provenance_sha256=build.digest(root / "provenance.json"),
        bridge_binary_sha256=build.digest(root / "glm_reconstruction_bridge_v1140.so"),
    )
    receipt = root / "multi-route-build.json"
    receipt.write_text(json.dumps(manifest))
    return manifest, receipt


@pytest.mark.parametrize(
    "name", ("glm_fused_down.bin", "candidate.patch", "glm_reconstruction_bridge_v1140.so", "provenance.json")
)
def test_candidate_admission_detects_changed_payload(tmp_path, name, monkeypatch):
    _, receipt = candidate_fixture(tmp_path)
    monkeypatch.setattr(adapter, "BRIDGE_SHA256", build.digest(tmp_path / "glm_reconstruction_bridge_v1140.so"))
    checksum = build.digest(receipt)
    adapter.authenticate_candidate(tmp_path, checksum)
    (tmp_path / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="candidate file changed"):
        adapter.authenticate_candidate(tmp_path, checksum)


@pytest.mark.parametrize(
    "field,value",
    (
        ("build_complete", False),
        ("activation_bits", 8),
        ("rows", 17),
        ("swiglu_limit", 0.0),
        ("original_provenance_sha256", "0" * 64),
    ),
)
def test_candidate_admission_rejects_incompatible_contract(tmp_path, field, value, monkeypatch):
    manifest, receipt = candidate_fixture(tmp_path)
    monkeypatch.setattr(adapter, "BRIDGE_SHA256", build.digest(tmp_path / "glm_reconstruction_bridge_v1140.so"))
    manifest[field] = value
    receipt.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="math/ABI/provenance"):
        adapter.authenticate_candidate(tmp_path, build.digest(receipt))


def test_probe_help_is_device_dependency_free():
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard.multi_route_probe", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert all(option in result.stdout for option in ("--config", "--output", "--rank"))


@pytest.mark.parametrize("rank", (0, 1))
def test_adapter_keeps_one_real_row_and_masks_peer_contributions(rank):
    calls = []

    class Owner:
        def __call__(self, *args):
            calls.append(("fallback", args))
            return "qualified"

    owner = Owner()
    owner.rank, owner.torch, owner.banks = rank, torch, {18: (object(), object(), object(), object())}
    candidate = object.__new__(adapter.MultiRouteDecode)
    candidate.actor, candidate.torch = owner, torch

    def native(*args):
        calls.append(("candidate", args))
        return "unpadded"

    candidate.native = native
    for case in route_cases(rank):
        hidden = torch.zeros((1, 4096), dtype=torch.float16)
        ids, weights = torch.tensor([case["ids"]]), torch.tensor([case["weights"]])
        assert candidate(18, hidden, ids, weights) == "unpadded"
        kind, args = calls[-1]
        assert kind == "candidate" and args[0] is hidden and args[0].shape == (1, 4096)
        assert all(actual is expected for actual, expected in zip(args[1:5], owner.banks[18], strict=True))
        assert args[-1] == rank * 144
        masked_weights, masked_ids = args[-3:-1]
        owned = (ids >= rank * 144) & (ids < (rank + 1) * 144)
        assert torch.equal(masked_weights, torch.where(owned, weights, 0))
        assert masked_ids.dtype == torch.int32
        assert set(route_order(masked_ids[0].tolist(), masked_weights[0].tolist(), rank)) == set(
            route_order(case["ids"], case["weights"], rank)
        )
    assert candidate(18, hidden.expand(4, -1), ids.expand(4, -1), weights.expand(4, -1)) == "qualified"
    assert calls[-1][0] == "fallback"


@pytest.mark.parametrize("rank", (0, 1))
def test_raw_reference_submits_each_live_duplicate_separately(rank):
    case = next(case for case in route_cases(rank) if case["name"] == "duplicate-peer-zero")
    submitted = []

    def native(hidden, *args):
        weights, ids, offset = args[-3:]
        expert = int(ids[0, 0])
        assert weights.shape == (1, 1) and weights[0, 0] == 1 and offset == rank * 144
        submitted.append(expert)
        return torch.full((1, 4096), float(expert - offset + 1))

    candidate = SimpleNamespace(torch=torch, actor=SimpleNamespace(rank=rank, banks={18: (None,) * 4}), native=native)
    actual = raw_reference(candidate, 18, torch.zeros((1, 4096)).half(), case["ids"], case["weights"])
    assert submitted == [case["ids"][slot] for slot in route_order(case["ids"], case["weights"], rank)]
    assert submitted.count(rank * 144) == 2
    expected = 0.125 + 0.875 + 144 * (1.375 + 0.625)
    assert torch.equal(actual, torch.full((1, 4096), expected))
