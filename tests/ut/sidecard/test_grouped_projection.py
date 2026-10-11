# SPDX-License-Identifier: Apache-2.0
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sidecard import batch_projection_gate, device_backend
from tools.sidecard.device_backend import CudaOracle, expert_tiles


@pytest.mark.parametrize(
    "experts,expected",
    [
        ((), ()),
        ((5,), ((0, 1),)),
        ((5,) * 17, ((0, 16), (16, 1))),
        ((1, 1, 3, 1, 1), ((0, 2), (2, 1), (3, 2))),
        ((0,) * 32 + (143,), ((0, 16), (16, 16), (32, 1))),
    ],
)
def test_tiles_cover_routes_once_without_mixing_experts(experts, expected):
    assert expert_tiles(experts) == expected
    covered = []
    for start, count in expert_tiles(experts):
        assert 1 <= count <= 16
        assert len(set(experts[start : start + count])) == 1
        covered.extend(range(start, start + count))
    assert covered == list(range(len(experts)))


@pytest.mark.parametrize("rows,available,grouped", [(31, True, False), (32, True, True), (32, False, False)])
def test_decode_and_legacy_library_use_scalar_path(rows, available, grouped):
    oracle = CudaOracle.__new__(CudaOracle)
    oracle.max_routes = 64
    oracle.buffers = {
        name: name
        for name in (
            "input",
            "ids",
            "input_q",
            "input_scales",
            "gate_up",
            "hidden",
            "hidden_q",
            "hidden_scales",
            "output",
            "expert_tiles",
        )
    }
    oracle.bank_pointers = {3: ("gate", "gs", "down", "ds")}
    calls, uploads = [], []
    oracle.launch = lambda *args: calls.append(("scalar", args))
    oracle.split_launch = None
    oracle.grouped_launch = (lambda *args: calls.append(("grouped", args))) if available else None
    oracle.upload = lambda pointer, body: uploads.append((pointer, body))
    oracle.cuda = SimpleNamespace(check=lambda result: None)
    experts = (0,) * 16 + (143,) * (rows - 16)
    oracle.project_bytes(3, bytes(rows * 4096 * 2), experts, 4)
    assert calls[0][0] == ("grouped" if grouped else "scalar")
    if grouped:
        assert uploads[-1] == ("expert_tiles", struct.pack("<4i", 0, 16, 16, 16))
        assert calls[0][1][-6:] == ("expert_tiles", 32, 4096, 2048, 4, 2)


def test_grouped_path_still_rejects_oversized_route_payload():
    oracle = CudaOracle.__new__(CudaOracle)
    oracle.max_routes = 32
    with pytest.raises(ValueError, match="geometry"):
        oracle.project_bytes(3, bytes(33 * 4096 * 2), (0,) * 33, 4)


@pytest.mark.parametrize("split,expected", [(True, "split"), (False, "scalar")])
def test_short_decode_uses_bounded_split_scratch(split, expected):
    oracle = CudaOracle.__new__(CudaOracle)
    oracle.max_routes = 32
    names = (
        "input",
        "ids",
        "input_q",
        "input_scales",
        "gate_up",
        "hidden",
        "hidden_q",
        "hidden_scales",
        "output",
        "split_groups",
    )
    oracle.buffers = dict(zip(names, names, strict=True))
    oracle.bank_pointers = {3: ("gate", "gs", "down", "ds")}
    calls = []
    oracle.launch = lambda *args: calls.append("scalar")
    oracle.split_launch = (lambda *args: calls.append("split")) if split else None
    oracle.grouped_launch = None
    oracle.upload = lambda *args: None
    oracle.cuda = SimpleNamespace(check=lambda result: None)
    oracle.project_bytes(3, bytes(4 * 4096 * 2), (0, 2, 8, 143), 4)
    assert calls == [expected]


def test_sparse_experts_skip_tensor_core_padding():
    oracle = CudaOracle.__new__(CudaOracle)
    oracle.max_routes = 64
    names = ("input", "ids", "input_q", "input_scales", "gate_up", "hidden", "hidden_q", "hidden_scales", "output")
    oracle.buffers = dict(zip(names, names, strict=True))
    oracle.bank_pointers = {3: ("gate", "gs", "down", "ds")}
    calls = []
    oracle.launch = lambda *args: calls.append("scalar")
    oracle.grouped_launch = lambda *args: calls.append("grouped")
    oracle.split_launch = None
    oracle.upload = lambda *args: None
    oracle.cuda = SimpleNamespace(check=lambda result: None)
    oracle.project_bytes(3, bytes(32 * 4096 * 2), tuple(range(32)), 4)
    assert calls == ["scalar"]


@pytest.mark.parametrize("mismatch", [False, True])
def test_paired_gate_restores_kernel_selection_after_failure(monkeypatch, mismatch):
    monkeypatch.setattr(batch_projection_gate, "SHORT_ROUTES", (1, 32))
    grouped, split = object(), object()
    oracle = SimpleNamespace(grouped_launch=grouped, split_launch=split)
    oracle.compute_bytes = lambda *args: b"changed" if mismatch and oracle.split_launch is split else b"original"
    if mismatch:
        with pytest.raises(AssertionError, match="bitwise"):
            batch_projection_gate.short_projection_cases(oracle, 3, np.random.default_rng(0))
    else:
        cases = batch_projection_gate.short_projection_cases(oracle, 3, np.random.default_rng(0))
        assert {(case["routes"], case["activation_bits"]) for case in cases} == {(1, 4), (1, 8), (32, 4), (32, 8)}
        assert all(case["bitwise_equal"] for case in cases)
    assert (oracle.grouped_launch, oracle.split_launch) == (grouped, split)


@pytest.mark.parametrize("layout,selection", [("row_major_v1", True), ("k32_nibbles_n_v1", 1)])
def test_experimental_selection_is_rejected_before_allocating_cuda(monkeypatch, layout, selection):
    monkeypatch.setattr(device_backend, "digest_file", lambda path: "pinned")
    monkeypatch.setattr(device_backend, "CudaCopies", lambda *args: pytest.fail("invalid configuration allocated CUDA"))
    with pytest.raises(ValueError, match="explicit packed selection"):
        CudaOracle(
            {},
            dict(library="unused", library_sha256="pinned", weight_layout=layout, experimental_projection=selection),
            1,
        )
