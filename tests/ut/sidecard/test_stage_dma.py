# SPDX-License-Identifier: Apache-2.0
"""Complete-state layout and replay/admission checks without accelerator imports."""

import copy

import pytest
import torch

from tools.sidecard.layer_math import mhc_post, round_state
from tools.sidecard.stage_dma import CAPACITY, pack, validate_packet, views
from tools.sidecard.stage_options import enabled
from tools.sidecard.stage_wire import MAX_BODY_BYTES, descriptors, metadata


def supported_spec():
    return {"checkpoint_sha256": "a" * 64, "max_context": 131072}


@pytest.mark.parametrize("rows", [1, 4, 32])
def test_complete_state_roundtrip_and_bounded_slot(rows):
    packed = torch.full((CAPACITY,), 0xD3, dtype=torch.uint8)
    fields, size = descriptors(rows)
    source = tuple(
        torch.arange(torch.tensor(f["shape"]).prod()).reshape(f["shape"]).to(dtype)
        for f, dtype in zip(
            fields, [torch.float16, torch.float32, torch.float32, torch.float32, torch.int64], strict=True
        )
    )
    pack(packed, source[:4], source[-1], torch)
    result = views(packed, rows, torch)
    assert size <= MAX_BODY_BYTES and size < CAPACITY
    for expected, actual in zip(source, result, strict=True):
        assert torch.equal(expected, actual)
    assert bool((packed[size:] == 0xD3).all())


@pytest.mark.parametrize("change", ["dtype", "shape"])
def test_pack_rejects_incomplete_state(change):
    packed = torch.empty(CAPACITY, dtype=torch.uint8)
    state = list(views(packed.clone(), 1, torch))
    state[1] = state[1].half() if change == "dtype" else state[1][:, :3]
    with pytest.raises(ValueError):
        pack(packed, state[:4], state[4], torch)


def test_metadata_admits_full_context_boundary():
    packet = metadata(32, 7, supported_spec()["checkpoint_sha256"], request_context_limit=131072)
    assert validate_packet(packet, supported_spec(), 7, 131040, 131072) == 131072


@pytest.mark.parametrize("mutation", ["sequence", "checkpoint", "offset", "reservation", "position", "extra"])
def test_metadata_rejects_replay_geometry_and_cache_changes(mutation):
    packet = copy.deepcopy(metadata(4, 7, supported_spec()["checkpoint_sha256"], request_context_limit=8192))
    position = 100
    if mutation == "sequence":
        packet["sequence"] = 6
    elif mutation == "checkpoint":
        packet["checkpoint_sha256"] = "b" * 64
    elif mutation == "offset":
        packet["fields"][1]["offset"] += 8
    elif mutation == "reservation":
        packet["request_context_limit"] = 4096
    elif mutation == "position":
        position = 131072
    else:
        packet["first_position"] = position
    with pytest.raises(ValueError):
        validate_packet(packet, supported_spec(), 7, position, 8192)


@pytest.mark.parametrize(
    "spec", [{"stage_transport": "bad"}, {"npu_native_bf16_round": 1}, {"npu_native_bf16_round": True}]
)
def test_options_reject_unqualified_native_binding(spec):
    with pytest.raises(ValueError):
        enabled(spec)


@pytest.mark.parametrize("rows,width", [(1, 4096), (1, 4), (4, 4096), (32, 4096)])
def test_native_round_only_large_single_row_and_preserves_range(rows, width):
    calls = []
    value = torch.full((rows, width), 100000.0)

    def native(x):
        calls.append(x.shape)
        return x.bfloat16().float()

    actual = round_state(value, torch, native)
    assert torch.equal(actual, value.bfloat16().float())
    assert len(calls) == int(rows == 1 and width >= 4096)
    assert bool((actual > 65504).all())


def test_native_post_matches_orientation_without_fp16_residual_conversion():
    residual = torch.arange(4 * 4096, dtype=torch.float32).reshape(1, 4, 4096) * 20
    comb = torch.eye(4)[None]
    x = torch.ones(1, 4096).half()
    post = torch.ones(1, 4, 1)
    calls = []

    def native(value):
        calls.append(value)
        return value.bfloat16().float()

    actual = mhc_post(x, residual, post, comb, torch, native)
    assert torch.equal(actual, (residual + 1).bfloat16().float())
    assert len(calls) == 1


def close_fixture(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tools.sidecard import stage_dma

    trace = []
    owner = object.__new__(stage_dma.StageDMA)
    owner.closed, owner.output, owner.nonce = False, tmp_path, "epoch"
    owner.registered = {0, 1}
    owner.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: trace.append("cuda-drained")))

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            trace.append(self.name)

    owner.source_event = Resource("event-closed")
    owner.exports = [Resource("export-closed") for _ in range(2)]
    owner.allocations = [Resource("allocation-freed") for _ in range(2)]
    owner.buffers, owner.logs = [1, 2], []
    owner.connections = [
        SimpleNamespace(settimeout=lambda _: None, close=lambda: trace.append("socket-closed")) for _ in range(2)
    ]
    owner.processes = [SimpleNamespace(wait=lambda **kwargs: trace.append("worker-exited")) for _ in range(2)]
    monkeypatch.setattr(stage_dma, "send_control", lambda conn, value: trace.append(next(iter(value))))
    monkeypatch.setattr(
        stage_dma, "receive_control", lambda conn: trace.append("mapping-released") or {"closed": "epoch"}
    )
    return owner, trace


def test_close_drains_foreign_mappings_before_free_and_is_idempotent(monkeypatch, tmp_path):
    owner, trace = close_fixture(monkeypatch, tmp_path)
    owner.close()
    assert trace.count("mapping-released") == 2
    assert max(i for i, x in enumerate(trace) if x == "mapping-released") < trace.index("allocation-freed")
    assert trace.index("allocation-freed") < trace.index("released")
    assert trace.index("cuda-drained") < trace.index("close")
    previous = trace[:]
    owner.close()
    assert trace == previous


def test_missing_quiescence_preserves_exports_and_allocations(monkeypatch, tmp_path):
    from tools.sidecard import stage_dma

    owner, trace = close_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(stage_dma, "receive_control", lambda conn: {"closed": "wrong-epoch"})

    def retained(*args):
        raise RuntimeError("owner retained")

    monkeypatch.setattr(stage_dma, "preserve", retained)
    with pytest.raises(RuntimeError, match="owner retained"):
        owner.close()
    assert "allocation-freed" not in trace and "export-closed" not in trace
    assert not owner.closed


@pytest.mark.parametrize("rows", [33, 64, 512])
def test_complete_dma_rejects_larger_wire_admission_before_device_work(rows):
    packet = metadata(rows, 7, supported_spec()["checkpoint_sha256"], request_context_limit=8192)
    with pytest.raises(ValueError, match="dedicated slot"):
        validate_packet(packet, supported_spec(), 7, 100, 8192)
    with pytest.raises(ValueError, match="fixed slot"):
        views(torch.empty(CAPACITY, dtype=torch.uint8), rows, torch)
