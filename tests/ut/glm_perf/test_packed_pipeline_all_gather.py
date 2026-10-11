# SPDX-License-Identifier: Apache-2.0
"""Real CPU bytes over a synchronized multi-rank collective simulator."""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest
import torch
from test_pipeline_all_gather import Backend, Network


def candidate_module():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/device_communicators"
    # Avoid package imports that register the NPU plugin in a CPU protocol test.
    base_name = "vllm_ascend.distributed.device_communicators.pipeline_all_gather"
    old = sys.modules.get(base_name)
    spec = importlib.util.spec_from_file_location(base_name, root / "pipeline_all_gather.py")
    base = ModuleType(base_name)
    spec.loader.exec_module(base)
    sys.modules[base_name] = base
    try:
        spec = importlib.util.spec_from_file_location(
            "packed_transport_under_test", root / "packed_pipeline_all_gather.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            del sys.modules[base_name]
        else:
            sys.modules[base_name] = old


@pytest.mark.parametrize("tokens", [1, 4, 640, 1280])
@pytest.mark.parametrize("ranks", [(0, 2, 4), (4, 0, 2)])
def test_mhc_packet_one_collective_and_independent_live_bytes(tokens, ranks):
    cls = candidate_module().PackedPipelineAllGatherTransport
    network = Network()
    backends = [Backend(network, rank) for rank in ranks]
    transports = [cls("control", ranks, rank, "cpu", backend) for rank, backend in zip(ranks, backends)]
    random = torch.Generator().manual_seed(309 + tokens)
    packet = {
        "hidden_states": torch.randn(tokens, 32, dtype=torch.float16, generator=random),
        "residual": torch.randn(tokens, 4, 32, generator=random),
        "post": torch.randn(tokens, 4, 1, generator=random),
        "comb": torch.randn(tokens, 4, 4, generator=random).transpose(1, 2),
        "odd_bytes": torch.arange(3, dtype=torch.uint8),
        "scalar": torch.tensor(4.25, dtype=torch.float64),
        "empty": torch.empty(2, 0, dtype=torch.int32),
        "control": {"request": "first"},
        "none": None,
    }
    packet["residual"].view(-1)[:2] = torch.tensor([-0.0, float("nan")])
    second = {**packet, "hidden_states": packet["hidden_states"] + 1}
    received = []

    def worker(index):
        for value in (packet, second):
            value = value if index == 0 else transports[index].recv_tensor_dict()
            if index < len(ranks) - 1:
                transports[index].send_tensor_dict(value)
            else:
                received.append(value)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(worker, index) for index in range(3)]:
            future.result(timeout=20)
    for expected, actual in zip((packet, second), received):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert actual[key].shape == value.shape and actual[key].dtype == value.dtype
                assert torch.equal(
                    actual[key].reshape(-1).view(torch.int8), value.contiguous().reshape(-1).view(torch.int8)
                )
            else:
                assert actual[key] == value
    assert received[0]["residual"].data_ptr() != received[1]["residual"].data_ptr()
    for backend in backends:
        # Two complete dictionaries require exactly two collectives per edge.
        assert all(rounds == 2 for rounds in backend.rounds.values())


@pytest.mark.parametrize(
    "header",
    [
        {"version": 9, "bytes": 32, "entries": []},
        {"version": 1, "bytes": -32, "entries": []},
        {"version": 1, "bytes": 33, "entries": []},
        {"version": 1, "bytes": 64, "entries": [("x", "tensor", torch.float32, (8,), 32, 32)]},
        {"version": 1, "bytes": 32, "entries": [("x", "tensor", torch.float32, (8,), 0, 31)]},
        {"version": 1, "bytes": 32, "entries": [("x", "tensor", torch.float32, (-8,), 0, 32)]},
        {"version": 1, "bytes": 32, "entries": [("x", "tensor", torch.float32, (8,), 0, 32), ("x", "value", None)]},
        {
            "version": 1,
            "bytes": 32,
            "entries": [("x", "tensor", torch.float32, (8,), 0, 32), ("y", "tensor", torch.int8, (1,), 0, 1)],
        },
    ],
)
def test_invalid_header_rejected_before_device_collective(header):
    with pytest.raises(ValueError):
        candidate_module().validate_header(header, 32)


def test_payload_budget_rejects_before_sending_metadata():
    network = Network()
    transport = candidate_module().PackedPipelineAllGatherTransport(
        None, (0, 1), 0, "cpu", Backend(network, 0), max_payload_bytes=32
    )
    with pytest.raises(ValueError, match="payload size"):
        transport.send_tensor_dict({"x": torch.ones(64)})
    assert not network.messages and not network.collectives


def test_ownership_upgrade_reuses_groups_and_only_one_owner_destroys_them():
    module = candidate_module()
    network = Network()
    backend = Backend(network, 0)
    original = module.PipelineAllGatherTransport(None, (0, 1), 0, "cpu", backend)
    groups = original.groups
    transferred = module.PackedPipelineAllGatherTransport.take_ownership(original)
    assert transferred.groups is groups and original.groups == {}
    original.destroy()
    assert not backend.destroyed
    transferred.destroy()
    assert len(backend.destroyed) == 1
    with pytest.raises(ValueError, match="owned"):
        module.PackedPipelineAllGatherTransport.take_ownership(original)


def test_control_only_packet_does_not_allocate_collective_rounds():
    module = candidate_module()
    network = Network()
    backends = [Backend(network, rank) for rank in (0, 1)]
    transports = [
        module.PackedPipelineAllGatherTransport(None, (0, 1), rank, "cpu", backend)
        for rank, backend in enumerate(backends)
    ]
    packet = {"done": True, "empty": torch.empty(0, dtype=torch.float16)}
    transports[0].send_tensor_dict(packet)
    received = transports[1].recv_tensor_dict()
    assert received["done"] and received["empty"].dtype == torch.float16
    assert not network.collectives and all(not backend.rounds for backend in backends)


def test_full_glm_hidden_width_packet_matches_the_offline_byte_ledger():
    module = candidate_module()
    network = Network()
    backends = [Backend(network, rank) for rank in (0, 1)]
    transports = [
        module.PackedPipelineAllGatherTransport(None, (0, 1), rank, "cpu", backend)
        for rank, backend in enumerate(backends)
    ]
    tokens, hidden = 640, 4096
    packet = {
        "hidden": torch.ones(tokens, hidden, dtype=torch.float16),
        "residual": torch.ones(tokens, 4, hidden, dtype=torch.float32),
        "post": torch.ones(tokens, 4, 1, dtype=torch.float32),
        "comb": torch.ones(tokens, 4, 4, dtype=torch.float32),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        sent = pool.submit(transports[0].send_tensor_dict, packet)
        received = pool.submit(transports[1].recv_tensor_dict).result(timeout=20)
        sent.result(timeout=20)
    for key in packet:
        assert torch.equal(packet[key], received[key])
    expected_bytes = tokens * (hidden * 2 + 4 * hidden * 4 + (4 + 16) * 4)
    retained_bytes = received["hidden"].untyped_storage().nbytes()
    # Pairwise AllGather owns both ranks' slices, even though fields view only
    # the sender's half. Account for the actual retained allocation.
    assert retained_bytes == 2 * expected_bytes
    assert all(all(rounds == 1 for rounds in backend.rounds.values()) for backend in backends)
