# SPDX-License-Identifier: Apache-2.0
"""CPU protocol/byte tests; real HCCL gates are separate hardware evidence."""

import copy
import importlib.util
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch


def transport_class():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/device_communicators/pipeline_all_gather.py"
    spec = importlib.util.spec_from_file_location("pipeline_transport_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PipelineAllGatherTransport


class Network:
    def __init__(self):
        self.condition = threading.Condition()
        self.messages = {}
        self.collectives = {}
        self.members = set()


class Backend:
    def __init__(self, network, rank):
        self.network = network
        self.rank = rank
        self.rounds = {}
        self.destroyed = []
        self.network.members.add(rank)
        self.control_round = 0

    def new_group(self, members, *, backend, use_local_synchronization):
        assert backend == "hccl" and use_local_synchronization
        return tuple(members)

    def send_object_list(self, objects, *, dst, group):
        with self.network.condition:
            mailbox = self.network.messages.setdefault((self.rank, dst), queue.Queue())
        mailbox.put(copy.deepcopy(objects))

    def recv_object_list(self, objects, *, src, group):
        with self.network.condition:
            mailbox = self.network.messages.setdefault((src, self.rank), queue.Queue())
        objects[:] = mailbox.get(timeout=10)

    def all_gather_into_tensor(self, output, tensor, *, group):
        ordinal = self.rounds.get(group, 0)
        self.rounds[group] = ordinal + 1
        key = (group, ordinal)
        with self.network.condition:
            rows = self.network.collectives.setdefault(key, {})
            rows[self.rank] = (output, tensor.clone())
            if len(rows) == len(group):
                result = torch.cat([rows[rank][1] for rank in group])
                for destination, _ in rows.values():
                    destination.copy_(result)
                self.network.condition.notify_all()
            else:
                assert self.network.condition.wait_for(lambda: len(rows) == len(group), timeout=10)

    def destroy_process_group(self, group):
        self.destroyed.append(group)

    def barrier(self, *, group):
        key = ("control", group, self.control_round)
        self.control_round += 1
        with self.network.condition:
            arrived = self.network.collectives.setdefault(key, set())
            arrived.add(self.rank)
            self.network.condition.notify_all()
            assert self.network.condition.wait_for(lambda: arrived == self.network.members, timeout=10)


@pytest.mark.parametrize("ranks", [(0, 2, 4), (4, 0, 2), (3, 1)])
@pytest.mark.parametrize("tokens", [1, 17, 640])
def test_pipeline_packet_bytes_and_live_storage(ranks, tokens):
    cls = transport_class()
    network = Network()
    backends = [Backend(network, rank) for rank in ranks]
    transports = [cls("cpu-control", ranks, rank, "cpu", backend) for rank, backend in zip(ranks, backends)]
    generator = torch.Generator().manual_seed(tokens)
    packet = {
        "hidden_states": torch.randn(tokens, 8, generator=generator, dtype=torch.float16),
        "residual": torch.randn(tokens, 4, 8, generator=generator),
        "post": torch.randn(tokens, 4, 1, generator=generator),
        "comb": torch.randn(tokens, 4, 4, generator=generator),
        "empty": torch.empty(0, 8),
        "control": {"request": "first", "finished": False},
        "none": None,
    }
    packet["residual"].reshape(-1)[:2] = torch.tensor([-0.0, float("nan")])
    # A strided sender must become a contiguous packet without changing values.
    packet["comb"] = packet["comb"].transpose(1, 2)
    second = {**packet, "hidden_states": packet["hidden_states"] + 1, "control": {"request": "second"}}
    received = []

    def worker(index):
        transports[index].warmup()
        for original in (packet, second):
            value = original if index == 0 else transports[index].recv_tensor_dict()
            if index < len(ranks) - 1:
                transports[index].send_tensor_dict(value)
            else:
                received.append(value)

    with ThreadPoolExecutor(max_workers=len(ranks)) as executor:
        futures = [executor.submit(worker, index) for index in range(len(ranks))]
        for future in futures:
            future.result(timeout=20)
    for expected, actual in zip((packet, second), received):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert actual[key].dtype == value.dtype and actual[key].shape == value.shape
                assert torch.equal(
                    actual[key].reshape(-1).view(torch.int8), value.contiguous().reshape(-1).view(torch.int8)
                )
            else:
                assert actual[key] == value
    assert received[0]["hidden_states"].data_ptr() != received[1]["hidden_states"].data_ptr()
    for transport, backend in zip(transports, backends):
        count = len(transport.groups)
        transport.destroy()
        transport.destroy()
        assert len(backend.destroyed) == count


@pytest.mark.parametrize("ranks,rank", [((0,), 0), ((0, 0), 0), ((0, 1), 2)])
def test_invalid_pipeline_members_rejected(ranks, rank):
    with pytest.raises(ValueError, match="distinct ranks"):
        transport_class()(None, ranks, rank, "cpu", Backend(Network(), rank))


@pytest.mark.parametrize("peer", [-1, 4, 0, True, 2])
def test_invalid_or_nonadjacent_peer_rejected(peer):
    transport = transport_class()(None, (0, 1, 2, 3), 0, "cpu", Backend(Network(), 0))
    with pytest.raises(ValueError, match="peer|adjacent"):
        transport.send_tensor_dict({}, peer)


def test_pipeline_does_not_warm_or_use_the_bootstrap_closing_edge():
    transport = transport_class()(None, (0, 2, 4), 0, "cpu", Backend(Network(), 0))
    assert transport.pairs == ((0, 2), (2, 4))
    with pytest.raises(ValueError, match="adjacent"):
        transport.send_tensor_dict({}, 2)


def test_world_order_includes_nonmembers_and_both_tp_lanes():
    cls = transport_class()
    records = []

    class WorldBackend:
        def __init__(self):
            self.calls = []

        def new_group(self, members, *, backend, use_local_synchronization):
            assert backend == "hccl" and use_local_synchronization is False
            self.calls.append(tuple(members))
            return tuple(members)

    for rank in range(6):
        backend = WorldBackend()
        transport = cls(None, tuple(range(rank % 2, 6, 2)), rank, "cpu", backend, world_size=6)
        records.append(backend.calls)
        assert all(rank in pair for pair in transport.groups)
    assert records == [[(0, 2), (0, 4), (1, 3), (1, 5), (2, 4), (3, 5)]] * 6


@pytest.mark.parametrize("ranks,world", [((0, 2, 5), 6), ((4, 2, 0), 6), ((0, 2, 4), 7)])
def test_world_order_rejects_inconsistent_layout_before_group_creation(ranks, world):
    with pytest.raises(ValueError, match="pipeline"):
        transport_class()(None, ranks, ranks[0], "cpu", Backend(Network(), ranks[0]), world_size=world)
