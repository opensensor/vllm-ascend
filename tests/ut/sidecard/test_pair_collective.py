# SPDX-License-Identifier: Apache-2.0
"""Adversarial stream scheduling checks; these are not NPU qualification."""

import random
from collections import deque

import pytest

from tools.sidecard.pair_collective import BOUNDARIES, MAX_PENDING, PairConfig, PairEndpoint, PairQuarantined
from tools.sidecard.pair_collective_acl import Bindings, _key


class Fabric:
    def __init__(self):
        self.queues = (deque(), deque())
        self.flags = ({"ready": False, "consumed": False}, {"ready": False, "consumed": False})
        self.inputs, self.inbox, self.output = [None, None], [None, None], [None, None]
        self.observed, self.calls = ([], []), ([], [])

    def run(self, seed=0):
        rng = random.Random(seed)
        while any(self.queues):
            ready = [r for r in range(2) if self.queues[r] and self.queues[r][0][0]()]
            if not ready:
                raise RuntimeError("simulated stream deadlock")
            rank = rng.choice(ready)
            _, action = self.queues[rank].popleft()
            action()


class Backend:
    def __init__(self, fabric, rank):
        self.fabric, self.rank, self.fail = fabric, rank, None

    def prepare(self, config, pid):
        self.config, self.peer_pid = config, pid
        return {"rank": self.rank}

    def connect(self, keys):
        assert keys["rank"] == 1 - self.rank

    def stage(self, values):
        f, rank = self.fabric, self.rank
        f.queues[rank].append((lambda: True, lambda: f.inputs.__setitem__(rank, tuple(values))))

    def enqueue(self, timeout_raw):
        f, rank, peer = self.fabric, self.rank, 1 - self.rank
        if self.fail:
            raise RuntimeError("injected partial native submission failure")

        def push():
            f.inbox[peer] = f.inputs[rank]

        def signal(name):
            assert not f.flags[peer][name], "binary Notify records coalesced"
            f.flags[peer][name] = True

        def reduce():
            other = f.inputs[peer] if self.config.exchange_mode == "remote-read" else f.inbox[rank]
            f.output[rank] = tuple(a + b for a, b in zip(f.inputs[rank], other, strict=True))

        queue = f.queues[rank]
        if self.config.exchange_mode == "push":
            queue.append((lambda: True, push))
        queue.append((lambda: True, lambda: signal("ready")))
        queue.append((lambda: f.flags[rank]["ready"], lambda: f.flags[rank].__setitem__("ready", False)))
        queue.append((lambda: True, reduce))
        queue.append((lambda: True, lambda: signal("consumed")))
        queue.append((lambda: f.flags[rank]["consumed"], lambda: f.flags[rank].__setitem__("consumed", False)))

    def consume(self):
        f, rank = self.fabric, self.rank
        f.queues[rank].append((lambda: True, lambda: f.observed[rank].append(f.output[rank])))

    def synchronize(self):
        assert not any(self.fabric.queues), "test must progress both streams before local drain"

    def release_imports(self):
        self.fabric.calls[self.rank].append("release imports")

    def close(self):
        self.fabric.calls[self.rank].append("close exports")


def connected(rows=1, exchange_mode="push"):
    fabric = Fabric()
    backends = tuple(Backend(fabric, rank) for rank in range(2))
    endpoints = tuple(
        PairEndpoint(PairConfig("a" * 64, rows, exchange_mode=exchange_mode), rank, backends[rank]) for rank in range(2)
    )
    exports = [e.prepare(123 + rank) for rank, e in enumerate(endpoints)]
    imports = [e.connect(exports[1 - rank]) for rank, e in enumerate(endpoints)]
    for rank, endpoint in enumerate(endpoints):
        endpoint.activate(imports[1 - rank])
    return fabric, backends, endpoints


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("rows", (1, 4, 32))
@pytest.mark.parametrize("exchange_mode", ("push", "remote-read"))
def test_repeated_exchange_with_skew_does_not_overwrite_inbox_or_coalesce_notify(seed, rows, exchange_mode):
    fabric, backends, endpoints = connected(rows, exchange_mode)
    # Queue a complete rank far ahead of its peer, then advance device tasks
    # under random scheduling. Real byte/numerical coverage is a hardware gate.
    for window in range(3):
        for rank in range(2):
            for index in range(30):
                sequence = window * 30 + index
                backends[rank].stage((sequence, rank, 2 * rank + sequence))
                endpoints[rank].submit(sequence, BOUNDARIES[index])
                backends[rank].consume()
        fabric.run(seed + window)
        for endpoint in endpoints:
            endpoint.synchronize()
    expected = [(2 * s, 1, 2 * s + 2) for s in range(90)]
    assert fabric.observed[0] == fabric.observed[1] == expected
    assert all(not any(flags.values()) for flags in fabric.flags)


def test_shutdown_requires_both_drains_then_import_release_before_export_free():
    fabric, _, endpoints = connected()
    drained = [e.drain() for e in endpoints]
    with pytest.raises(RuntimeError):
        endpoints[0].close(drained[1])
    closed = [e.release_imports(drained[1 - r]) for r, e in enumerate(endpoints)]
    for rank, endpoint in enumerate(endpoints):
        endpoint.close(closed[1 - rank])
        assert fabric.calls[rank] == ["release imports", "close exports"]
        assert endpoint.state == "closed"
        with pytest.raises(RuntimeError):
            endpoint.submit(0, BOUNDARIES[0])


def test_handshake_identity_mismatch_rejected_before_importing_peer_objects():
    fabric, backend, endpoints = connected()
    fresh = PairEndpoint(PairConfig("a" * 64, handshake="ping-pong"), 0, Backend(Fabric(), 0))
    packet = fresh.prepare(123)
    assert fresh.config.identity != endpoints[1].config.identity
    # A fresh counterpart is needed because connected() has already activated.
    other = PairEndpoint(PairConfig("a" * 64), 1, Backend(Fabric(), 1))
    other.prepare(124)
    with pytest.raises(ValueError, match="configuration"):
        other.connect(packet)
    assert other.state == "exported"


@pytest.mark.parametrize("field,value", (("rank", False), ("session", "b" * 64), ("next_sequence", 1)))
def test_stale_or_mismatched_shutdown_receipt_retains_owners(field, value):
    fabric, _, endpoints = connected()
    drained = [e.drain() for e in endpoints]
    drained[1][field] = value
    with pytest.raises(ValueError):
        endpoints[0].release_imports(drained[1])
    assert fabric.calls[0] == []


def test_failure_quarantines_without_free_or_fallback():
    fabric, backends, endpoints = connected()
    backends[0].fail = True
    with pytest.raises(PairQuarantined):
        endpoints[0].submit(0, BOUNDARIES[0])
    assert endpoints[0].sequence == 0
    for operation in (endpoints[0].drain, lambda: endpoints[0].submit(0, BOUNDARIES[0])):
        with pytest.raises(PairQuarantined):
            operation()
    assert not fabric.calls[0]


@pytest.mark.parametrize("sequence,boundary", ((True, BOUNDARIES[0]), (1, BOUNDARIES[0]), (0, BOUNDARIES[1])))
def test_invalid_submission_does_not_enqueue(sequence, boundary):
    fabric, _, endpoints = connected()
    with pytest.raises(ValueError):
        endpoints[0].submit(sequence, boundary)
    assert not fabric.queues[0]


def test_bounded_enqueue_window_requires_explicit_drain():
    _, _, endpoints = connected()
    for sequence in range(MAX_PENDING):
        endpoints[0].submit(sequence, BOUNDARIES[sequence % len(BOUNDARIES)])
    with pytest.raises(RuntimeError, match="bounded"):
        endpoints[0].submit(MAX_PENDING, BOUNDARIES[MAX_PENDING % len(BOUNDARIES)])


@pytest.mark.parametrize("row", (True, 0, 33, 1.0))
def test_invalid_geometry_fails_offline(row):
    with pytest.raises(ValueError):
        PairConfig("a" * 64, row)


@pytest.mark.parametrize("value", ("", "a" * 65, "a\0b", 1, "é"))
def test_private_key_validation(value):
    with pytest.raises((ValueError, UnicodeError)):
        _key(value)


def test_abi_geometry_and_budget():
    import ctypes

    assert ctypes.sizeof(Bindings) == 104
    assert PairConfig("a" * 64).nbytes == 16384
    assert PairConfig("a" * 64, 32).nbytes == 524288
