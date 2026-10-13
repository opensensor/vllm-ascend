# SPDX-License-Identifier: Apache-2.0
import ctypes
import hashlib
import os
import socket
import threading
from types import SimpleNamespace

import pytest
import torch

from tests.ut.sidecard.test_dma_dispatch import dispatcher
from tools.sidecard import completion_results
from tools.sidecard.completion_probe import HbmSequence, fixture, probe_rows
from tools.sidecard.completion_protocol import (
    RESULT_OFFSET,
    SEQUENCE_OFFSET,
    RemoteResult,
    enabled,
    epoch,
    input_record,
    receive_fd,
    result_header,
    send_fd,
)
from tools.sidecard.completion_results import CompletionResults, validate_capacity
from tools.sidecard.dma_protocol import COMMAND
from tools.sidecard.pcie.cuda_context import CudaContextRuntime
from tools.sidecard.pcie.mailbox import Invocation


def test_result_packet_header_has_an_unready_word_and_exact_signal_source():
    body = result_header(7, 8, 18, 32, 1)
    assert len(body) == RESULT_OFFSET == 72 and body[:8] == bytes(8)
    assert int.from_bytes(body[SEQUENCE_OFFSET : SEQUENCE_OFFSET + 8], "little") == 9
    assert Invocation.decode(body[8:], epoch=7, sequence=9, slots=2, rank=1) == Invocation(7, 9, 18, 32, 1, 1)


def test_probe_sequence_subrange_keeps_the_actual_allocation_leased():
    base = SimpleNamespace(pointer=1024, runtime=object(), host=False, size=100, leases=0)
    view = HbmSequence(base)
    assert view.pointer == 1048 and view.size == 48
    view.leases += 1
    assert base.leases == 1
    view.leases -= 1


def test_probe_fixture_detects_reassociation_of_rank_addition():
    hot = torch.tensor([1e20, 1.0, -1e20, 0.03125]).repeat(4096 // 4)
    partial = [torch.frombuffer(bytearray(fixture(rank, 1)), dtype=torch.float32) for rank in range(2)]
    ordered = (hot + partial[0]) + partial[1]
    combined = hot + (partial[0] + partial[1])
    assert ordered[0] == 1 and combined[0] == 0


def test_fd_registration_duplicates_one_descriptor_and_sets_close_on_exec(tmp_path):
    path = tmp_path / "handle"
    path.write_bytes(b"fixture")
    descriptor = os.open(path, os.O_RDONLY)
    a, b = socket.socketpair()
    try:
        send_fd(a, descriptor)
        duplicate = receive_fd(b)
        assert os.fstat(duplicate).st_ino == os.fstat(descriptor).st_ino
        assert not os.get_inheritable(duplicate)
        os.close(duplicate)
    finally:
        a.close()
        b.close()
        os.close(descriptor)


def test_fd_registration_rejects_an_unaccompanied_or_wrong_marker():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(b"R")
        with pytest.raises(ValueError, match="exactly one"):
            receive_fd(b)


def test_foreign_result_keeps_fd_until_kernel_cache_lease_is_released(tmp_path):
    path = tmp_path / "handle"
    path.write_bytes(b"fixture")
    descriptor = os.open(path, os.O_RDONLY)
    result = RemoteResult(SimpleNamespace(check_owner=lambda: None), descriptor)
    result.allocation.leases += 1
    with pytest.raises(RuntimeError, match="retained"):
        result.close()
    os.fstat(descriptor)
    result.allocation.leases -= 1
    result.close()
    result.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_completion_selection_rejects_ambiguous_configuration(value):
    with pytest.raises(ValueError, match="boolean"):
        enabled(dict(direct_dma=dict(cuda_completion_contract=value)))


def test_completion_packet_capacity_includes_header():
    assert not enabled({})
    assert enabled(dict(prefill_chunk_size=32, direct_dma=dict(cuda_completion_contract=True)))
    with pytest.raises(ValueError, match="allocation"):
        enabled(
            dict(expert_transport="direct-dma", prefill_chunk_size=128, direct_dma=dict(cuda_completion_contract=True))
        )
    assert epoch("0" * 64) == 1


def test_separate_control_admits_exact_2mib_results_but_packed_control_does_not():
    spec = dict(
        expert_transport="direct-dma",
        prefill_window_size=128,
        direct_dma=dict(cuda_completion_contract=True, completion_input_record=True),
    )
    assert enabled(spec) and input_record(spec)
    spec["direct_dma"]["completion_input_record"] = False
    with pytest.raises(ValueError, match="allocation"):
        enabled(spec)


@pytest.mark.parametrize("capacity", [0, 64, 128, 256])
def test_native_binary_capacity_is_checked_before_future_waits(capacity):
    query = lambda separate: capacity if separate else 64
    native = SimpleNamespace(sidecard_glm_completion_max_rows=query)
    if capacity == 128:
        assert validate_capacity(native, 128, True) == 128
    else:
        with pytest.raises(ValueError, match="capacity"):
            validate_capacity(native, 128, True)
    assert validate_capacity(native, 64, False) == 64
    assert validate_capacity(SimpleNamespace(), 64, True) == 64
    with pytest.raises(ValueError, match="capacity"):
        validate_capacity(SimpleNamespace(), 128, True)


def test_transport_probe_covers_large_payload_boundaries_only_when_admitted():
    assert probe_rows({}) == (1, 4, 17, 32)
    assert probe_rows(dict(expert_transport="direct-dma", prefill_window_size=128)) == (1, 4, 17, 32, 64, 96, 128)


def test_old_consumer_rejection_precedes_context_and_export_allocation(tmp_path, monkeypatch):
    library = tmp_path / "old.so"
    library.write_bytes(b"authenticated old 64-row binary")
    spec = dict(
        expert_transport="direct-dma",
        prefill_window_size=128,
        direct_dma=dict(
            cuda_completion_contract=True,
            completion_input_record=True,
            completion_library=str(library),
            completion_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(completion_results.ctypes, "CDLL", lambda name: SimpleNamespace())

    def fail_allocation(*args):
        raise AssertionError("context/export allocation preceded capacity admission")

    monkeypatch.setattr(completion_results, "CudaContextRuntime", fail_allocation)
    with pytest.raises(ValueError, match="capacity"):
        completion_results.CompletionResults(spec, "1" * 64, None, None, 128)


def test_serialized_context_runtime_allows_asgi_threads_only_in_the_same_context():
    runtime = object.__new__(CudaContextRuntime)
    runtime.pid, runtime.owner_context = os.getpid(), 123

    def get_current(pointer):
        pointer._obj.value = 123
        return 0

    runtime.current_context = get_current
    runtime.check_owner()
    errors = []

    def other_thread():
        try:
            runtime.check_owner()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert not errors
    runtime.owner_context = 124
    with pytest.raises(RuntimeError, match="context"):
        runtime.check_owner()


def test_cuda_merge_queues_wait_and_rank_consumer_before_any_host_reply():
    results = object.__new__(CompletionResults)
    events = []
    results.epoch, results.calls, results.waits = 7, 0, 0
    results.runtime = SimpleNamespace(check_owner=lambda: None)
    results.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=123)))
    results.allocations = [SimpleNamespace(pointer=100), SimpleNamespace(pointer=200)]
    results.scratch = torch.empty((1, 4096))

    def enqueue(flag, sequence, stream):
        events.append(("wait", flag.pointer, sequence))
        return SimpleNamespace(seal=lambda: events.append(("seal", flag.pointer)))

    results.queue = SimpleNamespace(enqueue=enqueue)

    def merge(source, destination, packet, generation, sequence, layer, rows, rank, stream):
        assert (generation, layer, rows, stream) == (7, 18, 1, 123)
        events.append(("merge", rank))
        src = torch.frombuffer((ctypes.c_float * 4096).from_address(source), dtype=torch.float32)
        dst = torch.frombuffer((ctypes.c_float * 4096).from_address(destination), dtype=torch.float32)
        dst.copy_(src + rank + 1)
        return 0

    results.merge_rank = merge
    hot = torch.ones(1, 4096)
    output = results.merge(hot, [(0, COMMAND.pack(4, 18, 1)), (1, COMMAND.pack(9, 18, 1))], 18, 1)
    assert torch.equal(output, torch.full_like(hot, 4))
    assert events == [("wait", 100, 5), ("merge", 0), ("seal", 100), ("wait", 200, 10), ("merge", 1), ("seal", 200)]
    assert results.waits == 2 and results.calls == 1


def test_glm_dispatch_prequeues_contract_merge_and_omits_legacy_flush():
    actor, events = dispatcher()

    def merge(hot, pending, layer, rows):
        events.append(("device_merge", None))
        for rank, tag in pending:
            hot = hot + actor.outputs[rank][: rows * 16384].view(torch.float32).reshape(rows, 4096)
        return hot

    actor.completion_results = SimpleNamespace(retire=lambda: events.append(("retire", None)), merge=merge)
    x = torch.ones(1, 4096).half()
    ids = torch.tensor([[1, 2, 3, 4, 144, 145, 146, 210]])
    actual, _ = actor.compute(18, x, ids, torch.ones(1, 8), lambda: None)
    torch.testing.assert_close(actual, x.float() * (ids + 1).sum(-1, keepdim=True))
    assert events.index(("device_merge", None)) < events.index(("control_receive", 0))
    assert ("write_visibility", None) not in events


def test_pending_completion_close_retains_gpu_owner_without_synchronizing(tmp_path, monkeypatch):
    from tools.sidecard import dma_dispatch

    actor, events = dispatcher()
    actor.output = tmp_path
    actor.torch.cuda.synchronize = lambda: events.append(("unsafe_sync", None))
    ticket = SimpleNamespace(sealed=True, complete=SimpleNamespace(ready=lambda: False))
    actor.completion_results = SimpleNamespace(queue=SimpleNamespace(failed=False, active={1: ticket}))
    monkeypatch.setattr(dma_dispatch.signal, "signal", lambda *args: None)

    def stop(seconds):
        raise InterruptedError("unit test stops quarantine loop")

    monkeypatch.setattr(dma_dispatch.time, "sleep", stop)
    with pytest.raises(InterruptedError, match="quarantine"):
        actor.close()
    assert not actor.closed and ("unsafe_sync", None) not in events
    assert actor.outputs and actor.connections


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_input_record_rejects_ambiguous_configuration(value):
    with pytest.raises(ValueError, match="input-record"):
        input_record(dict(direct_dma=dict(cuda_completion_contract=True, completion_input_record=value)))


def test_input_record_requires_authenticated_completion_selection():
    assert not input_record({})
    with pytest.raises(ValueError, match="input-record"):
        input_record(dict(direct_dma=dict(completion_input_record=True)))
    assert input_record(dict(direct_dma=dict(cuda_completion_contract=True, completion_input_record=True)))


def test_input_record_precedes_source_event_and_result_wait():
    actor, events = dispatcher()
    actor.completion_results = SimpleNamespace(
        input_record=True,
        retire=lambda: None,
        prepare=lambda rank, sequence, layer, rows, packed: events.append(("input_record", rank)),
        merge=lambda hot, pending, layer, rows: hot,
    )
    ids = torch.tensor([[1, 2, 3, 4, 144, 145, 146, 210]])
    actor.compute(18, torch.ones(1, 4096).half(), ids, torch.ones(1, 8))
    for rank in (0, 1):
        assert events.index(("input_record", rank)) < events.index(("source_ready", rank))


def test_input_sequence_subrange_pins_uploaded_record_allocation():
    base = SimpleNamespace(pointer=1024, runtime=object(), host=False, size=2097152, leases=0)
    view = HbmSequence(base, 32 * 8256 + 16)
    assert view.pointer == base.pointer + 32 * 8256 + 16 and view.size == 48
    view.leases += 1
    assert base.leases == 1
