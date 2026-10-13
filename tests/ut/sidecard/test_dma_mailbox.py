# SPDX-License-Identifier: Apache-2.0
from copy import copy
from types import SimpleNamespace

import pytest

from tools.sidecard.pcie.mailbox import RECORD, CudaCompletionWait, Invocation
from tools.sidecard.pcie.mailbox_probe import NEGATIVE_RECORDS, PROBE_EPOCH, invocation_record


def test_mailbox_exact_record_and_stale_session_rejection():
    value = Invocation(7, 9, 18, 32, 1, 0)
    body = value.encode(slots=2)
    assert len(body) == 64
    assert Invocation.decode(body, epoch=7, sequence=9, slots=2, rank=0) == value
    for changes in ({"epoch": 8}, {"sequence": 8}, {"rank": 1}, {"slots": 1}):
        with pytest.raises(ValueError):
            parameters = dict(epoch=7, sequence=9, slots=2, rank=0)
            parameters.update(changes)
            Invocation.decode(body, **parameters)


@pytest.mark.parametrize("rejection", NEGATIVE_RECORDS)
def test_hardware_negative_fixture_changes_only_the_targeted_record_field(rejection):
    parameters = dict(sequence=2, slot=1, rank=0, size=264192)
    good = invocation_record(**parameters, rejection=None)
    bad = invocation_record(**parameters, rejection=rejection)
    changed = [i for i, (a, b) in enumerate(zip(RECORD.unpack(good), RECORD.unpack(bad), strict=True)) if a != b]
    assert changed == [
        {
            "stale_epoch": 1,
            "stale_sequence": 2,
            "foreign_rank": 6,
            "foreign_layer": 3,
            "changed_geometry": 7,
            "reserved_field": 9,
        }[rejection]
    ]
    if rejection == "foreign_layer":
        # This layer is valid in the general codec, but does not belong to
        # the probe's prequeued layer-18 consumer. CUDA rejects it separately.
        assert Invocation.decode(bad, epoch=PROBE_EPOCH, sequence=2, slots=2, rank=0).layer == 19
    else:
        with pytest.raises(ValueError):
            Invocation.decode(bad, epoch=PROBE_EPOCH, sequence=2, slots=2, rank=0)


@pytest.mark.parametrize("index,value", [(0, b"INVALID!"), (7, 1), (8, 1), (9, 1)])
def test_mailbox_rejects_changed_magic_sizes_or_reserved_fields(index, value):
    fields = list(RECORD.unpack(Invocation(7, 9, 18, 32, 1, 0).encode(slots=2)))
    fields[index] = value
    with pytest.raises(ValueError):
        Invocation.decode(RECORD.pack(*fields), epoch=7, sequence=9, slots=2, rank=0)


def wait_queue(*, memory_operations=1, ordering=100):
    events = []

    class Function:
        def __call__(self, *args):
            return 0

    functions = SimpleNamespace(cuDeviceGetAttribute=Function(), cuStreamWaitValue64_v2=Function())

    def call(name, *args):
        if name == "cuDeviceGetAttribute":
            args[0]._obj.value = memory_operations if args[1] == 122 else ordering
        else:
            events.append((name, args))

    driver = SimpleNamespace(library=functions, call=call)
    event = SimpleNamespace(
        record=lambda stream: events.append("record"), ready=lambda: False, close=lambda: events.append("close")
    )
    runtime = SimpleNamespace(kind="cuda", device=0, event=lambda: event, check_owner=lambda: None)
    stream = SimpleNamespace(pointer=123, require=lambda r: None)

    class Flag:
        host = imported = False
        size = 64
        pointer = 128
        leases = 0

    flag = Flag()
    flag.runtime = runtime
    queue = CudaCompletionWait(driver, runtime)
    queue.register(flag)
    return queue, flag, stream, event, events


@pytest.mark.parametrize("capabilities", [dict(memory_operations=0), dict(ordering=0)])
def test_unsupported_memory_wait_or_ordering_is_rejected_before_submission(capabilities):
    with pytest.raises(RuntimeError, match="native owner ordering"):
        wait_queue(**capabilities)


def test_completion_ticket_prevents_reuse_and_free_before_consumers_complete():
    queue, flag, stream, event, events = wait_queue()
    ticket = queue.enqueue(flag, 1, stream)
    assert flag.leases == 2
    assert events[0][0] == "cuStreamWaitValue64_v2"
    with pytest.raises(ValueError, match="leased"):
        queue.enqueue(flag, 2, stream)
    with pytest.raises(RuntimeError):
        queue.close()
    with pytest.raises(RuntimeError):
        ticket.release()
    ticket.seal()
    with pytest.raises(RuntimeError):
        ticket.release()
    event.ready = lambda: True
    ticket.release()
    assert flag.leases == 1
    with pytest.raises(ValueError, match="stale"):
        queue.enqueue(flag, 1, stream)
    with pytest.raises(RuntimeError):
        ticket.release()
    queue.close()
    assert flag.leases == 0


def test_uncertain_wait_submission_keeps_original_flag_and_event_owners():
    queue, flag, stream, event, events = wait_queue()

    def failed(*args):
        raise RuntimeError("submission uncertain")

    queue.driver.call = failed
    with pytest.raises(RuntimeError):
        queue.enqueue(flag, 1, stream)
    assert flag.leases == 2 and queue.failed and len(queue.active) == 1
    with pytest.raises(RuntimeError):
        queue.close()


def test_independent_flags_complete_out_of_order_without_releasing_other_slot():
    queue, first, stream, event, events = wait_queue()
    second = copy(first)
    second.leases, second.pointer = 0, 256
    second_event = SimpleNamespace(record=lambda stream: None, ready=lambda: False, close=lambda: None)
    queue.runtime.event = lambda: second_event
    queue.register(second)
    earlier = queue.enqueue(first, 1, stream)
    later = queue.enqueue(second, 1, stream)
    earlier.seal()
    later.seal()
    with pytest.raises(RuntimeError, match="before queuing"):
        queue.register(copy(first))
    second_event.ready = lambda: True
    later.release()
    assert first in queue.active and first.leases == 2
    assert second not in queue.active and second.leases == 1
    with pytest.raises(RuntimeError):
        queue.close()
    event.ready = lambda: True
    earlier.release()
    queue.close()
    assert first.leases == second.leases == 0


def test_foreign_runtime_thread_cannot_mutate_completion_ownership():
    queue, flag, stream, event, events = wait_queue()

    def reject():
        raise RuntimeError("foreign thread")

    queue.runtime.check_owner = reject
    for operation in (lambda: queue.enqueue(flag, 1, stream), queue.close):
        with pytest.raises(RuntimeError, match="foreign thread"):
            operation()
    assert flag.leases == 1 and flag in queue.events and not queue.active


@pytest.mark.parametrize("rows", [128, 255, 256, 511, 512, 513, 1023, 1024])
def test_mailbox_large_cold_geometry_round_trips_without_changing_cuda_chunks(rows):
    value = Invocation(7, 9, 18, rows, 1, 0)
    body = value.encode(slots=2)
    assert Invocation.decode(body, epoch=7, sequence=9, slots=2, rank=0) == value
    fields = RECORD.unpack(body)
    assert fields[7:9] == (rows * 8256, rows * 16384)


@pytest.mark.parametrize("rows", [0, 1025, 2048])
def test_mailbox_rejects_rows_above_registered_logical_capacity(rows):
    with pytest.raises(ValueError, match="registered contract"):
        Invocation(7, 9, 18, rows, 1, 0).encode(slots=2)
