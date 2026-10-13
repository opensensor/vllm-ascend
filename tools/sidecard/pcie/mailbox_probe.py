# SPDX-License-Identifier: Apache-2.0
"""Prove a pre-queued CUDA consumer is released by an Ascend DMA flag.

DMA submission and the Ascend producer remain host-driven in this prototype.
On uncertain DMA or a pending CUDA wait, retain the original owners.
"""

import argparse
import ctypes
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .mailbox import RECORD, CudaCompletionWait, Invocation
from .peer import ASCEND_TO_CUDA, Driver, KernelPeer
from .peer_benchmark import GUARD_POISON, pattern
from .runtime import Runtime

CAPACITY = 524288
SIZES = (8256, 16384, 264192, CAPACITY)
ITERATIONS = 12
CONSUMER_TIMEOUT_SECONDS = 10
PROBE_SLOTS = 2
PROBE_EPOCH = 20261012
NEGATIVE_RECORDS = (
    "stale_epoch",
    "stale_sequence",
    "foreign_rank",
    "foreign_layer",
    "changed_geometry",
    "reserved_field",
)
REJECTED_RECORD = 0xEE


def invocation_record(sequence, slot, rank, size, rejection):
    rows = 1 if size in SIZES[:2] else 32
    value = Invocation(PROBE_EPOCH, sequence, 18, rows, slot, rank)
    body = value.encode(slots=PROBE_SLOTS)
    fields = list(RECORD.unpack(body))
    if rejection == "stale_epoch":
        fields[1] -= 1
    elif rejection == "stale_sequence":
        fields[2] -= 1
    elif rejection == "foreign_rank":
        fields[6] = 1 - rank
    elif rejection == "foreign_layer":
        fields[3] += 1
    elif rejection == "changed_geometry":
        fields[7] += 1
    elif rejection == "reserved_field":
        fields[9] = 1
    elif rejection is not None:
        raise ValueError("unknown mailbox rejection fixture")
    return RECORD.pack(*fields)


def probe(spec, rank, output):
    cuda = Runtime("cuda", spec["cuda_backend"]["cuda_runtime"])
    ascend = Runtime("ascend", "/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so", rank)
    driver = Driver()
    queue = CudaCompletionWait(driver, cuda)
    # Allocate and register every resource before any future flag can block
    # CUDA. Allocation and lazy module loading can synchronize the context.
    slots = []
    for index in range(PROBE_SLOTS):
        slot = SimpleNamespace(
            index=index,
            gpu=cuda.allocate(CAPACITY),
            flag=cuda.allocate(CAPACITY),
            snapshot=cuda.allocate(CAPACITY),
            npu=ascend.allocate(CAPACITY),
            npu_flag=ascend.allocate(CAPACITY),
            npu_record=ascend.allocate(CAPACITY),
            chost=cuda.allocate(CAPACITY, host=True),
            nhost=ascend.allocate(CAPACITY, host=True),
            guard=ascend.allocate(CAPACITY, host=True),
            cs=cuda.stream(),
            ns=ascend.stream(),
            ce=cuda.event(),
            ne=ascend.event(),
            peer=KernelPeer(rank + 2, persistent_gpu_mappings=True),
        )
        slot.exports = [driver.export(value) for value in (slot.gpu, slot.flag)]
        queue.register(slot.flag)
        slots.append(slot)
    native_ordering = spec.get("mailbox_native_owner_ordering", False)
    if native_ordering:
        # Probe-only: all exported allocations belong to this CUDA runtime.
        # Imported aliases and cross-context consumers are outside this gate.
        def owner_ordered_flush(*, all_devices=False):
            cuda.check_owner()
            if all_devices:
                raise ValueError("mailbox probe only qualifies its CUDA owner context")

        driver.flush = owner_ordered_flush
    library = Path(spec["mailbox_consumer_library"])
    if hashlib.sha256(library.read_bytes()).hexdigest() != spec["mailbox_consumer_sha256"]:
        raise ValueError("mailbox consumer binary failed authentication")
    native = ctypes.CDLL(str(library))
    preload = native.sidecard_mailbox_preload
    preload.argtypes, preload.restype = [], ctypes.c_int
    if preload():
        raise RuntimeError("CUDA consumer preload failed before queuing any waits")
    consumer = native.sidecard_mailbox_consume
    consumer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    consumer.restype = ctypes.c_int
    report = dict(
        passed=False,
        physical_chip=rank + 2,
        state="running",
        cases=[],
        device_consumer="pre-queued CUDA record validation and XOR kernel then validation D2H",
        dma_submission="blocking host ioctl",
        ascend_producer="host-initialized fixture",
        host_free=False,
        socket_commands_in_hot_path=0,
        independent_slots=PROBE_SLOTS,
        remote_flush_api_used=not native_ordering,
        capabilities=queue.capabilities,
    )

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    for slot in slots:
        ctypes.memset(slot.guard.pointer, GUARD_POISON, CAPACITY)
        ctypes.memset(slot.chost.pointer, 0, CAPACITY)
        cuda.copy(slot.flag.pointer, slot.chost.pointer, CAPACITY, "h2d", slot.cs)
        slot.ce.record(slot.cs)
        slot.ce.wait()
    save()
    try:
        for sequence in range(1, ITERATIONS + len(NEGATIVE_RECORDS) + 1):
            size = SIZES[(sequence - 1) % len(SIZES)]
            rejection = NEGATIVE_RECORDS[sequence - ITERATIONS - 1] if sequence > ITERATIONS else None
            for slot in slots:
                slot.expected = pattern(size, sequence + slot.index * ITERATIONS)
                ctypes.memmove(slot.nhost.pointer, slot.expected, size)
                ascend.copy(slot.npu.pointer, slot.nhost.pointer, size, "h2d", slot.ns)
                slot.ne.record(slot.ns)
                slot.ne.wait()
                ctypes.memset(slot.nhost.pointer, 0, CAPACITY)
                ctypes.c_uint64.from_address(slot.nhost.pointer).value = sequence
                ascend.copy(slot.npu_flag.pointer, slot.nhost.pointer, 8, "h2d", slot.ns)
                slot.ne.record(slot.ns)
                slot.ne.wait()
                record = invocation_record(sequence, slot.index, rank, size, rejection)
                ctypes.memset(slot.nhost.pointer, 0, CAPACITY)
                ctypes.memmove(slot.nhost.pointer + 8, record, len(record))
                ascend.copy(slot.npu_record.pointer, slot.nhost.pointer, 8 + len(record), "h2d", slot.ns)
                slot.ne.record(slot.ns)
                slot.ne.wait()
            for slot in slots:
                slot.ticket = queue.enqueue(slot.flag, sequence, slot.cs)
                status = consumer(
                    slot.gpu.pointer,
                    slot.snapshot.pointer,
                    size,
                    slot.flag.pointer,
                    PROBE_EPOCH,
                    sequence,
                    slot.index,
                    rank,
                    slot.cs.pointer,
                )
                if status:
                    raise RuntimeError(f"CUDA mailbox consumer launch failed: {status}")
                cuda.copy(slot.chost.pointer, slot.snapshot.pointer, size, "d2h", slot.cs)
                slot.ticket.seal()
                if slot.ticket.complete.ready():
                    raise AssertionError("consumer completed before its future flag")
            # Complete the second slot first. Its consumer must finish while
            # the first stream still waits, proving there is no shared barrier.
            for slot in reversed(slots):
                payload = slot.peer.copy(
                    ASCEND_TO_CUDA, slot.exports[0], slot.npu, slot.guard, size, source_ready=slot.ne
                )
                descriptor = slot.peer.copy(
                    ASCEND_TO_CUDA, slot.exports[1], slot.npu_record, slot.guard, 8 + RECORD.size, source_ready=slot.ne
                )
                if slot.ticket.complete.ready():
                    raise AssertionError("record DMA published completion before its final flag")
                completion = slot.peer.copy(
                    ASCEND_TO_CUDA, slot.exports[1], slot.npu_flag, slot.guard, 8, source_ready=slot.ne
                )
                deadline = time.monotonic() + CONSUMER_TIMEOUT_SECONDS
                while not slot.ticket.complete.ready():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CUDA consumer did not complete after its DMA flag")
                    time.sleep(0.001)
                other_waiting = slot.index == 0 or not slots[0].ticket.complete.ready()
                expected = (
                    bytes([REJECTED_RECORD]) * size if rejection else bytes(value ^ 0xA7 for value in slot.expected)
                )
                equal = ctypes.string_at(slot.chost.pointer, size) == expected
                slot.ticket.release()
                report["cases"].append(
                    dict(
                        sequence=sequence,
                        slot=slot.index,
                        size=size,
                        wait_blocked_before_flag=True,
                        other_slot_still_waiting=other_waiting if slot.index else None,
                        payload_equal=equal,
                        rejection_fixture=rejection,
                        record_validated_on_cuda=True,
                        payload_dma=payload,
                        record_dma=descriptor,
                        flag_dma=completion,
                    )
                )
                save()
                if not equal or not other_waiting:
                    raise AssertionError("queued consumer observed stale payload or another slot's completion")
        report["guard_poison_exact"] = all(
            ctypes.string_at(slot.guard.pointer, CAPACITY) == bytes([GUARD_POISON]) * CAPACITY for slot in slots
        )
        if not report["guard_poison_exact"]:
            raise AssertionError("unused lifetime guard changed")
        report["passed"] = True
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        if not any(slot.peer.failed for slot in slots):
            for pending in list(queue.active.values()):
                if pending.sealed and pending.complete.ready():
                    pending.release()
        if any(slot.peer.failed for slot in slots) or queue.active:
            report["state"] = "quarantined"
            save()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            while True:
                signal.pause()
        raise
    finally:
        if not any(slot.peer.failed for slot in slots) and not queue.active:
            queue.close()
            for slot in slots:
                slot.peer.close()
                for exported in slot.exports:
                    exported.close()
                for name in (
                    "ce",
                    "ne",
                    "cs",
                    "ns",
                    "gpu",
                    "flag",
                    "snapshot",
                    "npu",
                    "npu_flag",
                    "npu_record",
                    "chost",
                    "nhost",
                    "guard",
                ):
                    getattr(slot, name).close()
            report["state"] = "closed"
            save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.config.read_bytes())
    if args.bootstrap:
        environment = dict(os.environ)
        environment.update(json.loads(Path(spec["npu_environment"]).read_bytes()))
        environment["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
        environment["PYTHONPATH"] = str(args.config.resolve().parent / "source")
        command = [
            spec["npu_python"],
            "-m",
            "tools.sidecard.pcie.mailbox_probe",
            *[v for v in sys.argv[1:] if v != "--bootstrap"],
        ]
        os.execve(command[0], command, environment)
    if os.geteuid() != 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("probe requires root and only assigned chips 2,3")
    report = probe(spec, args.rank, args.output)
    print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]), state=report["state"])), flush=True)


if __name__ == "__main__":
    main()
