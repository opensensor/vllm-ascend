# SPDX-License-Identifier: Apache-2.0
"""Actual aggregate C++ with a CPU-only SDK/exchange implementation.

These tests cover submission, storage, ownership and error contracts. Immediate
CPU copies do not qualify peer synchronization, NPU execution or graph capture.
The Torch adapter is compiled against the real pinned SDK by the device owner.
"""

import concurrent.futures
import ctypes as c
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MAX_BYTES = 64 * 1024 * 1024
MAX_STEPS = 128
MAX_EVENTS = MAX_STEPS * 2 + 2
STUB_HEADER = r"""
#pragma once
#include <cstddef>
constexpr int ACL_MEMCPY_DEVICE_TO_DEVICE = 3;
int aclrtMemcpyAsync(void*, size_t, const void*, size_t, int, void*);
"""
STUB_SOURCE = r"""
#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <new>
struct State {
    int allocations, fail_allocation, exchange_destroys;
};
struct Stream {
    int count, fail_at;
    int kind[258];
    uintptr_t destination[258], source[258], exchange[258];
    uint64_t bytes[258];
    uint32_t operation[258], timeout[258];
};
struct Exchange {
    void* input;
    void* output;
    const void* peer;
    uint64_t bytes;
    uint32_t dtype;
};
extern "C" State* stub_state() { static State s{}; return &s; }
extern "C" void* __real__Znwm(size_t);
extern "C" void* __wrap__Znwm(size_t bytes) {
    auto* state = stub_state();
    if (++state->allocations == state->fail_allocation) throw std::bad_alloc();
    return __real__Znwm(bytes);
}
int record(Stream* stream, int kind, void* dst, const void* src, uint64_t bytes,
           uintptr_t exchange = 0, uint32_t operation = 0, uint32_t timeout = 0) {
    const int i = stream->count++;
    if (i >= 258) return 82;
    stream->kind[i] = kind;
    stream->destination[i] = reinterpret_cast<uintptr_t>(dst);
    stream->source[i] = reinterpret_cast<uintptr_t>(src);
    stream->exchange[i] = exchange;
    stream->bytes[i] = bytes;
    stream->operation[i] = operation;
    stream->timeout[i] = timeout;
    return stream->count == stream->fail_at ? 71 : 0;
}
int aclrtMemcpyAsync(void* dst, size_t capacity, const void* src, size_t bytes, int kind, void* ptr) {
    if (capacity != bytes || kind != ACL_MEMCPY_DEVICE_TO_DEVICE) return 81;
    const int status = record(static_cast<Stream*>(ptr), 1, dst, src, bytes);
    if (!status) std::memcpy(dst, src, bytes);
    return status;
}
extern "C" int sidecard_exchange_enqueue_io(void* ptr, const void* input, void* output,
                                           void* stream, uint32_t timeout, uint32_t operation) {
    auto* exchange = static_cast<Exchange*>(ptr);
    const int status = record(static_cast<Stream*>(stream), 2, output, input,
                              exchange->bytes, reinterpret_cast<uintptr_t>(ptr), operation, timeout);
    if (status) return status;
    if (operation) {
        std::memcpy(output, operation == 1 ? input : exchange->peer, exchange->bytes);
    } else if (exchange->dtype == 0) {
        auto* a = static_cast<const float*>(input);
        auto* b = static_cast<const float*>(exchange->peer);
        auto* out = static_cast<float*>(output);
        for (size_t i = 0; i < exchange->bytes / 4; ++i) out[i] = a[i] + b[i];
    } else {
        auto* a = static_cast<const _Float16*>(input);
        auto* b = static_cast<const _Float16*>(exchange->peer);
        auto* out = static_cast<_Float16*>(output);
        for (size_t i = 0; i < exchange->bytes / 2; ++i) out[i] = a[i] + b[i];
    }
    return 0;
}
extern "C" int sidecard_exchange_destroy(void**, uint32_t) {
    ++stub_state()->exchange_destroys;
    return 0;
}
"""


class Step(c.Structure):
    _fields_ = [("exchange", c.c_void_p), ("operation", c.c_uint32), ("reserved", c.c_uint32)]


class Exchange(c.Structure):
    _fields_ = [(name, c.c_void_p) for name in ("input", "output", "peer")] + [
        ("bytes", c.c_uint64),
        ("dtype", c.c_uint32),
    ]


class State(c.Structure):
    _fields_ = [(name, c.c_int) for name in ("allocations", "fail_allocation", "exchange_destroys")]


class Stream(c.Structure):
    _fields_ = (
        [("count", c.c_int), ("fail_at", c.c_int), ("kind", c.c_int * MAX_EVENTS)]
        + [(name, c.c_size_t * MAX_EVENTS) for name in ("destination", "source", "exchange")]
        + [("bytes", c.c_uint64 * MAX_EVENTS)]
        + [(name, c.c_uint32 * MAX_EVENTS) for name in ("operation", "timeout")]
    )


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("group ABI tests require a CPU C++ compiler")
    root = tmp_path_factory.mktemp("group-native")
    (root / "acl").mkdir()
    (root / "acl/acl.h").write_text(STUB_HEADER)
    (root / "stub.cpp").write_text(STUB_SOURCE)
    output = root / "group.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wl,--wrap=_Znwm",
            "-I",
            str(root),
            str(ROOT / "csrc/sidecard_collectives/npu_group.cpp"),
            str(root / "stub.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    lib = c.CDLL(str(output))
    lib.sidecard_group_abi.restype = c.c_uint64
    lib.sidecard_group_create.argtypes = [
        c.c_void_p,
        c.c_void_p,
        c.c_uint64,
        c.c_uint32,
        c.c_int32,
        c.POINTER(Step),
        c.c_uint64,
        c.POINTER(c.c_void_p),
    ]
    lib.sidecard_group_metadata.argtypes = [
        c.c_void_p,
        c.POINTER(c.c_uint64),
        c.POINTER(c.c_uint32),
        c.POINTER(c.c_int32),
    ]
    lib.sidecard_group_enqueue.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint32]
    lib.sidecard_group_destroy.argtypes = [c.POINTER(c.c_void_p), c.c_uint32]
    lib.sidecard_group_graph_lease_acquire.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
    lib.sidecard_group_graph_lease_release.argtypes = [c.POINTER(c.c_void_p), c.c_uint32]
    lib.sidecard_group_graph_lease_count.argtypes = [c.c_void_p, c.POINTER(c.c_uint64)]
    lib.stub_state.restype = c.POINTER(State)
    return lib


@pytest.fixture
def native(compiled):
    state = compiled.stub_state().contents
    c.memset(c.addressof(state), 0, c.sizeof(state))
    return compiled, state


def geometry(operations=(0, 1, 2), dtype=0, elements=8):
    item = c.c_float if dtype == 0 else c.c_uint16
    arrays = [(item * elements)() for _ in range(7)]
    for index, array in enumerate(arrays):
        if dtype == 0:
            array[:] = [float(index + value % 4 - 2) for value in range(elements)]
        else:
            array[:] = [0x3C00 + index * 0x400] * elements
    a, b, input_tensor, output_tensor, *peers = arrays
    exchanges = [
        Exchange(c.addressof(a), c.addressof(b), c.addressof(peers[index % 3]), c.sizeof(a), dtype)
        for index in range(len(operations))
    ]
    steps = (Step * len(operations))(
        *[Step(c.addressof(exchange), operation, 0) for exchange, operation in zip(exchanges, operations)]
    )
    return dict(
        arrays=arrays,
        exchanges=exchanges,
        steps=steps,
        create=[c.addressof(a), c.addressof(b), c.sizeof(a), dtype, 2, steps, len(operations)],
        input=c.addressof(input_tensor),
        output=c.addressof(output_tensor),
    )


def create(lib, case):
    handle = c.c_void_p()
    assert lib.sidecard_group_create(*case["create"], c.byref(handle)) == 0
    return handle


def destroy(lib, handle):
    assert lib.sidecard_group_destroy(c.byref(handle), 1) == 0
    assert handle.value is None


def metadata(lib, handle):
    size, dtype, device = c.c_uint64(99), c.c_uint32(99), c.c_int32(99)
    status = lib.sidecard_group_metadata(handle, c.byref(size), c.byref(dtype), c.byref(device))
    return status, size.value, dtype.value, device.value


def lease_count(lib, handle):
    count = c.c_uint64(999)
    assert lib.sidecard_group_graph_lease_count(handle, c.byref(count)) == 0
    return count.value


def acquire_lease(lib, handle):
    lease = c.c_void_p()
    assert lib.sidecard_group_graph_lease_acquire(handle, c.byref(lease)) == 0
    assert lease.value
    return lease


def expected_bytes(input_tensor, peers, operations, dtype):
    fmt = "<f" if dtype == 0 else "<e"
    result = bytes(input_tensor)
    for index, operation in enumerate(operations):
        peer = bytes(peers[index % len(peers)])
        if operation == 2:
            result = peer
        elif operation == 0:
            result = b"".join(
                struct.pack(fmt, a[0] + b[0])
                for a, b in zip(struct.iter_unpack(fmt, result), struct.iter_unpack(fmt, peer))
            )
    return result


def test_abi_geometry_and_creation_use_no_sdk(native):
    lib, state = native
    case = geometry()
    handle = create(lib, case)
    assert lib.sidecard_group_abi() == 1
    assert c.sizeof(Step) == 16 and Step.reserved.offset == 12
    assert metadata(lib, handle) == (0, 32, 0, 2)
    assert state.exchange_destroys == 0
    destroy(lib, handle)
    assert state.exchange_destroys == 0


@pytest.mark.parametrize("dtype", [0, 1])
@pytest.mark.parametrize("operations", [(0,), (1,), (2,), (0, 1, 2), (0, 0, 0), (2, 0, 1, 0)])
def test_owned_steps_order_direct_io_and_repeated_submission(native, dtype, operations):
    lib, state = native
    case = geometry(operations, dtype)
    a, b, input_tensor, output_tensor, *peers = case["arrays"]
    handle = create(lib, case)
    # The aggregate owns a copy, independent of the caller's mutable step list.
    for step in case["steps"]:
        step.exchange = None
        step.operation = 99
        step.reserved = 99
    allocations = state.allocations
    for iteration in range(3):
        stream = Stream()
        input_tensor[0] = iteration + 2 if dtype == 0 else 0x3C00 + iteration * 0x400
        original = bytes(input_tensor)
        expected = expected_bytes(input_tensor, peers, operations, dtype)
        assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 37) == 0
        assert list(stream.kind)[: stream.count] == [2] * len(operations)
        assert stream.source[0] == case["input"]
        assert stream.destination[stream.count - 1] == case["output"]
        scratch = [c.addressof(a), c.addressof(b)]
        for index in range(stream.count - 1):
            assert stream.destination[index] == scratch[index % 2]
            assert stream.source[index + 1] == scratch[index % 2]
        assert list(stream.bytes)[: stream.count] == [c.sizeof(a)] * stream.count
        assert [stream.exchange[index] for index in range(stream.count) if stream.kind[index] == 2] == [
            c.addressof(exchange) for exchange in case["exchanges"]
        ]
        assert [stream.operation[index] for index in range(stream.count) if stream.kind[index] == 2] == list(operations)
        assert [stream.timeout[index] for index in range(stream.count) if stream.kind[index] == 2] == [37] * len(
            operations
        )
        assert bytes(input_tensor) == original and bytes(output_tensor) == expected
        assert state.allocations == allocations and state.exchange_destroys == 0
    destroy(lib, handle)
    assert peers and state.exchange_destroys == 0


@pytest.mark.parametrize("dtype", [0, 1])
@pytest.mark.parametrize("operation", [1, 2])
def test_broadcast_and_identity_preserve_payload_bits(native, dtype, operation):
    lib, _ = native
    case = geometry((operation,), dtype, 4)
    kind = c.c_uint32 if dtype == 0 else c.c_uint16
    patterns = (0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000) if dtype == 0 else (0x8000, 0x7E23, 0x7C00, 0xFC00)
    source = case["arrays"][2 if operation == 1 else 4]
    c.memmove(c.addressof(source), (kind * 4)(*patterns), c.sizeof(source))
    handle, stream = create(lib, case), Stream()
    assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 10) == 0
    assert bytes(case["arrays"][3]) == bytes(source)
    destroy(lib, handle)
    identity = geometry((), dtype, 4)
    c.memmove(identity["input"], source, c.sizeof(source))
    handle, stream = create(lib, identity), Stream()
    assert lib.sidecard_group_enqueue(handle, identity["input"], identity["output"], c.byref(stream), 10) == 0
    assert stream.count == 1 and bytes(identity["arrays"][3]) == bytes(source)
    destroy(lib, handle)


@pytest.mark.parametrize("count", [0, 1, MAX_STEPS])
def test_boundary_step_counts_and_exact_inplace_caller_alias(native, count):
    lib, _ = native
    case = geometry((1,) * count)
    handle, stream = create(lib, case), Stream()
    original = bytes(case["arrays"][2])
    assert lib.sidecard_group_enqueue(handle, case["input"], case["input"], c.byref(stream), 1) == 0
    assert stream.count == (0 if count == 0 else 2 if count == 1 else count)
    assert bytes(case["arrays"][2]) == original
    destroy(lib, handle)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_every_sdk_failure_quarantines_without_releasing_borrowed_resources(native, count):
    lib, state = native
    for failure in range(1, (1 if count == 0 else count) + 1):
        case = geometry((0,) * count)
        handle, stream = create(lib, case), Stream()
        before = bytes(case["arrays"][3])
        allocations = state.allocations
        stream.fail_at = failure
        assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 7) == 71
        assert stream.count == failure and bytes(case["arrays"][3]) == before
        assert metadata(lib, handle) == (-3, 99, 99, 99)
        stream.fail_at = 0
        assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 7) == -3
        assert stream.count == failure and state.allocations == allocations and state.exchange_destroys == 0
        destroy(lib, handle)
        assert state.exchange_destroys == 0


@pytest.mark.parametrize("dtype", [0, 1])
@pytest.mark.parametrize("failure", [0, 1, 2])
def test_single_step_alias_uses_only_scratch_and_copyback_and_quarantines(native, dtype, failure):
    lib, _ = native
    case = geometry((0,), dtype)
    handle, stream = create(lib, case), Stream()
    before = bytes(case["arrays"][2])
    expected = expected_bytes(case["arrays"][2], case["arrays"][4:], (0,), dtype)
    stream.fail_at = failure
    assert lib.sidecard_group_enqueue(handle, case["input"], case["input"], c.byref(stream), 7) == (
        71 if failure else 0
    )
    assert stream.count == (failure if failure else 2)
    assert stream.source[0] == case["input"] and stream.destination[0] == case["create"][0]
    if not failure:
        assert list(stream.kind)[:2] == [2, 1]
        assert stream.source[1] == case["create"][0] and stream.destination[1] == case["input"]
    assert bytes(case["arrays"][2]) == (before if failure else expected)
    assert metadata(lib, handle)[0] == (-3 if failure else 0)
    destroy(lib, handle)


@pytest.mark.parametrize("dtype", [0, 1])
def test_changed_caller_allocations_use_direct_io_without_rebinding_group(native, dtype):
    lib, state = native
    case = geometry((0, 0), dtype)
    handle = create(lib, case)
    alternatives = [geometry((), dtype) for _ in range(3)]
    allocations = state.allocations
    for alternative in alternatives:
        stream = Stream()
        expected = expected_bytes(alternative["arrays"][2], case["arrays"][4:], (0, 0), dtype)
        assert lib.sidecard_group_enqueue(handle, alternative["input"], alternative["output"], c.byref(stream), 8) == 0
        assert stream.count == 2 and list(stream.kind)[:2] == [2, 2]
        assert stream.source[0] == alternative["input"] and stream.destination[1] == alternative["output"]
        assert bytes(alternative["arrays"][3]) == expected and state.allocations == allocations
    destroy(lib, handle)


@pytest.mark.parametrize("failure", [1, 2, 3])
def test_host_allocation_failure_leaves_no_handle_or_sdk_ownership(native, failure):
    lib, state = native
    case = geometry()
    state.fail_allocation = failure
    handle = c.c_void_p(123)
    assert lib.sidecard_group_create(*case["create"], c.byref(handle)) == -2
    assert handle.value is None and state.allocations == failure and state.exchange_destroys == 0


@pytest.mark.parametrize(
    "field,value", [(2, 0), (2, 1), (2, 3), (2, MAX_BYTES + 4), (3, 2), (4, -1), (6, MAX_STEPS + 1)]
)
def test_invalid_geometry_precedes_allocations_and_sdk(native, field, value):
    lib, state = native
    case = geometry()
    case["create"][field] = value
    handle = c.c_void_p(123)
    assert lib.sidecard_group_create(*case["create"], c.byref(handle)) == -1
    assert handle.value is None and state.allocations == 0 and state.exchange_destroys == 0


@pytest.mark.parametrize("mutation", ["null", "null-steps", "unaligned-steps", "operation", "reserved"])
def test_invalid_step_metadata_precedes_allocations(native, mutation):
    lib, state = native
    case = geometry()
    if mutation == "null":
        case["steps"][0].exchange = None
    elif mutation == "null-steps":
        case["create"][5] = None
    elif mutation == "unaligned-steps":
        case["create"][5] = c.cast(c.addressof(case["steps"]) + 1, c.POINTER(Step))
    elif mutation == "operation":
        case["steps"][0].operation = 3
    else:
        case["steps"][0].reserved = 1
    handle = c.c_void_p()
    assert lib.sidecard_group_create(*case["create"], c.byref(handle)) == -1
    assert handle.value is None and state.allocations == 0


@pytest.mark.parametrize("dtype", [0, 1])
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("invalid", ["null", "unaligned", "overflow", "equal", "overlap"])
def test_work_pointer_validation(native, dtype, index, invalid):
    lib, state = native
    case = geometry(dtype=dtype)
    address = case["create"][index]
    other = case["create"][1 - index]
    case["create"][index] = {
        "null": 0,
        "unaligned": address + 1,
        "overflow": 2 ** (8 * c.sizeof(c.c_void_p)) - 4,
        "equal": other,
        "overlap": other + (2 if dtype else 4),
    }[invalid]
    handle = c.c_void_p()
    assert lib.sidecard_group_create(*case["create"], c.byref(handle)) == -1
    assert handle.value is None and state.allocations == 0


@pytest.mark.parametrize("dtype", [0, 1])
@pytest.mark.parametrize("which", ["input", "output"])
@pytest.mark.parametrize("invalid", ["null", "unaligned", "overflow", "work-a", "work-b", "partial-alias"])
def test_caller_pointer_validation_precedes_sdk_and_does_not_quarantine(native, dtype, which, invalid):
    lib, _ = native
    case = geometry(dtype=dtype)
    handle, stream = create(lib, case), Stream()
    pointers = {"input": case["input"], "output": case["output"]}
    pointers[which] = {
        "null": 0,
        "unaligned": pointers[which] + 1,
        "overflow": 2 ** (8 * c.sizeof(c.c_void_p)) - 4,
        "work-a": case["create"][0],
        "work-b": case["create"][1],
        "partial-alias": pointers["output" if which == "input" else "input"] + (2 if dtype else 4),
    }[invalid]
    assert lib.sidecard_group_enqueue(handle, pointers["input"], pointers["output"], c.byref(stream), 1) == -1
    assert stream.count == 0 and metadata(lib, handle)[0] == 0
    assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 1) == 0
    destroy(lib, handle)


@pytest.mark.parametrize("dtype,size", [(0, 4), (1, 2), (0, MAX_BYTES), (1, MAX_BYTES)])
def test_minimum_and_maximum_bytes_without_device_dereference(native, dtype, size):
    lib, _ = native
    handle = c.c_void_p()
    assert lib.sidecard_group_create(0x10000000, 0x20000000, size, dtype, 0, None, 0, c.byref(handle)) == 0
    assert metadata(lib, handle) == (0, size, dtype, 0)
    destroy(lib, handle)


def test_fp16_two_byte_aligned_storage_and_null_identity_steps(native):
    lib, _ = native
    storage = [(c.c_uint16 * 5)() for _ in range(4)]
    addresses = [c.addressof(array) + 2 for array in storage]
    handle = c.c_void_p()
    assert lib.sidecard_group_create(*addresses[:2], 6, 1, 1, None, 0, c.byref(handle)) == 0
    c.memmove(addresses[2], b"\x00\x80\x23\x7e\x00\x7c", 6)
    stream = Stream()
    assert lib.sidecard_group_enqueue(handle, *addresses[2:], c.byref(stream), 1) == 0
    assert c.string_at(addresses[2], 6) == c.string_at(addresses[3], 6)
    destroy(lib, handle)


def test_invalid_entry_arguments_leave_live_plan_usable(native):
    lib, state = native
    case = geometry()
    handle, stream = create(lib, case), Stream()
    assert lib.sidecard_group_create(*case["create"], None) == -1
    assert lib.sidecard_group_enqueue(None, case["input"], case["output"], c.byref(stream), 1) == -1
    assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], None, 1) == -1
    assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 0) == -1
    assert lib.sidecard_group_metadata(None, None, None, None) == -1
    outputs = (c.c_uint64(), c.c_uint32(), c.c_int32())
    for missing in range(3):
        pointers = [c.byref(output) for output in outputs]
        pointers[missing] = None
        assert lib.sidecard_group_metadata(handle, *pointers) == -1
    assert stream.count == 0
    for proof in (0, 2, 2**32 - 1):
        assert lib.sidecard_group_destroy(c.byref(handle), proof) == -1 and handle.value
    assert lib.sidecard_group_destroy(None, 1) == -1
    assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 1) == 0
    destroy(lib, handle)
    assert lib.sidecard_group_destroy(c.byref(handle), 1) == -1 and state.exchange_destroys == 0


def test_creating_process_ownership_survives_thread_change_and_rejects_fork(native):
    lib, _ = native
    case = geometry()
    handle, stream = create(lib, case), Stream()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        assert (
            executor.submit(
                lib.sidecard_group_enqueue, handle, case["input"], case["output"], c.byref(stream), 1
            ).result()
            == 0
        )
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        child_stream = Stream()
        results = [
            lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(child_stream), 1),
            metadata(lib, handle)[0],
            lib.sidecard_group_destroy(c.byref(handle), 1),
            child_stream.count,
            int(bool(handle.value)),
        ]
        os.write(write_fd, repr(results).encode())
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    raw = os.read(read_fd, 128)
    os.close(read_fd)
    assert os.waitpid(pid, 0)[1] == 0
    assert raw == b"[-4, -4, -4, 0, 1]"
    assert metadata(lib, handle)[0] == 0
    destroy(lib, handle)


def test_torch_adapter_queues_after_metadata_without_baseline_or_cpu_fallback():
    source = (ROOT / "csrc/sidecard_collectives/npu_torch.cpp").read_text()
    assert source.index("sidecard_group_metadata(group") < source.index("OptionalNPUGuard")
    assert source.index("sidecard_group_enqueue(") > source.index("auto handler =")
    assert "getCurrentNPUStream(input.get_device()).stream(false)" in source
    assert "input_owner = input, output_owner = output" in source
    assert "command.SetCustomHandler(handler);" in source and "command.Run();" in source
    assert "TORCH_LIBRARY_FRAGMENT(sidecard_comm, ops)" in source
    assert 'ops.def("all_reduce(Tensor input, int handle, int timeout) -> Tensor")' in source
    assert "TORCH_LIBRARY_IMPL(sidecard_comm, PrivateUse1" in source
    assert "TORCH_LIBRARY_IMPL(sidecard_comm, Meta" in source
    assert "HCCL" not in source and "ctypes" not in source and ".synchronize(" not in source


@pytest.mark.parametrize("count", [0, 1, 3, 257])
def test_graph_leases_block_destruction_until_each_exact_lease_is_released(native, count):
    lib, state = native
    case = geometry((0,))
    handle = create(lib, case)
    leases = [acquire_lease(lib, handle) for _ in range(count)]
    assert lease_count(lib, handle) == count
    assert len({lease.value for lease in leases}) == count
    original = handle.value
    for remaining, lease in zip(range(count, 0, -1), reversed(leases), strict=True):
        # The group cannot infer whether a native graph replays without host
        # callbacks. Even a drained-proof destroy must respect each live lease.
        assert lib.sidecard_group_destroy(c.byref(handle), 1) == -5
        assert handle.value == original and lease_count(lib, handle) == remaining
        before = lease.value
        for proof in (0, 2, 2**32 - 1):
            assert lib.sidecard_group_graph_lease_release(c.byref(lease), proof) == -1
            assert lease.value == before and lease_count(lib, handle) == remaining
        assert lib.sidecard_group_graph_lease_release(c.byref(lease), 1) == 0
        assert lease.value is None
        assert lib.sidecard_group_graph_lease_release(c.byref(lease), 1) == -1
    assert lease_count(lib, handle) == 0 and state.exchange_destroys == 0
    destroy(lib, handle)
    assert state.exchange_destroys == 0


def test_release_order_and_group_association_preserve_other_graph_leases(native):
    lib, _ = native
    cases = [geometry(), geometry()]
    handles = [create(lib, case) for case in cases]
    leases = [acquire_lease(lib, handles[0]) for _ in range(3)]
    other = acquire_lease(lib, handles[1])
    for index, remaining in ((1, 2), (0, 1), (2, 0)):
        assert lib.sidecard_group_graph_lease_release(c.byref(leases[index]), 1) == 0
        assert lease_count(lib, handles[0]) == remaining and lease_count(lib, handles[1]) == 1
    destroy(lib, handles[0])
    assert lib.sidecard_group_destroy(c.byref(handles[1]), 1) == -5
    assert lib.sidecard_group_graph_lease_release(c.byref(other), 1) == 0
    destroy(lib, handles[1])


def test_acquisition_rejects_occupied_slot_without_losing_its_original_owner(native):
    lib, state = native
    cases = [geometry(), geometry()]
    handles = [create(lib, case) for case in cases]
    lease = acquire_lease(lib, handles[0])
    before, allocations = lease.value, state.allocations
    assert lib.sidecard_group_graph_lease_acquire(handles[1], c.byref(lease)) == -1
    assert lease.value == before and state.allocations == allocations
    assert lease_count(lib, handles[0]) == 1 and lease_count(lib, handles[1]) == 0
    assert lib.sidecard_group_destroy(c.byref(handles[0]), 1) == -5
    destroy(lib, handles[1])
    assert lib.sidecard_group_graph_lease_release(c.byref(lease), 1) == 0
    destroy(lib, handles[0])


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", [1, 2])
def test_cold_lease_allocation_failure_keeps_group_and_existing_leases(native, existing, failure):
    lib, state = native
    case = geometry()
    handle = create(lib, case)
    first = acquire_lease(lib, handle) if existing else None
    state.fail_allocation = state.allocations + failure
    failed = c.c_void_p()
    assert lib.sidecard_group_graph_lease_acquire(handle, c.byref(failed)) == -2
    assert failed.value is None and lease_count(lib, handle) == int(existing)
    assert state.exchange_destroys == 0 and metadata(lib, handle)[0] == 0
    state.fail_allocation = 0
    # Host allocation failure submitted no device work, so cold acquisition can
    # be retried without an SDK retry, quarantine bypass or lost prior ticket.
    recovered = acquire_lease(lib, handle)
    assert lease_count(lib, handle) == int(existing) + 1
    assert lib.sidecard_group_graph_lease_release(c.byref(recovered), 1) == 0
    if first is not None:
        assert lib.sidecard_group_graph_lease_release(c.byref(first), 1) == 0
    destroy(lib, handle)


def test_quarantine_blocks_new_leases_but_explicit_retirement_can_release_old_lease(native):
    lib, state = native
    case = geometry((0,))
    handle, stream = create(lib, case), Stream()
    lease = acquire_lease(lib, handle)
    stream.fail_at = 1
    assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 1) == 71
    failed = c.c_void_p()
    assert lib.sidecard_group_graph_lease_acquire(handle, c.byref(failed)) == -3
    assert failed.value is None and lease_count(lib, handle) == 1
    assert lib.sidecard_group_destroy(c.byref(handle), 1) == -5 and state.exchange_destroys == 0
    # This is the caller's proof, not native inspection of a reset or stream.
    # A real failure without that proof must retain this lease and every owner.
    assert lib.sidecard_group_graph_lease_release(c.byref(lease), 1) == 0
    assert lease.value is None and lease_count(lib, handle) == 0
    destroy(lib, handle)


def test_graph_ownership_has_no_submission_allocations_or_replay_counters(native):
    lib, state = native
    case = geometry((1,))
    handle, stream = create(lib, case), Stream()
    lease = acquire_lease(lib, handle)
    allocations = state.allocations
    for _ in range(20):
        assert lib.sidecard_group_enqueue(handle, case["input"], case["output"], c.byref(stream), 3) == 0
    assert stream.count == 20 and state.allocations == allocations
    assert lease_count(lib, handle) == 1 and state.exchange_destroys == 0
    assert lib.sidecard_group_destroy(c.byref(handle), 1) == -5
    assert lib.sidecard_group_graph_lease_release(c.byref(lease), 1) == 0
    destroy(lib, handle)


def test_invalid_graph_lease_arguments_do_not_allocate_or_mutate_group(native):
    lib, state = native
    case = geometry()
    handle = create(lib, case)
    before = state.allocations
    output = c.c_void_p()
    count = c.c_uint64(999)
    assert lib.sidecard_group_graph_lease_acquire(None, c.byref(output)) == -1 and output.value is None
    assert lib.sidecard_group_graph_lease_acquire(handle, None) == -1
    assert lib.sidecard_group_graph_lease_release(None, 1) == -1
    assert lib.sidecard_group_graph_lease_release(c.byref(output), 1) == -1
    assert lib.sidecard_group_graph_lease_count(None, c.byref(count)) == -1 and count.value == 999
    assert lib.sidecard_group_graph_lease_count(handle, None) == -1
    assert state.allocations == before and lease_count(lib, handle) == 0
    destroy(lib, handle)


def test_graph_lease_cannot_be_manipulated_by_foreign_pid_but_can_move_host_threads(native):
    lib, _ = native
    case = geometry()
    handle = create(lib, case)
    lease = acquire_lease(lib, handle)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        foreign = c.c_void_p()
        count = c.c_uint64(999)
        statuses = [
            lib.sidecard_group_graph_lease_acquire(handle, c.byref(foreign)),
            lib.sidecard_group_graph_lease_count(handle, c.byref(count)),
            lib.sidecard_group_graph_lease_release(c.byref(lease), 1),
            lib.sidecard_group_destroy(c.byref(handle), 1),
        ]
        statuses.extend((int(foreign.value is None), count.value, int(bool(lease.value)), int(bool(handle.value))))
        os.write(write_fd, repr(statuses).encode())
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    raw = os.read(read_fd, 256)
    os.close(read_fd)
    assert os.waitpid(pid, 0)[1] == 0
    assert raw == b"[-4, -4, -4, -4, 1, 999, 1, 1]"
    assert lease_count(lib, handle) == 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(lib.sidecard_group_graph_lease_release, c.byref(lease), 1).result() == 0
    assert lease.value is None and lease_count(lib, handle) == 0
    destroy(lib, handle)
