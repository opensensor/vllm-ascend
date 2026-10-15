# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual exchange C ABI against a CPU-only SDK implementation.

These tests qualify metadata, byte semantics and submission order. The stub
does not implement peer rendezvous, real asynchronous execution or NPU math.
Deferred math models executor address snapshots; the actual SDK must establish
that guarantee in changed-buffer device tests before candidate admission.
"""

import ctypes as c
import itertools
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DATA_FIELDS = ("input", "output", "inbox0", "inbox1", "peer_inbox0", "peer_inbox1")
STUB_CAPACITY = 32
STUB_HEADER = r"""
#pragma once
#include <cstdint>
#include <cstddef>
enum aclDataType { ACL_FLOAT = 0, ACL_FLOAT16 = 1 };
constexpr int ACL_FORMAT_ND = 2, ACL_MEMCPY_DEVICE_TO_DEVICE = 3;
constexpr int TEST_CAPACITY = 32;
struct aclTensor { void* data; int64_t size; aclDataType dtype; };
struct aclScalar { float value; };
struct aclOpExecutor { aclTensor a, b, out; bool used; };
struct State {
    int make_calls, fail_make_at, destroys, fail_destroy_at;
    int plans, plan_status;
    uint64_t workspace;
    int scalar_dtype;
    float scalar_value;
    int tensor_dtypes[TEST_CAPACITY];
    uintptr_t tensor_pointers[TEST_CAPACITY];
    int setters, fail_setter_at, defer_add, pending_count;
    uintptr_t set_pointers[TEST_CAPACITY];
};
struct Stream {
    int count, fail_at;
    int operations[TEST_CAPACITY];
    uintptr_t destinations[TEST_CAPACITY], sources[TEST_CAPACITY];
    uint64_t bytes[TEST_CAPACITY];
    uint32_t timeouts[TEST_CAPACITY];
};
aclTensor* aclCreateTensor(const int64_t*, uint64_t, aclDataType, const int64_t*,
                         int64_t, int, const int64_t*, uint64_t, void*);
aclScalar* aclCreateScalar(void*, aclDataType);
int aclSetRawTensorAddr(aclTensor*, void*);
int aclDestroyTensor(aclTensor*);
int aclDestroyScalar(aclScalar*);
int aclrtMemcpyAsync(void*, size_t, const void*, size_t, int, void*);
int aclrtRecordNotify(void*, void*);
int aclrtWaitAndResetNotify(void*, void*, uint32_t);
int aclnnAddGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclScalar*,
                           aclTensor*, uint64_t*, aclOpExecutor**);
int aclnnAdd(void*, uint64_t, aclOpExecutor*, void*);
"""
STUB_SOURCE = r"""
#include <acl/acl.h>
#include <cstring>
extern "C" State* stub_state() { static State s{}; return &s; }
int operation(void* ptr, int code, void* dst, const void* src, size_t bytes, uint32_t timeout = 0) {
    auto* stream = static_cast<Stream*>(ptr);
    const int i = stream->count++;
    if (i >= TEST_CAPACITY) return 96;
    stream->operations[i] = code;
    stream->destinations[i] = reinterpret_cast<uintptr_t>(dst);
    stream->sources[i] = reinterpret_cast<uintptr_t>(src);
    stream->bytes[i] = bytes;
    stream->timeouts[i] = timeout;
    return stream->count == stream->fail_at ? 71 : 0;
}
aclTensor* aclCreateTensor(const int64_t* n, uint64_t rank, aclDataType dtype,
                         const int64_t* stride, int64_t offset, int format,
                         const int64_t* storage, uint64_t storage_rank, void* data) {
    auto& s = *stub_state();
    const int i = s.make_calls++;
    if (s.make_calls == s.fail_make_at) return nullptr;
    if (i >= TEST_CAPACITY) return nullptr;
    if (rank != 1 || *stride != 1 || offset || format != ACL_FORMAT_ND ||
        *storage != *n || storage_rank != 1) return nullptr;
    s.tensor_dtypes[i] = dtype;
    s.tensor_pointers[i] = reinterpret_cast<uintptr_t>(data);
    return new aclTensor{data, *n, dtype};
}
aclScalar* aclCreateScalar(void* data, aclDataType dtype) {
    auto& s = *stub_state();
    ++s.make_calls;
    if (s.make_calls == s.fail_make_at) return nullptr;
    s.scalar_dtype = dtype;
    s.scalar_value = *static_cast<float*>(data);
    return new aclScalar{s.scalar_value};
}
int aclDestroyTensor(aclTensor* t) {
    auto& s = *stub_state();
    if (++s.destroys == s.fail_destroy_at) return 81;
    delete t; return 0;
}
int aclDestroyScalar(aclScalar* t) {
    auto& s = *stub_state();
    if (++s.destroys == s.fail_destroy_at) return 81;
    delete t; return 0;
}
int aclSetRawTensorAddr(aclTensor* tensor, void* pointer) {
    auto& s = *stub_state();
    const int index = s.setters++;
    if (index >= TEST_CAPACITY) return 84;
    s.set_pointers[index] = reinterpret_cast<uintptr_t>(pointer);
    if (s.setters == s.fail_setter_at) return 84;
    tensor->data = pointer;
    return 0;
}
int aclrtMemcpyAsync(void* dst, size_t capacity, const void* src, size_t bytes, int kind, void* stream) {
    if (capacity != bytes || kind != ACL_MEMCPY_DEVICE_TO_DEVICE) return 93;
    const int status = operation(stream, 1, dst, src, bytes);
    if (!status) std::memcpy(dst, src, bytes);
    return status;
}
int aclrtRecordNotify(void* notify, void* stream) {
    return operation(stream, 2, notify, nullptr, 0);
}
int aclrtWaitAndResetNotify(void* notify, void* stream, uint32_t timeout) {
    return operation(stream, 3, notify, nullptr, 0, timeout);
}
int aclnnAddGetWorkspaceSize(const aclTensor* a, const aclTensor* b, const aclScalar* alpha,
                           aclTensor* out, uint64_t* workspace, aclOpExecutor** executor) {
    auto& s = *stub_state();
    ++s.plans;
    if (s.plan_status) return s.plan_status;
    if (a->dtype != b->dtype || a->dtype != out->dtype || alpha->value != 1.0F) return 94;
    if (s.plans > TEST_CAPACITY) return 94;
    static aclOpExecutor snapshots[TEST_CAPACITY];
    auto& current = snapshots[s.plans - 1];
    current = {*a, *b, *out, false};
    *workspace = s.workspace;
    *executor = &current;
    return 0;
}
void add(const aclOpExecutor* e) {
    if (e->a.dtype == ACL_FLOAT) {
        auto* a = static_cast<float*>(e->a.data);
        auto* b = static_cast<float*>(e->b.data);
        auto* out = static_cast<float*>(e->out.data);
        for (int64_t i = 0; i < e->a.size; ++i) out[i] = a[i] + b[i];
    } else {
        auto* a = static_cast<_Float16*>(e->a.data);
        auto* b = static_cast<_Float16*>(e->b.data);
        auto* out = static_cast<_Float16*>(e->out.data);
        for (int64_t i = 0; i < e->a.size; ++i) out[i] = a[i] + b[i];
    }
}
aclOpExecutor* pending[TEST_CAPACITY];
extern "C" void stub_flush() {
    auto& s = *stub_state();
    for (int index = 0; index < s.pending_count; ++index) add(pending[index]);
    s.pending_count = 0;
}
int aclnnAdd(void*, uint64_t workspace, aclOpExecutor* e, void* stream) {
    if (e->used || workspace != stub_state()->workspace) return 95;
    e->used = true;
    const size_t item_bytes = e->a.dtype == ACL_FLOAT ? 4 : 2;
    const int status = operation(stream, 4, e->out.data, e->b.data, e->a.size * item_bytes);
    if (status) return status;
    if (stub_state()->defer_add) {
        if (stub_state()->pending_count < 0 || stub_state()->pending_count >= TEST_CAPACITY) return 95;
        pending[stub_state()->pending_count++] = e;
    }
    else add(e);
    return 0;
}
"""


class Bindings(c.Structure):
    _fields_ = (
        [(name, c.c_uint64) for name in ("abi", "elements", "dtype")]
        + [(name, c.c_void_p) for name in DATA_FIELDS + ("ready0", "ready1", "peer_ready0", "peer_ready1", "workspace")]
        + [("workspace_bytes", c.c_uint64)]
    )


class State(c.Structure):
    _fields_ = (
        [
            (name, c.c_int)
            for name in ("make_calls", "fail_make_at", "destroys", "fail_destroy_at", "plans", "plan_status")
        ]
        + [("workspace", c.c_uint64), ("scalar_dtype", c.c_int), ("scalar_value", c.c_float)]
        + [("tensor_dtypes", c.c_int * STUB_CAPACITY), ("tensor_pointers", c.c_size_t * STUB_CAPACITY)]
        + [(name, c.c_int) for name in ("setters", "fail_setter_at", "defer_add", "pending_count")]
        + [("set_pointers", c.c_size_t * STUB_CAPACITY)]
    )


class Stream(c.Structure):
    _fields_ = [
        ("count", c.c_int),
        ("fail_at", c.c_int),
        ("operations", c.c_int * STUB_CAPACITY),
        ("destinations", c.c_size_t * STUB_CAPACITY),
        ("sources", c.c_size_t * STUB_CAPACITY),
        ("bytes", c.c_uint64 * STUB_CAPACITY),
        ("timeouts", c.c_uint32 * STUB_CAPACITY),
    ]


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("exchange ABI tests require a CPU C++ compiler")
    root = tmp_path_factory.mktemp("exchange-native")
    for name in ("acl/acl.h", "aclnnop/aclnn_add.h"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(STUB_HEADER if name == "acl/acl.h" else "#include <acl/acl.h>\n")
    (root / "stub.cpp").write_text(STUB_SOURCE)
    output = root / "exchange.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root),
            str(ROOT / "csrc/sidecard_collectives/npu_exchange.cpp"),
            str(root / "stub.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    lib = c.CDLL(str(output))
    lib.sidecard_exchange_abi.restype = c.c_uint64
    lib.sidecard_exchange_bindings_bytes.restype = c.c_uint64
    lib.sidecard_exchange_create.argtypes = [c.POINTER(Bindings), c.POINTER(c.c_void_p)]
    lib.sidecard_exchange_set_fixed_handshake.argtypes = [c.c_void_p]
    lib.sidecard_exchange_enqueue.argtypes = [c.c_void_p, c.c_void_p, c.c_uint32, c.c_uint32]
    lib.sidecard_exchange_enqueue_io.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint32, c.c_uint32]
    lib.sidecard_exchange_destroy.argtypes = [c.POINTER(c.c_void_p), c.c_uint32]
    lib.stub_state.restype = c.POINTER(State)
    lib.stub_flush.restype = None
    return lib


@pytest.fixture
def native(compiled):
    state = compiled.stub_state().contents
    c.memset(c.addressof(state), 0, c.sizeof(state))
    return compiled, state


def geometry(dtype=0, elements=17):
    kind = c.c_float if dtype == 0 else c.c_uint16
    arrays = [(kind * elements)() for _ in DATA_FIELDS]
    workspace = (c.c_ubyte * (1024 * 1024))()
    bindings = Bindings(
        1, elements, dtype, *[c.addressof(a) for a in arrays], 11, 12, 21, 22, c.addressof(workspace), len(workspace)
    )
    for index, array in enumerate(arrays):
        values = [float((i % 7) - 3 + index) for i in range(elements)]
        if dtype == 1:
            array[:] = [int.from_bytes(struct.pack("<e", v), "little") for v in values]
        else:
            array[:] = values
    return arrays, workspace, bindings


def values(array, dtype):
    if dtype == 0:
        return list(array)
    return [struct.unpack("<e", value.to_bytes(2, "little"))[0] for value in array]


def create(lib, bindings):
    handle = c.c_void_p()
    assert lib.sidecard_exchange_create(c.byref(bindings), c.byref(handle)) == 0
    return handle


def destroy(lib, handle):
    assert lib.sidecard_exchange_destroy(c.byref(handle), 1) == 0
    assert handle.value is None


def test_exact_abi_and_no_metadata_getters(native):
    lib, state = native
    assert lib.sidecard_exchange_abi() == 1
    assert lib.sidecard_exchange_bindings_bytes() == c.sizeof(Bindings) == 120
    assert Bindings.workspace_bytes.offset == 112
    assert state.make_calls == 0


def test_two_live_exchanges_record_all_descriptors_without_corrupting_stub_state(native):
    lib, state = native
    cases = [geometry(dtype=0), geometry(dtype=1)]
    handles = [create(lib, case[2]) for case in cases]
    # make_calls includes each scalar, so the second owner's four descriptors
    # occupy slots 5..8. A four-slot stub previously overwrote its deferred-add
    # flags/pending count here, producing allocator-dependent full-suite crashes.
    assert state.make_calls == 10
    assert list(state.tensor_dtypes)[:10] == [0] * 5 + [1] * 4 + [0]
    assert list(state.tensor_pointers)[:10] == [
        cases[0][2].input,
        cases[0][2].output,
        cases[0][2].inbox0,
        cases[0][2].inbox1,
        0,
        cases[1][2].input,
        cases[1][2].output,
        cases[1][2].inbox0,
        cases[1][2].inbox1,
        0,
    ]
    assert (state.setters, state.fail_setter_at, state.defer_add, state.pending_count) == (0, 0, 0, 0)
    for dtype, ((arrays, _workspace, _bindings), handle) in enumerate(zip(cases, handles, strict=True)):
        stream = Stream()
        wanted = [a + b for a, b in zip(values(arrays[0], dtype), values(arrays[2], dtype))]
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 31, 0) == 0
        assert values(arrays[1], dtype) == wanted
    assert state.pending_count == state.defer_add == 0 and state.plans == 2 and state.setters == 4
    for handle in handles:
        destroy(lib, handle)
    assert state.destroys == 10


def test_stub_metadata_history_exhaustion_returns_partial_owner_without_overwrite(native):
    lib, state = native
    cases = [geometry() for _ in range(7)]
    handles = []
    for index, case in enumerate(cases):
        handle = c.c_void_p()
        status = lib.sidecard_exchange_create(c.byref(case[2]), c.byref(handle))
        handles.append(handle)
        assert handle.value is not None
        assert status == (0 if index < 6 else -2)
    assert state.make_calls == STUB_CAPACITY + 1
    assert (state.setters, state.fail_setter_at, state.defer_add, state.pending_count) == (0, 0, 0, 0)
    for handle in handles:
        destroy(lib, handle)
    assert state.destroys == STUB_CAPACITY


@pytest.mark.parametrize("dtype", (0, 1))
@pytest.mark.parametrize("operation", (0, 1, 2))
@pytest.mark.parametrize("elements", (1, 17, 4096))
def test_three_calls_preserve_dtype_slots_stream_and_local_only_math(native, dtype, operation, elements):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype, elements)
    handle, stream = create(lib, bindings), Stream()
    assert state.make_calls == 5
    assert list(state.tensor_dtypes)[:4] == [dtype] * 4
    assert list(state.tensor_pointers)[:4] == [bindings.input, bindings.output, bindings.inbox0, bindings.inbox1]
    assert state.scalar_dtype == 0 and state.scalar_value == 1.0
    for iteration, slot in enumerate((0, 1, 0)):
        # Distinct bit patterns across calls catch stale descriptor/executor use.
        arrays[0][0] = iteration + 1 if dtype == 0 else (0x3C00 + iteration * 0x400)
        inputs = bytes(arrays[0])
        inbox = arrays[2 + slot]
        inboxes = (bytes(arrays[2]), bytes(arrays[3]))
        other_peer = bytes(arrays[4 + (1 - slot)])
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 37, operation) == 0
        start = iteration * 4
        assert stream.count == start + 4
        assert list(stream.operations)[start : start + 4] == [1, 2, 3, 4 if operation == 0 else 1]
        assert list(stream.destinations)[start : start + 4] == [
            c.addressof(arrays[4 + slot]),
            21 + slot,
            11 + slot,
            bindings.output,
        ]
        expected_source = bindings.input if operation == 1 else c.addressof(inbox)
        assert list(stream.sources)[start : start + 4] == [bindings.input, 0, 0, expected_source]
        assert list(stream.timeouts)[start : start + 4] == [0, 0, 37, 0]
        assert stream.bytes[start] == stream.bytes[start + 3] == elements * (4 if dtype == 0 else 2)
        assert bytes(arrays[0]) == bytes(arrays[4 + slot]) == inputs
        assert (bytes(arrays[2]), bytes(arrays[3])) == inboxes
        assert bytes(arrays[4 + (1 - slot)]) == other_peer
        if operation == 0:
            assert values(arrays[1], dtype) == [a + b for a, b in zip(values(arrays[0], dtype), values(inbox, dtype))]
        else:
            assert bytes(arrays[1]) == (inputs if operation == 1 else bytes(inbox))
    assert state.plans == (3 if operation == 0 else 0)
    assert state.destroys == 0 and workspace
    destroy(lib, handle)
    assert state.destroys == 5


@pytest.mark.parametrize("dtype", (0, 1))
@pytest.mark.parametrize("operation", (1, 2))
def test_broadcast_preserves_signed_zero_nan_payload_and_infinity_bits(native, dtype, operation):
    lib, _ = native
    arrays, workspace, bindings = geometry(dtype, 4)
    bit_kind = c.c_uint32 if dtype == 0 else c.c_uint16
    patterns = (0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000) if dtype == 0 else (0x8000, 0x7E23, 0x7C00, 0xFC00)
    source = arrays[0] if operation == 1 else arrays[2]
    c.memmove(c.addressof(source), (bit_kind * 4)(*patterns), c.sizeof(source))
    handle, stream = create(lib, bindings), Stream()
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, operation) == 0
    assert bytes(arrays[1]) == bytes(source)
    destroy(lib, handle)
    assert workspace


@pytest.mark.parametrize(
    "field,value",
    (
        ("abi", 0),
        ("abi", 2),
        ("elements", 0),
        ("elements", 16 * 1024 * 1024 + 1),
        ("dtype", 2),
        ("workspace_bytes", 0),
        ("workspace_bytes", 1024 * 1024 + 1),
        ("ready0", 0),
        ("ready1", 0),
        ("peer_ready0", 0),
        ("peer_ready1", 0),
        ("ready1", 11),
        ("peer_ready1", 21),
    ),
)
def test_invalid_geometry_is_rejected_before_metadata(native, field, value):
    lib, state = native
    arrays, workspace, bindings = geometry()
    setattr(bindings, field, value)
    handle = c.c_void_p(123)
    assert lib.sidecard_exchange_create(c.byref(bindings), c.byref(handle)) == -1
    assert handle.value is None and state.make_calls == 0 and arrays and workspace


@pytest.mark.parametrize("dtype", (0, 1))
@pytest.mark.parametrize("field", DATA_FIELDS + ("workspace",))
@pytest.mark.parametrize("invalid", ("null", "unaligned", "overflow"))
def test_pointer_validation_precedes_sdk(native, dtype, field, invalid):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype)
    address = {"null": 0, "unaligned": getattr(bindings, field) + 1, "overflow": 2 ** (8 * c.sizeof(c.c_void_p)) - 4}[
        invalid
    ]
    setattr(bindings, field, address)
    handle = c.c_void_p()
    assert lib.sidecard_exchange_create(c.byref(bindings), c.byref(handle)) == -1
    assert handle.value is None and state.make_calls == 0 and arrays and workspace


@pytest.mark.parametrize("first,second", tuple(itertools.combinations(DATA_FIELDS + ("workspace",), 2)))
@pytest.mark.parametrize("offset", (0, 2, -2))
def test_all_storage_ranges_are_fully_disjoint(native, first, second, offset):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype=1)
    setattr(bindings, first, getattr(bindings, second) + offset)
    handle = c.c_void_p()
    assert lib.sidecard_exchange_create(c.byref(bindings), c.byref(handle)) == -1
    assert handle.value is None and state.make_calls == 0 and arrays and workspace


def test_maximum_elements_and_minimum_workspace_admitted_without_dereference(native):
    lib, _ = native
    # Fake ranges are used only by descriptor creation/destruction in the stub.
    bindings = Bindings(1, 16 * 1024 * 1024, 0, *[0x10000000 * i for i in range(1, 7)], 11, 12, 21, 22, 0x70000000, 1)
    handle = create(lib, bindings)
    destroy(lib, handle)


@pytest.mark.parametrize("failure", range(1, 6))
def test_partial_metadata_failure_retains_handle_until_explicit_proven_destroy(native, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    state.fail_make_at = failure
    handle, stream = c.c_void_p(), Stream()
    assert lib.sidecard_exchange_create(c.byref(bindings), c.byref(handle)) == -2
    assert handle.value and state.make_calls == failure and state.destroys == 0
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, 0) == -1
    assert stream.count == 0 and state.plans == 0
    assert lib.sidecard_exchange_destroy(c.byref(handle), 0) == -1
    assert handle.value and state.destroys == 0
    destroy(lib, handle)
    assert state.destroys == failure - 1 and arrays and workspace


@pytest.mark.parametrize("operation", (0, 1, 2))
@pytest.mark.parametrize("slot", (0, 1))
@pytest.mark.parametrize("failure", (1, 2, 3, 4))
def test_sdk_submission_failures_short_circuit_and_quarantine(native, operation, slot, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    handle, stream = create(lib, bindings), Stream()
    if slot:
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, operation) == 0
    start = stream.count
    output_before = bytes(arrays[1])
    stream.fail_at = start + failure
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, operation) == 71
    assert stream.count == start + failure
    assert list(stream.operations)[start : stream.count] == [1, 2, 3, 4 if operation == 0 else 1][:failure]
    assert bytes(arrays[1]) == output_before
    assert handle.value and state.destroys == 0
    stream.fail_at = 0
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, operation) == -1
    assert stream.count == start + failure
    destroy(lib, handle)
    assert workspace


@pytest.mark.parametrize("failure", ("planner", "workspace"))
def test_planning_failure_precedes_any_queued_peer_operation(native, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    state.plan_status = 91 if failure == "planner" else 0
    state.workspace = 1024 * 1024 + 1 if failure == "workspace" else 0
    handle, stream = create(lib, bindings), Stream()
    original = [bytes(a) for a in arrays]
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, 0) == (91 if failure == "planner" else -3)
    assert stream.count == 0 and [bytes(a) for a in arrays] == original
    assert state.destroys == 0
    destroy(lib, handle)
    assert workspace


@pytest.mark.parametrize("failure", range(1, 6))
def test_destroy_failure_retains_only_remaining_metadata_and_disallows_enqueue(native, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    handle, stream = create(lib, bindings), Stream()
    state.fail_destroy_at = failure
    assert lib.sidecard_exchange_destroy(c.byref(handle), 1) == 81
    assert handle.value and state.destroys == failure
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, 0) == -1
    assert stream.count == state.plans == 0
    state.fail_destroy_at = 0
    destroy(lib, handle)
    assert state.destroys == 6 and arrays and workspace


def test_invalid_entry_arguments_do_not_touch_sdk_or_consume_a_slot(native):
    lib, state = native
    arrays, workspace, bindings = geometry()
    handle, stream = create(lib, bindings), Stream()
    assert lib.sidecard_exchange_create(None, None) == -1
    assert lib.sidecard_exchange_create(c.byref(bindings), None) == -1
    assert lib.sidecard_exchange_enqueue(None, c.byref(stream), 30, 0) == -1
    assert lib.sidecard_exchange_enqueue(handle, None, 30, 0) == -1
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 0, 0) == -1
    for operation in (3, 2**32 - 1):
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, operation) == -1
    assert stream.count == state.plans == 0
    for proof in (0, 2, 2**32 - 1):
        assert lib.sidecard_exchange_destroy(c.byref(handle), proof) == -1
    assert lib.sidecard_exchange_destroy(None, 1) == -1
    assert state.destroys == 0 and handle.value
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 30, 2) == 0
    assert stream.destinations[0] == bindings.peer_inbox0
    destroy(lib, handle)
    assert lib.sidecard_exchange_destroy(c.byref(handle), 1) == -1
    assert arrays and workspace


@pytest.mark.parametrize("dtype", (0, 1))
@pytest.mark.parametrize("operation", (0, 1, 2))
def test_dynamic_io_uses_requested_pointers_and_permits_original_work_buffers(native, dtype, operation):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype)
    handle, stream = create(lib, bindings), Stream()
    # Reverse the original two work buffers, keeping both disjoint from inboxes.
    source, output = arrays[1], arrays[0]
    original = bytes(source)
    expected = [a + b for a, b in zip(values(source, dtype), values(arrays[2], dtype))]
    assert (
        lib.sidecard_exchange_enqueue_io(
            handle, c.addressof(source), c.addressof(output), c.byref(stream), 37, operation
        )
        == 0
    )
    assert state.setters == 2 and list(state.set_pointers)[:2] == [c.addressof(source), c.addressof(output)]
    assert stream.sources[0] == c.addressof(source) and stream.destinations[3] == c.addressof(output)
    assert bytes(source) == bytes(arrays[4]) == original
    if operation == 0:
        assert values(output, dtype) == expected
    else:
        assert bytes(output) == (original if operation == 1 else bytes(arrays[2]))
    assert state.make_calls == 5 and workspace
    destroy(lib, handle)


@pytest.mark.parametrize("dtype", (0, 1))
def test_different_dynamic_io_before_drain_keeps_single_use_executor_snapshots(native, dtype):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype)
    handle, stream = create(lib, bindings), Stream()
    state.defer_add = 1
    alternatives = [geometry(dtype)[0][:2] for _ in range(3)]
    expected = []
    originals = []
    for index, (source, output) in enumerate(alternatives):
        source[0] = index + 7 if dtype == 0 else 0x3C00 + index * 0x400
        originals.append(bytes(source))
        expected.append([a + b for a, b in zip(values(source, dtype), values(arrays[2 + index % 2], dtype))])
        before = bytes(output)
        assert (
            lib.sidecard_exchange_enqueue_io(handle, c.addressof(source), c.addressof(output), c.byref(stream), 41, 0)
            == 0
        )
        assert bytes(output) == before and state.pending_count == index + 1
    assert state.make_calls == 5 and state.plans == 3 and state.setters == 6
    assert list(state.set_pointers)[:6] == [c.addressof(array) for pair in alternatives for array in pair]
    lib.stub_flush()
    assert state.pending_count == 0
    for index, (source, output) in enumerate(alternatives):
        assert bytes(source) == originals[index] and values(output, dtype) == expected[index]
    # A legacy call after dynamic rebinding restores its original fixed I/O.
    state.defer_add = 0
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 41, 1) == 0
    assert bytes(arrays[1]) == bytes(arrays[0]) and workspace
    destroy(lib, handle)


@pytest.mark.parametrize("operation", (0, 1, 2))
@pytest.mark.parametrize("slot", (0, 1))
@pytest.mark.parametrize("failure", (1, 2))
def test_each_descriptor_setter_failure_precedes_planning_and_copy_and_quarantines(native, operation, slot, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    handle, stream = create(lib, bindings), Stream()
    if slot:
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 31, operation) == 0
    start, plans, setters = stream.count, state.plans, state.setters
    state.fail_setter_at = setters + failure
    before = [bytes(array) for array in arrays]
    assert (
        lib.sidecard_exchange_enqueue_io(handle, bindings.input, bindings.output, c.byref(stream), 31, operation) == 84
    )
    assert stream.count == start and state.plans == plans and state.setters == setters + failure
    assert [bytes(array) for array in arrays] == before and state.destroys == 0
    assert (
        lib.sidecard_exchange_enqueue_io(handle, bindings.input, bindings.output, c.byref(stream), 31, operation) == -1
    )
    assert state.setters == setters + failure and stream.count == start
    destroy(lib, handle)
    assert workspace


@pytest.mark.parametrize("dtype", (0, 1))
@pytest.mark.parametrize("which", ("input", "output"))
@pytest.mark.parametrize("forbidden", ("inbox0", "inbox1", "peer_inbox0", "peer_inbox1", "workspace"))
@pytest.mark.parametrize("offset", (-1, 0, 1))
def test_dynamic_io_rejects_inbox_and_workspace_overlap_before_any_sdk(native, dtype, which, forbidden, offset):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype)
    handle, stream = create(lib, bindings), Stream()
    pointers = dict(input=bindings.input, output=bindings.output)
    pointers[which] = getattr(bindings, forbidden) + offset * (4 if dtype == 0 else 2)
    assert lib.sidecard_exchange_enqueue_io(handle, pointers["input"], pointers["output"], c.byref(stream), 2, 0) == -1
    assert state.setters == state.plans == stream.count == 0
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 2, 0) == 0
    destroy(lib, handle)
    assert arrays and workspace


@pytest.mark.parametrize("dtype", (0, 1))
@pytest.mark.parametrize("which", ("input", "output"))
@pytest.mark.parametrize("invalid", ("null", "unaligned", "overflow", "same", "partial-positive", "partial-negative"))
def test_dynamic_io_checks_actual_full_ranges_without_quarantining_invalid_arguments(native, dtype, which, invalid):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype)
    handle, stream = create(lib, bindings), Stream()
    pointers = dict(input=bindings.input, output=bindings.output)
    other = pointers["output" if which == "input" else "input"]
    item_bytes = 4 if dtype == 0 else 2
    pointers[which] = {
        "null": 0,
        "unaligned": pointers[which] + 1,
        "overflow": 2 ** (8 * c.sizeof(c.c_void_p)) - item_bytes,
        "same": other,
        "partial-positive": other + item_bytes,
        "partial-negative": other - item_bytes,
    }[invalid]
    assert lib.sidecard_exchange_enqueue_io(handle, pointers["input"], pointers["output"], c.byref(stream), 2, 0) == -1
    assert state.setters == state.plans == stream.count == 0
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 2, 0) == 0
    destroy(lib, handle)
    assert arrays and workspace


@pytest.mark.parametrize("dtype", (0, 1))
@pytest.mark.parametrize("operation", (0, 1, 2))
@pytest.mark.parametrize("calls", (1, 3, 5))
def test_fixed_handshake_repeats_exact_ready_consumed_addresses_across_odd_calls(native, dtype, operation, calls):
    lib, state = native
    arrays, workspace, bindings = geometry(dtype)
    handle, stream = create(lib, bindings), Stream()
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == 0
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == -1
    assert state.setters == state.plans == stream.count == 0
    unused = bytes(arrays[3]), bytes(arrays[5])
    for iteration in range(calls):
        arrays[0][0] = iteration + 1 if dtype == 0 else 0x3C00 + iteration * 0x400
        expected = [a + b for a, b in zip(values(arrays[0], dtype), values(arrays[2], dtype))]
        start = stream.count
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 43, operation) == 0
        assert list(stream.operations)[start : stream.count] == [1, 2, 3, 4 if operation == 0 else 1, 2, 3]
        assert list(stream.destinations)[start : stream.count] == [
            bindings.peer_inbox0,
            bindings.peer_ready0,
            bindings.ready0,
            bindings.output,
            bindings.peer_ready1,
            bindings.ready1,
        ]
        assert list(stream.timeouts)[start : stream.count] == [0, 0, 43, 0, 0, 43]
        assert stream.sources[start] == bindings.input
        assert stream.sources[start + 3] == (bindings.input if operation == 1 else bindings.inbox0)
        if operation == 0:
            assert values(arrays[1], dtype) == expected
        else:
            assert bytes(arrays[1]) == bytes(arrays[0 if operation == 1 else 2])
        assert (bytes(arrays[3]), bytes(arrays[5])) == unused
        assert lib.sidecard_exchange_set_fixed_handshake(handle) == -1
    assert state.make_calls == 5 and state.destroys == 0 and workspace
    destroy(lib, handle)


@pytest.mark.parametrize("operation", (0, 1, 2))
@pytest.mark.parametrize("prior_calls", (0, 1))
@pytest.mark.parametrize("failure", range(1, 7))
def test_fixed_handshake_all_submission_failures_stop_and_quarantine(native, operation, prior_calls, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    handle, stream = create(lib, bindings), Stream()
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == 0
    if prior_calls:
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 47, operation) == 0
    start, setters = stream.count, state.setters
    stream.fail_at = start + failure
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 47, operation) == 71
    assert stream.count == start + failure
    assert list(stream.operations)[start : stream.count] == [1, 2, 3, 4 if operation == 0 else 1, 2, 3][:failure]
    assert state.setters == setters + 2 and state.destroys == 0
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == -1
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 47, operation) == -1
    assert stream.count == start + failure
    destroy(lib, handle)
    assert arrays and workspace


@pytest.mark.parametrize("failure", ("setter-input", "setter-output", "planner", "workspace"))
def test_fixed_handshake_metadata_failures_precede_copy_and_prevent_mode_changes(native, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    handle, stream = create(lib, bindings), Stream()
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == 0
    if failure.startswith("setter"):
        state.fail_setter_at = 1 if failure == "setter-input" else 2
        expected_status = 84
    elif failure == "planner":
        state.plan_status = expected_status = 91
    else:
        state.workspace = 1024 * 1024 + 1
        expected_status = -3
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 47, 0) == expected_status
    assert stream.count == 0 and state.destroys == 0
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == -1
    destroy(lib, handle)
    assert arrays and workspace


@pytest.mark.parametrize("failure", range(1, 6))
def test_partial_exchange_cannot_enable_fixed_handshake(native, failure):
    lib, state = native
    arrays, workspace, bindings = geometry()
    state.fail_make_at = failure
    handle = c.c_void_p()
    assert lib.sidecard_exchange_create(c.byref(bindings), c.byref(handle)) == -2
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == -1
    assert state.setters == state.plans == 0
    destroy(lib, handle)
    assert arrays and workspace


@pytest.mark.parametrize("prior", ("valid", "invalid-stream", "invalid-timeout", "invalid-operation", "invalid-io"))
def test_protocol_selection_is_one_shot_and_frozen_after_any_enqueue_attempt(native, prior):
    lib, state = native
    arrays, workspace, bindings = geometry()
    handle, stream = create(lib, bindings), Stream()
    assert lib.sidecard_exchange_set_fixed_handshake(None) == -1
    if prior == "invalid-io":
        assert lib.sidecard_exchange_enqueue_io(handle, None, bindings.output, c.byref(stream), 3, 0) == -1
    else:
        stream_arg = None if prior == "invalid-stream" else c.byref(stream)
        timeout = 0 if prior == "invalid-timeout" else 3
        operation = 3 if prior == "invalid-operation" else 0
        assert lib.sidecard_exchange_enqueue(handle, stream_arg, timeout, operation) == (0 if prior == "valid" else -1)
    before = stream.count, state.setters, state.plans
    assert lib.sidecard_exchange_set_fixed_handshake(handle) == -1
    assert (stream.count, state.setters, state.plans) == before
    # The failed mode mutation never changes the default ping-pong schedule.
    start = stream.count
    assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 3, 1) == 0
    assert stream.count == start + 4
    assert stream.destinations[start] == (bindings.peer_inbox1 if prior == "valid" else bindings.peer_inbox0)
    destroy(lib, handle)
    assert arrays and workspace


def captured_steps(stream):
    return [
        tuple(
            field[index]
            for field in (stream.operations, stream.destinations, stream.sources, stream.bytes, stream.timeouts)
        )
        for index in range(stream.count)
    ]


def replay_static_tasks(traces, bindings, *, repeats, preferred_rank):
    """Adversarial CPU stream model, driven by actual native submitted tasks.

    It runs the preferred peer until it blocks, then the other. Notify is a
    one-bit record/reset token. Every inbox write is tagged with its call index;
    reading an overwritten or stale call fails independently of result values.
    This models task order only, never actual SDK capture/replay semantics.
    """
    positions, memory, notifications, sums = [0, 0], {}, {}, [[], []]
    lengths = [len(trace) for trace in traces]
    while any(positions[rank] < lengths[rank] * repeats for rank in range(2)):
        advanced = False
        for rank in (preferred_rank, 1 - preferred_rank):
            if positions[rank] == lengths[rank] * repeats:
                continue
            call, offset = divmod(positions[rank], lengths[rank])
            kind, destination, source, _bytes, _timeout = traces[rank][offset]
            if kind == 3 and not notifications.get(destination, False):
                continue
            if kind == 1:
                assert source == bindings[rank].input
                memory[destination] = (call, rank * 100 + call + 1)
            elif kind == 2:
                assert not notifications.get(destination, False), "Notify overwritten before reset"
                notifications[destination] = True
            elif kind == 3:
                notifications[destination] = False
            else:
                generation, peer_value = memory[source]
                assert generation == call, "inbox overwritten before consumption or stale replay"
                sums[rank].append(rank * 100 + call + 1 + peer_value)
            positions[rank] += 1
            advanced = True
            break
        assert advanced, "static task replay deadlocked"
    assert all(not pending for pending in notifications.values())
    assert sums == [[102 + 2 * call for call in range(repeats)]] * 2


@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("preferred_rank", (0, 1))
def test_static_native_task_replay_with_odd_calls_and_eager_interleave(native, fixed, preferred_rank):
    lib, _ = native
    geometries = [geometry(elements=1), geometry(elements=1)]
    bindings = [case[2] for case in geometries]
    for rank, binding in enumerate(bindings):
        peer = bindings[1 - rank]
        binding.peer_inbox0, binding.peer_inbox1 = peer.inbox0, peer.inbox1
        binding.ready0, binding.ready1 = 11 + rank * 10, 12 + rank * 10
        binding.peer_ready0, binding.peer_ready1 = 21 - rank * 10, 22 - rank * 10
    handles, traces = [], []
    for binding in bindings:
        handle, stream = create(lib, binding), Stream()
        handles.append(handle)
        if fixed:
            assert lib.sidecard_exchange_set_fixed_handshake(handle) == 0
        assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 31, 0) == 0
        traces.append(captured_steps(stream))
    if fixed:
        # Replaying a single static capture for three calls, one eager call and
        # another replay uses the exact same six-task address schedule.
        replay_static_tasks(traces, bindings, repeats=5, preferred_rank=preferred_rank)
        for handle, trace in zip(handles, traces):
            stream = Stream()
            assert lib.sidecard_exchange_enqueue(handle, c.byref(stream), 31, 0) == 0
            assert captured_steps(stream) == trace
    else:
        # Negative control: replaying the first ping-pong slot without the
        # consumed handshake lets a faster peer overwrite an unread inbox.
        with pytest.raises(AssertionError, match="overwritten"):
            replay_static_tasks(traces, bindings, repeats=5, preferred_rank=preferred_rank)
    for handle in handles:
        destroy(lib, handle)
