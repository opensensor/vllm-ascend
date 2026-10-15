# SPDX-License-Identifier: Apache-2.0
"""Compile the actual shim against a CPU SDK stub to check its C ABI/order.

The stub performs no hardware synchronization; the separate adversarial stream
model tests notification/reuse ordering. Neither constitutes NPU validation.
"""

import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.sidecard.pair_collective_acl import Bindings

ROOT = Path(__file__).resolve().parents[3]
STUB_HEADER = r"""
#pragma once
#include <cstdint>
#include <cstddef>
constexpr int ACL_FLOAT = 0, ACL_FORMAT_ND = 2, ACL_MEMCPY_DEVICE_TO_DEVICE = 3;
struct aclTensor { float* data; int64_t size; };
struct aclScalar { float value; };
struct aclOpExecutor { const aclTensor *a, *b; aclTensor* out; };
struct TestStream {
    int count; int fail_at; int operations[16];
    uintptr_t input, other, output;
    uintptr_t resources[16];
    uint32_t timeouts[16];
};
aclTensor* aclCreateTensor(const int64_t*, uint64_t, int, const int64_t*, int64_t,
                           int, const int64_t*, uint64_t, void*);
aclScalar* aclCreateScalar(void*, int);
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
extern "C" int* stub_tensor_creations() { static int count = 0; return &count; }
extern "C" int* stub_tensor_destructions() { static int count = 0; return &count; }
extern "C" int* stub_fail_tensor_at() { static int count = 0; return &count; }
int operation(void* ptr, int code, void* resource = nullptr, uint32_t timeout = 0) {
    auto* stream = static_cast<TestStream*>(ptr);
    stream->resources[stream->count] = reinterpret_cast<uintptr_t>(resource);
    stream->timeouts[stream->count] = timeout;
    stream->operations[stream->count++] = code;
    return stream->count == stream->fail_at ? 55 : 0;
}
aclTensor* aclCreateTensor(const int64_t* n, uint64_t, int, const int64_t*, int64_t,
                           int, const int64_t*, uint64_t, void* p) {
    ++*stub_tensor_creations();
    if (*stub_tensor_creations() == *stub_fail_tensor_at()) return nullptr;
    return new aclTensor{static_cast<float*>(p), *n};
}
aclScalar* aclCreateScalar(void* v, int) { return new aclScalar{*static_cast<float*>(v)}; }
int aclDestroyTensor(aclTensor* t) { ++*stub_tensor_destructions(); delete t; return 0; }
int aclDestroyScalar(aclScalar* s) { delete s; return 0; }
int aclrtMemcpyAsync(void* dst, size_t, const void* src, size_t n, int, void* s) {
    int status = operation(s, 1, dst);
    if (!status) std::memcpy(dst, src, n);
    return status;
}
int aclrtRecordNotify(void* notify, void* s) { return operation(s, 2, notify); }
int aclrtWaitAndResetNotify(void* notify, void* s, uint32_t timeout) { return operation(s, 3, notify, timeout); }
int aclnnAddGetWorkspaceSize(const aclTensor* a, const aclTensor* b, const aclScalar*,
                            aclTensor* out, uint64_t* bytes, aclOpExecutor** executor) {
    if (a->data[0] < 0) return 66;
    *bytes = static_cast<uint64_t>(a->data[0]);
    *executor = new aclOpExecutor{a, b, out};
    return 0;
}
int aclnnAdd(void*, uint64_t, aclOpExecutor* e, void* s) {
    auto* stream = static_cast<TestStream*>(s);
    stream->input = reinterpret_cast<uintptr_t>(e->a->data);
    stream->other = reinterpret_cast<uintptr_t>(e->b->data);
    stream->output = reinterpret_cast<uintptr_t>(e->out->data);
    int status = operation(s, 4, e->b->data);
    if (!status) for (int64_t i = 0; i < e->a->size; ++i)
        e->out->data[i] = e->a->data[i] + e->b->data[i];
    delete e;
    return status;
}
"""


class Stream(c.Structure):
    _fields_ = [
        ("count", c.c_int),
        ("fail_at", c.c_int),
        ("operations", c.c_int * 16),
        ("input", c.c_size_t),
        ("other", c.c_size_t),
        ("output", c.c_size_t),
        ("resources", c.c_size_t * 16),
        ("timeouts", c.c_uint32 * 16),
    ]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native ABI tests require a C++ compiler")
    root = tmp_path_factory.mktemp("pair-native-stub")
    for name in ("acl/acl.h", "aclnnop/aclnn_add.h"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(STUB_HEADER if name.startswith("acl/") else "#include <acl/acl.h>\n")
    (root / "stub.cpp").write_text(STUB_SOURCE)
    output = root / "pair.so"
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
            str(ROOT / "csrc/sidecard_collectives/pair_sum.cpp"),
            str(root / "stub.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    lib = c.CDLL(str(output))
    lib.stub_tensor_creations.restype = c.POINTER(c.c_int)
    lib.stub_tensor_destructions.restype = c.POINTER(c.c_int)
    lib.stub_fail_tensor_at.restype = c.POINTER(c.c_int)
    lib.sidecard_pair_abi.restype = c.c_uint64
    lib.sidecard_pair_bindings_bytes.restype = c.c_uint64
    lib.sidecard_pair_create.argtypes = [c.POINTER(Bindings), c.POINTER(c.c_void_p)]
    lib.sidecard_pair_enqueue.argtypes = [c.c_void_p, c.c_void_p, c.c_uint32]
    lib.sidecard_pair_destroy.argtypes = [c.c_void_p]
    lib.sidecard_pair_set_local_add.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p]
    lib.sidecard_pair_enable_ping_pong.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p]
    return lib


@pytest.mark.parametrize("failure", (False, True))
def test_bound_local_add_bypasses_aclnn_planning_and_preserves_handshake(native, failure):
    arrays, bindings = buffers()
    arrays[0][0] = -1  # Stub ACLNN planning fails for this value if called.
    handle, stream = c.c_void_p(), Stream()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0

    @c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)
    def local_add(add_handle, native_stream):
        assert add_handle == 123 and native_stream == c.addressof(stream)
        stream.operations[stream.count] = 4
        stream.count += 1
        if failure:
            return 55
        for index in range(len(arrays[0])):
            arrays[2][index] = arrays[0][index] + arrays[1][index]
        return 0

    callback = c.cast(local_add, c.c_void_p)
    assert native.sidecard_pair_set_local_add(handle, 123, callback) == 0
    assert native.sidecard_pair_set_local_add(handle, 123, callback) == -1
    assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == (55 if failure else 0)
    assert list(stream.operations)[: stream.count] == ([1, 2, 3, 4] if failure else [1, 2, 3, 4, 2, 3])
    if not failure:
        assert list(arrays[2]) == [arrays[0][i] + arrays[1][i] for i in range(len(arrays[0]))]
    assert native.sidecard_pair_set_local_add(handle, 123, callback) == -1
    # CPU stub has no outstanding accelerator operations.
    native.sidecard_pair_destroy(handle)


def buffers(rows=1, mode=0):
    arrays = [(c.c_float * (4096 * rows))() for _ in range(4)]
    for index in range(4096 * rows):
        arrays[0][index], arrays[1][index], arrays[3][index] = index, 2 * index, 4 * index
    arrays.append((c.c_ubyte * (1024 * 1024))())
    # Native unit-stub SDK uses the first source value as requested workspace.
    pointers = [c.cast(array, c.c_void_p) for array in arrays[:4]]
    bindings = Bindings(2, 4096 * rows, *pointers, 1, 2, 3, 4, c.addressof(arrays[4]), 1024 * 1024, mode)
    return arrays, bindings


def test_native_abi_getters_match_python_size_and_tail_layout(native):
    assert native.sidecard_pair_abi() == 2
    assert native.sidecard_pair_bindings_bytes() == c.sizeof(Bindings) == 104
    assert Bindings.exchange_mode.offset == 96
    assert Bindings.peer_data.offset == 40


@pytest.mark.parametrize("rows", (1, 4, 32))
def test_real_native_shim_abi_fp32_buffers_and_enqueue_order(native, rows):
    arrays, bindings = buffers(rows)
    original_input, original_inbox = bytes(arrays[0]), bytes(arrays[1])
    handle, stream = c.c_void_p(), Stream()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == 0
    assert list(stream.operations)[: stream.count] == [1, 2, 3, 4, 2, 3]
    assert list(arrays[2]) == [3 * i for i in range(4096 * rows)]
    assert bytes(arrays[0]) == original_input and bytes(arrays[1]) == original_inbox
    assert bytes(arrays[3]) == original_input
    assert stream.input == c.addressof(arrays[0])
    assert stream.other == c.addressof(arrays[1])
    assert stream.output == c.addressof(arrays[2])
    assert native.sidecard_pair_destroy(handle) == 0


@pytest.mark.parametrize("failure", range(1, 7))
def test_partial_native_failure_stops_without_retry_or_later_signal(native, failure):
    arrays, bindings = buffers()
    original_input = bytes(arrays[0])
    handle, stream = c.c_void_p(), Stream(fail_at=failure)
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == 55
    assert stream.count == failure
    expected = [1, 2, 3, 4, 2, 3]
    assert list(stream.operations)[: stream.count] == expected[:failure]
    assert bytes(arrays[0]) == original_input
    # CPU stub has no live device work; production must retain/quarantine owners.
    assert native.sidecard_pair_destroy(handle) == 0


@pytest.mark.parametrize("first,expected", ((-1, 66), (2**21, -3)))
def test_workspace_or_executor_failure_precedes_peer_write(native, first, expected):
    arrays, bindings = buffers()
    arrays[0][0] = first
    handle, stream = c.c_void_p(), Stream()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == expected
    assert stream.count == 0
    native.sidecard_pair_destroy(handle)


@pytest.mark.parametrize(
    "field,value",
    (
        ("abi", 1),
        ("abi", 3),
        ("elements", 1),
        ("elements", 0),
        ("elements", 4096 * 33),
        ("input", 0),
        ("inbox", 0),
        ("output", 0),
        ("peer_data", 0),
        ("workspace_bytes", 0),
        ("exchange_mode", 2),
        ("exchange_mode", 2**64 - 1),
    ),
)
def test_native_rejects_bad_bindings(native, field, value):
    _, bindings = buffers()
    setattr(bindings, field, value)
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == -1
    assert handle.value is None


@pytest.mark.parametrize("field", ("input", "inbox", "output", "peer_data", "workspace"))
def test_rejects_unaligned_ranges_before_descriptor_allocation(native, field):
    arrays, bindings = buffers()
    setattr(bindings, field, getattr(bindings, field) + 1)
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == -1
    assert handle.value is None
    assert arrays


@pytest.mark.parametrize(
    "field,other",
    (
        ("input", "inbox"),
        ("input", "output"),
        ("input", "peer_data"),
        ("inbox", "output"),
        ("inbox", "peer_data"),
        ("output", "peer_data"),
        ("workspace", "input"),
        ("workspace", "inbox"),
        ("workspace", "output"),
        ("workspace", "peer_data"),
    ),
)
@pytest.mark.parametrize("offset", (0, 4))
def test_rejects_data_or_workspace_alias_and_partial_overlap(native, field, other, offset):
    arrays, bindings = buffers()
    setattr(bindings, field, getattr(bindings, other) + offset)
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == -1
    assert handle.value is None
    assert arrays


@pytest.mark.parametrize("field", ("input", "inbox", "output", "peer_data", "workspace"))
def test_rejects_address_range_wrap(native, field):
    arrays, bindings = buffers()
    setattr(bindings, field, 2 ** (8 * c.sizeof(c.c_void_p)) - 4)
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == -1
    assert handle.value is None
    assert arrays


@pytest.mark.parametrize("rows", (1, 4, 32))
def test_remote_read_rejected_before_sdk_tensor_creation_or_queue_work(native, rows):
    arrays, bindings = buffers(rows, mode=1)
    original = [bytes(array) for array in arrays]
    count_before = native.stub_tensor_creations().contents.value
    handle, stream = c.c_void_p(1234), Stream()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == -4
    assert handle.value is None
    assert native.stub_tensor_creations().contents.value == count_before
    assert stream.count == 0 and [bytes(array) for array in arrays] == original


def test_remote_read_reservation_is_not_an_invalid_or_legacy_abi(native):
    arrays, bindings = buffers(mode=1)
    count_before = native.stub_tensor_creations().contents.value
    for abi, mode, expected in ((2, 1, -4), (2, 2, -1), (1, 1, -1)):
        bindings.abi, bindings.exchange_mode = abi, mode
        handle = c.c_void_p()
        assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == expected
        assert handle.value is None
    assert native.stub_tensor_creations().contents.value == count_before and arrays


def ping_pong_buffers(rows=1):
    arrays, bindings = buffers(rows)
    extra = [(c.c_float * bindings.elements)() for _ in range(2)]
    for index in range(bindings.elements):
        extra[0][index], extra[1][index] = -index, -2 * index
    return arrays, bindings, extra


@pytest.mark.parametrize("rows", (1, 4, 32))
def test_ping_pong_three_calls_alternate_copy_add_and_notify_slots(native, rows):
    arrays, bindings, extra = ping_pong_buffers(rows)
    handle, stream = c.c_void_p(), Stream()
    creates = native.stub_tensor_creations().contents.value
    destroys = native.stub_tensor_destructions().contents.value
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    assert native.sidecard_pair_enable_ping_pong(handle, c.addressof(extra[0]), c.addressof(extra[1])) == 0
    assert native.stub_tensor_creations().contents.value == creates + 4
    try:
        for iteration, slot in enumerate((0, 1, 0)):
            # Changing all source values catches reuse of a prior executor/data.
            for index in range(bindings.elements):
                arrays[0][index] = index + iteration + 1
            local_inbox = extra[0] if slot else arrays[1]
            peer_inbox = extra[1] if slot else arrays[3]
            other_peer_inbox = arrays[3] if slot else extra[1]
            source_before = bytes(arrays[0])
            inbox_before = (bytes(arrays[1]), bytes(extra[0]))
            other_before = bytes(other_peer_inbox)
            assert native.sidecard_pair_enqueue(handle, c.byref(stream), 37) == 0
            begin = iteration * 4
            assert stream.count == begin + 4
            assert list(stream.operations)[begin : begin + 4] == [1, 2, 3, 4]
            assert list(stream.resources)[begin : begin + 4] == [
                c.addressof(peer_inbox),
                bindings.peer_consumed if slot else bindings.peer_ready,
                bindings.consumed if slot else bindings.ready,
                c.addressof(local_inbox),
            ]
            assert list(stream.timeouts)[begin : begin + 4] == [0, 0, 37, 0]
            assert stream.input == c.addressof(arrays[0])
            assert stream.other == c.addressof(local_inbox)
            assert stream.output == c.addressof(arrays[2])
            assert list(arrays[2]) == [arrays[0][i] + local_inbox[i] for i in range(bindings.elements)]
            assert bytes(arrays[0]) == bytes(peer_inbox) == source_before
            assert (bytes(arrays[1]), bytes(extra[0])) == inbox_before
            assert bytes(other_peer_inbox) == other_before
            assert native.stub_tensor_destructions().contents.value == destroys
    finally:
        # CPU calls finish immediately. Production requires drained proof.
        assert native.sidecard_pair_destroy(handle) == 0
    assert native.stub_tensor_destructions().contents.value == destroys + 4


@pytest.mark.parametrize("extra_index", (0, 1))
@pytest.mark.parametrize("invalid", ("null", "unaligned", "wrap"))
def test_ping_pong_rejects_invalid_new_range_before_sdk(native, extra_index, invalid):
    arrays, bindings, extra = ping_pong_buffers()
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    pointers = [c.addressof(array) for array in extra]
    pointers[extra_index] = {
        "null": 0,
        "unaligned": pointers[extra_index] + 1,
        "wrap": 2 ** (8 * c.sizeof(c.c_void_p)) - 4,
    }[invalid]
    creates = native.stub_tensor_creations().contents.value
    assert native.sidecard_pair_enable_ping_pong(handle, *pointers) == -1
    assert native.stub_tensor_creations().contents.value == creates
    assert native.sidecard_pair_destroy(handle) == 0 and arrays


@pytest.mark.parametrize("extra_index", (0, 1))
@pytest.mark.parametrize("existing", ("input", "inbox", "output", "peer_data", "workspace"))
@pytest.mark.parametrize("offset", (0, 4, -4))
def test_ping_pong_rejects_new_range_overlap_with_all_existing_storage(native, extra_index, existing, offset):
    arrays, bindings, extra = ping_pong_buffers()
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    pointers = [c.addressof(array) for array in extra]
    pointers[extra_index] = getattr(bindings, existing) + offset
    creates = native.stub_tensor_creations().contents.value
    assert native.sidecard_pair_enable_ping_pong(handle, *pointers) == -1
    assert native.stub_tensor_creations().contents.value == creates
    assert native.sidecard_pair_destroy(handle) == 0 and arrays


@pytest.mark.parametrize("offset", (0, 4, -4))
def test_ping_pong_rejects_overlap_between_extra_ranges(native, offset):
    arrays, bindings, extra = ping_pong_buffers()
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    creates = native.stub_tensor_creations().contents.value
    address = c.addressof(extra[0])
    assert native.sidecard_pair_enable_ping_pong(handle, address, address + offset) == -1
    assert native.stub_tensor_creations().contents.value == creates
    assert native.sidecard_pair_destroy(handle) == 0 and arrays


def test_ping_pong_requires_live_handle_and_only_one_configuration(native):
    arrays, bindings, extra = ping_pong_buffers()
    pointers = [c.addressof(array) for array in extra]
    creates = native.stub_tensor_creations().contents.value
    assert native.sidecard_pair_enable_ping_pong(None, *pointers) == -1
    assert native.stub_tensor_creations().contents.value == creates
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    assert native.sidecard_pair_enable_ping_pong(handle, *pointers) == 0
    creates = native.stub_tensor_creations().contents.value
    assert native.sidecard_pair_enable_ping_pong(handle, *pointers) == -1
    assert native.stub_tensor_creations().contents.value == creates
    assert native.sidecard_pair_destroy(handle) == 0 and arrays


@pytest.mark.parametrize("planning_failure", (False, True))
def test_ping_pong_cannot_be_enabled_after_first_submission_attempt(native, planning_failure):
    arrays, bindings, extra = ping_pong_buffers()
    if planning_failure:
        arrays[0][0] = -1
    handle, stream = c.c_void_p(), Stream()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == (66 if planning_failure else 0)
    creates = native.stub_tensor_creations().contents.value
    assert native.sidecard_pair_enable_ping_pong(handle, *[c.addressof(array) for array in extra]) == -1
    assert native.stub_tensor_creations().contents.value == creates
    assert native.sidecard_pair_destroy(handle) == 0


@pytest.mark.parametrize("ping_pong_first", (False, True))
def test_ping_pong_and_fixed_local_add_callback_are_mutually_exclusive(native, ping_pong_first):
    arrays, bindings, extra = ping_pong_buffers()
    handle = c.c_void_p()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0

    @c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)
    def local_add(add_handle, stream):
        pytest.fail("configuration must not execute a local add")
        return 55

    callback = c.cast(local_add, c.c_void_p)
    pointers = [c.addressof(array) for array in extra]
    if ping_pong_first:
        assert native.sidecard_pair_enable_ping_pong(handle, *pointers) == 0
        assert native.sidecard_pair_set_local_add(handle, 123, callback) == -1
    else:
        assert native.sidecard_pair_set_local_add(handle, 123, callback) == 0
        creates = native.stub_tensor_creations().contents.value
        assert native.sidecard_pair_enable_ping_pong(handle, *pointers) == -1
        assert native.stub_tensor_creations().contents.value == creates
    assert native.sidecard_pair_destroy(handle) == 0 and arrays


def test_ping_pong_descriptor_failure_retains_original_pair_ownership(native):
    arrays, bindings, extra = ping_pong_buffers()
    handle = c.c_void_p()
    destroys = native.stub_tensor_destructions().contents.value
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    creates = native.stub_tensor_creations().contents.value
    native.stub_fail_tensor_at().contents.value = creates + 1
    try:
        assert native.sidecard_pair_enable_ping_pong(handle, *[c.addressof(array) for array in extra]) == -2
        assert handle.value is not None
        assert native.stub_tensor_creations().contents.value == creates + 1
        assert native.stub_tensor_destructions().contents.value == destroys
    finally:
        native.stub_fail_tensor_at().contents.value = 0
        # Only the CPU stub is drained here; no production failure retry.
        assert native.sidecard_pair_destroy(handle) == 0
    assert native.stub_tensor_destructions().contents.value == destroys + 3 and arrays


@pytest.mark.parametrize("slot", (0, 1))
@pytest.mark.parametrize("failure", range(1, 5))
def test_ping_pong_partial_failure_short_circuits_without_advancing_slot(native, slot, failure):
    arrays, bindings, extra = ping_pong_buffers()
    handle, stream = c.c_void_p(), Stream()
    assert native.sidecard_pair_create(c.byref(bindings), c.byref(handle)) == 0
    assert native.sidecard_pair_enable_ping_pong(handle, *[c.addressof(array) for array in extra]) == 0
    if slot:
        assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == 0
    start = stream.count
    stream.fail_at = start + failure
    output_before = bytes(arrays[2])
    assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == 55
    assert stream.count == start + failure
    assert list(stream.operations)[start : stream.count] == [1, 2, 3, 4][:failure]
    assert bytes(arrays[2]) == output_before
    # This synchronous CPU-only retry inspects slot bookkeeping. Production
    # quarantines on any partial failure and must never retry queued SDK work.
    stream.fail_at = 0
    retry_start = stream.count
    assert native.sidecard_pair_enqueue(handle, c.byref(stream), 30) == 0
    local_inbox = extra[0] if slot else arrays[1]
    peer_inbox = extra[1] if slot else arrays[3]
    assert list(stream.resources)[retry_start : stream.count] == [
        c.addressof(peer_inbox),
        bindings.peer_consumed if slot else bindings.peer_ready,
        bindings.consumed if slot else bindings.ready,
        c.addressof(local_inbox),
    ]
    assert native.sidecard_pair_destroy(handle) == 0
