# SPDX-License-Identifier: Apache-2.0
"""Compile the real HCCL shim against CPU stubs; never initialize an NPU.

The stub's two identical contributions check FP32/out-of-place ABI wiring only.
They do not validate HCCL arithmetic, rendezvous, stream completion, or speed.
"""

import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ROOT_BYTES = 4108
STUB_HEADER = r"""
#pragma once
#include <cstdint>
typedef void* aclrtStream;
typedef void* HcclComm;
typedef enum { HCCL_SUCCESS = 0, HCCL_TEST_STATUS = 20000 } HcclResult;
typedef enum { HCCL_DATA_TYPE_FP32 = 4 } HcclDataType;
typedef enum { HCCL_REDUCE_SUM = 0 } HcclReduceOp;
struct HcclRootInfo { char internal[4108]; };
extern "C" {
HcclResult HcclGetRootInfo(HcclRootInfo*) __attribute__((weak));
HcclResult HcclCommInitRootInfo(uint32_t, const HcclRootInfo*, uint32_t, HcclComm*) __attribute__((weak));
HcclResult HcclAllReduce(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
HcclResult HcclCommDestroy(HcclComm) __attribute__((weak));
}
"""
STUB_SOURCE = r"""
#include <hccl/hccl.h>
#include <cstring>
struct StubState {
    int status[4];
    uint32_t calls[4];
    uint64_t count;
    uint32_t ranks, rank, dtype, reduction;
    uintptr_t comm, stream, input, output;
    int null_comm, partial_comm;
    unsigned char root[4108];
};
extern "C" StubState* stub_state() {
    static StubState state{};
    return &state;
}
#ifndef OMIT_ROOT_API
extern "C" HcclResult HcclGetRootInfo(HcclRootInfo* info) {
    auto& s = *stub_state();
    ++s.calls[0];
    if (s.status[0]) return static_cast<HcclResult>(s.status[0]);
    for (unsigned i = 0; i < sizeof(info->internal); ++i)
        info->internal[i] = static_cast<char>((i * 37 + 11) % 256);
    return HCCL_SUCCESS;
}
#endif
extern "C" HcclResult HcclCommInitRootInfo(uint32_t ranks, const HcclRootInfo* info,
                                          uint32_t rank, HcclComm* comm) {
    auto& s = *stub_state();
    ++s.calls[1];
    s.ranks = ranks;
    s.rank = rank;
    std::memcpy(s.root, info, sizeof(*info));
    if (!s.null_comm && (!s.status[1] || s.partial_comm))
        *comm = reinterpret_cast<void*>(0x4000 + rank);
    return static_cast<HcclResult>(s.status[1]);
}
extern "C" HcclResult HcclAllReduce(void* input, void* output, uint64_t count,
                                   HcclDataType dtype, HcclReduceOp reduction,
                                   HcclComm comm, aclrtStream stream) {
    auto& s = *stub_state();
    ++s.calls[2];
    s.input = reinterpret_cast<uintptr_t>(input);
    s.output = reinterpret_cast<uintptr_t>(output);
    s.count = count;
    s.dtype = dtype;
    s.reduction = reduction;
    s.comm = reinterpret_cast<uintptr_t>(comm);
    s.stream = reinterpret_cast<uintptr_t>(stream);
    if (s.status[2]) return static_cast<HcclResult>(s.status[2]);
    auto* source = static_cast<float*>(input);
    auto* destination = static_cast<float*>(output);
    for (uint64_t i = 0; i < count; ++i) destination[i] = source[i] + source[i];
    return HCCL_SUCCESS;
}
extern "C" HcclResult HcclCommDestroy(HcclComm comm) {
    auto& s = *stub_state();
    ++s.calls[3];
    s.comm = reinterpret_cast<uintptr_t>(comm);
    return static_cast<HcclResult>(s.status[3]);
}
"""


class State(c.Structure):
    _fields_ = [
        ("status", c.c_int * 4),
        ("calls", c.c_uint32 * 4),
        ("count", c.c_uint64),
        ("ranks", c.c_uint32),
        ("rank", c.c_uint32),
        ("dtype", c.c_uint32),
        ("reduction", c.c_uint32),
        ("comm", c.c_size_t),
        ("stream", c.c_size_t),
        ("input", c.c_size_t),
        ("output", c.c_size_t),
        ("null_comm", c.c_int),
        ("partial_comm", c.c_int),
        ("root", c.c_ubyte * ROOT_BYTES),
    ]


@pytest.fixture(scope="module", params=(False, True), ids=("complete-sdk", "missing-root-api"))
def compiled(request, tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("HCCL shim ABI tests require a CPU C++ compiler")
    root = tmp_path_factory.mktemp("hccl-reference-stub")
    include = root / "hccl"
    include.mkdir()
    (include / "hccl.h").write_text(STUB_HEADER)
    (root / "stub.cpp").write_text(STUB_SOURCE)
    output = root / "reference.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            *(["-DOMIT_ROOT_API"] if request.param else []),
            "-I",
            str(root),
            str(ROOT / "csrc/sidecard_collectives/hccl_reference.cpp"),
            str(root / "stub.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    lib = c.CDLL(str(output))
    lib.sidecard_hccl_abi.restype = c.c_uint64
    lib.sidecard_hccl_root_info_bytes.restype = c.c_uint64
    lib.sidecard_hccl_get_root_info.argtypes = [c.c_void_p, c.c_uint64]
    lib.sidecard_hccl_create.argtypes = [c.c_void_p, c.c_uint64, c.c_uint32, c.POINTER(c.c_void_p)]
    lib.sidecard_hccl_all_reduce.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint64, c.c_void_p]
    lib.sidecard_hccl_destroy.argtypes = [c.POINTER(c.c_void_p), c.c_uint32]
    for name in ("get_root_info", "create", "all_reduce", "destroy"):
        getattr(lib, "sidecard_hccl_" + name).restype = c.c_int
    lib.stub_state.restype = c.POINTER(State)
    return lib, request.param


@pytest.fixture
def native(compiled):
    lib, missing_root = compiled
    state = lib.stub_state().contents
    c.memset(c.addressof(state), 0, c.sizeof(state))
    return lib, state, missing_root


def test_c_abi_and_installed_root_size(native):
    lib, _, _ = native
    assert lib.sidecard_hccl_abi() == 1
    assert lib.sidecard_hccl_root_info_bytes() == ROOT_BYTES


def test_root_info_bytes_are_copied_exactly_and_missing_api_fails_closed(native):
    lib, state, missing_root = native
    blob = (c.c_ubyte * ROOT_BYTES)(*([251] * ROOT_BYTES))
    status = lib.sidecard_hccl_get_root_info(blob, ROOT_BYTES)
    if missing_root:
        assert status == -3 and list(blob) == [251] * ROOT_BYTES and state.calls[0] == 0
    else:
        assert status == 0 and list(blob) == [(i * 37 + 11) % 256 for i in range(ROOT_BYTES)]
        assert state.calls[0] == 1


@pytest.mark.parametrize("size", (0, 1, ROOT_BYTES - 1, ROOT_BYTES + 1, 2**64 - 1))
def test_root_info_exact_length_admission(native, size):
    lib, state, _ = native
    blob, handle = (c.c_ubyte * ROOT_BYTES)(), c.c_void_p(1234)
    assert lib.sidecard_hccl_get_root_info(blob, size) == -1
    assert lib.sidecard_hccl_create(blob, size, 0, c.byref(handle)) == -1
    assert handle.value is None and list(state.calls) == [0] * 4


@pytest.mark.parametrize("rank", (0, 1))
def test_init_copies_unaligned_serialized_root_and_uses_exact_two_rank_domain(native, rank):
    lib, state, _ = native
    storage = (c.c_ubyte * (ROOT_BYTES + 1))()
    payload = [(i * 13 + rank) % 256 for i in range(ROOT_BYTES)]
    storage[1:] = payload
    handle = c.c_void_p()
    assert lib.sidecard_hccl_create(c.byref(storage, 1), ROOT_BYTES, rank, c.byref(handle)) == 0
    assert handle.value == 0x4000 + rank and state.ranks == 2 and state.rank == rank
    assert list(state.root) == payload
    assert lib.sidecard_hccl_destroy(c.byref(handle), 1) == 0
    assert handle.value is None


@pytest.mark.parametrize("rank", (2, 3, 2**32 - 1))
def test_invalid_rank_rejected_without_sdk_call(native, rank):
    lib, state, _ = native
    blob, handle = (c.c_ubyte * ROOT_BYTES)(), c.c_void_p()
    assert lib.sidecard_hccl_create(blob, ROOT_BYTES, rank, c.byref(handle)) == -1
    assert handle.value is None and state.calls[1] == 0


@pytest.mark.parametrize("count", (1, 4096, 4 * 4096, 32 * 4096))
def test_raw_out_of_place_fp32_sum_and_exact_stream_passthrough(native, count):
    lib, state, _ = native
    source, output = (c.c_float * count)(), (c.c_float * count)()
    for i in range(count):
        source[i] = (i % 127 - 63) * 0.125
        output[i] = float("nan")
    before = bytes(source)
    assert lib.sidecard_hccl_all_reduce(0x4001, source, output, count, 0x7330) == 0
    assert bytes(source) == before and list(output) == [2 * value for value in source]
    assert (state.count, state.dtype, state.reduction) == (count, 4, 0)
    assert (state.comm, state.stream) == (0x4001, 0x7330)
    assert (state.input, state.output) == (c.addressof(source), c.addressof(output))
    assert list(state.calls) == [0, 0, 1, 0]


def test_null_arguments_and_bad_counts_do_not_enter_sdk(native):
    lib, state, _ = native
    source, output = (c.c_float * 8)(), (c.c_float * 8)()
    for arguments in (
        (None, source, output, 8, 7),
        (3, None, output, 8, 7),
        (3, source, None, 8, 7),
        (3, source, output, 8, None),
        (3, source, output, 0, 7),
        (3, source, output, 2**64 - 1, 7),
    ):
        assert lib.sidecard_hccl_all_reduce(*arguments) == -1
    handle = c.c_void_p()
    assert lib.sidecard_hccl_get_root_info(None, ROOT_BYTES) == -1
    assert lib.sidecard_hccl_create(None, ROOT_BYTES, 0, c.byref(handle)) == -1
    assert lib.sidecard_hccl_create(source, ROOT_BYTES, 0, None) == -1
    assert lib.sidecard_hccl_destroy(None, 1) == -1
    assert lib.sidecard_hccl_destroy(c.byref(handle), 1) == -1
    assert list(state.calls) == [0] * 4


def test_rejects_inplace_partial_overlap_unaligned_and_wrapping_ranges(native):
    lib, state, _ = native
    storage, output = (c.c_float * 16)(), (c.c_float * 16)()
    source_address, output_address = c.addressof(storage), c.addressof(output)
    maximum = (1 << (8 * c.sizeof(c.c_void_p))) - 1
    for source, destination, count in (
        (source_address, source_address, 8),
        (source_address, source_address + 4, 8),
        (source_address + 4, source_address, 8),
        (source_address + 1, output_address, 8),
        (source_address, output_address + 1, 8),
        (maximum - 3, output_address, 1),
        (source_address, maximum - 3, 1),
    ):
        assert lib.sidecard_hccl_all_reduce(3, source, destination, count, 7) == -1
    assert state.calls[2] == 0


@pytest.mark.parametrize("partial", (False, True))
def test_sdk_init_failure_propagates_and_preserves_partial_handle_for_quarantine(native, partial):
    lib, state, _ = native
    state.status[1], state.partial_comm = 17, partial
    blob, handle = (c.c_ubyte * ROOT_BYTES)(), c.c_void_p()
    assert lib.sidecard_hccl_create(blob, ROOT_BYTES, 1, c.byref(handle)) == 17
    assert handle.value == (0x4001 if partial else None)
    assert list(state.calls) == [0, 1, 0, 0]


def test_sdk_success_without_communicator_is_rejected(native):
    lib, state, _ = native
    state.null_comm = 1
    blob, handle = (c.c_ubyte * ROOT_BYTES)(), c.c_void_p()
    assert lib.sidecard_hccl_create(blob, ROOT_BYTES, 0, c.byref(handle)) == -2
    assert handle.value is None


def test_sdk_error_statuses_propagate_without_hidden_cleanup_or_retry(native):
    lib, state, missing_root = native
    if not missing_root:
        state.status[0] = 9
        blob = (c.c_ubyte * ROOT_BYTES)(*([249] * ROOT_BYTES))
        assert lib.sidecard_hccl_get_root_info(blob, ROOT_BYTES) == 9
        assert list(blob) == [249] * ROOT_BYTES and state.calls[0] == 1
    state.status[2] = 15
    source, output = (c.c_float * 8)(*range(8)), (c.c_float * 8)(*([37.0] * 8))
    assert lib.sidecard_hccl_all_reduce(0x4000, source, output, 8, 7) == 15
    assert list(output) == [37.0] * 8 and state.calls[2] == 1
    handle = c.c_void_p(0x4000)
    state.status[3] = 20
    assert lib.sidecard_hccl_destroy(c.byref(handle), 1) == 20
    assert handle.value == 0x4000 and state.calls[3] == 1


@pytest.mark.parametrize("proof", (0, 2, 2**32 - 1))
def test_destroy_requires_exact_explicit_drained_proof(native, proof):
    lib, state, _ = native
    handle = c.c_void_p(0x4000)
    assert lib.sidecard_hccl_destroy(c.byref(handle), proof) == -1
    assert handle.value == 0x4000 and state.calls[3] == 0


def test_destroy_success_clears_handle_and_rejects_second_destroy(native):
    lib, state, _ = native
    handle = c.c_void_p(0x4000)
    assert lib.sidecard_hccl_destroy(c.byref(handle), 1) == 0
    assert handle.value is None and state.calls[3] == 1
    assert lib.sidecard_hccl_destroy(c.byref(handle), 1) == -1
    assert state.calls[3] == 1
