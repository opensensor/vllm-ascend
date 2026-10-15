# SPDX-License-Identifier: Apache-2.0
"""Compile the actual raw N-rank comparator against a CPU-only HCCL stub.

The stub copies payload bytes and records the HCCL datatype, rank geometry and
arguments. It does not implement a collective or qualify device performance.
"""

import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ROOT_BYTES = 64
COMM = 0x12345678
SDK_FP32 = 7
SDK_FP16 = 9
SDK_SUM = 13
STUB_HEADER = r"""
#pragma once
#include <cstdint>
using aclrtStream = void*;
using HcclComm = void*;
using HcclResult = int;
enum HcclDataType { HCCL_DATA_TYPE_FP32 = 7, HCCL_DATA_TYPE_FP16 = 9 };
enum HcclReduceOp { HCCL_REDUCE_SUM = 13 };
constexpr int HCCL_SUCCESS = 0;
struct HcclRootInfo { unsigned char bytes[64]; };
extern "C" {
HcclResult HcclGetRootInfo(HcclRootInfo*) __attribute__((weak));
HcclResult HcclCommInitRootInfo(uint32_t, const HcclRootInfo*, uint32_t, HcclComm*) __attribute__((weak));
HcclResult HcclCommDestroy(HcclComm) __attribute__((weak));
HcclResult HcclAllReduce(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
}
"""
STUB_SOURCE = r"""
#include <hccl/hccl.h>
#include <cstring>
struct State {
    int root_calls, init_calls, reduce_calls, destroy_calls;
    int root_status, init_status, reduce_status, destroy_status, null_comm, skip_copy;
    uint32_t world, rank;
    int dtype, reduction;
    uintptr_t input, output, comm, stream;
    uint64_t elements;
    unsigned char root[64];
};
extern "C" State* stub_state() { static State s{}; return &s; }
#ifndef MISSING_WEAK_APIS
extern "C" HcclResult HcclGetRootInfo(HcclRootInfo* root) {
    auto* s = stub_state();
    ++s->root_calls;
    for (unsigned i = 0; i < 64; ++i) root->bytes[i] = static_cast<unsigned char>(i + 17);
    return s->root_status;
}
extern "C" HcclResult HcclCommInitRootInfo(uint32_t world, const HcclRootInfo* root,
                                        uint32_t rank, HcclComm* comm) {
    auto* s = stub_state();
    ++s->init_calls;
    s->world = world;
    s->rank = rank;
    std::memcpy(s->root, root->bytes, 64);
    *comm = s->null_comm ? nullptr : reinterpret_cast<void*>(0x12345678);
    return s->init_status;
}
extern "C" HcclResult HcclCommDestroy(HcclComm comm) {
    auto* s = stub_state();
    ++s->destroy_calls;
    s->comm = reinterpret_cast<uintptr_t>(comm);
    return s->destroy_status;
}
#endif
extern "C" HcclResult HcclAllReduce(void* input, void* output, uint64_t elements,
                                  HcclDataType dtype, HcclReduceOp reduction,
                                  HcclComm comm, aclrtStream stream) {
    auto* s = stub_state();
    ++s->reduce_calls;
    s->dtype = dtype;
    s->reduction = reduction;
    s->input = reinterpret_cast<uintptr_t>(input);
    s->output = reinterpret_cast<uintptr_t>(output);
    s->comm = reinterpret_cast<uintptr_t>(comm);
    s->stream = reinterpret_cast<uintptr_t>(stream);
    s->elements = elements;
    if (!s->reduce_status && !s->skip_copy) {
        std::memcpy(output, input, elements * (dtype == HCCL_DATA_TYPE_FP32 ? 4 : 2));
    }
    return s->reduce_status;
}
"""


class State(c.Structure):
    _fields_ = (
        [
            (name, c.c_int)
            for name in (
                "root_calls",
                "init_calls",
                "reduce_calls",
                "destroy_calls",
                "root_status",
                "init_status",
                "reduce_status",
                "destroy_status",
                "null_comm",
                "skip_copy",
            )
        ]
        + [("world", c.c_uint32), ("rank", c.c_uint32), ("dtype", c.c_int), ("reduction", c.c_int)]
        + [(name, c.c_size_t) for name in ("input", "output", "comm", "stream")]
        + [("elements", c.c_uint64), ("root", c.c_ubyte * ROOT_BYTES)]
    )


@pytest.fixture(scope="module", params=[False, True], ids=["sdk-present", "weak-sdk-absent"])
def compiled(request, tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("raw HCCL group ABI tests require a CPU C++ compiler")
    missing = request.param
    root = tmp_path_factory.mktemp("hccl-group-native")
    (root / "hccl").mkdir()
    (root / "hccl/hccl.h").write_text(STUB_HEADER)
    (root / "stub.cpp").write_text(STUB_SOURCE)
    output = root / "hccl-group.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            *(["-DMISSING_WEAK_APIS=1"] if missing else []),
            "-I",
            str(root),
            str(ROOT / "csrc/sidecard_collectives/hccl_group_reference.cpp"),
            str(root / "stub.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    lib = c.CDLL(str(output))
    lib.sidecard_hccl_group_abi.restype = c.c_uint64
    lib.sidecard_hccl_group_root_info_bytes.restype = c.c_uint64
    lib.sidecard_hccl_group_get_root_info.argtypes = [c.c_void_p, c.c_uint64]
    lib.sidecard_hccl_group_create.argtypes = [c.c_void_p, c.c_uint64, c.c_uint32, c.c_uint32, c.POINTER(c.c_void_p)]
    lib.sidecard_hccl_group_all_reduce.argtypes = [
        c.c_void_p,
        c.c_void_p,
        c.c_void_p,
        c.c_uint64,
        c.c_uint32,
        c.c_void_p,
    ]
    lib.sidecard_hccl_group_destroy.argtypes = [c.POINTER(c.c_void_p), c.c_uint32]
    lib.stub_state.restype = c.POINTER(State)
    return lib, missing


@pytest.fixture
def native(compiled):
    lib, missing = compiled
    state = lib.stub_state().contents
    c.memset(c.addressof(state), 0, c.sizeof(state))
    return lib, state, missing


def root_info():
    return (c.c_ubyte * ROOT_BYTES)(*range(ROOT_BYTES))


def buffers(dtype):
    kind = c.c_uint32 if dtype == 0 else c.c_uint16
    values = (0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000) if dtype == 0 else (0x8000, 0x7E23, 0x7C00, 0xFC00)
    return (kind * 4)(*values), (kind * 4)()


def test_exact_abi_getters_execute_no_sdk(native):
    lib, state, _ = native
    assert lib.sidecard_hccl_group_abi() == 1
    assert lib.sidecard_hccl_group_root_info_bytes() == ROOT_BYTES
    assert state.root_calls == state.init_calls == state.reduce_calls == state.destroy_calls == 0


def test_root_info_copies_only_on_sdk_success(native):
    lib, state, missing = native
    root = root_info()
    original = bytes(root)
    assert lib.sidecard_hccl_group_get_root_info(root, ROOT_BYTES) == (-3 if missing else 0)
    assert bytes(root) == (original if missing else bytes(range(17, ROOT_BYTES + 17)))
    assert state.root_calls == (0 if missing else 1)
    for status in (41, -71):
        state.root_status = status
        before = bytes(root)
        assert lib.sidecard_hccl_group_get_root_info(root, ROOT_BYTES) == (-3 if missing else status)
        assert bytes(root) == before


@pytest.mark.parametrize("world", [1, 2, 3, 4, 65, 2**32 - 1])
@pytest.mark.parametrize("last_rank", [False, True])
def test_arbitrary_world_and_rank_boundaries_preserve_serialized_bytes(native, world, last_rank):
    lib, state, missing = native
    root = root_info()
    rank = world - 1 if last_rank else 0
    comm = c.c_void_p(999)
    assert lib.sidecard_hccl_group_create(root, ROOT_BYTES, world, rank, c.byref(comm)) == (-3 if missing else 0)
    assert comm.value == (None if missing else COMM)
    assert state.init_calls == (0 if missing else 1)
    if not missing:
        assert (state.world, state.rank, bytes(state.root)) == (world, rank, bytes(root))
        assert lib.sidecard_hccl_group_destroy(c.byref(comm), 1) == 0
        assert comm.value is None and state.comm == COMM and state.destroy_calls == 1


@pytest.mark.parametrize("world,rank", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (65, 65), (65, 2**32 - 1)])
def test_invalid_world_and_rank_reject_before_sdk(native, world, rank):
    lib, state, _ = native
    comm = c.c_void_p(999)
    assert lib.sidecard_hccl_group_create(root_info(), ROOT_BYTES, world, rank, c.byref(comm)) == -1
    assert comm.value is None and state.init_calls == 0


@pytest.mark.parametrize("dtype", [0, 1])
def test_strong_all_reduce_dtype_pointer_stream_and_payload_passthrough(native, dtype):
    lib, state, _ = native
    source, destination = buffers(dtype)
    before = bytes(source)
    stream = c.c_uint64(0)
    assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, len(source), dtype, c.byref(stream)) == 0
    assert state.reduce_calls == 1 and state.elements == len(source)
    assert (state.input, state.output, state.comm, state.stream) == (
        c.addressof(source),
        c.addressof(destination),
        COMM,
        c.addressof(stream),
    )
    assert state.dtype == (SDK_FP32 if dtype == 0 else SDK_FP16) and state.reduction == SDK_SUM
    assert bytes(source) == bytes(destination) == before
    assert state.root_calls == state.init_calls == state.destroy_calls == 0


@pytest.mark.parametrize("status", [0, 77, -83])
@pytest.mark.parametrize("null_comm", [False, True])
def test_initialization_retains_partial_communicator_and_raw_status(native, status, null_comm):
    lib, state, missing = native
    state.init_status, state.null_comm = status, null_comm
    comm = c.c_void_p(999)
    expected = -3 if missing else (-2 if status == 0 and null_comm else status)
    assert lib.sidecard_hccl_group_create(root_info(), ROOT_BYTES, 4, 3, c.byref(comm)) == expected
    assert comm.value == (None if missing or null_comm else COMM)
    assert state.init_calls == (0 if missing else 1) and state.destroy_calls == 0


@pytest.mark.parametrize("status", [21, -31])
@pytest.mark.parametrize("dtype", [0, 1])
def test_collective_sdk_error_propagates_without_hidden_calls_or_output_write(native, status, dtype):
    lib, state, _ = native
    state.reduce_status = status
    source, destination = buffers(dtype)
    before = bytes(destination)
    assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, len(source), dtype, 123) == status
    assert state.reduce_calls == 1 and bytes(destination) == before
    assert state.root_calls == state.init_calls == state.destroy_calls == 0


@pytest.mark.parametrize("status", [0, 57, -67])
def test_destroy_requires_explicit_proof_and_retains_failed_sdk_handle(native, status):
    lib, state, missing = native
    state.destroy_status = status
    comm = c.c_void_p(COMM)
    for proof in (0, 2, 2**32 - 1):
        assert lib.sidecard_hccl_group_destroy(c.byref(comm), proof) == -1
        assert comm.value == COMM and state.destroy_calls == 0
    assert lib.sidecard_hccl_group_destroy(c.byref(comm), 1) == (-3 if missing else status)
    assert comm.value == (None if not missing and status == 0 else COMM)
    assert state.destroy_calls == (0 if missing else 1)


@pytest.mark.parametrize("dtype", [0, 1])
@pytest.mark.parametrize("which", ["input", "output"])
@pytest.mark.parametrize("invalid", ["null", "unaligned", "overflow", "same", "positive-overlap", "negative-overlap"])
def test_invalid_full_pointer_ranges_precede_strong_sdk_call(native, dtype, which, invalid):
    lib, state, _ = native
    source, destination = buffers(dtype)
    pointers = dict(input=c.addressof(source), output=c.addressof(destination))
    item_bytes = 4 if dtype == 0 else 2
    other = pointers["output" if which == "input" else "input"]
    pointers[which] = {
        "null": 0,
        "unaligned": pointers[which] + 1,
        "overflow": 2 ** (8 * c.sizeof(c.c_void_p)) - item_bytes,
        "same": other,
        "positive-overlap": other + item_bytes,
        "negative-overlap": other - item_bytes,
    }[invalid]
    assert lib.sidecard_hccl_group_all_reduce(COMM, pointers["input"], pointers["output"], 4, dtype, 123) == -1
    assert state.reduce_calls == 0


@pytest.mark.parametrize("dtype", [0, 1])
@pytest.mark.parametrize("elements", [0, 2**63, 2**64 - 1])
def test_element_byte_multiplication_is_checked_before_sdk(native, dtype, elements):
    lib, state, _ = native
    source, destination = buffers(dtype)
    assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, elements, dtype, 123) == -1
    assert state.reduce_calls == 0


def test_fp16_two_byte_alignment_is_accepted_and_fp32_is_rejected(native):
    lib, state, _ = native
    arrays = [(c.c_uint16 * 5)() for _ in range(2)]
    source, destination = [c.addressof(array) + 2 for array in arrays]
    c.memmove(source, b"\x00\x80\x23\x7e\x00\x7c", 6)
    assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, 3, 1, 123) == 0
    assert c.string_at(source, 6) == c.string_at(destination, 6) and state.dtype == SDK_FP16
    assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, 1, 0, 123) == -1
    assert state.reduce_calls == 1


@pytest.mark.parametrize("dtype", [0, 1])
def test_adjacent_disjoint_ranges_and_single_element_are_accepted(native, dtype):
    lib, state, _ = native
    array = (c.c_uint32 * 8)(*range(8))
    item_bytes = 4 if dtype == 0 else 2
    source, destination = c.addressof(array), c.addressof(array) + item_bytes
    assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, 1, dtype, 123) == 0
    assert c.string_at(source, item_bytes) == c.string_at(destination, item_bytes)
    assert state.reduce_calls == 1


def test_invalid_entry_arguments_execute_no_sdk(native):
    lib, state, _ = native
    root = root_info()
    comm = c.c_void_p(999)
    source, destination = buffers(0)
    assert lib.sidecard_hccl_group_get_root_info(None, ROOT_BYTES) == -1
    assert lib.sidecard_hccl_group_create(root, ROOT_BYTES, 1, 0, None) == -1
    assert lib.sidecard_hccl_group_create(None, ROOT_BYTES, 1, 0, c.byref(comm)) == -1
    assert comm.value is None
    for size in (0, ROOT_BYTES - 1, ROOT_BYTES + 1, 2**64 - 1):
        assert lib.sidecard_hccl_group_get_root_info(root, size) == -1
        assert lib.sidecard_hccl_group_create(root, size, 1, 0, c.byref(comm)) == -1
        assert comm.value is None
    assert lib.sidecard_hccl_group_all_reduce(None, source, destination, 4, 0, 123) == -1
    assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, 4, 0, None) == -1
    for dtype in (2, 2**32 - 1):
        assert lib.sidecard_hccl_group_all_reduce(COMM, source, destination, 4, dtype, 123) == -1
    assert lib.sidecard_hccl_group_destroy(None, 1) == -1
    assert lib.sidecard_hccl_group_destroy(c.byref(comm), 1) == -1
    assert state.root_calls == state.init_calls == state.reduce_calls == state.destroy_calls == 0
