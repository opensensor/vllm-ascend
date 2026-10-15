# SPDX-License-Identifier: Apache-2.0
"""CPU-compile the real local-add kernel and runtime; no SDK/device execution.

Fake AscendC copies/adds exercise complete tiling and FP32 storage on the CPU.
They cannot establish vector instructions, pipeline fences, or NPU eligibility.
"""

import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ACL_HEADER = r"""
#pragma once
#include <cstdint>
#include <cstddef>
using aclError = int;
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
using aclrtStream = void*;
struct aclrtLaunchKernelCfg {};
constexpr int ACL_RT_BINARY_LOAD_OPT_MAGIC = 2;
constexpr uint32_t ACL_RT_BINARY_MAGIC_ELF_AICORE = 0x43554245U;
struct aclrtBinaryLoadOption { int type; union { uint32_t magic; } value; };
struct aclrtBinaryLoadOptions { aclrtBinaryLoadOption* options; size_t count; };
aclError aclrtBinaryLoadFromData(const void*, size_t, const aclrtBinaryLoadOptions*, aclrtBinHandle*);
aclError aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle*);
aclError aclrtLaunchKernelWithArgsArray(void*, uint32_t, aclrtStream, aclrtLaunchKernelCfg*, void**);
aclError aclrtBinaryUnLoad(aclrtBinHandle);
"""
KERNEL_HEADER = r"""
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#define __global__
#define __aicore__
#define __gm__
using GM_ADDR = unsigned char*;
constexpr int PIPE_ALL = 0;
namespace AscendC {
enum class TPosition { VECCALC };
template<class T> struct LocalTensor { T* data; };
template<TPosition> struct TBuf {
    std::vector<unsigned char> storage;
    template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(storage.data())}; }
};
struct TPipe {
    template<TPosition P> void InitBuffer(TBuf<P>& buffer, uint64_t bytes) { buffer.storage.resize(bytes); }
};
template<class T> struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* pointer) { data = pointer; }
    T* operator[](uint64_t offset) { return data + offset; }
};
inline void InitSocState() {}
template<int> inline void PipeBarrier() {}
template<class T> void DataCopy(LocalTensor<T> destination, const T* source, uint64_t count) {
    std::memcpy(destination.data, source, count * sizeof(T));
}
template<class T> void DataCopy(T* destination, LocalTensor<T> source, uint64_t count) {
    std::memcpy(destination, source.data, count * sizeof(T));
}
template<class T> void Add(LocalTensor<T> output, LocalTensor<T> a, LocalTensor<T> b, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) output.data[i] = a.data[i] + b.data[i];
}
uint32_t GetBlockIdx();
uint32_t GetBlockNum();
}
"""
STUB_SOURCE = r"""
#include <acl/acl.h>
#include "kernel_operator.h"
#include <cstring>
struct StubState {
    int status[4];
    uint32_t calls[4];
    uintptr_t image;
    uint64_t image_bytes;
    uintptr_t binary, function, stream;
    uintptr_t data[3], slots[3];
    uint32_t blocks, block_index, lookup_rows, magic, option_type;
    int null_binary, null_function, partial_binary;
};
extern "C" StubState* stub_state() { static StubState state{}; return &state; }
namespace AscendC {
uint32_t GetBlockIdx() { return stub_state()->block_index; }
uint32_t GetBlockNum() { return stub_state()->blocks; }
}
extern "C" void sidecard_pair_add_r1_v1(GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" void sidecard_pair_add_r4_v1(GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" void sidecard_pair_add_r32_v1(GM_ADDR, GM_ADDR, GM_ADDR);
aclError aclrtBinaryLoadFromData(const void* image, size_t bytes, const aclrtBinaryLoadOptions* options,
                                aclrtBinHandle* binary) {
    auto& s = *stub_state();
    ++s.calls[0];
    s.image = reinterpret_cast<uintptr_t>(image);
    s.image_bytes = bytes;
    s.magic = options->options[0].value.magic;
    s.option_type = options->options[0].type;
    if (options->count != 1) return 91;
    if (!s.null_binary && (!s.status[0] || s.partial_binary)) *binary = reinterpret_cast<void*>(0x8000);
    return s.status[0];
}
aclError aclrtBinaryGetFunction(aclrtBinHandle binary, const char* symbol, aclrtFuncHandle* function) {
    auto& s = *stub_state();
    ++s.calls[1];
    s.binary = reinterpret_cast<uintptr_t>(binary);
    if (s.status[1]) return s.status[1];
    using Kernel = void (*)(GM_ADDR, GM_ADDR, GM_ADDR);
    Kernel kernel = nullptr;
    if (!std::strcmp(symbol, "sidecard_pair_add_r1_v1")) { s.lookup_rows = 1; kernel = sidecard_pair_add_r1_v1; }
    if (!std::strcmp(symbol, "sidecard_pair_add_r4_v1")) { s.lookup_rows = 4; kernel = sidecard_pair_add_r4_v1; }
    if (!std::strcmp(symbol, "sidecard_pair_add_r32_v1")) { s.lookup_rows = 32; kernel = sidecard_pair_add_r32_v1; }
    auto* image = reinterpret_cast<const unsigned char*>(s.image);
    if (!kernel || image[8] != s.lookup_rows) return 73;
    if (!s.null_function) *function = reinterpret_cast<void*>(kernel);
    return 0;
}
aclError aclrtLaunchKernelWithArgsArray(void* function, uint32_t blocks, aclrtStream stream,
                                       aclrtLaunchKernelCfg* config, void** args) {
    auto& s = *stub_state();
    ++s.calls[2];
    s.function = reinterpret_cast<uintptr_t>(function);
    s.stream = reinterpret_cast<uintptr_t>(stream);
    s.blocks = blocks;
    if (config != nullptr) return 92;
    for (unsigned i = 0; i < 3; ++i) {
        s.slots[i] = reinterpret_cast<uintptr_t>(args[i]);
        s.data[i] = reinterpret_cast<uintptr_t>(*static_cast<void**>(args[i]));
    }
    if (s.status[2]) return s.status[2];
    auto kernel = reinterpret_cast<void (*)(GM_ADDR, GM_ADDR, GM_ADDR)>(function);
    for (s.block_index = 0; s.block_index < blocks; ++s.block_index)
        kernel(reinterpret_cast<GM_ADDR>(s.data[0]), reinterpret_cast<GM_ADDR>(s.data[1]),
                reinterpret_cast<GM_ADDR>(s.data[2]));
    return 0;
}
aclError aclrtBinaryUnLoad(aclrtBinHandle binary) {
    auto& s = *stub_state();
    ++s.calls[3];
    if (binary != reinterpret_cast<void*>(0x8000)) return 93;
    return s.status[3];
}
"""


class State(c.Structure):
    _fields_ = [
        ("status", c.c_int * 4),
        ("calls", c.c_uint32 * 4),
        ("image", c.c_size_t),
        ("image_bytes", c.c_uint64),
        ("binary", c.c_size_t),
        ("function", c.c_size_t),
        ("stream", c.c_size_t),
        ("data", c.c_size_t * 3),
        ("slots", c.c_size_t * 3),
        ("blocks", c.c_uint32),
        ("block_index", c.c_uint32),
        ("lookup_rows", c.c_uint32),
        ("magic", c.c_uint32),
        ("option_type", c.c_uint32),
        ("null_binary", c.c_int),
        ("null_function", c.c_int),
        ("partial_binary", c.c_int),
    ]


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native local-add ABI tests require a CPU C++ compiler")
    root = tmp_path_factory.mktemp("pair-local-add-stub")
    (root / "acl").mkdir()
    (root / "acl/acl.h").write_text(ACL_HEADER)
    (root / "kernel_operator.h").write_text(KERNEL_HEADER)
    (root / "stub.cpp").write_text(STUB_SOURCE)
    sources = ROOT / "csrc/sidecard_collectives"
    flags = [compiler, "-std=c++17", "-fPIC", "-Wall", "-Wextra", "-Werror", "-I", str(root)]
    objects = []
    for rows in (1, 4, 32):
        output = root / f"kernel-r{rows}.o"
        subprocess.run(
            [*flags, f"-DSIDECARD_PAIR_ROWS={rows}", "-c", str(sources / "pair_add_kernel.cpp"), "-o", str(output)],
            check=True,
            capture_output=True,
        )
        objects.append(str(output))
    output = root / "local-add.so"
    subprocess.run(
        [*flags, "-shared", str(sources / "pair_add_runtime.cpp"), str(root / "stub.cpp"), *objects, "-o", str(output)],
        check=True,
        capture_output=True,
    )
    library = c.CDLL(str(output))
    library.sidecard_add_abi.restype = c.c_uint64
    library.sidecard_add_create.argtypes = [
        c.c_void_p,
        c.c_uint64,
        c.c_uint64,
        c.c_uint32,
        c.c_void_p,
        c.c_void_p,
        c.c_void_p,
        c.POINTER(c.c_void_p),
    ]
    library.sidecard_add_enqueue.argtypes = [c.c_void_p, c.c_void_p]
    library.sidecard_add_destroy.argtypes = [c.POINTER(c.c_void_p), c.c_uint32]
    library.stub_state.restype = c.POINTER(State)
    return library


@pytest.fixture
def native(compiled):
    state = compiled.stub_state().contents
    c.memset(c.addressof(state), 0, c.sizeof(state))
    return compiled, state


def fixture(rows=1):
    image = (c.c_ubyte * 64)()
    image[:4], image[8] = b"\x7fELF", rows
    arrays = [(c.c_float * (rows * 4096))() for _ in range(3)]
    for index in range(rows * 4096):
        arrays[0][index] = (index % 127 - 63) * 0.125 + 65536
        arrays[1][index] = (index * 13 % 31 - 15) * 0.0625
        arrays[2][index] = float("nan")
    return image, arrays


def create(lib, image, arrays, rows=1, blocks=8):
    handle = c.c_void_p()
    status = lib.sidecard_add_create(image, len(image), rows * 4096, blocks, *arrays, c.byref(handle))
    return status, handle


@pytest.mark.parametrize("rows", (1, 4, 32))
@pytest.mark.parametrize("blocks", (1, 2, 4, 8))
def test_actual_kernel_tiling_complete_fp32_output_and_stream_passthrough(native, rows, blocks):
    lib, state = native
    image, arrays = fixture(rows)
    expected = [c.c_float(a + b).value for a, b in zip(arrays[0], arrays[1], strict=True)]
    original = [bytes(array) for array in arrays[:2]]
    status, handle = create(lib, image, arrays, rows, blocks)
    assert status == 0 and lib.sidecard_add_abi() == 1
    assert state.lookup_rows == rows and state.magic == 0x43554245 and state.option_type == 2
    assert lib.sidecard_add_enqueue(handle, 0x7110) == 0
    assert state.stream == 0x7110 and state.blocks == blocks
    assert list(state.data) == [c.addressof(array) for array in arrays]
    assert list(arrays[2]) == expected and [bytes(array) for array in arrays[:2]] == original
    assert list(state.calls) == [1, 1, 1, 0]
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 0 and handle.value is None


def test_retains_image_and_fixed_argument_slots_across_changed_inputs(native):
    lib, state = native
    image, arrays = fixture()
    copied = bytes(image)
    status, handle = create(lib, image, arrays)
    assert status == 0 and state.image != c.addressof(image)
    image[:] = bytes(len(image))
    assert c.string_at(state.image, state.image_bytes) == copied
    slots = None
    for iteration in range(4):
        arrays[0][0], arrays[1][0] = 70000 + iteration, 13.5 + iteration
        assert lib.sidecard_add_enqueue(handle, 0x7210) == 0
        assert arrays[2][0] == c.c_float(arrays[0][0] + arrays[1][0]).value
        if slots is None:
            slots = list(state.slots)
        assert list(state.slots) == slots
    assert list(state.calls) == [1, 1, 4, 0]
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 0


@pytest.mark.parametrize("rows", (0, 2, 3, 33))
def test_rejects_unsupported_fixed_shape_without_sdk_calls(native, rows):
    lib, state = native
    image, arrays = fixture()
    status, handle = create(lib, image, arrays, rows)
    assert status == -1 and handle.value is None and list(state.calls) == [0] * 4


@pytest.mark.parametrize("blocks", (0, 3, 5, 7, 9, 2**32 - 1))
def test_rejects_unsupported_grid_without_sdk_calls(native, blocks):
    lib, state = native
    image, arrays = fixture()
    status, handle = create(lib, image, arrays, blocks=blocks)
    assert status == -1 and handle.value is None and list(state.calls) == [0] * 4


def test_rejects_bad_image_and_null_arguments_without_sdk_calls(native):
    lib, state = native
    image, arrays = fixture()
    handle = c.c_void_p()
    for pointer, size in ((None, 64), (image, 0), (image, 15), (image, 64 * 1024**2 + 1)):
        assert lib.sidecard_add_create(pointer, size, 4096, 8, *arrays, c.byref(handle)) == -1
    image[0] = 0
    assert create(lib, image, arrays)[0] == -1
    image[0] = 127
    assert lib.sidecard_add_create(image, len(image), 4096, 8, *arrays, None) == -1
    for index in range(3):
        bad = list(arrays)
        bad[index] = None
        assert create(lib, image, bad)[0] == -1
    assert lib.sidecard_add_enqueue(None, 5) == -1
    assert lib.sidecard_add_destroy(None, 1) == -1
    assert list(state.calls) == [0] * 4


@pytest.mark.parametrize("case", ("unaligned", "alias", "partial-overlap", "wrap"))
def test_invalid_local_buffer_ranges_fail_before_sdk(native, case):
    lib, state = native
    image, arrays = fixture()
    addresses = [c.addressof(array) for array in arrays]
    if case == "unaligned":
        addresses[1] += 1
    elif case == "alias":
        addresses[2] = addresses[0]
    elif case == "partial-overlap":
        addresses[2] = addresses[0] + 4
    else:
        addresses[0] = 2 ** (8 * c.sizeof(c.c_void_p)) - 4
    status, handle = create(lib, image, addresses)
    assert status == -1 and handle.value is None and list(state.calls) == [0] * 4


@pytest.mark.parametrize("failure", (0, 1))
def test_load_lookup_errors_preserve_ownership_for_explicit_cleanup(native, failure):
    lib, state = native
    image, arrays = fixture()
    state.status[failure], state.partial_binary = 310007, 1
    status, handle = create(lib, image, arrays)
    assert status == 310007 and handle.value is not None
    assert state.calls[3] == 0 and lib.sidecard_add_enqueue(handle, 7) == -1
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 0 and handle.value is None


def test_wrong_shape_image_fails_symbol_lookup_without_launch(native):
    lib, state = native
    image, arrays = fixture()
    image[8] = 4
    status, handle = create(lib, image, arrays)
    assert status == 73 and handle.value is not None and state.calls[2] == 0
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 0


@pytest.mark.parametrize("field", ("null_binary", "null_function"))
def test_sdk_success_with_null_handle_is_rejected(native, field):
    lib, state = native
    image, arrays = fixture()
    setattr(state, field, 1)
    status, handle = create(lib, image, arrays)
    assert status == -3 and handle.value is not None
    assert lib.sidecard_add_enqueue(handle, 7) == -1
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 0


def test_launch_and_unload_failures_propagate_and_keep_image_until_successful_teardown(native):
    lib, state = native
    image, arrays = fixture()
    status, handle = create(lib, image, arrays)
    assert status == 0
    original = bytes(arrays[2])
    state.status[2] = 310010
    assert lib.sidecard_add_enqueue(handle, 0x7211) == 310010 and bytes(arrays[2]) == original
    state.status[3] = 310011
    address = handle.value
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 310011 and handle.value == address
    assert c.string_at(state.image, len(image)) == bytes(image)
    state.status[3] = 0
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 0 and handle.value is None
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == -1


@pytest.mark.parametrize("proof", (0, 2, 2**32 - 1))
def test_teardown_requires_exact_drained_and_graph_retired_proof(native, proof):
    lib, state = native
    image, arrays = fixture()
    status, handle = create(lib, image, arrays)
    assert status == 0 and lib.sidecard_add_enqueue(handle, None) == -1
    assert lib.sidecard_add_destroy(c.byref(handle), proof) == -1 and state.calls[3] == 0
    assert lib.sidecard_add_destroy(c.byref(handle), 1) == 0
