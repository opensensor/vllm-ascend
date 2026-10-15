// SPDX-License-Identifier: Apache-2.0
// Raw, out-of-place FP32 SUM reference on an existing caller-owned context.
// This shim never initializes ACL, selects a device, creates a stream, fences,
// resets hardware, or changes process/global configuration. The caller owns
// matched two-rank collective order, buffers, context, stream, and completion.
#include <hccl/hccl.h>

#include <cstdint>
#include <cstring>
#include <limits>

namespace {
constexpr uint64_t REFERENCE_ABI = 1;
constexpr uint32_t RANKS = 2;
constexpr int INVALID_ARGUMENT = -1;
constexpr int INVALID_SDK_RESULT = -2;
constexpr int MISSING_SDK_API = -3;

bool valid_buffers(const void* input, void* output, uint64_t elements)
{
    constexpr uintptr_t LIMIT = std::numeric_limits<uintptr_t>::max();
    if (!input || !output || elements == 0 || elements > LIMIT / sizeof(float)) return false;
    const uintptr_t bytes = static_cast<uintptr_t>(elements) * sizeof(float);
    const uintptr_t source = reinterpret_cast<uintptr_t>(input);
    const uintptr_t destination = reinterpret_cast<uintptr_t>(output);
    if (source % alignof(float) || destination % alignof(float) || source > LIMIT - bytes ||
        destination > LIMIT - bytes) {
        return false;
    }
    // Keep the reference's storage contract equal to the custom pair path:
    // no in-place operation and no partial input/output overlap.
    return source + bytes <= destination || destination + bytes <= source;
}
}  // namespace

extern "C" uint64_t sidecard_hccl_abi()
{
    return REFERENCE_ABI;
}

extern "C" uint64_t sidecard_hccl_root_info_bytes()
{
    return sizeof(HcclRootInfo);
}

extern "C" int sidecard_hccl_get_root_info(void* serialized, uint64_t bytes)
{
    if (!serialized || bytes != sizeof(HcclRootInfo)) return INVALID_ARGUMENT;
    if (!HcclGetRootInfo) return MISSING_SDK_API;
    HcclRootInfo info{};
    const HcclResult status = HcclGetRootInfo(&info);
    if (status == HCCL_SUCCESS) std::memcpy(serialized, &info, sizeof(info));
    return static_cast<int>(status);
}

// comm is an output slot; the caller must not overwrite an owned communicator
// by reusing it. On SDK failure, preserve any partially initialized non-null
// communicator returned by that SDK for caller quarantine. Never destroy it
// implicitly or retry an initialization that may have submitted device work.
extern "C" int sidecard_hccl_create(const void* serialized, uint64_t bytes, uint32_t rank, void** comm)
{
    if (!comm) return INVALID_ARGUMENT;
    *comm = nullptr;
    if (!serialized || bytes != sizeof(HcclRootInfo) || rank >= RANKS) return INVALID_ARGUMENT;
    if (!HcclCommInitRootInfo) return MISSING_SDK_API;
    HcclRootInfo info{};
    std::memcpy(&info, serialized, sizeof(info));
    HcclComm initialized = nullptr;
    const HcclResult status = HcclCommInitRootInfo(RANKS, &info, rank, &initialized);
    *comm = initialized;
    if (status == HCCL_SUCCESS && initialized == nullptr) return INVALID_SDK_RESULT;
    return static_cast<int>(status);
}

extern "C" int sidecard_hccl_all_reduce(void* comm, const void* input, void* output, uint64_t elements, void* stream)
{
    if (!comm || !stream || !valid_buffers(input, output, elements)) return INVALID_ARGUMENT;
    // HCCL's installed C ABI uses a non-const send pointer. Out-of-place SUM
    // must preserve input storage; CPU tests check our raw pointer passthrough.
    return static_cast<int>(HcclAllReduce(const_cast<void*>(input), output, elements, HCCL_DATA_TYPE_FP32,
                                          HCCL_REDUCE_SUM, comm, static_cast<aclrtStream>(stream)));
}

// caller_drained==1 is the caller's explicit proof that all submitted work on
// this communicator is complete and both ranks have agreed to stop/release.
// The shim cannot verify this proof without adding synchronization to timing.
// Destroy errors preserve the handle for quarantine; no implicit retry occurs.
extern "C" int sidecard_hccl_destroy(void** comm, uint32_t caller_drained)
{
    if (!comm || !*comm || caller_drained != 1) return INVALID_ARGUMENT;
    if (!HcclCommDestroy) return MISSING_SDK_API;
    const HcclResult status = HcclCommDestroy(static_cast<HcclComm>(*comm));
    if (status == HCCL_SUCCESS) *comm = nullptr;
    return static_cast<int>(status);
}
