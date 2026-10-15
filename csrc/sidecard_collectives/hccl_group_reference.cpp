// SPDX-License-Identifier: Apache-2.0
// Raw, out-of-place FP16/FP32 SUM comparator on an existing owner context.
// No ACL/device/stream initialization, synchronization or global changes occur.
// The caller owns matched N-rank collective order, storage and completion.
#include <hccl/hccl.h>

#include <cstdint>
#include <cstring>
#include <limits>

namespace {
constexpr uint64_t REFERENCE_ABI = 1;
constexpr uint32_t DTYPE_FP32 = 0;
constexpr uint32_t DTYPE_FP16 = 1;
constexpr int INVALID_ARGUMENT = -1;
constexpr int INVALID_SDK_RESULT = -2;
constexpr int MISSING_SDK_API = -3;

bool valid_buffers(const void* input, void* output, uint64_t elements, uint32_t dtype) {
  constexpr uintptr_t LIMIT = std::numeric_limits<uintptr_t>::max();
  const uintptr_t item_bytes = dtype == DTYPE_FP32 ? sizeof(float) : sizeof(uint16_t);
  if (dtype > DTYPE_FP16 || !input || !output || !elements || elements > LIMIT / item_bytes) return false;
  const uintptr_t bytes = static_cast<uintptr_t>(elements) * item_bytes;
  const auto source = reinterpret_cast<uintptr_t>(input);
  const auto destination = reinterpret_cast<uintptr_t>(output);
  if (source % item_bytes || destination % item_bytes || source > LIMIT - bytes || destination > LIMIT - bytes)
    return false;
  // Equal storage and partial overlap are both forbidden, matching the custom
  // out-of-place comparator contract without adding fences to the timing path.
  return source + bytes <= destination || destination + bytes <= source;
}
}  // namespace

extern "C" uint64_t sidecard_hccl_group_abi() { return REFERENCE_ABI; }

extern "C" uint64_t sidecard_hccl_group_root_info_bytes() { return sizeof(HcclRootInfo); }

extern "C" int sidecard_hccl_group_get_root_info(void* serialized, uint64_t bytes) {
  if (!serialized || bytes != sizeof(HcclRootInfo)) return INVALID_ARGUMENT;
  if (!HcclGetRootInfo) return MISSING_SDK_API;
  HcclRootInfo info{};
  const HcclResult status = HcclGetRootInfo(&info);
  if (status == HCCL_SUCCESS) std::memcpy(serialized, &info, sizeof(info));
  return static_cast<int>(status);
}

// The output slot must not already own another communicator. Preserve partial
// SDK initialization even on error for explicit caller quarantine/drain.
extern "C" int sidecard_hccl_group_create(const void* serialized, uint64_t bytes, uint32_t world, uint32_t rank,
                                          void** comm) {
  if (!comm) return INVALID_ARGUMENT;
  *comm = nullptr;
  if (!serialized || bytes != sizeof(HcclRootInfo) || !world || rank >= world) return INVALID_ARGUMENT;
  if (!HcclCommInitRootInfo) return MISSING_SDK_API;
  HcclRootInfo info{};
  std::memcpy(&info, serialized, sizeof(info));
  HcclComm initialized = nullptr;
  const HcclResult status = HcclCommInitRootInfo(world, &info, rank, &initialized);
  *comm = initialized;
  if (status == HCCL_SUCCESS && !initialized) return INVALID_SDK_RESULT;
  return static_cast<int>(status);
}

extern "C" int sidecard_hccl_group_all_reduce(void* comm, const void* input, void* output, uint64_t elements,
                                              uint32_t dtype, void* stream) {
  if (!comm || !stream || !valid_buffers(input, output, elements, dtype)) return INVALID_ARGUMENT;
  // AllReduce remains a strong SDK symbol; missing collective support must fail
  // library loading rather than silently selecting a fallback implementation.
  const HcclDataType sdk_dtype = dtype == DTYPE_FP32 ? HCCL_DATA_TYPE_FP32 : HCCL_DATA_TYPE_FP16;
  return static_cast<int>(HcclAllReduce(const_cast<void*>(input), output, elements, sdk_dtype, HCCL_REDUCE_SUM, comm,
                                        static_cast<aclrtStream>(stream)));
}

// Proof exactly 1 covers completion and every rank's agreement to release.
// The shim never waits implicitly, retries or discards a failed SDK handle.
extern "C" int sidecard_hccl_group_destroy(void** comm, uint32_t caller_drained) {
  if (!comm || !*comm || caller_drained != 1) return INVALID_ARGUMENT;
  if (!HcclCommDestroy) return MISSING_SDK_API;
  const HcclResult status = HcclCommDestroy(static_cast<HcclComm>(*comm));
  if (status == HCCL_SUCCESS) *comm = nullptr;
  return static_cast<int>(status);
}
