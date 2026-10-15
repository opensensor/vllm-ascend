// SPDX-License-Identifier: Apache-2.0
#include "npu_group.h"

#include "npu_exchange.h"

#include <acl/acl.h>

#include <algorithm>
#include <limits>
#include <new>
#include <unistd.h>
#include <vector>

namespace {
constexpr uint64_t GROUP_ABI = 1;
constexpr uint64_t MAX_BYTES = 64 * 1024 * 1024;
constexpr uint64_t MAX_STEPS = 128;
constexpr int INVALID_ARGUMENT = -1;
constexpr int ALLOCATION_FAILURE = -2;
constexpr int QUARANTINED = -3;
constexpr int FOREIGN_PROCESS = -4;
constexpr int OUTSTANDING_GRAPH_LEASES = -5;

struct GraphLease;

struct Group {
  void* const work_input;
  void* const work_output;
  const uint64_t bytes;
  const uint32_t dtype;
  const int32_t device_index;
  const std::vector<SidecardGroupStep> steps;
  const pid_t pid = getpid();
  bool failed = false;
  std::vector<GraphLease*> graph_leases{};
};

struct GraphLease {
  Group* const group;
  const pid_t pid;
};

bool valid_range(const void* pointer, uint64_t bytes, uint32_t dtype) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  const uintptr_t alignment = dtype == SIDECARD_EXCHANGE_FP32 ? sizeof(float) : sizeof(uint16_t);
  return address && address % alignment == 0 && address <= std::numeric_limits<uintptr_t>::max() - bytes;
}

bool overlaps(const void* first, const void* second, uint64_t bytes) {
  const auto a = reinterpret_cast<uintptr_t>(first);
  const auto b = reinterpret_cast<uintptr_t>(second);
  return a < b + bytes && b < a + bytes;
}

int quarantine(Group* group, int status) {
  if (status) group->failed = true;
  return status;
}

int copy(Group* group, void* destination, const void* source, void* stream) {
  return quarantine(
      group, aclrtMemcpyAsync(destination, group->bytes, source, group->bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
}
}  // namespace

extern "C" uint64_t sidecard_group_abi() { return GROUP_ABI; }

extern "C" int sidecard_group_create(void* work_input, void* work_output, uint64_t bytes, uint32_t dtype,
                                     int32_t device_index, const SidecardGroupStep* steps, uint64_t count,
                                     void** handle) {
  if (!handle) return INVALID_ARGUMENT;
  *handle = nullptr;
  const uint64_t item_bytes = dtype == SIDECARD_EXCHANGE_FP32 ? sizeof(float) : sizeof(uint16_t);
  if (dtype > SIDECARD_EXCHANGE_FP16 || device_index < 0 || !bytes || bytes > MAX_BYTES || bytes % item_bytes ||
      !valid_range(work_input, bytes, dtype) || !valid_range(work_output, bytes, dtype) ||
      overlaps(work_input, work_output, bytes) || count > MAX_STEPS ||
      (count && (!steps || reinterpret_cast<uintptr_t>(steps) % alignof(SidecardGroupStep))))
    return INVALID_ARGUMENT;
  for (uint64_t index = 0; index < count; ++index) {
    if (!steps[index].exchange || steps[index].reserved || steps[index].operation > SIDECARD_EXCHANGE_BROADCAST_CHILD)
      return INVALID_ARGUMENT;
  }
  try {
    // Avoid pointer arithmetic on null for the identity plan.
    const std::vector<SidecardGroupStep> owned =
        count ? std::vector<SidecardGroupStep>(steps, steps + count) : std::vector<SidecardGroupStep>{};
    *handle = new Group{work_input, work_output, bytes, dtype, device_index, owned};
  } catch (const std::bad_alloc&) {
    return ALLOCATION_FAILURE;
  }
  return 0;
}

extern "C" int sidecard_group_metadata(void* handle, uint64_t* bytes, uint32_t* dtype, int32_t* device_index) {
  if (!handle || !bytes || !dtype || !device_index) return INVALID_ARGUMENT;
  const auto* group = static_cast<Group*>(handle);
  if (group->pid != getpid()) return FOREIGN_PROCESS;
  if (group->failed) return QUARANTINED;
  *bytes = group->bytes;
  *dtype = group->dtype;
  *device_index = group->device_index;
  return 0;
}

extern "C" int sidecard_group_enqueue(void* handle, const void* input, void* output, void* stream,
                                      uint32_t timeout_raw) {
  if (!handle || !stream || !timeout_raw) return INVALID_ARGUMENT;
  auto* group = static_cast<Group*>(handle);
  if (group->pid != getpid()) return FOREIGN_PROCESS;
  if (group->failed) return QUARANTINED;
  if (!valid_range(input, group->bytes, group->dtype) || !valid_range(output, group->bytes, group->dtype) ||
      (input != output && overlaps(input, output, group->bytes)) || overlaps(input, group->work_input, group->bytes) ||
      overlaps(input, group->work_output, group->bytes) || overlaps(output, group->work_input, group->bytes) ||
      overlaps(output, group->work_output, group->bytes))
    return INVALID_ARGUMENT;
  if (group->steps.empty()) return input == output ? 0 : copy(group, output, input, stream);
  const void* current = input;
  for (size_t index = 0; index < group->steps.size(); ++index) {
    const auto& step = group->steps[index];
    void* destination =
        index + 1 == group->steps.size() ? output : (index % 2 == 0 ? group->work_input : group->work_output);
    // Only a single-step exact caller alias needs copyback. Multi-step plans
    // already read scratch on their final step; no separate staging copies.
    if (current == destination) destination = group->work_input;
    const int status =
        sidecard_exchange_enqueue_io(step.exchange, current, destination, stream, timeout_raw, step.operation);
    if (status) return quarantine(group, status);
    current = destination;
  }
  return current == output ? 0 : copy(group, output, current, stream);
}

extern "C" int sidecard_group_destroy(void** handle, uint32_t both_peer_drained) {
  if (!handle || !*handle || both_peer_drained != 1) return INVALID_ARGUMENT;
  auto* group = static_cast<Group*>(*handle);
  if (group->pid != getpid()) return FOREIGN_PROCESS;
  if (!group->graph_leases.empty()) return OUTSTANDING_GRAPH_LEASES;
  delete group;
  *handle = nullptr;
  return 0;
}

extern "C" int sidecard_group_graph_lease_acquire(void* handle, void** lease) {
  if (!lease || *lease) return INVALID_ARGUMENT;
  if (!handle) return INVALID_ARGUMENT;
  auto* group = static_cast<Group*>(handle);
  if (group->pid != getpid()) return FOREIGN_PROCESS;
  if (group->failed) return QUARANTINED;
  GraphLease* acquired = nullptr;
  try {
    acquired = new GraphLease{group, getpid()};
    group->graph_leases.push_back(acquired);
  } catch (const std::bad_alloc&) {
    delete acquired;
    return ALLOCATION_FAILURE;
  }
  *lease = acquired;
  return 0;
}

extern "C" int sidecard_group_graph_lease_release(void** lease, uint32_t graph_reset_and_device_drained) {
  if (!lease || !*lease || graph_reset_and_device_drained != 1) return INVALID_ARGUMENT;
  auto* acquired = static_cast<GraphLease*>(*lease);
  if (acquired->pid != getpid()) return FOREIGN_PROCESS;
  auto* group = acquired->group;
  if (!group || group->pid != getpid()) return FOREIGN_PROCESS;
  const auto found = std::find(group->graph_leases.begin(), group->graph_leases.end(), acquired);
  if (found == group->graph_leases.end()) return INVALID_ARGUMENT;
  group->graph_leases.erase(found);
  delete acquired;
  *lease = nullptr;
  return 0;
}

extern "C" int sidecard_group_graph_lease_count(void* handle, uint64_t* count) {
  if (!handle || !count) return INVALID_ARGUMENT;
  const auto* group = static_cast<Group*>(handle);
  if (group->pid != getpid()) return FOREIGN_PROCESS;
  *count = group->graph_leases.size();
  return 0;
}
