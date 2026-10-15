// SPDX-License-Identifier: Apache-2.0
#include "npu_exchange.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>

#include <limits>
#include <new>

namespace {
constexpr uint64_t EXCHANGE_ABI = 1;
constexpr uint64_t MAX_ELEMENTS = 16 * 1024 * 1024;
constexpr uint64_t MAX_WORKSPACE_BYTES = 1024 * 1024;
constexpr int INVALID_ARGUMENT = -1;
constexpr int ALLOCATION_FAILURE = -2;
constexpr int WORKSPACE_EXCEEDED = -3;

struct Exchange {
  SidecardExchangeBindings bindings;
  aclTensor* input = nullptr;
  aclTensor* output = nullptr;
  aclTensor* inbox[2] = {nullptr, nullptr};
  aclScalar* alpha = nullptr;
  unsigned slot = 0;
  bool complete = false;
  bool failed = false;
  bool enqueue_started = false;
  bool fixed_handshake = false;
};

bool valid_storage(const SidecardExchangeBindings& b) {
  const uintptr_t item_bytes = b.dtype == SIDECARD_EXCHANGE_FP32 ? 4 : 2;
  const uintptr_t bytes = b.elements * item_bytes;
  constexpr uintptr_t LIMIT = std::numeric_limits<uintptr_t>::max();
  const void* data[] = {b.input, b.output, b.inbox0, b.inbox1, b.peer_inbox0, b.peer_inbox1};
  const auto workspace = reinterpret_cast<uintptr_t>(b.workspace);
  if (!workspace || workspace % item_bytes || b.workspace_bytes > LIMIT - workspace) return false;
  for (unsigned i = 0; i < 6; ++i) {
    const auto address = reinterpret_cast<uintptr_t>(data[i]);
    if (!address || address % item_bytes || address > LIMIT - bytes) return false;
    for (unsigned j = 0; j < i; ++j) {
      const auto previous = reinterpret_cast<uintptr_t>(data[j]);
      if (address < previous + bytes && previous < address + bytes) return false;
    }
    if (address < workspace + b.workspace_bytes && workspace < address + bytes) return false;
  }
  return true;
}

bool valid_io(const SidecardExchangeBindings& b, const void* input, void* output) {
  const uintptr_t item_bytes = b.dtype == SIDECARD_EXCHANGE_FP32 ? 4 : 2;
  const uintptr_t bytes = b.elements * item_bytes;
  constexpr uintptr_t LIMIT = std::numeric_limits<uintptr_t>::max();
  const void* data[] = {input, output};
  const void* inboxes[] = {b.inbox0, b.inbox1, b.peer_inbox0, b.peer_inbox1};
  for (const auto* pointer : data) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    if (!address || address % item_bytes || address > LIMIT - bytes) return false;
    for (const auto* inbox : inboxes) {
      const auto other = reinterpret_cast<uintptr_t>(inbox);
      if (address < other + bytes && other < address + bytes) return false;
    }
    const auto workspace = reinterpret_cast<uintptr_t>(b.workspace);
    if (address < workspace + b.workspace_bytes && workspace < address + bytes) return false;
  }
  const auto source = reinterpret_cast<uintptr_t>(input);
  const auto destination = reinterpret_cast<uintptr_t>(output);
  return source + bytes <= destination || destination + bytes <= source;
}

aclTensor* tensor(void* address, int64_t elements, aclDataType dtype) {
  const int64_t stride = 1;
  return aclCreateTensor(&elements, 1, dtype, &stride, 0, ACL_FORMAT_ND, &elements, 1, address);
}

int quarantine(Exchange* exchange, int status) {
  if (status) exchange->failed = true;
  return status;
}
}  // namespace

extern "C" uint64_t sidecard_exchange_abi() { return EXCHANGE_ABI; }

extern "C" uint64_t sidecard_exchange_bindings_bytes() { return sizeof(SidecardExchangeBindings); }

extern "C" int sidecard_exchange_create(const SidecardExchangeBindings* b, void** handle) {
  if (!handle) return INVALID_ARGUMENT;
  *handle = nullptr;
  if (!b || b->abi != EXCHANGE_ABI || !b->elements || b->elements > MAX_ELEMENTS || b->dtype > SIDECARD_EXCHANGE_FP16 ||
      !b->workspace_bytes || b->workspace_bytes > MAX_WORKSPACE_BYTES || !b->ready0 || !b->ready1 || !b->peer_ready0 ||
      !b->peer_ready1 || b->ready0 == b->ready1 || b->peer_ready0 == b->peer_ready1 || !valid_storage(*b))
    return INVALID_ARGUMENT;
  auto* exchange = new (std::nothrow) Exchange{*b};
  if (!exchange) return ALLOCATION_FAILURE;
  *handle = exchange;
  const aclDataType dtype = b->dtype == SIDECARD_EXCHANGE_FP32 ? ACL_FLOAT : ACL_FLOAT16;
  exchange->input = tensor(b->input, b->elements, dtype);
  if (!exchange->input) return ALLOCATION_FAILURE;
  exchange->output = tensor(b->output, b->elements, dtype);
  if (!exchange->output) return ALLOCATION_FAILURE;
  exchange->inbox[0] = tensor(b->inbox0, b->elements, dtype);
  if (!exchange->inbox[0]) return ALLOCATION_FAILURE;
  exchange->inbox[1] = tensor(b->inbox1, b->elements, dtype);
  if (!exchange->inbox[1]) return ALLOCATION_FAILURE;
  // ACLNN Add's alpha scalar is the FP32 value 1 for either input dtype.
  float one = 1.0F;
  exchange->alpha = aclCreateScalar(&one, ACL_FLOAT);
  if (!exchange->alpha) return ALLOCATION_FAILURE;
  exchange->complete = true;
  return 0;
}

extern "C" int sidecard_exchange_enqueue(void* handle, void* stream, uint32_t timeout_raw, uint32_t operation) {
  if (!handle) return INVALID_ARGUMENT;
  const auto& b = static_cast<Exchange*>(handle)->bindings;
  return sidecard_exchange_enqueue_io(handle, b.input, b.output, stream, timeout_raw, operation);
}

extern "C" int sidecard_exchange_set_fixed_handshake(void* handle) {
  if (!handle) return INVALID_ARGUMENT;
  auto* exchange = static_cast<Exchange*>(handle);
  if (!exchange->complete || exchange->failed || exchange->enqueue_started || exchange->fixed_handshake)
    return INVALID_ARGUMENT;
  exchange->fixed_handshake = true;
  return 0;
}

extern "C" int sidecard_exchange_enqueue_io(void* handle, const void* input, void* output, void* stream,
                                            uint32_t timeout_raw, uint32_t operation) {
  if (!handle) return INVALID_ARGUMENT;
  auto* exchange = static_cast<Exchange*>(handle);
  if (!exchange->complete || exchange->failed) return INVALID_ARGUMENT;
  // Even an invalid attempted submission freezes protocol selection. No SDK
  // state is changed until argument checks below pass.
  exchange->enqueue_started = true;
  if (!stream || !timeout_raw || operation > SIDECARD_EXCHANGE_BROADCAST_CHILD) return INVALID_ARGUMENT;
  const auto& b = exchange->bindings;
  if (!valid_io(b, input, output)) return INVALID_ARGUMENT;
  const unsigned slot = exchange->fixed_handshake ? 0 : exchange->slot;
  const size_t bytes = b.elements * (b.dtype == SIDECARD_EXCHANGE_FP32 ? 4 : 2);
  aclOpExecutor* executor = nullptr;
  uint64_t workspace_bytes = 0;
  int status = aclSetRawTensorAddr(exchange->input, const_cast<void*>(input));
  if (status) return quarantine(exchange, status);
  status = aclSetRawTensorAddr(exchange->output, output);
  if (status) return quarantine(exchange, status);
  if (operation == SIDECARD_EXCHANGE_SUM) {
    status = aclnnAddGetWorkspaceSize(exchange->input, exchange->inbox[slot], exchange->alpha, exchange->output,
                                      &workspace_bytes, &executor);
    if (status) return quarantine(exchange, status);
    if (workspace_bytes > b.workspace_bytes) return quarantine(exchange, WORKSPACE_EXCEEDED);
  }
  status =
      aclrtMemcpyAsync(slot ? b.peer_inbox1 : b.peer_inbox0, bytes, input, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status) return quarantine(exchange, status);
  status = aclrtRecordNotify(slot ? b.peer_ready1 : b.peer_ready0, stream);
  if (status) return quarantine(exchange, status);
  status = aclrtWaitAndResetNotify(slot ? b.ready1 : b.ready0, stream, timeout_raw);
  if (status) return quarantine(exchange, status);
  if (operation == SIDECARD_EXCHANGE_SUM) {
    status = aclnnAdd(b.workspace, workspace_bytes, executor, stream);
  } else {
    const void* source = operation == SIDECARD_EXCHANGE_BROADCAST_PARENT ? input : (slot ? b.inbox1 : b.inbox0);
    status = aclrtMemcpyAsync(output, bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  }
  if (status) return quarantine(exchange, status);
  if (exchange->fixed_handshake) {
    status = aclrtRecordNotify(b.peer_ready1, stream);
    if (status) return quarantine(exchange, status);
    status = aclrtWaitAndResetNotify(b.ready1, stream, timeout_raw);
    if (status) return quarantine(exchange, status);
  } else {
    exchange->slot ^= 1;
  }
  return 0;
}

extern "C" int sidecard_exchange_destroy(void** handle, uint32_t both_peer_drained) {
  if (!handle || !*handle || both_peer_drained != 1) return INVALID_ARGUMENT;
  auto* exchange = static_cast<Exchange*>(*handle);
  exchange->failed = true;
  aclTensor** tensors[] = {&exchange->input, &exchange->output, &exchange->inbox[0], &exchange->inbox[1]};
  for (auto** tensor_handle : tensors) {
    if (!*tensor_handle) continue;
    const int status = aclDestroyTensor(*tensor_handle);
    if (status) return status;
    *tensor_handle = nullptr;
  }
  if (exchange->alpha) {
    const int status = aclDestroyScalar(exchange->alpha);
    if (status) return status;
    exchange->alpha = nullptr;
  }
  delete exchange;
  *handle = nullptr;
  return 0;
}
