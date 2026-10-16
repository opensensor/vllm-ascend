// SPDX-License-Identifier: Apache-2.0
// Experimental synchronous transport. Caller MUST finish the producing NPU
// stream before fetch, and retain device allocations until close succeeds.
// This is not a stream operator, graph node, or autonomous device DMA service.
#include <acl/acl.h>
#include <ascend_hal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <chrono>
#include <fstream>
#include <iterator>
#include <new>
#include <vector>

#include "ple_host_gather.h"
#include "ple_host_mapping.h"

namespace {
struct Owner {
  int device = -1;
  size_t tokens = 0, table_bytes = 0;
  void* table = MAP_FAILED;
  std::shared_ptr<ple_host::SharedTable> shared_table;
  void* packet = nullptr;
  void* staging = nullptr;
  void* output = nullptr;
  void* packet_device = nullptr;
  aclrtStream producer_stream = nullptr;
  aclrtBinHandle binary = nullptr;
  aclrtFuncHandle function = nullptr;
  std::vector<char> code;
  void* addresses[7]{};
  void* arguments[7]{};
  unsigned blocks = 0;
  DMA_ADDR request{}, response{};
  bool request_valid = false, response_valid = false, poisoned = false;
};

uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int descriptor(Owner* owner, void* source, void* destination, size_t bytes, DMA_ADDR* result) {
  result->offsetAddr.devid = owner->device;
  int status = drvMemConvertAddr(reinterpret_cast<DVdeviceptr>(source), reinterpret_cast<DVdeviceptr>(destination),
                                 bytes, result);
  if (status) return status;
  if (result->fixed_size != bytes) {
    status = drvMemDestroyAddr(result);
    return status ? status : -EMSGSIZE;
  }
  return 0;
}

int destroy(Owner* owner) {
  // A failed submission can leave uncertain DMA ownership. Retain resources
  // for process teardown rather than freeing possibly in-flight buffers.
  if (owner->poisoned) return -EBUSY;
  int status;
  if (owner->producer_stream) {
    status = aclrtDestroyStream(owner->producer_stream);
    if (status) return status;
    owner->producer_stream = nullptr;
  }
  if (owner->binary) {
    status = aclrtBinaryUnLoad(owner->binary);
    if (status) return status;
    owner->binary = nullptr;
  }
  if (owner->response_valid) {
    status = drvMemDestroyAddr(&owner->response);
    if (status) return status;
    owner->response_valid = false;
  }
  if (owner->request_valid) {
    status = drvMemDestroyAddr(&owner->request);
    if (status) return status;
    owner->request_valid = false;
  }
  if (owner->staging) {
    status = aclrtFreeHost(owner->staging);
    if (status) return status;
    owner->staging = nullptr;
  }
  if (owner->packet) {
    status = aclrtFreeHost(owner->packet);
    if (status) return status;
    owner->packet = nullptr;
  }
  delete owner;
  return 0;
}
}  // namespace

static int create_owner(int device, std::shared_ptr<ple_host::SharedTable> table, size_t tokens, void* packet_device,
                        void* output_device, void** result) {
  if (!result) return -EINVAL;
  *result = nullptr;
  if (device < 0 || device >= ple_host::kDeviceCount || !tokens || tokens > ple_host::kMaxTokens || !packet_device ||
      !output_device)
    return -EINVAL;
  int current;
  int status = aclrtGetDevice(&current);
  if (status) return status;
  if (current != device) return -EXDEV;
  auto* owner = new (std::nothrow) Owner;
  if (!owner) return -ENOMEM;
  owner->device = device;
  owner->tokens = tokens;
  owner->output = output_device;
  owner->packet_device = packet_device;
  owner->shared_table = std::move(table);
  owner->table_bytes = owner->shared_table->bytes;
  owner->table = owner->shared_table->data;
  status = aclrtMallocHost(&owner->packet, tokens * ple_host::kPacketBytesPerToken);
  if (!status) status = aclrtMallocHost(&owner->staging, tokens * ple_host::kHeads * ple_host::kRowBytes);
  if (!status) {
    status = descriptor(owner, packet_device, owner->packet, tokens * ple_host::kPacketBytesPerToken, &owner->request);
    owner->request_valid = status == 0;
  }
  if (!status) {
    status = descriptor(owner, owner->staging, output_device, tokens * ple_host::kHeads * ple_host::kRowBytes,
                        &owner->response);
    owner->response_valid = status == 0;
  }
  if (status) {
    destroy(owner);
    return status;
  }
  *result = owner;
  return 0;
}

extern "C" int ple_host_create(int device, int fd, size_t tokens, void* packet_device, void* output_device,
                               void** result) {
  if (!result) return -EINVAL;
  *result = nullptr;
  if (device < 0 || device >= ple_host::kDeviceCount || !tokens || tokens > ple_host::kMaxTokens || !packet_device ||
      !output_device)
    return -EINVAL;
  int current;
  int status = aclrtGetDevice(&current);
  if (status) return status;
  if (current != device) return -EXDEV;
  std::shared_ptr<ple_host::SharedTable> table;
  status = ple_host::SharedTable::Open(fd, ple_host::kRowBytes, table);
  if (status) return status;
  return create_owner(device, std::move(table), tokens, packet_device, output_device, result);
}

// source must be a live handle created by this same library. Only the immutable
// host mapping is shared; DMA descriptors, staging and producer streams remain
// private. A child retains the mapping even after source is closed.
extern "C" int ple_host_create_shared(void* source, size_t tokens, void* packet_device, void* output_device,
                                      void** result) {
  if (!result) return -EINVAL;
  *result = nullptr;
  auto* parent = static_cast<Owner*>(source);
  if (!parent || !parent->shared_table) return -EINVAL;
  if (parent->poisoned) return -EBUSY;
  return create_owner(parent->device, parent->shared_table, tokens, packet_device, output_device, result);
}

// timings: request readback, validation + gather, response, complete call.
// direct_rows=1 is the comparison: convert/submit/destroy each scattered row.
extern "C" int ple_host_fetch(void* handle, int direct_rows, uint64_t* timings) {
  auto* owner = static_cast<Owner*>(handle);
  if (!owner || !timings || (direct_rows != 0 && direct_rows != 1)) return -EINVAL;
  if (owner->poisoned) return -EBUSY;
  int current;
  int status = aclrtGetDevice(&current);
  if (status) return status;
  if (current != owner->device) return -EXDEV;
  const auto start = now_ns();
  status = halMemcpySumbit(&owner->request, MEMCPY_SUMBIT_SYNC);
  if (status) {
    owner->poisoned = true;
    return status;
  }
  const auto received = now_ns();
  status = direct_rows
               ? ple_host::validate(owner->packet, owner->tokens, owner->table_bytes)
               : ple_host::gather(owner->table, owner->table_bytes, owner->packet, owner->tokens, owner->staging);
  if (status) return status;
  const auto gathered = now_ns();
  if (direct_rows) {
    const auto* rows = static_cast<const int64_t*>(owner->packet);
    for (size_t row = 0; row < owner->tokens * ple_host::kHeads; ++row) {
      DMA_ADDR entry{};
      status = descriptor(owner, static_cast<char*>(owner->table) + rows[row] * ple_host::kRowBytes,
                          static_cast<char*>(owner->output) + row * ple_host::kRowBytes, ple_host::kRowBytes, &entry);
      if (status) return status;
      status = halMemcpySumbit(&entry, MEMCPY_SUMBIT_SYNC);
      if (status) {
        owner->poisoned = true;
        return status;
      }
      status = drvMemDestroyAddr(&entry);
      if (status) {
        owner->poisoned = true;
        return status;
      }
    }
  } else {
    status = halMemcpySumbit(&owner->response, MEMCPY_SUMBIT_SYNC);
    if (status) {
      owner->poisoned = true;
      return status;
    }
  }
  const auto end = now_ns();
  timings[0] = received - start;
  timings[1] = gathered - received;
  timings[2] = end - gathered;
  timings[3] = end - start;
  return 0;
}

extern "C" int ple_host_close(void* handle) { return handle ? destroy(static_cast<Owner*>(handle)) : -EINVAL; }

// Isolated native producer comparison. Inputs must already be ready; this
// private stream does not join Torch's producer queue. Pointers stay owned by
// the caller through close. This kernel is never captured in an ACL graph.
extern "C" int ple_host_set_producer(void* handle, const char* path, void** addresses, unsigned blocks) {
  auto* owner = static_cast<Owner*>(handle);
  if (!owner || !path || !addresses || !blocks || blocks > 8 || owner->binary || owner->poisoned) return -EINVAL;
  for (size_t i = 0; i < 7; ++i)
    if (!addresses[i]) return -EINVAL;
  if (addresses[4] != owner->packet_device ||
      addresses[6] != static_cast<char*>(owner->packet_device) + owner->tokens * ple_host::kHeads * sizeof(int64_t))
    return -EINVAL;
  std::ifstream file(path, std::ios::binary);
  if (!file) return -ENOENT;
  owner->code.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  if (owner->code.empty()) return -EINVAL;
  aclrtBinaryLoadOption option{};
  option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
  option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
  aclrtBinaryLoadOptions options{&option, 1};
  int status = aclrtBinaryLoadFromData(owner->code.data(), owner->code.size(), &options, &owner->binary);
  if (status) return status;
  status = aclrtBinaryGetFunction(owner->binary, "qwen_ple_request_v1", &owner->function);
  if (status) return status;
  status = aclrtCreateStream(&owner->producer_stream);
  if (status) return status;
  owner->blocks = blocks;
  for (size_t i = 0; i < 7; ++i) {
    owner->addresses[i] = addresses[i];
    owner->arguments[i] = &owner->addresses[i];
  }
  return 0;
}

static int produce_fetch(void* handle, uint64_t* timings, unsigned poll_us) {
  auto* owner = static_cast<Owner*>(handle);
  if (!owner || !timings || !owner->producer_stream || !owner->blocks) return -EINVAL;
  if (owner->poisoned) return -EBUSY;
  int current;
  int status = aclrtGetDevice(&current);
  if (status) return status;
  if (current != owner->device) return -EXDEV;
  const auto start = now_ns();
  status =
      aclrtLaunchKernelWithArgsArray(owner->function, owner->blocks, owner->producer_stream, nullptr, owner->arguments);
  bool complete = false;
  uint64_t queries = 0;
  const auto submitted = now_ns();
  if (!status && poll_us) {
    do {
      aclrtStreamStatus state = ACL_STREAM_STATUS_RESERVED;
      status = aclrtStreamQuery(owner->producer_stream, &state);
      ++queries;
      if (status) break;
      if (state == ACL_STREAM_STATUS_COMPLETE) {
        complete = true;
        break;
      }
      if (state != ACL_STREAM_STATUS_NOT_READY) {
        status = -EINVAL;
        break;
      }
    } while (now_ns() - submitted < static_cast<uint64_t>(poll_us) * 1000);
  }
  if (!status && !complete) status = aclrtSynchronizeStream(owner->producer_stream);
  if (status) {
    owner->poisoned = true;
    return status;
  }
  const auto ready = now_ns();
  status = ple_host_fetch(handle, 0, timings);
  timings[4] = ready - start;
  timings[5] = now_ns() - start;
  timings[6] = queries;
  timings[7] = complete;
  return status;
}

extern "C" int ple_host_produce_fetch(void* handle, uint64_t* timings) { return produce_fetch(handle, timings, 0); }

extern "C" int ple_host_produce_fetch_poll(void* handle, uint64_t* timings, unsigned poll_us) {
  // Diagnostic host-runtime poll, separate from the driver's 5-us DMA knob.
  // Preemption remains enabled. Expiry falls back to ACL stream synchronize.
  constexpr unsigned kMaximumProducerPollUs = 50;
  if (poll_us > kMaximumProducerPollUs) return -EINVAL;
  return produce_fetch(handle, timings, poll_us);
}

// Serving's eager boundary joins the Torch stream before this call. Input
// tensors may move between invocations; their caller retains them until return.
extern "C" int ple_host_produce_fetch_inputs(void* handle, void* tokens, void* starts, void* history, uint64_t* timings,
                                             unsigned poll_us) {
  auto* owner = static_cast<Owner*>(handle);
  if (!owner || !tokens || !starts || !history || poll_us > 50) return -EINVAL;
  if (owner->poisoned) return -EBUSY;
  owner->addresses[0] = tokens;
  owner->addresses[1] = starts;
  owner->addresses[2] = history;
  return produce_fetch(handle, timings, poll_us);
}
