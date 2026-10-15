// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

// Private owner-local PUSH ABI. Six data ranges belong to the caller; peer
// ranges are SDMA destinations only. All local and imported mappings, Notify
// handles and stream must outlive both-peer drain and explicit destruction.
enum SidecardExchangeDtype : uint64_t {
  SIDECARD_EXCHANGE_FP32 = 0,
  SIDECARD_EXCHANGE_FP16 = 1,
};

enum SidecardExchangeOperation : uint32_t {
  SIDECARD_EXCHANGE_SUM = 0,
  SIDECARD_EXCHANGE_BROADCAST_PARENT = 1,
  SIDECARD_EXCHANGE_BROADCAST_CHILD = 2,
};

struct SidecardExchangeBindings {
  uint64_t abi;
  uint64_t elements;
  uint64_t dtype;
  void* input;
  void* output;
  void* inbox0;
  void* inbox1;
  void* peer_inbox0;
  void* peer_inbox1;
  void* ready0;
  void* ready1;
  void* peer_ready0;
  void* peer_ready1;
  void* workspace;
  uint64_t workspace_bytes;
};

static_assert(sizeof(SidecardExchangeBindings) == 120, "exchange ABI1 requires 64-bit pointers");
static_assert(offsetof(SidecardExchangeBindings, workspace_bytes) == 112, "exchange ABI1 tail moved");

extern "C" {
uint64_t sidecard_exchange_abi();
uint64_t sidecard_exchange_bindings_bytes();
// Bad arguments create no ownership. Once host allocation succeeds, *handle
// owns even partial SDK metadata on failure; enqueue rejects partial handles.
int sidecard_exchange_create(const SidecardExchangeBindings* bindings, void** handle);
// Optional immutable mode, selected exactly once before any enqueue attempt.
// Use inbox0 exclusively; ready0/peer_ready0 are READY and ready1/peer_ready1
// are CONSUMED. Each ordered call acknowledges consumption and notification
// reset before the next call can reuse that inbox. Both peers must agree.
// This fixes captured task addresses; it does not qualify SDK graph support.
int sidecard_exchange_set_fixed_handshake(void* handle);
// Both peers must agree on immutable phase/shape/dtype, preserve input until
// stream completion, and submit on a single ordered owner stream. Failure
// quarantines the handle: no retry, fallback, or implicit resource release.
int sidecard_exchange_enqueue(void* handle, void* stream, uint32_t timeout_raw, uint32_t operation);
// Per-submission local I/O with the same fixed shape/dtype. Actual I/O must be
// disjoint from each other, all local/peer inboxes and workspace; the original
// input/output buffers may be reused. Descriptors are rebound before creating
// the single-use executor. SDK snapshot/lifetime semantics need device gates.
int sidecard_exchange_enqueue_io(void* handle, const void* input, void* output, void* stream, uint32_t timeout_raw,
                                 uint32_t operation);
// Proof must be exactly 1 and cover both peers' queued operations and mapping
// lifetimes. Release failure retains remaining resources and *handle. A drained
// caller can retry destruction; *handle becomes null only on full success.
int sidecard_exchange_destroy(void** handle, uint32_t both_peer_drained);
}
