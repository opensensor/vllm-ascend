// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

// Every borrowed exchange must bind the same work_input/work_output ranges and
// byte geometry. The caller authenticates those bindings and retains exchanges,
// work buffers, peer mappings and stream until proven drain and destruction.
struct SidecardGroupStep {
  void* exchange;
  uint32_t operation;
  uint32_t reserved = 0;
};

static_assert(sizeof(SidecardGroupStep) == 16, "group ABI1 requires 64-bit pointers");
static_assert(offsetof(SidecardGroupStep, reserved) == 12, "group ABI1 tail moved");

extern "C" {
uint64_t sidecard_group_abi();
// Host metadata only. Work ranges must be aligned, disjoint and at least bytes
// long. Steps are copied; count zero represents a single-rank identity plan.
int sidecard_group_create(void* work_input, void* work_output, uint64_t bytes, uint32_t dtype, int32_t device_index,
                          const SidecardGroupStep* steps, uint64_t count, void** handle);
// Host metadata query for validating Torch payload geometry before queueing.
// Dtype 0 is FP32, 1 is FP16. Rejects foreign-process/quarantined plans.
int sidecard_group_metadata(void* handle, uint64_t* bytes, uint32_t* dtype, int32_t* device_index);
// One serialized owner stream, creating process only. Caller ranges must not
// overlap work storage. Exact input/output alias is permitted; partial alias is
// rejected. Any SDK/borrowed-exchange failure permanently quarantines the plan.
// No allocation, synchronization, device selection or fallback occurs here.
int sidecard_group_enqueue(void* handle, const void* input, void* output, void* stream, uint32_t timeout_raw);
// Cold-path graph ownership only: acquire for each needed group BEFORE capture
// and retain the lease throughout native replay. Neither replay nor enqueue
// consults the lease registry or updates any graph replay counter. Lease output
// slots must be fresh null pointers; occupied slots are rejected unchanged.
// Serialize cold ownership changes.
int sidecard_group_graph_lease_acquire(void* handle, void** lease);
// Proof exactly 1 promises that the corresponding native graph was explicitly
// reset and the device drained. C cannot verify that promise: this function
// performs no SDK reset, query or synchronization. Release remains permitted
// after quarantine with this explicit proof; failure preserves the lease slot.
int sidecard_group_graph_lease_release(void** lease, uint32_t graph_reset_and_device_drained);
// Metadata-only cleanup query, available even after quarantine.
int sidecard_group_graph_lease_count(void* handle, uint64_t* count);
// Metadata only; never destroys borrowed exchanges. Proof exactly 1 promises
// both-peer completion, queued Torch-handler retirement and storage lifetimes.
// Outstanding graph leases reject destruction even when proof is supplied.
int sidecard_group_destroy(void** handle, uint32_t both_peer_drained);
}
