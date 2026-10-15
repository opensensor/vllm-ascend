// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Private local-only FP32 add ABI1. The caller owns initialized context, local
// input/inbox/output allocations, their immutable addresses, and stream order.
// Never bind imported IPC peer memory: PUSH populates the local inbox by SDMA.
// No reset, ACL initialization, device selection, OPP registration, or host
// synchronization occurs here. Nonzero ACL errors propagate unchanged.
// Shim statuses: -1 invalid argument; -2 host allocation failure;
// -3 SDK success without the promised binary/function handle.
extern "C" {
uint64_t sidecard_add_abi();
int sidecard_add_create(const void* image, uint64_t image_bytes, uint64_t elements, uint32_t blocks,
                        const void* local_input, const void* local_inbox, void* local_output, void** handle);
int sidecard_add_enqueue(void* handle, void* caller_stream);
// Proof==1 requires all submitted work drained AND captured graph references
// retired. On unload failure retain the handle, code bytes and fixed arguments.
int sidecard_add_destroy(void** handle, uint32_t caller_drained);
}
