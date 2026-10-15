// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

// Private experimental ABI. All pointers and the stream belong to the calling
// SDK owner. No handle, address or allocation may be reused after enqueue failure.
enum SidecardPairExchangeMode : uint64_t {
    SIDECARD_PAIR_PUSH = 0,
    // Reserved for rejection receipts. Imported peer vector access is unsafe
    // on this 310P stack; native create rejects this mode with status -4.
    SIDECARD_PAIR_REMOTE_READ = 1,
};

struct SidecardPairBindings {
    uint64_t abi;
    uint64_t elements;
    void* input;
    void* inbox;
    void* output;
    // Imported peer inbox for PUSH. REMOTE_READ is never admitted.
    void* peer_data;
    void* ready;
    void* consumed;
    void* peer_ready;
    void* peer_consumed;
    void* workspace;
    uint64_t workspace_bytes;
    uint64_t exchange_mode;
};

static_assert(sizeof(SidecardPairBindings) == 104, "sidecard pair ABI2 requires 64-bit pointers");
static_assert(offsetof(SidecardPairBindings, exchange_mode) == 96, "sidecard pair ABI2 tail field moved");

extern "C" {
uint64_t sidecard_pair_abi();
uint64_t sidecard_pair_bindings_bytes();
int sidecard_pair_create(const SidecardPairBindings* bindings, void** handle);
int sidecard_pair_enqueue(void* handle, void* stream, uint32_t notify_timeout_raw);
// Optional authenticated LOCAL add implementation. Bind once before enqueue;
// the caller retains its code/handle until the pair is drained and destroyed.
int sidecard_pair_set_local_add(void* handle, void* add_handle,
                               int (*enqueue)(void*, void*));
// Optional two-inbox protocol, mutually exclusive with the local-add callback.
// Existing READY/CONSUMED objects become alternating READY[0]/READY[1].
// Must bind once before enqueue; caller retains both extra mappings until the
// same both-peer drain/import-close proof used for the original inbox.
int sidecard_pair_enable_ping_pong(void* handle, void* local_inbox_next,
                                   void* peer_data_next);
// Caller must prove local completion AND both peers' imported mappings closed.
int sidecard_pair_destroy(void* handle);
}
