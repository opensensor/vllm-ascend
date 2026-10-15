// SPDX-License-Identifier: Apache-2.0
#include "pair_sum.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>

#include <limits>
#include <new>

namespace {
constexpr uint64_t PAIR_ABI = 2;
constexpr uint64_t WIDTH = 4096;
constexpr uint64_t MAX_ELEMENTS = 32 * WIDTH;
constexpr int INVALID_ARGUMENT = -1;
constexpr int ALLOCATION_FAILURE = -2;
constexpr int WORKSPACE_EXCEEDED = -3;
constexpr int UNSUPPORTED_EXCHANGE_MODE = -4;

struct Pair {
    SidecardPairBindings bindings;
    aclTensor* input = nullptr;
    aclTensor* other = nullptr;
    aclTensor* output = nullptr;
    aclScalar* alpha = nullptr;
    void* add_handle = nullptr;
    int (*enqueue_add)(void*, void*) = nullptr;
    bool started = false;
    aclTensor* other_next = nullptr;
    void* inbox_next = nullptr;
    void* peer_data_next = nullptr;
    unsigned slot = 0;

    ~Pair()
    {
        if (input) aclDestroyTensor(input);
        if (other) aclDestroyTensor(other);
        if (other_next) aclDestroyTensor(other_next);
        if (output) aclDestroyTensor(output);
        if (alpha) aclDestroyScalar(alpha);
    }
};

aclTensor* tensor(void* address, int64_t elements)
{
    const int64_t stride = 1;
    return aclCreateTensor(&elements, 1, ACL_FLOAT, &stride, 0, ACL_FORMAT_ND,
                           &elements, 1, address);
}

bool valid_storage(const SidecardPairBindings& b)
{
    // Imported and local data are ordinary FP32 ranges in the owner's address
    // space. PUSH permits no input/output alias. The imported peer pointer is
    // used only as an SDMA destination, never as a vector add input.
    const void* data[] = {b.input, b.inbox, b.output, b.peer_data};
    const uintptr_t bytes = b.elements * sizeof(float);
    constexpr uintptr_t LIMIT = std::numeric_limits<uintptr_t>::max();
    for (unsigned i = 0; i < 4; ++i) {
        const auto begin = reinterpret_cast<uintptr_t>(data[i]);
        if (!begin || begin % alignof(float) || begin > LIMIT - bytes) return false;
        for (unsigned j = 0; j < i; ++j) {
            const auto previous = reinterpret_cast<uintptr_t>(data[j]);
            if (begin < previous + bytes && previous < begin + bytes) return false;
        }
    }
    const auto workspace = reinterpret_cast<uintptr_t>(b.workspace);
    if (!workspace || workspace % alignof(float) || b.workspace_bytes > LIMIT - workspace) return false;
    for (const auto* address : data) {
        const auto begin = reinterpret_cast<uintptr_t>(address);
        if (workspace < begin + bytes && begin < workspace + b.workspace_bytes) return false;
    }
    return true;
}
}  // namespace

extern "C" uint64_t sidecard_pair_abi()
{
    return PAIR_ABI;
}

extern "C" uint64_t sidecard_pair_bindings_bytes()
{
    return sizeof(SidecardPairBindings);
}

extern "C" int sidecard_pair_create(const SidecardPairBindings* b, void** handle)
{
    if (!handle) return INVALID_ARGUMENT;
    *handle = nullptr;
    if (!b || b->abi != PAIR_ABI || b->exchange_mode > SIDECARD_PAIR_REMOTE_READ) return INVALID_ARGUMENT;
    // Fail before creating SDK tensor descriptors or any queued work. A real
    // peer-read add timed out and faulted the driver; this is not a fallback.
    if (b->exchange_mode == SIDECARD_PAIR_REMOTE_READ) return UNSUPPORTED_EXCHANGE_MODE;
    if (b->elements < WIDTH ||
        b->elements > MAX_ELEMENTS || b->elements % WIDTH || !b->input ||
        !b->inbox || !b->output || !b->peer_data || !b->ready ||
        !b->consumed || !b->peer_ready || !b->peer_consumed ||
        !b->workspace || !b->workspace_bytes || b->ready == b->consumed ||
        b->peer_ready == b->peer_consumed ||
        !valid_storage(*b)) {
        return INVALID_ARGUMENT;
    }
    auto* pair = new (std::nothrow) Pair{*b};
    if (!pair) return ALLOCATION_FAILURE;
    pair->input = tensor(b->input, b->elements);
    pair->other = tensor(b->inbox, b->elements);
    pair->output = tensor(b->output, b->elements);
    float one = 1.0F;
    pair->alpha = aclCreateScalar(&one, ACL_FLOAT);
    if (!pair->input || !pair->other || !pair->output || !pair->alpha) {
        delete pair;
        return ALLOCATION_FAILURE;
    }
    *handle = pair;
    return 0;
}

extern "C" int sidecard_pair_enqueue(void* handle, void* stream,
                                    uint32_t notify_timeout_raw)
{
    if (!handle || !stream || !notify_timeout_raw) return INVALID_ARGUMENT;
    auto* pair = static_cast<Pair*>(handle);
    pair->started = true;
    auto& b = pair->bindings;
    const bool ping_pong = pair->other_next != nullptr;
    const unsigned slot = pair->slot;
    uint64_t workspace_bytes = 0;
    aclOpExecutor* executor = nullptr;
    // An executor is single-use; descriptors and external workspace persist.
    int status = 0;
    if (!pair->enqueue_add) {
        status = aclnnAddGetWorkspaceSize(pair->input, slot ? pair->other_next : pair->other, pair->alpha,
                                          pair->output, &workspace_bytes, &executor);
        if (status) return status;
        if (workspace_bytes > b.workspace_bytes) return WORKSPACE_EXCEEDED;
    }
    const size_t bytes = b.elements * sizeof(float);
    status = aclrtMemcpyAsync(slot ? pair->peer_data_next : b.peer_data, bytes, b.input, bytes,
                              ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (status) return status;
    // Each Notify is owned by its waiting rank and imported by its recording
    // peer. The add reads only local input and the local SDMA-populated inbox.
    // No IPC event, cross-device StreamWaitEvent, or host barrier is used.
    status = aclrtRecordNotify(slot ? b.peer_consumed : b.peer_ready, stream);
    if (status) return status;
    status = aclrtWaitAndResetNotify(slot ? b.consumed : b.ready, stream, notify_timeout_raw);
    if (status) return status;
    status = pair->enqueue_add
                 ? pair->enqueue_add(pair->add_handle, stream)
                 : aclnnAdd(b.workspace, workspace_bytes, executor, stream);
    if (status) return status;
    if (ping_pong) {
        // Reusing slot s at s+2 requires receiving peer READY(s+1). The peer
        // records that only after its wait/reset and local add at s complete.
        // This implicitly acknowledges BOTH the inbox read and Notify reset.
        pair->slot ^= 1;
        return 0;
    }
    status = aclrtRecordNotify(b.peer_consumed, stream);
    if (status) return status;
    // Keep local input immutable until both peers' adds complete.
    return aclrtWaitAndResetNotify(b.consumed, stream, notify_timeout_raw);
}

extern "C" int sidecard_pair_set_local_add(void* handle, void* add_handle,
                                          int (*enqueue)(void*, void*))
{
    if (!handle || !add_handle || !enqueue) return INVALID_ARGUMENT;
    auto* pair = static_cast<Pair*>(handle);
    if (pair->started || pair->enqueue_add || pair->other_next) return INVALID_ARGUMENT;
    pair->add_handle = add_handle;
    pair->enqueue_add = enqueue;
    return 0;
}

extern "C" int sidecard_pair_enable_ping_pong(void* handle, void* local_inbox_next,
                                              void* peer_data_next)
{
    if (!handle) return INVALID_ARGUMENT;
    auto* pair = static_cast<Pair*>(handle);
    if (pair->started || pair->enqueue_add || pair->other_next) return INVALID_ARGUMENT;
    const auto& b = pair->bindings;
    const void* data[] = {b.input, b.inbox, b.output, b.peer_data, local_inbox_next, peer_data_next};
    const uintptr_t bytes = b.elements * sizeof(float);
    constexpr uintptr_t LIMIT = std::numeric_limits<uintptr_t>::max();
    for (unsigned i = 0; i < 6; ++i) {
        const auto address = reinterpret_cast<uintptr_t>(data[i]);
        if (!address || address % alignof(float) || address > LIMIT - bytes) return INVALID_ARGUMENT;
        for (unsigned j = 0; j < i; ++j) {
            const auto previous = reinterpret_cast<uintptr_t>(data[j]);
            if (address < previous + bytes && previous < address + bytes) return INVALID_ARGUMENT;
        }
        const auto workspace = reinterpret_cast<uintptr_t>(b.workspace);
        if (address < workspace + b.workspace_bytes && workspace < address + bytes) return INVALID_ARGUMENT;
    }
    pair->other_next = tensor(local_inbox_next, b.elements);
    if (!pair->other_next) return ALLOCATION_FAILURE;
    pair->inbox_next = local_inbox_next;
    pair->peer_data_next = peer_data_next;
    return 0;
}

extern "C" int sidecard_pair_destroy(void* handle)
{
    if (!handle) return INVALID_ARGUMENT;
    delete static_cast<Pair*>(handle);
    return 0;
}
