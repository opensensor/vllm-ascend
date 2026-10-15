// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cerrno>
#include <cstdint>

enum aclrtStreamStatus { ACL_STREAM_STATUS_COMPLETE, ACL_STREAM_STATUS_NOT_READY, ACL_STREAM_STATUS_RESERVED };
struct Owner {
    void* producer_stream = reinterpret_cast<void*>(1);
    unsigned blocks = 1;
    bool poisoned = false;
    int device = 0;
    void* function = nullptr;
    void* arguments[7]{};
    void* addresses[7]{};
};
static unsigned queries, syncs, fetches, ready_after;
static int query_error, sync_error, launch_error, fetch_error;
static bool invalid_state;
static uint64_t clock_ns;
static uint64_t now_ns() { clock_ns += 1000; return clock_ns; }
static int aclrtGetDevice(int* result) { *result = 0; return 0; }
static int aclrtLaunchKernelWithArgsArray(void*, unsigned, void*, void*, void**) { return launch_error; }
static int aclrtStreamQuery(void*, aclrtStreamStatus* state) {
    ++queries;
    *state = invalid_state ? ACL_STREAM_STATUS_RESERVED : queries >= ready_after ?
        ACL_STREAM_STATUS_COMPLETE : ACL_STREAM_STATUS_NOT_READY;
    return query_error;
}
static int aclrtSynchronizeStream(void*) { ++syncs; return sync_error; }
static int ple_host_fetch(void*, int, uint64_t*) { ++fetches; return fetch_error; }
#include "producer_wait.inc"

static void reset() {
    queries = syncs = fetches = 0;
    ready_after = 2;
    query_error = sync_error = launch_error = fetch_error = 0;
    invalid_state = false;
    clock_ns = 0;
}
int main() {
    uint64_t timings[8]{};
    reset(); Owner blocking;
    assert(produce_fetch(&blocking, timings, 0) == 0);
    assert(queries == 0 && syncs == 1 && fetches == 1 && timings[7] == 0);
    reset(); Owner polling;
    assert(produce_fetch(&polling, timings, 20) == 0);
    assert(queries == 2 && syncs == 0 && fetches == 1 && timings[6] == 2 && timings[7] == 1);
    reset(); Owner expires; ready_after = 1000;
    assert(produce_fetch(&expires, timings, 20) == 0);
    assert(queries > 0 && queries <= 20 && syncs == 1 && fetches == 1 && timings[7] == 0);
    reset(); Owner failed_query; query_error = 7;
    assert(produce_fetch(&failed_query, timings, 20) == 7);
    assert(failed_query.poisoned && syncs == 0 && fetches == 0);
    assert(produce_fetch(&failed_query, timings, 20) == -EBUSY);
    reset(); Owner failed_launch; launch_error = 8;
    assert(produce_fetch(&failed_launch, timings, 20) == 8);
    assert(failed_launch.poisoned && queries == 0 && syncs == 0 && fetches == 0);
    reset(); Owner failed_sync; sync_error = 9;
    assert(produce_fetch(&failed_sync, timings, 0) == 9);
    assert(failed_sync.poisoned && fetches == 0);
    reset(); Owner invalid; invalid_state = true;
    assert(produce_fetch(&invalid, timings, 20) == -EINVAL);
    assert(invalid.poisoned && fetches == 0);
    reset(); Owner rejected; fetch_error = -ERANGE;
    assert(produce_fetch(&rejected, timings, 20) == -ERANGE);
    assert(!rejected.poisoned && fetches == 1);
    reset(); Owner other; other.device = 1;
    assert(produce_fetch(&other, timings, 20) == -EXDEV);
    assert(queries == 0 && fetches == 0);
    assert(produce_fetch(nullptr, timings, 20) == -EINVAL);
    reset(); Owner bounded;
    assert(ple_host_produce_fetch_poll(&bounded, timings, 51) == -EINVAL);
    assert(queries == 0 && syncs == 0 && fetches == 0);
    assert(ple_host_produce_fetch_poll(&bounded, timings, 40) == 0);
    assert(queries == 2 && syncs == 0 && fetches == 1);
    reset(); Owner rebound;
    int tokens, starts, history;
    assert(ple_host_produce_fetch_inputs(&rebound, &tokens, &starts, &history, timings, 40) == 0);
    assert(rebound.addresses[0] == &tokens && rebound.addresses[1] == &starts && rebound.addresses[2] == &history);
    assert(queries == 2 && fetches == 1);
    reset();
    assert(ple_host_produce_fetch_inputs(&rebound, nullptr, &starts, &history, timings, 40) == -EINVAL);
    assert(ple_host_produce_fetch_inputs(&rebound, &tokens, &starts, &history, timings, 51) == -EINVAL);
    rebound.poisoned = true;
    assert(ple_host_produce_fetch_inputs(&rebound, &tokens, &starts, &history, timings, 40) == -EBUSY);
    assert(queries == 0 && fetches == 0);
    return 0;
}
