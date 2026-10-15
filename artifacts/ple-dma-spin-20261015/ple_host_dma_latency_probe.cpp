// SPDX-License-Identifier: Apache-2.0
// Host submission microbenchmark. This is not complete PLE lookup latency.
#include <acl/acl.h>
#include <ascend_hal.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr size_t kAllocationBytes = 16384;
constexpr size_t kRowBytes = 320;
constexpr size_t kTokenBytes = 16 * kRowBytes;
constexpr size_t kPageCrossingOffset = 3840;
constexpr size_t kWarmup = 32;
constexpr size_t kIterations = 1000;

static void require(int status, const char* operation) {
    if (status != 0) {
        std::fprintf(stderr, "%s: %d\n", operation, status);
        std::exit(2);
    }
}

static void measure(void* source, void* destination, int rank, size_t bytes, size_t offset) {
    DMA_ADDR descriptor{};
    descriptor.offsetAddr.devid = rank;
    auto* expected = static_cast<unsigned char*>(source) + offset;
    require(drvMemConvertAddr(reinterpret_cast<DVdeviceptr>(expected),
                              reinterpret_cast<DVdeviceptr>(destination), bytes, &descriptor), "convert");
    require(descriptor.fixed_size == bytes ? 0 : 1, "full descriptor");
    for (size_t i = 0; i < kWarmup; ++i)
        require(halMemcpySumbit(&descriptor, MEMCPY_SUMBIT_SYNC), "warmup submit");
    std::vector<long long> samples(kIterations);
    rusage usage_before{}, usage_after{};
    require(getrusage(RUSAGE_THREAD, &usage_before), "read initial CPU usage");
    const auto loop_start = std::chrono::steady_clock::now();
    for (auto& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        const int status = halMemcpySumbit(&descriptor, MEMCPY_SUMBIT_SYNC);
        const auto end = std::chrono::steady_clock::now();
        require(status, "measured submit");
        sample = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    const auto loop_end = std::chrono::steady_clock::now();
    require(getrusage(RUSAGE_THREAD, &usage_after), "read final CPU usage");
    const auto cpu_ns = [](const rusage& usage) {
        return (static_cast<long long>(usage.ru_utime.tv_sec) + usage.ru_stime.tv_sec) * 1000000000LL +
               (static_cast<long long>(usage.ru_utime.tv_usec) + usage.ru_stime.tv_usec) * 1000LL;
    };
    const long long caller_cpu_ns = cpu_ns(usage_after) - cpu_ns(usage_before);
    const auto loop_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - loop_start).count();
    std::vector<unsigned char> actual(bytes);
    require(aclrtMemcpy(actual.data(), bytes, destination, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "readback");
    const bool exact = std::memcmp(actual.data(), expected, bytes) == 0;
    require(drvMemDestroyAddr(&descriptor), "destroy descriptor");
    std::sort(samples.begin(), samples.end());
    std::printf("{\"rank\":%d,\"bytes\":%zu,\"offset\":%zu,\"samples\":%zu,\"exact\":%s,"
                "\"min_ns\":%lld,\"p50_ns\":%lld,\"p95_ns\":%lld,\"p99_ns\":%lld,\"max_ns\":%lld,"
                "\"caller_cpu_ns_per_submit\":%lld,\"voluntary_switches\":%ld,\"involuntary_switches\":%ld,"
                "\"loop_wall_ns\":%lld,"
                "\"scope\":\"host synchronous submit with reused descriptor\"}\n", rank, bytes, offset,
                samples.size(), exact ? "true" : "false", samples.front(), samples[kIterations / 2 - 1],
                samples[kIterations * 95 / 100 - 1], samples[kIterations * 99 / 100 - 1], samples.back(),
                caller_cpu_ns / static_cast<long long>(kIterations), usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                usage_after.ru_nivcsw - usage_before.ru_nivcsw, static_cast<long long>(loop_wall_ns));
    require(exact ? 0 : 1, "exact payload");
}

int main(int argc, char** argv) {
    if (argc != 2 || std::strlen(argv[1]) != 1 || argv[1][0] < '0' || argv[1][0] > '3') return 1;
    const int rank = argv[1][0] - '0';
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(aclInit(nullptr), "initialize");
    require(aclrtSetDevice(rank), "select rank");
    void* source = mmap(nullptr, kAllocationBytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (source == MAP_FAILED) return 2;
    for (size_t i = 0; i < kAllocationBytes; ++i)
        static_cast<unsigned char*>(source)[i] = (i * 17 + 13) % 251;
    void* destination = nullptr;
    require(aclrtMalloc(&destination, kAllocationBytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate");
    for (size_t bytes : {kRowBytes, kTokenBytes})
        for (size_t offset : {size_t{0}, kPageCrossingOffset})
            measure(source, destination, rank, bytes, offset);
    require(aclrtFree(destination), "free destination");
    require(munmap(source, kAllocationBytes), "unmap source");
    require(aclFinalize(), "finalize");
    return 0;
}
