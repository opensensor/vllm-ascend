// SPDX-License-Identifier: Apache-2.0
// Bounded same-device probe for device-resident DMA descriptors. No peer reads.
#include <acl/acl.h>
#include <runtime/mem.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr size_t kBytes = 320;
constexpr size_t kAllocationBytes = 4096;
constexpr int32_t kWaitMilliseconds = 2000;

static void require(int status, const char* operation) {
    if (status) {
        std::fprintf(stderr, "%s failed: %d\n", operation, status);
        std::exit(2);  // Never recycle an allocation after uncertain submission.
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const int rank = std::atoi(argv[1]);
    if (rank < 0 || rank > 3) return 1;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(aclInit(nullptr), "initialize");
    require(aclrtSetDevice(rank), "select private process device");
    aclrtStream stream = nullptr;
    require(aclrtCreateStream(&stream), "create private stream");
    void *source = nullptr, *destination = nullptr, *descriptor = nullptr;
    require(aclrtMalloc(&source, kAllocationBytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate source");
    require(aclrtMalloc(&destination, kAllocationBytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate destination");
    require(aclrtMalloc(&descriptor, kAllocationBytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate descriptor");
    unsigned char expected[kBytes], actual[kBytes];
    for (size_t i = 0; i < kBytes; ++i) expected[i] = (i * 19 + 13) % 251;
    require(aclrtMemcpy(source, kAllocationBytes, expected, kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "initialize source");
    rtMemcpyAddrInfo info{};
    info.len = kBytes;
    info.src = reinterpret_cast<uint64_t>(source);
    info.dst = reinterpret_cast<uint64_t>(destination);
    static_assert(sizeof(info) == 32);
    require(aclrtMemcpy(descriptor, kAllocationBytes, &info, sizeof(info), ACL_MEMCPY_HOST_TO_DEVICE),
            "initialize descriptor");
    for (int version = 1; version <= 2; ++version) {
        require(aclrtMemset(destination, kAllocationBytes, 0xff, kAllocationBytes), "poison destination");
        const int status = version == 1
            ? rtMemcpyAsyncPtr(descriptor, kAllocationBytes, kBytes, RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream, 0)
            : rtMemcpyAsyncPtrV2(descriptor, kAllocationBytes, kBytes, RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, stream, nullptr);
        std::printf("{\"device\":%d,\"version\":%d,\"operation\":\"submit\",\"status\":%d}\n", rank, version, status);
        bool exact = false;
        // Synchronize even a rejected call before releasing any owned resources.
        require(aclrtSynchronizeStreamWithTimeout(stream, kWaitMilliseconds), "drain private stream");
        if (status == 0) {
            require(aclrtMemcpy(actual, kBytes, destination, kBytes, ACL_MEMCPY_DEVICE_TO_HOST), "read result");
            exact = std::memcmp(expected, actual, kBytes) == 0;
        }
        std::printf("{\"device\":%d,\"version\":%d,\"runtime_kind\":5,\"status\":%d,\"exact\":%s}\n",
                    rank, version, status, exact ? "true" : "false");
        if (status == 0 && !exact) return 3;
    }
    require(aclrtFree(descriptor), "free descriptor");
    require(aclrtFree(destination), "free destination");
    require(aclrtFree(source), "free source");
    require(aclrtDestroyStream(stream), "destroy stream");
    require(aclFinalize(), "finalize private runtime");
    return 0;
}
