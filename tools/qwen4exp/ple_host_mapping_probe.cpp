// SPDX-License-Identifier: Apache-2.0
// Registration-only diagnostic: never dereference an unqualified device pointer.
// No driver changes, device reset, graph capture, or custom DMA submission.
#include <acl/acl.h>
#include <ascend_hal.h>
#include <sys/mman.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr size_t kPageBytes = 4096;
constexpr size_t kProbeBytes = 2 * kPageBytes;

static void require(int rc, const char* operation) {
    if (rc) {
        std::fprintf(stderr, "%s failed: %d\n", operation, rc);
        std::exit(2);
    }
}

static void acl_probe(const char* source, void* host, bool v2, unsigned flags) {
    void* device = nullptr;
    int rc = v2 ? aclrtHostRegisterV2(host, kProbeBytes, flags)
                : aclrtHostRegister(host, kProbeBytes, ACL_HOST_REGISTER_MAPPED, &device);
    std::printf("{\"kind\":\"register\",\"source\":\"%s\",\"api\":\"%s\","
                "\"flags\":%u,\"status\":%d,\"device_pointer\":%s}\n",
                source, v2 ? "aclrtHostRegisterV2" : "aclrtHostRegister", flags, rc,
                device ? "true" : "false");
    if (rc) return;
    if (v2) {
        rc = aclrtHostGetDevicePointer(host, &device, 0);
        std::printf("{\"kind\":\"pointer\",\"source\":\"%s\",\"flags\":%u,"
                    "\"status\":%d,\"device_pointer\":%s}\n",
                    source, flags, rc, device ? "true" : "false");
    }
    // No work was launched, so a successful mapping can be unregistered now.
    require(aclrtHostUnregister(host), "unregister ACL mapping");
}

static void hal_probe(const char* source, void* host, unsigned flags, unsigned rank) {
    void* device = nullptr;
    int rc = halHostRegister(host, kProbeBytes, flags, rank, &device);
    std::printf("{\"kind\":\"register\",\"source\":\"%s\",\"api\":\"halHostRegister\","
                "\"flags\":%u,\"status\":%d,\"device_pointer\":%s}\n",
                source, flags, rc, device ? "true" : "false");
    if (!rc) require(halHostUnregister(host, rank), "unregister HAL mapping");
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    char* end = nullptr;
    long rank = std::strtol(argv[1], &end, 10);
    if (!*argv[1] || *end || rank < 0 || rank > 3) return 1;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(aclInit(nullptr), "initialize ACL");
    require(aclrtSetDevice(static_cast<int>(rank)), "select device");
    std::printf("{\"kind\":\"identity\",\"rank\":%ld,\"soc\":\"%s\","
                "\"bytes\":%zu,\"device_read_tested\":false}\n", rank, aclrtGetSocName(), kProbeBytes);
    const aclrtHacType engines[] = {ACL_RT_HAC_TYPE_AICPU, ACL_RT_HAC_TYPE_AIC,
        ACL_RT_HAC_TYPE_AIV, ACL_RT_HAC_TYPE_PCIEDMA, ACL_RT_HAC_TYPE_SDMA};
    for (auto engine : engines) {
        auto capability = static_cast<aclrtHostMemMapCapability>(-1);
        int rc = aclrtHostMemMapCapabilities(rank, engine, &capability);
        std::printf("{\"kind\":\"capability\",\"engine\":%d,\"status\":%d,\"value\":%d}\n",
                    static_cast<int>(engine), rc, static_cast<int>(capability));
    }
    void* ordinary = mmap(nullptr, kProbeBytes, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ordinary == MAP_FAILED) return 2;
    std::memset(ordinary, 0x5a, kProbeBytes);
    acl_probe("shared_mmap", ordinary, false, ACL_HOST_REGISTER_MAPPED);
    acl_probe("shared_mmap", ordinary, true, ACL_HOST_REG_MAPPED);
    acl_probe("shared_mmap", ordinary, true, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
    hal_probe("shared_mmap", ordinary, HOST_MEM_MAP_DEV, rank);
    hal_probe("shared_mmap", ordinary, HOST_MEM_MAP_DEV_PCIE_TH, rank);
    require(munmap(ordinary, kProbeBytes), "unmap ordinary allocation");

    void* allocation = nullptr;
    require(aclrtMallocHost(&allocation, kProbeBytes + kPageBytes), "allocate pinned host memory");
    uintptr_t aligned = (reinterpret_cast<uintptr_t>(allocation) + kPageBytes - 1) & ~(kPageBytes - 1);
    void* pinned = reinterpret_cast<void*>(aligned);
    std::memset(pinned, 0xa5, kProbeBytes);
    acl_probe("acl_pinned", pinned, false, ACL_HOST_REGISTER_MAPPED);
    // ACL_HOST_REG_PINNED must NOT be used for an already-pinned allocation.
    acl_probe("acl_pinned", pinned, true, ACL_HOST_REG_MAPPED);
    hal_probe("acl_pinned", pinned, HOST_SVM_MAP_DEV, rank);
    hal_probe("acl_pinned", pinned, HOST_MEM_MAP_DEV_PCIE_TH, rank);

    void* device = nullptr;
    require(aclrtMalloc(&device, kProbeBytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate control buffer");
    require(aclrtMemcpy(device, kProbeBytes, pinned, kProbeBytes, ACL_MEMCPY_HOST_TO_DEVICE), "control H2D");
    unsigned char actual[kProbeBytes];
    require(aclrtMemcpy(actual, sizeof(actual), device, kProbeBytes, ACL_MEMCPY_DEVICE_TO_HOST), "control D2H");
    bool exact = std::memcmp(actual, pinned, kProbeBytes) == 0;
    std::printf("{\"kind\":\"host_submitted_copy_control\",\"exact\":%s}\n", exact ? "true" : "false");
    require(aclrtFree(device), "free control buffer");
    require(aclrtFreeHost(allocation), "free pinned allocation");

    // Torch-NPU's pinned_mem_register path requests the special VA allocation
    // first, then registers it with MAPPED only. Exercise that exact sequence.
    aclrtMallocAttribute attribute{};
    attribute.attr = ACL_RT_MEM_ATTR_VA_FLAG;
    attribute.value.vaFlag = 1;
    aclrtMallocConfig config{};
    config.attrs = &attribute;
    config.numAttrs = 1;
    void* configured = nullptr;
    int rc = aclrtMallocHostWithCfg(&configured, kProbeBytes, &config);
    std::printf("{\"kind\":\"allocate\",\"source\":\"acl_pinned_va1\",\"status\":%d}\n", rc);
    if (!rc) {
        acl_probe("acl_pinned_va1", configured, true, ACL_HOST_REG_MAPPED);
        require(aclrtFreeHost(configured), "free configured host allocation");
    }
    require(aclFinalize(), "finalize ACL");
    return exact ? 0 : 3;
}
