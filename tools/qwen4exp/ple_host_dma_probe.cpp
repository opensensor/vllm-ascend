// SPDX-License-Identifier: Apache-2.0
// Private-buffer capability probe. All DMA submission is deliberately on CPU.
#include <acl/acl.h>
#include <ascend_hal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr size_t kBufferBytes = 8192;
constexpr size_t kRowBytes = 320;
constexpr size_t kRowOffset = 3840;  // Exercise a row crossing a host page.

static void require(int status, const char* operation) {
    if (status != 0) {
        std::fprintf(stderr, "%s failed: %d\n", operation, status);
        std::exit(2);  // Retain uncertain resources until process teardown.
    }
}

static void probe(const char* name, void* host, void* device, int device_id, size_t offset = kRowOffset) {
    DMA_ADDR descriptor{};
    descriptor.offsetAddr.devid = device_id;
    const int converted = drvMemConvertAddr(
        reinterpret_cast<DVdeviceptr>(static_cast<char*>(host) + offset),
        reinterpret_cast<DVdeviceptr>(device), kRowBytes, &descriptor);
    int submitted = -1;
    bool exact = false;
    if (converted == 0) {
        if (descriptor.fixed_size != kRowBytes) {
            require(drvMemDestroyAddr(&descriptor), "destroy partial descriptor");
            std::printf("{\"source\":\"%s\",\"convert\":0,\"partial\":true}\n", name);
            return;
        }
        submitted = halMemcpySumbit(&descriptor, MEMCPY_SUMBIT_SYNC);
        // Do not release an uncertain DMA lease after an error.
        require(submitted, "submit host DMA");
        unsigned char actual[kRowBytes];
        require(aclrtMemcpy(actual, sizeof(actual), device, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST),
                "read DMA result");
        exact = std::memcmp(actual, static_cast<char*>(host) + offset, kRowBytes) == 0;
        require(drvMemDestroyAddr(&descriptor), "destroy descriptor");
    }
    std::printf("{\"source\":\"%s\",\"convert\":%d,\"submit\":%d,\"exact\":%s,"
                "\"host_submitted\":true,\"device_initiated\":false}\n",
                name, converted, submitted, exact ? "true" : "false");
    if (converted == 0 && !exact) std::exit(3);
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 1;
    const int device_id = std::atoi(argv[1]);
    if (device_id < 0 || device_id > 3) return 1;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(aclInit(nullptr), "initialize ACL");
    require(aclrtSetDevice(device_id), "set private process device");
    size_t descriptor_bytes = 0;
    const int capability = aclrtGetMemcpyDescSize(ACL_MEMCPY_INNER_DEVICE_TO_DEVICE, &descriptor_bytes);
    std::printf("{\"device\":%d,\"indirect_kind\":6,\"descriptor_status\":%d,\"descriptor_bytes\":%zu}\n",
                device_id, capability, descriptor_bytes);
    void* device = nullptr;
    require(aclrtMalloc(&device, kBufferBytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate private device buffer");
    if (argc == 3) {
        const int fd = std::atoi(argv[2]);
        struct stat info{};
        require(fstat(fd, &info), "stat shared table");
        if (info.st_size < static_cast<off_t>(kBufferBytes)) return 2;
        void* shared = mmap(nullptr, info.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (shared == MAP_FAILED) return 2;
        const size_t rows = static_cast<size_t>(info.st_size) / kRowBytes;
        const size_t selected[] = {12, 0, rows / 2, rows - 1, 12};
        for (size_t row : selected) probe("external_shared_memfd", shared, device, device_id, row * kRowBytes);
        require(munmap(shared, info.st_size), "unmap imported shared table");
        require(aclrtFree(device), "free private device buffer");
        require(aclFinalize(), "finalize private runtime");
        return 0;
    }
    void* host = mmap(nullptr, kBufferBytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (host == MAP_FAILED) return 2;
    for (size_t i = 0; i < kBufferBytes; ++i) static_cast<unsigned char*>(host)[i] = (i * 17 + 13) % 251;
    probe("shared_mmap_rw", host, device, device_id);
    const int shared_fd = memfd_create("ple-dma-private", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (shared_fd < 0 || write(shared_fd, host, kBufferBytes) != static_cast<ssize_t>(kBufferBytes)) return 2;
    require(fcntl(shared_fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL),
            "seal private shared table");
    void* sealed = mmap(nullptr, kBufferBytes, PROT_READ, MAP_SHARED, shared_fd, 0);
    if (sealed == MAP_FAILED) return 2;
    probe("sealed_memfd_ro", sealed, device, device_id);
    require(munmap(sealed, kBufferBytes), "unmap shared table");
    require(close(shared_fd), "close shared table");
    require(mprotect(host, kBufferBytes, PROT_READ), "protect private host source");
    probe("shared_mmap_ro", host, device, device_id);
    require(munmap(host, kBufferBytes), "unmap host source");
    void* pinned = nullptr;
    require(aclrtMallocHost(&pinned, kBufferBytes), "allocate native host buffer");
    for (size_t i = 0; i < kBufferBytes; ++i) static_cast<unsigned char*>(pinned)[i] = (i * 19 + 29) % 253;
    probe("acl_pinned", pinned, device, device_id);
    require(aclrtFreeHost(pinned), "free native host buffer");
    require(aclrtFree(device), "free private device buffer");
    require(aclFinalize(), "finalize private runtime");
    return 0;
}
