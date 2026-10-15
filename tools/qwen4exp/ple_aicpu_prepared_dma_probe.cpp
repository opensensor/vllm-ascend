// SPDX-License-Identifier: Apache-2.0
// Diagnostic only: host-prepared descriptors passed to the stock AI CPU HAL.
// The cross-context descriptor ABI is not established. Only call the exact
// known unsupported stub; a changed implementation requires a fresh ABI audit.
#include <aicpu/aicpu_task_struct.h>
#include <ascend_hal.h>

#include <cstdint>
#include <cstring>

constexpr uint64_t kCookie = 0x504c455052455041ULL;
constexpr size_t kRowBytes = 320;
constexpr size_t kAllocationBytes = 8192;
constexpr size_t kDescriptorCount = 2;
constexpr size_t kSourceOffsets[kDescriptorCount] = {0, 3840};
constexpr uint32_t kNotCalled = 0xffffffffU;

struct Payload {
    DMA_ADDR descriptors[kDescriptorCount];
    uint64_t cookie;
    uint64_t seed;
    uint64_t selected;
    uint32_t submit_status;
    uint32_t matched_stub;
};

#pragma pack(push, 1)
struct Arguments {
    aicpu::AicpuParamHead header;
    uint64_t payload;
};
#pragma pack(pop)

#ifdef PLE_AICPU_DEVICE
#include <dlfcn.h>

extern "C" uint32_t PlePreparedDma(void* arguments) {
    if (arguments == nullptr) return 1;
    Arguments args{};
    std::memcpy(&args, arguments, sizeof(args));
    if (args.header.length != sizeof(args) || args.header.ioAddrNum != 1 ||
        args.header.extInfoLength != 0 || args.payload == 0) return 2;
    auto* payload = reinterpret_cast<Payload*>(args.payload);
    if (payload->cookie != kCookie) return 3;
    payload->selected = payload->seed % kDescriptorCount;
    payload->submit_status = kNotCalled;
    payload->matched_stub = 0;
    auto* entry = dlsym(RTLD_DEFAULT, "halMemcpySumbit");
    auto* inner = dlsym(RTLD_DEFAULT, "halMemcpySumbitInner");
    // Exact AArch64 instructions from the installed 26.0.rc1 device HAL.
    // The wrapper BL targets the adjacent inner function returning 0xfffe.
    constexpr uint32_t kWrapper[] = {
        0xa9be7bfd, 0x910003fd, 0xf9000fa0, 0xb90017a1, 0xb94017a1,
        0xf9400fa0, 0x97fffff4, 0xa8c27bfd, 0xd65f03c0};
    constexpr uint32_t kInner[] = {
        0xd10043ff, 0xf90007e0, 0xb90007e1, 0x529fffc0, 0x910043ff, 0xd65f03c0};
    if (entry == nullptr || inner == nullptr ||
        reinterpret_cast<uintptr_t>(entry) != reinterpret_cast<uintptr_t>(inner) + sizeof(kInner) ||
        std::memcmp(entry, kWrapper, sizeof(kWrapper)) != 0 ||
        std::memcmp(inner, kInner, sizeof(kInner)) != 0) return 0;
    payload->matched_stub = 1;
    auto submit = reinterpret_cast<decltype(&halMemcpySumbit)>(entry);
    payload->submit_status = submit(&payload->descriptors[payload->selected], MEMCPY_SUMBIT_SYNC);
    return 0;
}
#else
#include <acl/acl.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

static void require(int status, const char* operation) {
    if (status != 0) {
        std::fprintf(stderr, "%s failed: %d\n", operation, status);
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc != 3 || std::strlen(argv[2]) != 1 || argv[2][0] < '0' || argv[2][0] > '3') return 1;
    const int rank = argv[2][0] - '0';
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input.good()) return 1;
    std::vector<char> binary{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    require(aclInit(nullptr), "initialize");
    require(aclrtSetDevice(rank), "select rank");
    aclrtStream stream = nullptr;
    require(aclrtCreateStream(&stream), "create stream");
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 2;
    aclrtBinaryLoadOptions options{&option, 1};
    aclrtBinHandle handle = nullptr;
    require(aclrtBinaryLoadFromData(binary.data(), binary.size(), &options, &handle), "load AI CPU binary");
    aclrtFuncHandle function = nullptr;
    require(aclrtRegisterCpuFunc(handle, "PlePreparedDma", "PlePreparedDma", &function), "register function");
    void *source = nullptr, *destination = nullptr, *device_payload = nullptr;
    require(aclrtMallocHost(&source, kAllocationBytes), "allocate pinned source");
    require(aclrtMalloc(&destination, kAllocationBytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate destination");
    require(aclrtMalloc(&device_payload, sizeof(Payload), ACL_MEM_MALLOC_NORMAL_ONLY), "allocate payload");
    auto* source_bytes = static_cast<unsigned char*>(source);
    for (size_t i = 0; i < kAllocationBytes; ++i) source_bytes[i] = (i * 17 + 13) % 251;
    Payload prepared{};
    prepared.cookie = kCookie;
    for (size_t i = 0; i < kDescriptorCount; ++i) {
        auto& descriptor = prepared.descriptors[i];
        descriptor.offsetAddr.devid = rank;
        require(drvMemConvertAddr(reinterpret_cast<DVdeviceptr>(source_bytes + kSourceOffsets[i]),
                                  reinterpret_cast<DVdeviceptr>(destination), kRowBytes, &descriptor),
                "prepare host descriptor");
        require(descriptor.fixed_size == kRowBytes ? 0 : 1, "complete descriptor");
    }
    bool all_controls_exact = true;
    bool all_rejected = true;
    for (size_t i = 0; i < kDescriptorCount; ++i) {
        require(halMemcpySumbit(&prepared.descriptors[i], MEMCPY_SUMBIT_SYNC), "host control submit");
        unsigned char actual[kRowBytes]{};
        require(aclrtMemcpy(actual, sizeof(actual), destination, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST), "read control");
        const bool host_exact = std::memcmp(actual, source_bytes + kSourceOffsets[i], kRowBytes) == 0;
        all_controls_exact &= host_exact;
        require(host_exact ? 0 : 1, "host control exact");
        require(aclrtMemset(destination, kAllocationBytes, 0, kAllocationBytes), "clear destination");
        prepared.seed = kDescriptorCount + i;
        prepared.submit_status = kNotCalled;
        require(aclrtMemcpy(device_payload, sizeof(Payload), &prepared, sizeof(Payload), ACL_MEMCPY_HOST_TO_DEVICE), "copy descriptors");
        Arguments args{};
        args.header.length = sizeof(args);
        args.header.ioAddrNum = 1;
        args.payload = reinterpret_cast<uint64_t>(device_payload);
        require(aclrtLaunchKernelWithHostArgs(function, 1, stream, nullptr, &args, sizeof(args), nullptr, 0), "launch submission probe");
        require(aclrtSynchronizeStreamWithTimeout(stream, 5000), "complete AI CPU task");
        Payload result{};
        require(aclrtMemcpy(&result, sizeof(result), device_payload, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST), "read receipt");
        require(result.cookie == kCookie && result.selected == i ? 0 : 1, "validate selection receipt");
        require(aclrtMemcpy(actual, sizeof(actual), destination, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST), "read destination");
        const unsigned char zeros[kRowBytes]{};
        const bool unchanged = std::memcmp(actual, zeros, kRowBytes) == 0;
        const bool rejected = result.matched_stub && result.submit_status == 65534 && unchanged;
        all_rejected &= rejected;
        std::printf("{\"rank\":%d,\"selected\":%zu,\"row_bytes\":%zu,\"host_control_exact\":%s,"
                    "\"device_stub_matched\":%s,\"device_submit_status\":%u,\"destination_unchanged\":%s,"
                    "\"device_transfer_proven\":false}\n", rank, i, kRowBytes, host_exact ? "true" : "false",
                    result.matched_stub ? "true" : "false", result.submit_status, unchanged ? "true" : "false");
    }
    for (auto& descriptor : prepared.descriptors) require(drvMemDestroyAddr(&descriptor), "destroy host descriptor");
    require(aclrtFree(device_payload), "free payload");
    require(aclrtFree(destination), "free destination");
    require(aclrtFreeHost(source), "free source");
    require(aclrtBinaryUnLoad(handle), "unload binary");
    require(aclrtDestroyStream(stream), "destroy stream");
    require(aclFinalize(), "finalize");
    return all_controls_exact && all_rejected ? 0 : 3;
}
#endif
