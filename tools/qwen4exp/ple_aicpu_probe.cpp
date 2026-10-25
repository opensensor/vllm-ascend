// SPDX-License-Identifier: Apache-2.0
// Inspect on-device DMA entry points without submitting DMA or touching MMIO.
#include <aicpu/aicpu_task_struct.h>
#include <ascend_hal.h>
#include <cstdint>
#include <cstring>

constexpr uint64_t kCookie = 0x504c454149435055ULL;
constexpr size_t kReceiptWords = 8192;
constexpr size_t kInventoryStart = 64;
#pragma pack(push, 1)
struct ProbeArguments {
    aicpu::AicpuParamHead header;
    uint64_t receipt;
    uint64_t destination;
    uint64_t cookie;
    uint64_t host_source;
    uint64_t mode;
};
#pragma pack(pop)

#ifdef PLE_AICPU_DEVICE
#include <cerrno>
#include <cstdio>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <initializer_list>
#include <link.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include "ple_probe_text.h"
#include "ple_device_access.h"

struct InventoryBuffer {
    char* data;
    size_t capacity;
    size_t used;
};

static int list_library(dl_phdr_info* info, size_t, void* context) {
    auto* buffer = static_cast<InventoryBuffer*>(context);
    int count = std::snprintf(buffer->data + buffer->used, buffer->capacity - buffer->used,
                              "LIBRARY %s\n", info->dlpi_name);
    if (count < 0 || static_cast<size_t>(count) >= buffer->capacity - buffer->used) return 1;
    buffer->used += static_cast<size_t>(count);
    return 0;
}

extern "C" uint32_t PleDmaCapability(void* args) {
    if (args == nullptr) return 1;
    ProbeArguments values{};
    std::memcpy(&values, args, sizeof(values));
    if (values.header.length != sizeof(values) || values.header.ioAddrNum != 2 ||
        values.header.extInfoLength != 0 || values.cookie != kCookie || values.receipt == 0) return 2;
    auto* receipt = reinterpret_cast<uint64_t*>(values.receipt);
    receipt[0] = kCookie;
    receipt[1] = sizeof(void*);
    receipt[2] = static_cast<uint64_t>(getpid());
    receipt[3] = dlsym(RTLD_DEFAULT, "drvMemConvertAddr") != nullptr;
    receipt[4] = dlsym(RTLD_DEFAULT, "halMemcpySumbit") != nullptr;
    receipt[5] = dlsym(RTLD_DEFAULT, "halMemcpyWait") != nullptr;
    receipt[6] = dlsym(RTLD_DEFAULT, "halHostRegister") != nullptr;
    receipt[7] = 1;
    if (values.mode == 1) {
        auto convert = reinterpret_cast<decltype(&drvMemConvertAddr)>(dlsym(RTLD_DEFAULT, "drvMemConvertAddr"));
        auto destroy = reinterpret_cast<decltype(&drvMemDestroyAddr)>(dlsym(RTLD_DEFAULT, "drvMemDestroyAddr"));
        receipt[8] = 0xffffffffU;
        receipt[9] = 0xffffffffU;
        if (convert != nullptr && destroy != nullptr && values.host_source && values.destination) {
            DMA_ADDR descriptor{};
            descriptor.offsetAddr.devid = 0;
            receipt[8] = convert(values.host_source, values.destination, 320, &descriptor);
            Dl_info origin{};
            receipt[13] = dladdr(reinterpret_cast<void*>(convert), &origin);
            std::memcpy(&receipt[14], reinterpret_cast<void*>(convert), sizeof(uint64_t));
            std::memcpy(receipt + 32, reinterpret_cast<void*>(convert), 32);
            receipt[12] = access("/dev/devmm_svm", F_OK) == 0;
            if (receipt[13] && origin.dli_fname != nullptr) {
                auto* name = reinterpret_cast<char*>(receipt + 16);
                std::strncpy(name, origin.dli_fname, 16 * sizeof(uint64_t) - 1);
                name[16 * sizeof(uint64_t) - 1] = 0;
            }
            if (receipt[8] == 0) {
                receipt[10] = descriptor.fixed_size;
                receipt[11] = descriptor.phyAddr.flag;
                receipt[9] = destroy(&descriptor);
            }
        }
    }
    if (values.mode == 2) {
        // Read-only inventory of this AI CPU process; never open device nodes.
        char* output = reinterpret_cast<char*>(receipt + kInventoryStart);
        const size_t capacity = (kReceiptWords - kInventoryStart) * sizeof(uint64_t);
        size_t used = 0;
        const char* paths[] = {"/proc/modules", "/proc/version", "/proc/self/status"};
        for (const char* path : paths) {
            int fd = open(path, O_RDONLY | O_CLOEXEC);
            int open_error = fd < 0 ? errno : 0;
            int n = std::snprintf(output + used, capacity - used,
                                  "\nFILE %s open_errno=%d\n", path, open_error);
            if (n < 0 || static_cast<size_t>(n) >= capacity - used) {
                if (fd >= 0) close(fd);
                break;
            }
            used += static_cast<size_t>(n);
            if (fd >= 0) {
                // One bounded read per file; inventory is diagnostic, not I/O timing.
                size_t limit = capacity - used - 1;
                if (limit > 3072) limit = 3072;
                ssize_t count = read(fd, output + used, limit);
                if (count > 0) used += static_cast<size_t>(count);
                close(fd);
            }
        }
        DIR* directory = opendir("/dev");
        int open_error = directory == nullptr ? errno : 0;
        int n = std::snprintf(output + used, capacity - used,
                              "\nDIRECTORY /dev open_errno=%d\n", open_error);
        if (n > 0 && static_cast<size_t>(n) < capacity - used) used += static_cast<size_t>(n);
        if (directory != nullptr) {
            for (size_t i = 0; i < 128; ++i) {
                dirent* entry = readdir(directory);
                if (entry == nullptr) break;
                n = std::snprintf(output + used, capacity - used, "%s\n", entry->d_name);
                if (n < 0 || static_cast<size_t>(n) >= capacity - used) break;
                used += static_cast<size_t>(n);
            }
            closedir(directory);
        }
        InventoryBuffer buffer{output, capacity, used};
        dl_iterate_phdr(list_library, &buffer);
        receipt[36] = buffer.used;
    }
    if (values.mode == 3) {
        PleProbeText output(reinterpret_cast<char*>(receipt + kInventoryStart),
                            (kReceiptWords - kInventoryStart) * sizeof(uint64_t));
        const char* symbols[] = {
            "drvMemConvertAddr", "drvMemDestroyAddr", "halMemcpySumbit", "halMemcpyWait",
            "halMemcpySumbitInner", "halMemcpyWaitInner", "drvMemcpyInner", "halMemAllocInner",
            "drvMemcpy", "halMemcpy", "drvMemAddressTranslate", "halHostRegister",
            "halMemAlloc", "halMemMap", "halMemCreate", "halResAddrMap", "halResMap",
            "halAsyncDmaCreate", "halAsyncDmaWqeCreate", "halAsyncDmaJettyCreate",
            "halAsyncDmaJettyWqeFill", "drvHdcDmaMap", "drvHdcDmaUnMap",
            "halHdcFastSend", "halHdcFastRecv", "hdcPcieOpen",
            "halSdmaCopy", "halSdmaCopyInner", "halQueryDevpid", "halGetDeviceInfo"
        };
        for (const char* name : symbols) {
            void* address = dlsym(RTLD_DEFAULT, name);
            Dl_info info{};
            bool located = address && dladdr(address, &info);
            output.append("SYMBOL %s present=%d library=%s offset=%llu bytes=", name,
                          address != nullptr, located ? info.dli_fname : "unknown",
                          located ? static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address) -
                                                                    reinterpret_cast<uintptr_t>(info.dli_fbase)) : 0);
            if (address) {
                const auto* code = static_cast<const unsigned char*>(address);
                size_t code_bytes = std::strcmp(name, "drvMemcpyInner") == 0 ? 768 : 128;
                for (size_t i = 0; i < code_bytes; ++i) output.append("%02x", code[i]);
            }
            output.append("\n");
        }
        // Open only known driver nodes read-only. Never issue an ioctl or mmap.
        const char* nodes[] = {"/dev/devmm_svm", "/dev/davinci_manager", "/dev/davinci0",
                               "/dev/svm0", "/dev/ascend_device_dma", "/dev/mem",
                               "/dev/hisi_hdc", "/dev/sdma"};
        for (const char* path : nodes) {
            errno = 0;
            int fd = open(path, O_RDONLY | O_CLOEXEC);
            int error = fd < 0 ? errno : 0;
            output.append("NODE %s open_errno=%d\n", path, error);
            if (fd >= 0) close(fd);
        }
        const char* paths[] = {"/proc/modules", "/proc/self/maps"};
        for (const char* path : paths) {
            errno = 0;
            int fd = open(path, O_RDONLY | O_CLOEXEC);
            output.append("FILE %s open_errno=%d\n", path, fd < 0 ? errno : 0);
            if (fd < 0) continue;
            char chunk[1025];
            for (size_t batch = 0; batch < 20; ++batch) {
                errno = 0;
                ssize_t count = read(fd, chunk, sizeof(chunk) - 1);
                if (count <= 0) {
                    output.append("END read_errno=%d\n", count < 0 ? errno : 0);
                    break;
                }
                chunk[count] = 0;
                if (!output.append("%s", chunk)) break;
            }
            close(fd);
        }
        receipt[36] = output.size();
        receipt[37] = output.truncated();
    }
    if (values.mode == 4) {
        PleProbeText output(reinterpret_cast<char*>(receipt + kInventoryStart),
                            (kReceiptWords - kInventoryStart) * sizeof(uint64_t));
        ple_device_access_report(output);
        receipt[36] = output.size();
        receipt[37] = output.truncated();
    }
    if (values.mode == 5) {
        // Identify the kernel owner without opening a device node or issuing ioctls.
        PleProbeText output(reinterpret_cast<char*>(receipt + kInventoryStart),
                            (kReceiptWords - kInventoryStart) * sizeof(uint64_t));
        const char* nodes[] = {"/dev/davinci0", "/dev/davinci_manager", "/dev/svm0"};
        for (const char* node : nodes) {
            struct stat info{};
            errno = 0;
            const int status = stat(node, &info);
            const int stat_error = status < 0 ? errno : 0;
            output.append("NODE %s stat_errno=%d", node, stat_error);
            if (status < 0) {
                output.append("\n");
                continue;
            }
            const unsigned major_number = major(info.st_rdev);
            const unsigned minor_number = minor(info.st_rdev);
            output.append(" major=%u minor=%u uid=%lu gid=%lu mode=%04o\n", major_number,
                          minor_number, static_cast<unsigned long>(info.st_uid),
                          static_cast<unsigned long>(info.st_gid),
                          static_cast<unsigned>(info.st_mode & 07777));
            char path[128];
            for (const char* suffix : {"", "/device/driver", "/subsystem"}) {
                const int length = std::snprintf(path, sizeof(path), "/sys/dev/char/%u:%u%s",
                                                 major_number, minor_number, suffix);
                if (length < 0 || static_cast<size_t>(length) >= sizeof(path)) continue;
                char target[512];
                errno = 0;
                const ssize_t count = readlink(path, target, sizeof(target) - 1);
                const int link_error = count < 0 ? errno : 0;
                if (count >= 0) target[count] = '\0';
                output.append("LINK %s errno=%d target=%s\n", path, link_error,
                              count >= 0 ? target : "unavailable");
            }
        }
        const char* paths[] = {"/proc/devices", "/sys/devices/virtual/tsdrv-class/davinci0/uevent"};
        for (const char* path : paths) ple_access_file(output, path);
        receipt[36] = output.size();
        receipt[37] = output.truncated();
    }
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
    std::printf("{\"operation\":\"%s\",\"status\":%d}\n", operation, status);
    if (status) std::exit(2);
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 1;
    if (argc == 3 && std::strcmp(argv[2], "convert") != 0 &&
        std::strcmp(argv[2], "inventory") != 0 && std::strcmp(argv[2], "interfaces") != 0 &&
        std::strcmp(argv[2], "access") != 0 && std::strcmp(argv[2], "identity") != 0) return 1;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::ifstream file(argv[1], std::ios::binary);
    if (!file.good()) return 1;
    std::vector<char> binary{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    require(aclInit(nullptr), "initialize");
    require(aclrtSetDevice(0), "select private device context");
    aclrtStream stream = nullptr;
    require(aclrtCreateStream(&stream), "create private stream");
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 2;
    aclrtBinaryLoadOptions options{&option, 1};
    aclrtBinHandle handle = nullptr;
    require(aclrtBinaryLoadFromData(binary.data(), binary.size(), &options, &handle), "load private AI CPU binary");
    aclrtFuncHandle function = nullptr;
    require(aclrtRegisterCpuFunc(handle, "PleDmaCapability", "PleDmaCapability", &function), "register AI CPU probe");
    void* device = nullptr;
    constexpr size_t bytes = kReceiptWords * sizeof(uint64_t);
    require(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate private receipt");
    require(aclrtMemset(device, bytes, 0, bytes), "clear receipt");
    void *host_source = nullptr, *destination = nullptr;
    require(aclrtMallocHost(&host_source, 4096), "allocate private host source");
    require(aclrtMalloc(&destination, 4096, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate private destination");
    std::memset(host_source, 0x39, 4096);
    require(aclrtMemset(destination, 4096, 0, 4096), "clear private destination");
    ProbeArguments args{};
    args.header.length = sizeof(args);
    args.header.ioAddrNum = 2;
    args.receipt = reinterpret_cast<uint64_t>(device);
    args.destination = reinterpret_cast<uint64_t>(destination);
    args.cookie = kCookie;
    args.host_source = reinterpret_cast<uint64_t>(host_source);
    args.mode = argc == 3 ? (std::strcmp(argv[2], "convert") == 0 ? 1 :
        (std::strcmp(argv[2], "inventory") == 0 ? 2 :
        (std::strcmp(argv[2], "interfaces") == 0 ? 3 :
        (std::strcmp(argv[2], "access") == 0 ? 4 : 5)))) : 0;
    require(aclrtLaunchKernelWithHostArgs(function, 1, stream, nullptr, &args, sizeof(args), nullptr, 0), "launch AI CPU probe");
    require(aclrtSynchronizeStreamWithTimeout(stream, 5000), "complete AI CPU probe");
    uint64_t result[kReceiptWords]{};
    require(aclrtMemcpy(result, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read private receipt");
    if (result[0] != kCookie || result[7] != 1) return 3;
    std::printf("{\"passed\":true,\"device_pid\":%llu,\"convert_symbol\":%llu,\"submit_symbol\":%llu,"
                "\"wait_symbol\":%llu,\"host_register_symbol\":%llu,\"dma_submitted\":false}\n",
                static_cast<unsigned long long>(result[2]), static_cast<unsigned long long>(result[3]),
                static_cast<unsigned long long>(result[4]), static_cast<unsigned long long>(result[5]),
                static_cast<unsigned long long>(result[6]));
    if (args.mode == 1) {
        std::printf("{\"device_convert_status\":%llu,\"device_destroy_status\":%llu,\"fixed_size\":%llu,"
                    "\"chain\":%llu,\"dma_submitted\":false}\n",
                    static_cast<unsigned long long>(result[8]), static_cast<unsigned long long>(result[9]),
                    static_cast<unsigned long long>(result[10]), static_cast<unsigned long long>(result[11]));
        std::printf("{\"convert_library\":\"%s\",\"entry_first_8_bytes_hex\":\"%016llx\","
                    "\"usable_descriptor\":%s}\n", reinterpret_cast<const char*>(result + 16),
                    static_cast<unsigned long long>(result[14]), result[10] == 320 ? "true" : "false");
        std::printf("{\"device_svm_node_exists\":%s,\"convert_first_32_bytes\":\"", result[12] ? "true" : "false");
        const auto* code = reinterpret_cast<const unsigned char*>(result + 32);
        for (size_t i = 0; i < 32; ++i) std::printf("%02x", code[i]);
        std::printf("\"}\n");
        if (result[8] == 0 && result[9] != 0) return 4;
    }
    if (args.mode == 2 || args.mode == 3 || args.mode == 4 || args.mode == 5) {
        if (result[36] > (kReceiptWords - kInventoryStart) * sizeof(uint64_t)) return 5;
        if (result[37]) return 6;  // Never accept a truncated interface inventory.
        std::printf("{\"inventory_hex\":\"");
        const auto* data = reinterpret_cast<const unsigned char*>(result + kInventoryStart);
        for (size_t i = 0; i < result[36]; ++i) std::printf("%02x", data[i]);
        std::printf("\"}\n");
    }
    require(aclrtFreeHost(host_source), "free private source");
    require(aclrtFree(destination), "free private destination");
    require(aclrtFree(device), "free receipt");
    require(aclrtBinaryUnLoad(handle), "unload private binary");
    require(aclrtDestroyStream(stream), "destroy private stream");
    require(aclFinalize(), "finalize private runtime");
    return 0;
}
#endif
