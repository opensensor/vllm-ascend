// SPDX-License-Identifier: Apache-2.0
#include "pair_add_runtime.h"

#include <acl/acl.h>

#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {
constexpr uint64_t ADD_ABI = 1;
constexpr uint64_t WIDTH = 4096;
constexpr uint64_t MAX_IMAGE_BYTES = 64 * 1024 * 1024;
constexpr int INVALID_ARGUMENT = -1;
constexpr int ALLOCATION_FAILURE = -2;
constexpr int INVALID_SDK_RESULT = -3;

struct LocalAdd {
    std::vector<char> image;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    uint32_t blocks = 0;
    // The SDK argument array points to stable pointer-value slots, not directly
    // to tensor data. Neither array is rebuilt or allocated on the hot path.
    void* addresses[3] = {};
    void* arguments[3] = {};
};

const char* entry(uint64_t elements)
{
    switch (elements) {
        case WIDTH: return "sidecard_pair_add_r1_v1";
        case 4 * WIDTH: return "sidecard_pair_add_r4_v1";
        case 32 * WIDTH: return "sidecard_pair_add_r32_v1";
        default: return nullptr;
    }
}

bool valid_buffers(const void* input, const void* inbox, void* output, uint64_t elements)
{
    const void* data[] = {input, inbox, output};
    const uintptr_t bytes = elements * sizeof(float);
    constexpr uintptr_t LIMIT = std::numeric_limits<uintptr_t>::max();
    for (unsigned i = 0; i < 3; ++i) {
        const auto address = reinterpret_cast<uintptr_t>(data[i]);
        if (!address || address % alignof(float) || address > LIMIT - bytes) return false;
        for (unsigned j = 0; j < i; ++j) {
            const auto previous = reinterpret_cast<uintptr_t>(data[j]);
            if (address < previous + bytes && previous < address + bytes) return false;
        }
    }
    return true;
}
}  // namespace

extern "C" uint64_t sidecard_add_abi()
{
    return ADD_ABI;
}

extern "C" int sidecard_add_create(const void* image, uint64_t image_bytes, uint64_t elements, uint32_t blocks,
                                   const void* local_input, const void* local_inbox, void* local_output, void** handle)
{
    if (!handle) return INVALID_ARGUMENT;
    *handle = nullptr;
    const char* symbol = entry(elements);
    if (!image || image_bytes < 16 || image_bytes > MAX_IMAGE_BYTES || std::memcmp(image, "\x7f" "ELF", 4) ||
        !symbol || (blocks != 1 && blocks != 2 && blocks != 4 && blocks != 8) ||
        !valid_buffers(local_input, local_inbox, local_output, elements)) {
        return INVALID_ARGUMENT;
    }
    auto* add = new (std::nothrow) LocalAdd;
    if (!add) return ALLOCATION_FAILURE;
    try {
        const auto* bytes = static_cast<const char*>(image);
        add->image.assign(bytes, bytes + image_bytes);
    } catch (const std::bad_alloc&) {
        delete add;
        return ALLOCATION_FAILURE;
    }
    add->blocks = blocks;
    add->addresses[0] = const_cast<void*>(local_input);
    add->addresses[1] = const_cast<void*>(local_inbox);
    add->addresses[2] = local_output;
    for (unsigned i = 0; i < 3; ++i) add->arguments[i] = &add->addresses[i];
    // Expose partial ownership before SDK calls. Failed load/lookup must never
    // discard code storage that the SDK may retain; caller destroys explicitly.
    *handle = add;
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions options{&option, 1};
    aclError status = aclrtBinaryLoadFromData(add->image.data(), add->image.size(), &options, &add->binary);
    if (status) return status;
    if (!add->binary) return INVALID_SDK_RESULT;
    status = aclrtBinaryGetFunction(add->binary, symbol, &add->function);
    if (status) return status;
    return add->function ? 0 : INVALID_SDK_RESULT;
}

extern "C" int sidecard_add_enqueue(void* handle, void* caller_stream)
{
    if (!handle || !caller_stream) return INVALID_ARGUMENT;
    auto* add = static_cast<LocalAdd*>(handle);
    if (!add->binary || !add->function) return INVALID_ARGUMENT;
    return aclrtLaunchKernelWithArgsArray(add->function, add->blocks, static_cast<aclrtStream>(caller_stream),
                                         nullptr, add->arguments);
}

extern "C" int sidecard_add_destroy(void** handle, uint32_t caller_drained)
{
    if (!handle || !*handle || caller_drained != 1) return INVALID_ARGUMENT;
    auto* add = static_cast<LocalAdd*>(*handle);
    if (add->binary) {
        const aclError status = aclrtBinaryUnLoad(add->binary);
        if (status) return status;
    }
    delete add;
    *handle = nullptr;
    return 0;
}
