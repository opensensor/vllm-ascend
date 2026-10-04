#include <acl/acl.h>
#include <acl/acl_rt_compile.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int kBlocks = 8;
constexpr int kWarmups = 20;
constexpr int kRepeats = 200;

bool Check(aclError result, const char* operation)
{
    if (result == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %d %s\n", operation, result, aclGetRecentErrMsg());
    return false;
}

uint32_t RoundBits(uint32_t bits)
{
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0) {
        return (bits & 0x80000000U) | 0x7fc00000U;
    }
    return (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s SOURCE.asc DEVICE_ID ELEMENTS\n", argv[0]);
        return 2;
    }
    const int device_id = std::atoi(argv[2]);
    const int count = std::atoi(argv[3]);
    if (count <= 0) {
        std::fprintf(stderr, "ELEMENTS must be positive\n");
        return 2;
    }
    std::ifstream source_file(argv[1]);
    if (!source_file) {
        std::perror(argv[1]);
        return 2;
    }
    const std::string source((std::istreambuf_iterator<char>(source_file)), std::istreambuf_iterator<char>());
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    aclrtcProg program = nullptr;
    if (!Check(aclrtcCreateProg(&program, source.c_str(), "round.asc", 0, nullptr, nullptr),
               "aclrtcCreateProg")) {
        return 1;
    }
    const char* options[] = {
        "--npu-arch=dav-2002",
        "--sysroot=/usr/local/Ascend/cann-9.1.0/tools/hcc/sysroot",
        "-isystem/usr/local/Ascend/cann-9.1.0/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0",
        "-isystem/usr/local/Ascend/cann-9.1.0/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu",
    };
    const aclError compiled = aclrtcCompileProg(program, 4, options);
    if (compiled != ACL_SUCCESS) {
        size_t log_size = 0;
        (void)aclrtcGetCompileLogSize(program, &log_size);
        std::vector<char> log(log_size + 1);
        (void)aclrtcGetCompileLog(program, log.data());
        std::fprintf(stderr, "compile failed (%d): %s\n", compiled, log.data());
        (void)aclrtcDestroyProg(&program);
        return 1;
    }
    size_t binary_size = 0;
    if (!Check(aclrtcGetBinDataSize(program, &binary_size), "aclrtcGetBinDataSize")) {
        return 1;
    }
    std::vector<char> binary(binary_size);
    if (!Check(aclrtcGetBinData(program, binary.data()), "aclrtcGetBinData")) {
        return 1;
    }

    if (!Check(aclrtSetDevice(device_id), "aclrtSetDevice")) {
        return 1;
    }
    aclrtStream stream = nullptr;
    if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        return 1;
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    void* input_device = nullptr;
    void* output_device = nullptr;
    if (!Check(aclrtMalloc(&input_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc input") ||
        !Check(aclrtMalloc(&output_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc output")) {
        return 1;
    }
    std::vector<uint32_t> input(count);
    std::vector<uint32_t> output(count, 0);
    std::mt19937 generator(3103);
    for (int index = 0; index < count; ++index) {
        uint32_t bits = generator();
        if ((bits & 0x7f800000U) == 0x7f800000U) {
            bits ^= 0x00800000U;
        }
        if (index % 17 == 0) {
            bits = (bits & 0xffff0000U) | 0x8000U;
        }
        input[index] = bits;
    }
    if (count >= 4) {
        input[0] = 0x7f800001U;
        input[1] = 0xff800001U;
        input[2] = 0x7f800000U;
        input[3] = 0xff800000U;
    }
    if (!Check(aclrtMemcpy(input_device, bytes, input.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy input")) {
        return 1;
    }
    aclrtBinaryLoadOption binary_option{};
    binary_option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    binary_option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions load_options{};
    load_options.numOpt = 1;
    load_options.options = &binary_option;
    aclrtBinHandle binary_handle = nullptr;
    if (!Check(aclrtBinaryLoadFromData(binary.data(), binary_size, &load_options, &binary_handle),
               "aclrtBinaryLoadFromData")) {
        return 1;
    }
    aclrtFuncHandle function_handle = nullptr;
    if (!Check(aclrtBinaryGetFunction(binary_handle, "round_bf16_fp32", &function_handle),
               "aclrtBinaryGetFunction")) {
        return 1;
    }
    void* kernel_args[] = {&input_device, &output_device, const_cast<int*>(&count)};
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        if (!Check(aclrtLaunchKernelWithArgsArray(function_handle, kBlocks, stream, nullptr, kernel_args),
                   "warmup launch")) {
            return 1;
        }
    }
    if (!Check(aclrtSynchronizeStream(stream), "warmup sync")) {
        return 1;
    }
    aclrtEvent begin = nullptr;
    aclrtEvent end = nullptr;
    if (!Check(aclrtCreateEvent(&begin), "create begin event") ||
        !Check(aclrtCreateEvent(&end), "create end event") ||
        !Check(aclrtRecordEvent(begin, stream), "record begin")) {
        return 1;
    }
    const auto wall_begin = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        if (!Check(aclrtLaunchKernelWithArgsArray(function_handle, kBlocks, stream, nullptr, kernel_args),
                   "benchmark launch")) {
            return 1;
        }
    }
    if (!Check(aclrtRecordEvent(end, stream), "record end") ||
        !Check(aclrtSynchronizeStream(stream), "benchmark sync")) {
        return 1;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    float device_ms = 0;
    if (!Check(aclrtEventElapsedTime(&device_ms, begin, end), "elapsed time") ||
        !Check(aclrtMemcpy(output.data(), bytes, output_device, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy output")) {
        return 1;
    }
    size_t mismatches = 0;
    for (int index = 0; index < count; ++index) {
        if (output[index] != RoundBits(input[index])) {
            if (mismatches < 4) {
                std::printf("mismatch %d: in=%08x out=%08x expected=%08x\n", index,
                            input[index], output[index], RoundBits(input[index]));
            }
            ++mismatches;
        }
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    std::printf("elements=%d mismatches=%zu/%d device_ms_per_call=%.6f wall_ms_per_call=%.6f\n",
                count, mismatches, count, device_ms / kRepeats, wall_ms / kRepeats);
    (void)aclrtDestroyEvent(begin);
    (void)aclrtDestroyEvent(end);
    (void)aclrtBinaryUnLoad(binary_handle);
    (void)aclrtFree(input_device);
    (void)aclrtFree(output_device);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(device_id);
    (void)aclFinalize();
    (void)aclrtcDestroyProg(&program);
    return mismatches == 0 ? 0 : 1;
}
