// SPDX-License-Identifier: Apache-2.0
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

}
int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s SOURCE.cpp OUTPUT.bin\n", argv[0]);
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

    std::ofstream output(argv[2], std::ios::binary);
    output.write(binary.data(), binary.size());
    (void)aclrtcDestroyProg(&program);
    return output.good() ? 0 : 1;
}
