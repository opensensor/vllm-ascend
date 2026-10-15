// SPDX-License-Identifier: Apache-2.0
// Private fixed-target ACLRTC compiler. Initializes ACL without selecting a
// device or submitting work; run only while the installed driver is healthy.
#include <acl/acl.h>
#include <acl/acl_rt_compile.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s SOURCE OUTPUT [compiler options...]\n", argv[0]);
        return 2;
    }
    std::ifstream source_file(argv[1], std::ios::binary);
    if (!source_file.good()) return 2;
    const std::string source((std::istreambuf_iterator<char>(source_file)), std::istreambuf_iterator<char>());
    if (source.empty() || aclInit(nullptr) != ACL_SUCCESS) return 1;
    aclrtcProg program = nullptr;
    if (aclrtcCreateProg(&program, source.c_str(), "sidecard_pair_add.asc", 0, nullptr, nullptr) != ACL_SUCCESS)
        return 1;
    std::vector<const char*> options;
    for (int i = 3; i < argc; ++i) options.push_back(argv[i]);
    const aclError compiled = aclrtcCompileProg(program, options.size(), options.data());
    if (compiled != ACL_SUCCESS) {
        size_t size = 0;
        aclrtcGetCompileLogSize(program, &size);
        std::vector<char> log(size + 1, 0);
        if (size) aclrtcGetCompileLog(program, log.data());
        std::fprintf(stderr, "ACLRTC failed (%d): %s\n", compiled, log.data());
        aclrtcDestroyProg(&program);
        return 1;
    }
    size_t size = 0;
    const aclError sized = aclrtcGetBinDataSize(program, &size);
    if (sized != ACL_SUCCESS || size < 16) {
        aclrtcDestroyProg(&program);
        return 1;
    }
    std::vector<char> binary(size);
    const aclError extracted = aclrtcGetBinData(program, binary.data());
    const aclError destroyed = aclrtcDestroyProg(&program);
    if (extracted != ACL_SUCCESS || destroyed != ACL_SUCCESS) return 1;
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    output.write(binary.data(), binary.size());
    return output.good() ? 0 : 1;
}
