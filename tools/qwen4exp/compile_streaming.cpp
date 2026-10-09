// SPDX-License-Identifier: Apache-2.0
// Host ACLRTC compiler. No ACL runtime initialization, device selection or load.
#include <acl/acl.h>
#include <acl/acl_rt_compile.h>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s SOURCE OUTPUT [compiler options...]\n", argv[0]);
    return 2;
  }
  std::ifstream sourceFile(argv[1]);
  if (!sourceFile.good()) return 2;
  const std::string source((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
  aclrtcProg program = nullptr;
  if (aclrtcCreateProg(&program, source.c_str(), "qwen_streaming.asc", 0, nullptr, nullptr) != ACL_SUCCESS) return 1;
  std::vector<const char*> options;
  for (int i = 3; i < argc; ++i) options.push_back(argv[i]);
  const aclError compiled = aclrtcCompileProg(program, options.size(), options.data());
  if (compiled != ACL_SUCCESS) {
    size_t size = 0;
    aclrtcGetCompileLogSize(program, &size);
    std::vector<char> log(size + 1);
    aclrtcGetCompileLog(program, log.data());
    std::fprintf(stderr, "ACLRTC failed (%d): %s\n", compiled, log.data());
    aclrtcDestroyProg(&program);
    return 1;
  }
  size_t size = 0;
  if (aclrtcGetBinDataSize(program, &size) != ACL_SUCCESS || size == 0) {
    aclrtcDestroyProg(&program);
    return 1;
  }
  std::vector<char> binary(size);
  const aclError extracted = aclrtcGetBinData(program, binary.data());
  aclrtcDestroyProg(&program);
  if (extracted != ACL_SUCCESS) return 1;
  std::ofstream output(argv[2], std::ios::binary);
  output.write(binary.data(), binary.size());
  return output.good() ? 0 : 1;
}
