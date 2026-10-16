// SPDX-License-Identifier: Apache-2.0
#include "native_local_down_projection.cpp"
extern "C" int inspect_projection_entry(int64_t* descriptor, int64_t* ends, int64_t* selection) {
  AscendC::regions = {{reinterpret_cast<uintptr_t>(ends), 128 * sizeof(int64_t), 128 * sizeof(int64_t)}};
  native_int4::calls = native_int4::selectedM = native_int4::selectedN = 0;
  try {
    qwen_local_down_projection_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ends, nullptr,
                                  descriptor);
    selection[0] = native_int4::calls;
    selection[1] = native_int4::selectedM;
    selection[2] = native_int4::selectedN;
    return 0;
  } catch (const std::runtime_error&) {
    return 1;
  }
}
