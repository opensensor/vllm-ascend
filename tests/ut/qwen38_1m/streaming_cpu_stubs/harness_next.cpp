// SPDX-License-Identifier: Apache-2.0
// Invoke the actual native entry/body with synchronous, bounds-checked CPU stubs.
#include "native_streaming_next.cpp"
#include <exception>

namespace {
std::string lastError;
}
int Run(void** arguments, int64_t* lengths, bool columns) {
  using namespace AscendC;
  lastError.clear();
  globals.clear();
  pending = {};
  peaks = {};
  for (uint32_t i = 0; i < 11; ++i)
    globals.push_back({reinterpret_cast<uint8_t*>(arguments[i]), static_cast<size_t>(lengths[i])});
  try {
    for (blockIndex = 0; blockIndex < 8; ++blockIndex) {
      used = {};
      auto entry = columns ? qwen_streaming_columns_v2 : qwen_streaming_projection_v2;
      entry(reinterpret_cast<GM_ADDR>(arguments[0]), reinterpret_cast<GM_ADDR>(arguments[1]),
            reinterpret_cast<GM_ADDR>(arguments[2]), reinterpret_cast<GM_ADDR>(arguments[3]),
            reinterpret_cast<GM_ADDR>(arguments[4]), reinterpret_cast<GM_ADDR>(arguments[5]),
            reinterpret_cast<GM_ADDR>(arguments[6]), reinterpret_cast<GM_ADDR>(arguments[7]),
            reinterpret_cast<GM_ADDR>(arguments[8]), reinterpret_cast<GM_ADDR>(arguments[9]),
            reinterpret_cast<GM_ADDR>(arguments[10]));
      for (auto direction : pending)
        for (bool signal : direction)
          if (signal) throw std::runtime_error("unmatched event at final drain");
    }
  } catch (const std::exception& error) {
    lastError = error.what();
    return 1;
  }
  return 0;
}
extern "C" int run_projection(void** arguments, int64_t* lengths) { return Run(arguments, lengths, false); }
extern "C" int run_columns(void** arguments, int64_t* lengths) { return Run(arguments, lengths, true); }
extern "C" const char* projection_error() { return lastError.c_str(); }
extern "C" uint64_t projection_peak(uint32_t space) { return AscendC::peaks.at(space); }
