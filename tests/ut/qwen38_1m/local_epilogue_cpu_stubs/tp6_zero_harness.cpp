// SPDX-License-Identifier: Apache-2.0
#define EVENT_ID0 0
#include "tp6_zero_local_pack.h"

extern "C" int run_tp6_zero(int64_t rows, int64_t active, int8_t* low, int8_t* high, float* scale, float* sum) {
  using namespace AscendC;
  globals = {{reinterpret_cast<uint8_t*>(low), static_cast<size_t>(rows * 320)},
             {reinterpret_cast<uint8_t*>(high), static_cast<size_t>(rows * 320)},
             {reinterpret_cast<uint8_t*>(scale), static_cast<size_t>(rows * 40 * sizeof(float))},
             {reinterpret_cast<uint8_t*>(sum), static_cast<size_t>(rows * 40 * sizeof(float))}};
  used.fill(0);
  peaks.fill(0);
  pending = {};
  try {
    TPipe pipe;
    tp6_pack::LocalPack pack;
    pack.Init(pipe, reinterpret_cast<GM_ADDR>(low), reinterpret_cast<GM_ADDR>(high), reinterpret_cast<GM_ADDR>(scale),
              reinterpret_cast<GM_ADDR>(sum));
    for (int group = 0; group < 5; ++group) pack.EmitZero(group, active, rows - active);
    for (const auto& event : pending)
      for (bool outstanding : event)
        if (outstanding) return 2;
    return peaks[0] > 32 * 1024 ? 3 : 0;
  } catch (const std::exception&) {
    return 1;
  }
}
