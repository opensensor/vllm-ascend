// SPDX-License-Identifier: Apache-2.0
// Exact bounded coordinate division without an AI-CPU task or dtype cast.
#include "kernel_operator.h"
namespace {
constexpr int64_t MAX_COORDINATE_ELEMENTS = 16;
}
extern "C" __global__ __aicore__ void glm_coordinate_div_v1(GM_ADDR input, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  const auto settings = reinterpret_cast<__gm__ int64_t*>(config);
  const int64_t elements = settings[0], divisor = settings[1];
  if (elements < 0 || elements > MAX_COORDINATE_ELEMENTS || divisor <= 0 || AscendC::GetBlockIdx() != 0) return;
  const auto source = reinterpret_cast<__gm__ int64_t*>(input);
  const auto destination = reinterpret_cast<__gm__ int64_t*>(output);
  for (int64_t i = 0; i < elements; ++i) {
    const int64_t value = source[i];
    int64_t quotient = value / divisor;
    if (value < 0 && value % divisor != 0) --quotient;
    destination[i] = quotient;
  }
}
