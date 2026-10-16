// SPDX-License-Identifier: Apache-2.0
// Entry-ABI test double only. This does not simulate the INT4 matrix schedule.
#pragma once
#include "kernel_operator.h"
namespace native_int4 {
inline int calls = 0, selectedM = 0, selectedN = 0;
template <uint32_t M, uint32_t N = 64>
class Schedule {
 public:
  template <typename Descriptor>
  void Init(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR routeWeights,
            GM_ADDR, const Descriptor* descriptor) {
    assert(routeWeights == nullptr && descriptor->nDim == 2560 && descriptor->kDim == 640);
  }
  void Process() {
    ++calls;
    selectedM = M;
    selectedN = N;
  }
};
}  // namespace native_int4
