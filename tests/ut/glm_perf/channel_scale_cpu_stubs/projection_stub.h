// SPDX-License-Identifier: Apache-2.0
// Replace only Cube algebra in the production down-prefill control test.
#include "kernel_operator.h"
using namespace AscendC;
constexpr unsigned ROWS = 64, N = 128, LANES = 8;
inline float projection_value(long row, long expert, long tile, long column) {
  return float((row * 19 + expert * 31 + tile * 7 + column) % 31 - 15);
}
class Projection {
 protected:
  bool Init(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR ends, GM_ADDR order, GM_ADDR keys, GM_ADDR,
            GM_ADDR, const int64_t* config) {
    currentCore_ = config[8];
    tokens_ = config[0];
    topK_ = config[1];
    experts_ = config[2];
    n_ = config[3];
    bits_ = config[6];
    originalTokens_ = config[7];
    ends_.SetGlobalBuffer(reinterpret_cast<int32_t*>(ends));
    order_.SetGlobalBuffer(reinterpret_cast<int64_t*>(order));
    sortedKeys_.SetGlobalBuffer(reinterpret_cast<int32_t*>(keys));
    pipe_.InitBuffer(outputUb_, ROWS * N * sizeof(half));
    return true;
  }
  void ProjectBlock(long row, unsigned count, long expert, long tile) {
    for (unsigned m = 0; m < count; ++m)
      for (unsigned c = 0; c < N; ++c)
        outputUb_.Get<half>().data[m * N + c] = projection_value(row + m, expert, tile, c);
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> outputUb_;
  GlobalTensor<int32_t> ends_, sortedKeys_;
  GlobalTensor<int64_t> order_;
  unsigned currentCore_;
  long tokens_, topK_, experts_, n_, bits_;
  bool originalTokens_;
};
