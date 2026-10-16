// SPDX-License-Identifier: Apache-2.0
#ifndef SIDECARD_MHC_POST_V310_H
#define SIDECARD_MHC_POST_V310_H
#include "kernel_operator.h"
#include "sidecard_mhc_post_v310_tiling_data.h"
namespace NsSidecardMhcPostV310 {
using namespace AscendC;
constexpr int32_t WIDTH = 4096;
constexpr int32_t STREAMS = 4;
constexpr int32_t TILE = 512;
constexpr int32_t TILES = WIDTH / TILE;
class Kernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR output,
                              const SidecardMhcPostV310TilingData* tiling, TPipe* pipe) {
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    residualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(residual));
    postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(post));
    combGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(comb));
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    rows_ = tiling->rows;
    blocks_ = tiling->blocks;
    pipe->InitBuffer(xBuf_, TILE * sizeof(half));
    pipe->InitBuffer(xFloatBuf_, TILE * sizeof(float));
    pipe->InitBuffer(residualBuf_, TILE * sizeof(float));
    pipe->InitBuffer(productBuf_, TILE * sizeof(float));
    pipe->InitBuffer(resultBuf_, TILE * sizeof(float));
  }
  __aicore__ inline void Process() {
    auto x = xBuf_.Get<half>();
    auto xf = xFloatBuf_.Get<float>();
    auto residual = residualBuf_.Get<float>();
    auto product = productBuf_.Get<float>();
    auto result = resultBuf_.Get<float>();
    for (int64_t task = GetBlockIdx(); task < rows_ * STREAMS * TILES; task += blocks_) {
      const int64_t tile = task % TILES;
      const int64_t stream = (task / TILES) % STREAMS;
      const int64_t row = task / (STREAMS * TILES);
      const int64_t column = tile * TILE;
      DataCopy(x, xGm_[row * WIDTH + column], TILE);
      PipeBarrier<PIPE_ALL>();
      Cast(xf, x, RoundMode::CAST_NONE, TILE);
      PipeBarrier<PIPE_ALL>();
      for (int32_t input = 0; input < STREAMS; ++input) {
        DataCopy(residual, residualGm_[(row * STREAMS + input) * WIDTH + column], TILE);
        PipeBarrier<PIPE_ALL>();
        const float coefficient = combGm_.GetValue(row * STREAMS * STREAMS + input * STREAMS + stream);
        Muls(input == 0 ? result : product, residual, coefficient, TILE);
        PipeBarrier<PIPE_ALL>();
        if (input > 0) {
          Add(result, result, product, TILE);
          PipeBarrier<PIPE_ALL>();
        }
      }
      Muls(product, xf, postGm_.GetValue(row * STREAMS + stream), TILE);
      PipeBarrier<PIPE_ALL>();
      Add(result, result, product, TILE);
      PipeBarrier<PIPE_ALL>();
      auto bits = result.ReinterpretCast<uint32_t>();
      for (int32_t index = 0; index < TILE; ++index) {
        const uint32_t value = bits.GetValue(index);
        const bool nan = (value & 0x7f800000U) == 0x7f800000U && (value & 0x007fffffU) != 0;
        bits.SetValue(index, nan ? ((value & 0x80000000U) | 0x7fc00000U)
                                 : ((value + 0x7fffU + ((value >> 16) & 1U)) & 0xffff0000U));
      }
      PipeBarrier<PIPE_ALL>();
      DataCopy(outputGm_[(row * STREAMS + stream) * WIDTH + column], result, TILE);
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  GlobalTensor<half> xGm_;
  GlobalTensor<float> residualGm_, postGm_, combGm_, outputGm_;
  TBuf<TPosition::VECCALC> xBuf_, xFloatBuf_, residualBuf_, productBuf_, resultBuf_;
  int64_t rows_ = 0;
  uint32_t blocks_ = 0;
};
}  // namespace NsSidecardMhcPostV310
#endif
