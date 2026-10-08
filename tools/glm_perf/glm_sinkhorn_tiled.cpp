// SPDX-License-Identifier: Apache-2.0
// Retain torch softmax; normalize independent 4x4 matrices in vector tiles.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr uint32_t MATRIX_ELEMENTS = 16;
constexpr uint32_t TILE_ROWS = 16;
constexpr uint32_t ELEMENTS = MATRIX_ELEMENTS * TILE_ROWS;
class TiledSinkhornNormalize {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR config, GM_ADDR indices) {
    auto params = reinterpret_cast<__gm__ int64_t*>(config);
    const int64_t rows = params[0];
    const int64_t iterations = params[1];
    const float eps = reinterpret_cast<__gm__ float*>(config)[4];
    const int64_t order = params[3];
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    target_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    indices_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(indices));
    pipe_.InitBuffer(storage_, ELEMENTS * sizeof(float) * 13);
    auto x = storage_.Get<float>();
    auto sum = x[ELEMENTS];
    auto scratch = x[2 * ELEMENTS];
    auto row = x[3 * ELEMENTS].ReinterpretCast<uint32_t>();
    auto col = x[7 * ELEMENTS].ReinterpretCast<uint32_t>();
    auto tmp = x[11 * ELEMENTS];
    auto tmp2 = x[12 * ELEMENTS];
    // One prepared index DMA replaces thousands of scalar index producers.
    DataCopy(row, indices_, 8 * ELEMENTS);
    PipeBarrier<PIPE_ALL>();
    for (int64_t base = GetBlockIdx() * TILE_ROWS; base < rows; base += GetBlockNum() * TILE_ROWS) {
      const uint32_t count = (rows - base < TILE_ROWS ? rows - base : TILE_ROWS) * MATRIX_ELEMENTS;
      DataCopy(x, source_[base * MATRIX_ELEMENTS], count);
      PipeBarrier<PIPE_ALL>();
      for (int64_t step = 0; step < 2 * iterations - 1; ++step) {
        auto idx = step % 2 == 0 ? col : row;
        Gather(sum, x, idx, static_cast<uint32_t>(0), count);
        Gather(scratch, x, idx[ELEMENTS], static_cast<uint32_t>(0), count);
        Gather(tmp, x, idx[2 * ELEMENTS], static_cast<uint32_t>(0), count);
        Gather(tmp2, x, idx[3 * ELEMENTS], static_cast<uint32_t>(0), count);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, scratch, count);
        PipeBarrier<PIPE_V>();
        if (order == 0 || (order == 2 && step % 2 == 1)) {
          Add(tmp, tmp, tmp2, count);
          PipeBarrier<PIPE_V>();
          Add(sum, sum, tmp, count);
        } else {
          Add(sum, sum, tmp, count);
          PipeBarrier<PIPE_V>();
          Add(sum, sum, tmp2, count);
        }
        PipeBarrier<PIPE_V>();
        Adds(sum, sum, eps, count);
        PipeBarrier<PIPE_V>();
        Div(x, x, sum, count);
        PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();
      DataCopy(target_[base * MATRIX_ELEMENTS], x, count);
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<float> source_, target_;
  GlobalTensor<uint32_t> indices_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_sinkhorn_tiled_v1(GM_ADDR input, GM_ADDR output, GM_ADDR config,
                                                            GM_ADDR indices) {
  AscendC::InitSocState();
  TiledSinkhornNormalize operation;
  operation.Run(input, output, config, indices);
}
