// SPDX-License-Identifier: Apache-2.0
// Normalization only: preserve the qualified torch softmax and its epsilon.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t ELEMENTS = 16;
class SinkhornNormalize {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR config) {
    auto params = reinterpret_cast<__gm__ int64_t*>(config);
    const int64_t rows = params[0];
    const int64_t iterations = params[1];
    const float eps = reinterpret_cast<__gm__ float*>(config)[4];
    const int64_t order = params[3];
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(storage_, ELEMENTS * sizeof(float) * 13);
    auto x = storage_.Get<float>();
    auto sum = x[ELEMENTS];
    auto scratch = x[2 * ELEMENTS];
    auto row = x[3 * ELEMENTS].ReinterpretCast<uint32_t>();
    auto col = x[7 * ELEMENTS].ReinterpretCast<uint32_t>();
    auto tmp = x[11 * ELEMENTS];
    auto tmp2 = x[12 * ELEMENTS];
    for (int i = 0; i < ELEMENTS; ++i) {
      for (int j = 0; j < 4; ++j) {
        row.SetValue(j * ELEMENTS + i, ((i / 4) * 4 + j) * sizeof(float));
        col.SetValue(j * ELEMENTS + i, (j * 4 + i % 4) * sizeof(float));
      }
    }
    PipeBarrier<PIPE_ALL>();
    for (int64_t r = GetBlockIdx(); r < rows; r += GetBlockNum()) {
      DataCopy(x, input_[r * ELEMENTS], ELEMENTS);
      PipeBarrier<PIPE_ALL>();
      for (int64_t step = 0; step < 2 * iterations - 1; ++step) {
        auto idx = (step % 2 == 0) ? col : row;
        Gather(sum, x, idx, static_cast<uint32_t>(0), ELEMENTS);
        Gather(scratch, x, idx[ELEMENTS], static_cast<uint32_t>(0), ELEMENTS);
        Gather(tmp, x, idx[2 * ELEMENTS], static_cast<uint32_t>(0), ELEMENTS);
        Gather(tmp2, x, idx[3 * ELEMENTS], static_cast<uint32_t>(0), ELEMENTS);
        PipeBarrier<PIPE_V>();
        Add(sum, sum, scratch, ELEMENTS);
        PipeBarrier<PIPE_V>();
        if (order == 0 || (order == 2 && step % 2 == 1)) {
          Add(tmp, tmp, tmp2, ELEMENTS);
          PipeBarrier<PIPE_V>();
          Add(sum, sum, tmp, ELEMENTS);
        } else {
          Add(sum, sum, tmp, ELEMENTS);
          PipeBarrier<PIPE_V>();
          Add(sum, sum, tmp2, ELEMENTS);
        }
        PipeBarrier<PIPE_V>();
        Adds(sum, sum, eps, ELEMENTS);
        PipeBarrier<PIPE_V>();
        Div(x, x, sum, ELEMENTS);
        PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();
      DataCopy(output_[r * ELEMENTS], x, ELEMENTS);
      PipeBarrier<PIPE_ALL>();
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<float> input_, output_;
};
}
extern "C" __global__ __aicore__ void glm_sinkhorn_normalize_v1(
    GM_ADDR input, GM_ADDR output, GM_ADDR config) {
  AscendC::InitSocState();
  SinkhornNormalize op;
  op.Run(input, output, config);
}
