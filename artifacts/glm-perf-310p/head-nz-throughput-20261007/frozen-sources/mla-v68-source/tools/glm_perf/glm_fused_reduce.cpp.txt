// SPDX-License-Identifier: Apache-2.0
// Stable FP32 reduction of bulk-prefill native INT4 down projections.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr uint32_t N = 128;
class Reduce {
 public:
  __aicore__ inline void Run(GM_ADDR workspace, GM_ADDR ranks, GM_ADDR ends, GM_ADDR output, const int64_t* config) {
    const int64_t rows = config[0], experts = config[1], width = config[2], tokens = config[6], topK = config[7];
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    ranks_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranks));
    ends_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    pipe_.InitBuffer(storage_, 2 * N * sizeof(float));
    auto accum = storage_.Get<float>(), value = accum[N];
    const int64_t localRows = ends_.GetValue(experts - 1), tiles = width / N;
    if (localRows < 0 || localRows > rows) return;
    for (int64_t task = GetBlockIdx(); task < tokens * tiles; task += GetBlockNum()) {
      const int64_t token = task / tiles, column = (task % tiles) * N;
      Duplicate(accum, 0.0f, N);
      PipeBarrier<PIPE_ALL>();
      // ranks are sorted by expert's stable route position within each token.
      // Never read the uninitialized peer/zero-weight suffix of the workspace.
      for (int64_t slot = 0; slot < topK; ++slot) {
        const int64_t row = ranks_.GetValue(token * topK + slot);
        if (row < 0 || row >= localRows) continue;
        DataCopy(value, input_[row * width + column], N);
        PipeBarrier<PIPE_ALL>();
        Add(accum, accum, value, N);
        PipeBarrier<PIPE_ALL>();
      }
      DataCopy(output_[token * width + column], accum, N);
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_;
  GlobalTensor<float> input_, output_;
  GlobalTensor<int32_t> ranks_;
  GlobalTensor<int64_t> ends_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_fused_reduce_v1(GM_ADDR workspace, GM_ADDR ranks, GM_ADDR ends,
                                                          GM_ADDR output, GM_ADDR tiling) {
  AscendC::InitSocState();
  int64_t config[8];
  const auto source = reinterpret_cast<__gm__ int64_t*>(tiling);
  for (unsigned i = 0; i < 8; ++i) config[i] = source[i];
  if (config[0] <= 0 || config[0] > 65536 || config[1] <= 0 || config[2] <= 0 || config[2] % N || config[6] <= 16 ||
      config[7] <= 0 || config[6] * config[7] != config[0])
    return;
  Reduce operation;
  operation.Run(workspace, ranks, ends, output, config);
}
