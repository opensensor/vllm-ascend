// SPDX-License-Identifier: Apache-2.0
// Bit-preserving FP16 carry copies: touch selected pages, never the whole bank.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t TILE = 4096;
class StateRows {
 public:
  __aicore__ inline void Run(GM_ADDR cache, GM_ADDR indices, GM_ADDR flags, GM_ADDR packed, GM_ADDR config) {
    auto values = reinterpret_cast<__gm__ int64_t*>(config);
    const int64_t rows = values[0], stride = values[1], payload = values[2], selected = values[3];
    const int64_t indexBytes = values[4];
    cache_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(cache));
    packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(packed));
    indices32_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
    indices64_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));
#ifndef GLM_STATE_ROWS_SCATTER
    flags_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(flags));
#endif
    pipe_.InitBuffer(buffer_, TILE * sizeof(uint16_t));
    auto local = buffer_.Get<uint16_t>();
    for (int64_t row = 0; row < selected; ++row) {
      int64_t slot = indexBytes == 4 ? static_cast<int64_t>(indices32_.GetValue(row)) : indices64_.GetValue(row);
      if (slot < 0) slot += rows;
#ifndef GLM_STATE_ROWS_SCATTER
      const bool fresh = !flags_.GetValue(row);
#endif
      // vLLM supplies bounded slots. Guard corrupt/padded ids against OOB DMA.
      const bool valid = slot >= 0 && slot < rows;
      for (int64_t offset = GetBlockIdx() * TILE; offset < payload; offset += GetBlockNum() * TILE) {
        const uint32_t count = static_cast<uint32_t>(payload - offset < TILE ? payload - offset : TILE);
#ifdef GLM_STATE_ROWS_SCATTER
        if (!valid) continue;
        DataCopy(local, packed_[row * payload + offset], count);
#else
        if (valid && !fresh)
          DataCopy(local, cache_[slot * stride + offset], count);
        else
          Duplicate(local.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), count);
#endif
        PipeBarrier<PIPE_ALL>();
#ifdef GLM_STATE_ROWS_SCATTER
        DataCopy(cache_[slot * stride + offset], local, count);
#else
        DataCopy(packed_[row * payload + offset], local, count);
#endif
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> buffer_;
  GlobalTensor<uint16_t> cache_, packed_;
  GlobalTensor<int32_t> indices32_;
  GlobalTensor<int64_t> indices64_;
  GlobalTensor<uint8_t> flags_;
};
}  // namespace
#ifdef GLM_STATE_ROWS_SCATTER
extern "C" __global__ __aicore__ void glm_state_rows_scatter_v1(GM_ADDR cache, GM_ADDR indices, GM_ADDR packed,
                                                                GM_ADDR config) {
  AscendC::InitSocState();
  StateRows operation;
  operation.Run(cache, indices, nullptr, packed, config);
}
#else
extern "C" __global__ __aicore__ void glm_state_rows_gather_v1(GM_ADDR cache, GM_ADDR indices, GM_ADDR flags,
                                                               GM_ADDR packed, GM_ADDR config) {
  AscendC::InitSocState();
  StateRows operation;
  operation.Run(cache, indices, flags, packed, config);
}
#endif
