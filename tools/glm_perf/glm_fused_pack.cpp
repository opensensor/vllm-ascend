// SPDX-License-Identifier: Apache-2.0
// Quantize each original token once; all expert/output tiles share its limbs.
#include "kernel_operator.h"
#include "glm_fused_quantize.h"
namespace {
using namespace AscendC;
class Pack {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR low, GM_ADDR high, GM_ADDR xs, int64_t groups, int64_t bits) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    low_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
    high_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    pipe_.InitBuffer(storage_, 16384);
    pipe_.InitBuffer(padded_, 1024);
    pipe_.InitBuffer(offsets_, 2048);
    auto storage = storage_.Get<uint8_t>();
    auto packed = storage.ReinterpretCast<int8_t>()[24 * GlmFusedQuant::ELEMENTS];
    constexpr uint32_t broadcastOffset =
        7 * GlmFusedQuant::ELEMENTS + 2 * GlmFusedQuant::BATCH + 3 * GlmFusedQuant::LANES;
    for (int64_t first = GetBlockIdx() * 8; first < groups; first += GetBlockNum() * 8) {
      DataCopy(storage.ReinterpretCast<half>(), input_[first * 32], 256);
      PipeBarrier<PIPE_ALL>();
      GlmFusedQuant::Run(storage, padded_.Get<half>(), offsets_.Get<uint32_t>(), bits);
      for (uint32_t group = 0; group < 8; ++group) {
        DataCopy(low_[(first + group) * 32], packed[group * 32], 32);
        if (bits == 8) DataCopy(high_[(first + group) * 32], packed[256 + group * 32], 32);
      }
      DataCopy(xs_[first * 8], storage.ReinterpretCast<float>()[broadcastOffset], 64);
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_, padded_, offsets_;
  GlobalTensor<half> input_;
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<float> xs_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_fused_pack_v1(GM_ADDR x, GM_ADDR low, GM_ADDR high, GM_ADDR xs,
                                                        GM_ADDR tiling) {
  AscendC::InitSocState();
  const auto config = reinterpret_cast<__gm__ int64_t*>(tiling);
  if (config[0] <= 0 || config[0] % 8 || (config[1] != 4 && config[1] != 8)) return;
  Pack operation;
  operation.Run(x, low, high, xs, config[0], config[1]);
}
