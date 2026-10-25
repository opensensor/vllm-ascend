// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int TILE = 256;
class Activation {
 public:
  __aicore__ inline void Run(GM_ADDR g, GM_ADDR u, GM_ADDR o, GM_ADDR cfg) {
    gate_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(g));
    up_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(u));
    out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(o));
    int64_t count = reinterpret_cast<__gm__ int64_t*>(cfg)[0];
    pipe_.InitBuffer(buf_, 24 * TILE);
    auto gh = buf_.Get<half>();
    auto uh = gh[TILE];
    auto gf = buf_.Get<float>()[TILE];
    auto uf = gf[TILE];
    auto sig = gf[2 * TILE];
    for (int64_t start = GetBlockIdx() * TILE; start < count; start += GetBlockNum() * TILE) {
      DataCopy(gh, gate_[start], TILE);
      DataCopy(uh, up_[start], TILE);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(gf, gh, RoundMode::CAST_NONE, TILE);
      Cast(uf, uh, RoundMode::CAST_NONE, TILE);
      PipeBarrier<PIPE_V>();
      Mins(gf, gf, 7.0f, TILE);
      Mins(uf, uf, 7.0f, TILE);
      PipeBarrier<PIPE_V>();
      Maxs(uf, uf, -7.0f, TILE);
      Muls(sig, gf, 1.702f, TILE);
      PipeBarrier<PIPE_V>();
      Adds(uf, uf, 1.0f, TILE);
      Sigmoid(sig, sig, buf_.Get<uint8_t>()[16 * TILE], TILE);
      PipeBarrier<PIPE_V>();
      Mul(gf, gf, sig, TILE);
      PipeBarrier<PIPE_V>();
      Mul(gf, gf, uf, TILE);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopy(out_[start], gf, TILE);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> buf_;
  GlobalTensor<half> gate_, up_;
  GlobalTensor<float> out_;
};
}  // namespace
extern "C" __global__ __aicore__ void minimax_clipped_swiglu(GM_ADDR g, GM_ADDR u, GM_ADDR o, GM_ADDR cfg) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  Activation op;
  op.Run(g, u, o, cfg);
}
