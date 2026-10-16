// SPDX-License-Identifier: Apache-2.0
// Staged FP16 residual + RMSNorm + INT8 NZ quantization, one row per core.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr uint32_t INNER = 4096, MAX_ROWS = 8, CELL = 32, ROW_BLOCK = 16, SCALE_STRIDE = 8;
constexpr float EPSILON = 1e-6f, QUANT_MAX = 127.0f;
class NormQuant {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR residual, GM_ADDR weight, GM_ADDR norm, GM_ADDR total, GM_ADDR quant,
                             GM_ADDR scales, GM_ADDR config) {
    const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
    const uint32_t row = GetBlockIdx();
    if (rows <= 0 || rows > MAX_ROWS || row >= rows) return;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    r_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight));
    n_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(norm));
    t_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(total));
    q_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quant));
    s_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
    pipe_.InitBuffer(half_a_, INNER * sizeof(half));
    pipe_.InitBuffer(half_b_, INNER * sizeof(half));
    pipe_.InitBuffer(float_a_, INNER * sizeof(float));
    pipe_.InitBuffer(float_b_, INNER * sizeof(float));
    pipe_.InitBuffer(integers_, INNER * sizeof(int32_t));
    pipe_.InitBuffer(codes_, INNER);
    pipe_.InitBuffer(reduction_, 256 * sizeof(float));
    pipe_.InitBuffer(scalar_, SCALE_STRIDE * sizeof(float));
    auto a = half_a_.Get<half>(), b = half_b_.Get<half>();
    auto fa = float_a_.Get<float>(), fb = float_b_.Get<float>();
    DataCopy(a, x_[row * INNER], INNER);
    DataCopy(b, r_[row * INNER], INNER);
    PipeBarrier<PIPE_ALL>();
    Cast(fa, a, RoundMode::CAST_NONE, INNER);
    Cast(fb, b, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Add(fa, fa, fb, INNER);
    PipeBarrier<PIPE_V>();
    Cast(a, fa, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_ALL>();
    DataCopy(t_[row * INNER], a, INNER);
    Mul(fb, fa, fa, INNER);
    PipeBarrier<PIPE_V>();
    ReduceSum(scalar_.Get<float>(), fb, reduction_.Get<float>(), INNER);
    PipeBarrier<PIPE_V>();
    Muls(scalar_.Get<float>(), scalar_.Get<float>(), 1.0f / INNER, 1);
    PipeBarrier<PIPE_V>();
    Adds(scalar_.Get<float>(), scalar_.Get<float>(), EPSILON, 1);
    PipeBarrier<PIPE_V>();
    Rsqrt(scalar_.Get<float>(), scalar_.Get<float>(), 1);
    PipeBarrier<PIPE_ALL>();
    const float inverse = scalar_.Get<float>().GetValue(0);
    DataCopy(b, w_, INNER);
    PipeBarrier<PIPE_ALL>();
    Cast(fb, b, RoundMode::CAST_NONE, INNER);
    Muls(fa, fa, inverse, INNER);
    PipeBarrier<PIPE_V>();
    Mul(fa, fa, fb, INNER);
    PipeBarrier<PIPE_V>();
    Cast(a, fa, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_ALL>();
    DataCopy(n_[row * INNER], a, INNER);
    Cast(fa, a, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Abs(fa, fa, INNER);
    PipeBarrier<PIPE_V>();
    ReduceMax(scalar_.Get<float>(), fa, reduction_.Get<float>(), INNER, false);
    PipeBarrier<PIPE_V>();
    Muls(scalar_.Get<float>(), scalar_.Get<float>(), 1.0f / QUANT_MAX, 1);
    PipeBarrier<PIPE_ALL>();
    const float scale = scalar_.Get<float>().GetValue(0);
    Duplicate(scalar_.Get<float>(), scale, SCALE_STRIDE);
    Muls(a, a, static_cast<half>(scale == 0 ? 0.0f : 1.0f / scale), INNER);
    PipeBarrier<PIPE_V>();
    Cast(fa, a, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Cast(integers_.Get<int32_t>(), fa, RoundMode::CAST_ROUND, INNER);
    PipeBarrier<PIPE_V>();
    Cast(fa, integers_.Get<int32_t>(), RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Cast(a, fa, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_V>();
    Cast(codes_.Get<int8_t>(), a, RoundMode::CAST_NONE, INNER);
    PipeBarrier<PIPE_ALL>();
    const DataCopyParams write{INNER / CELL, 1, 0, ROW_BLOCK - 1};
    DataCopy(q_[row * CELL], codes_.Get<int8_t>(), write);
    DataCopy(s_[row * SCALE_STRIDE], scalar_.Get<float>(), SCALE_STRIDE);
    PipeBarrier<PIPE_ALL>();
  }

 private:
  TPipe pipe_;
  GlobalTensor<half> x_, r_, w_, n_, t_;
  GlobalTensor<int8_t> q_;
  GlobalTensor<float> s_;
  TBuf<TPosition::VECCALC> half_a_, half_b_, float_a_, float_b_, integers_, codes_, reduction_, scalar_;
};
}  // namespace
extern "C" __global__ __aicore__ void rtsp_norm_quant(GM_ADDR x, GM_ADDR r, GM_ADDR w, GM_ADDR n, GM_ADDR t, GM_ADDR q,
                                                      GM_ADDR s, GM_ADDR config) {
  NormQuant op;
  op.Run(x, r, w, n, t, q, s, config);
}
