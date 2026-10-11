// SPDX-License-Identifier: Apache-2.0
// Exact block32 activation/projection contract with coalesced INT8 DP4A.
// Unpacked INT8 weights stay resident; packed weights remain follow-up work.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

constexpr int kThreads = 128;
constexpr int kBlock = 32;
constexpr float kHalfMax = 65504.0F;
constexpr int kDotLanes = 8;
constexpr int kDotValuesPerLane = 4;

__global__ void quantize(const __half* x, signed char* q, float* scales, int rows, int width, int bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * width / kBlock) return;
  float maximum = 0.0F;
  for (int lane = 0; lane < kBlock; ++lane) maximum = fmaxf(maximum, fabsf(__half2float(x[index * kBlock + lane])));
  int limit = (1 << (bits - 1)) - 1;
  float scale = maximum > 0.0F ? __fdiv_rn(maximum, static_cast<float>(limit)) : 1.0F;
  scales[index] = scale;
  for (int lane = 0; lane < kBlock; ++lane) {
    float value = rintf(__fdiv_rn(__half2float(x[index * kBlock + lane]), scale));
    q[index * kBlock + lane] = static_cast<signed char>(fminf(limit, fmaxf(-limit, value)));
  }
}

__global__ void project(const signed char* q, const float* xs, const signed char* weights, const __half* ws,
                        const int* expert_ids, __half* out, int rows, int n, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * n) return;
  int row = index / n;
  int channel = index % n;
  int expert = expert_ids[row];
  float result = 0.0F;
  for (int group = 0; group < k / kBlock; ++group) {
    int dot = 0;
    for (int lane = 0; lane < kBlock; ++lane) {
      dot += static_cast<int>(q[row * k + group * kBlock + lane]) *
             static_cast<int>(weights[(expert * n + channel) * k + group * kBlock + lane]);
    }
    float factor = __fmul_rn(xs[row * (k / kBlock) + group],
                             __half2float(ws[(expert * (n / kBlock) + channel / kBlock) * (k / kBlock) + group]));
    result = __fadd_rn(result, __fmul_rn(static_cast<float>(dot), factor));
  }
  out[index] = __float2half_rn(result);
}

// Eight lanes read one contiguous 32-value weight group. DP4A and the
// integer subgroup reduction preserve the exact block dot product; the
// leader retains the scalar kernel's ordered FP32 scale/multiply/add steps.
__global__ void project_dp4a(const signed char* q, const float* xs, const signed char* weights, const __half* ws,
                             const int* expert_ids, __half* out, int rows, int n, int k) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int index = thread / kDotLanes;
  if (index >= rows * n) return;
  int lane = threadIdx.x % kDotLanes;
  int row = index / n, channel = index % n;
  int expert = expert_ids[row];
  float result = 0.0F;
  for (int group = 0; group < k / kBlock; ++group) {
    int offset = group * kBlock + lane * kDotValuesPerLane;
    int activation = *reinterpret_cast<const int*>(q + row * k + offset);
    int weight = *reinterpret_cast<const int*>(weights + (expert * n + channel) * k + offset);
    int dot = __dp4a(activation, weight, 0);
    for (int delta = kDotLanes / 2; delta > 0; delta /= 2) {
      dot += __shfl_down_sync(0xffffffffU, dot, delta, kDotLanes);
    }
    if (lane == 0) {
      float factor = __fmul_rn(xs[row * (k / kBlock) + group],
                               __half2float(ws[(expert * (n / kBlock) + channel / kBlock) * (k / kBlock) + group]));
      result = __fadd_rn(result, __fmul_rn(static_cast<float>(dot), factor));
    }
  }
  if (lane == 0) out[index] = __float2half_rn(result);
}

__global__ void swiglu(const __half* gate_up, __half* hidden, int rows, int intermediate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * intermediate) return;
  int row = index / intermediate, channel = index % intermediate;
  float gate = __half2float(gate_up[row * 2 * intermediate + channel]);
  float up = __half2float(gate_up[row * 2 * intermediate + intermediate + channel]);
  float silu = __fdiv_rn(gate, __fadd_rn(1.0F, expf(-gate)));
  hidden[index] = __float2half_rn(fminf(kHalfMax, fmaxf(-kHalfMax, __fmul_rn(silu, up))));
}

extern "C" int sidecard_oracle_forward(const void* input, const void* ids, const void* gate, const void* gs,
                                       const void* down, const void* ds, void* input_q, void* input_scales,
                                       void* gate_up, void* hidden, void* hidden_q, void* hidden_scales, void* output,
                                       int rows, int width, int intermediate, int bits) {
  if (rows < 1 || rows > 256 || width != 4096 || intermediate != 2048 || (bits != 4 && bits != 8))
    return static_cast<int>(cudaErrorInvalidValue);
  quantize<<<(rows * width / kBlock + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(input), static_cast<signed char*>(input_q), static_cast<float*>(input_scales), rows,
      width, bits);
  project_dp4a<<<(rows * 2 * intermediate * kDotLanes + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const signed char*>(input_q), static_cast<const float*>(input_scales),
      static_cast<const signed char*>(gate), static_cast<const __half*>(gs), static_cast<const int*>(ids),
      static_cast<__half*>(gate_up), rows, 2 * intermediate, width);
  swiglu<<<(rows * intermediate + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(gate_up), static_cast<__half*>(hidden), rows, intermediate);
  quantize<<<(rows * intermediate / kBlock + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(hidden), static_cast<signed char*>(hidden_q), static_cast<float*>(hidden_scales), rows,
      intermediate, bits);
  project_dp4a<<<(rows * width * kDotLanes + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const signed char*>(hidden_q), static_cast<const float*>(hidden_scales),
      static_cast<const signed char*>(down), static_cast<const __half*>(ds), static_cast<const int*>(ids),
      static_cast<__half*>(output), rows, width, intermediate);
  return static_cast<int>(cudaGetLastError());
}

__global__ void combine_routes(const float* routes, const float* weights, const float* shared, float* output, int rows,
                               int top_k, int width, float scale) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * width) return;
  int row = index / width, channel = index % width;
  float total = 0.0F;
  for (int slot = 0; slot < top_k; ++slot) {
    float product = __fmul_rn(weights[row * top_k + slot], routes[(row * top_k + slot) * width + channel]);
    total = __fadd_rn(total, product);
  }
  output[index] = __fadd_rn(__fmul_rn(total, scale), shared[index]);
}

extern "C" int sidecard_oracle_combine(const void* routes, const void* weights, const void* shared, void* output,
                                       int rows, int top_k, int width, float scale) {
  if (rows < 1 || rows > 32 || top_k < 1 || top_k > 8 || width != 4096 || !std::isfinite(scale) || scale <= 0.0F)
    return static_cast<int>(cudaErrorInvalidValue);
  combine_routes<<<(rows * width + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const float*>(routes), static_cast<const float*>(weights), static_cast<const float*>(shared),
      static_cast<float*>(output), rows, top_k, width, scale);
  return static_cast<int>(cudaGetLastError());
}
