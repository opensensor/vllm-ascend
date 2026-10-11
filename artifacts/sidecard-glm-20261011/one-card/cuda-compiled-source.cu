// SPDX-License-Identifier: Apache-2.0
// Exact block32 activation/projection contract with coalesced INT8 DP4A.
// INT8 and lossless four-bit word planes share the same arithmetic contract.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

constexpr int kThreads = 128;
constexpr int kBlock = 32;
constexpr float kHalfMax = 65504.0F;
constexpr int kDotLanes = 8;
constexpr int kDotValuesPerLane = 4;
constexpr int kMaxRoutes = 5120;
constexpr int kCubeN = 128;
constexpr int kCubeK = 256;
constexpr int kCubeInnerK = 64;
constexpr int kMaxPackedExperts = 144;

// Translate the authenticated native Cube bytes directly into CUDA word
// planes. Signed integer codes and FP16 scales are never requantized.
__global__ void pack_cube_nibbles(const unsigned char* input, unsigned short* output, int experts, int n, int k,
                                  int bits) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t words = static_cast<size_t>(experts) * n * k / kDotValuesPerLane;
  if (index >= words) return;
  int channel = index % n;
  size_t remainder = index / n;
  int word = remainder % (kBlock / kDotValuesPerLane);
  remainder /= kBlock / kDotValuesPerLane;
  int group = remainder % (k / kBlock);
  int expert = remainder / (k / kBlock);
  int physical_channel = (channel % kCubeN) / 2 + (channel % 2) * (kCubeN / 2);
  unsigned int packed = 0;
#pragma unroll
  for (int lane = 0; lane < kDotValuesPerLane; ++lane) {
    int column = group * kBlock + word * kDotValuesPerLane + lane;
    size_t tile = (static_cast<size_t>(expert) * (n / kCubeN) + channel / kCubeN) * (k / kCubeK) + column / kCubeK;
    size_t base = tile * kCubeN * kCubeK * bits / 8;
    int position = ((column % kCubeK) / kCubeInnerK * kCubeN + physical_channel) * kCubeInnerK + column % kCubeInnerK;
    unsigned int code;
    if (bits == 4) {
      code = (input[base + position / 2] >> (4 * (position % 2))) & 15U;
    } else {
      int plane_values = kCubeN * kCubeK / (bits == 3 ? 8 : 4);
      int field = position / plane_values;
      size_t offset = base + position % plane_values;
      unsigned int value = input[offset];
      if (bits == 3)
        value |= static_cast<unsigned int>(input[offset + plane_values]) << 8 |
                 static_cast<unsigned int>(input[offset + 2 * plane_values]) << 16;
      code = (value >> (bits * field)) & ((1U << bits) - 1);
    }
    int sign = 1 << (bits - 1);
    int signed_code = static_cast<int>(code ^ sign) - sign;
    packed |= static_cast<unsigned int>(signed_code + 8) << (4 * lane);
  }
  output[index] = static_cast<unsigned short>(packed);
}

extern "C" int sidecard_oracle_pack_cube(const void* input, void* output, int experts, int n, int k, int bits) {
  if (experts < 1 || experts > kMaxPackedExperts || n < kCubeN || n % kCubeN || k < kCubeK || k % kCubeK ||
      (bits != 2 && bits != 3 && bits != 4))
    return static_cast<int>(cudaErrorInvalidValue);
  size_t words = static_cast<size_t>(experts) * n * k / kDotValuesPerLane;
  pack_cube_nibbles<<<(words + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const unsigned char*>(input), static_cast<unsigned short*>(output), experts, n, k, bits);
  return static_cast<int>(cudaGetLastError());
}

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
             static_cast<int>(weights[(static_cast<size_t>(expert) * n + channel) * k + group * kBlock + lane]);
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
    int weight = *reinterpret_cast<const int*>(weights + (static_cast<size_t>(expert) * n + channel) * k + offset);
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

// K32 word planes make neighboring output channels contiguous. Every lane
// retains the original ordered FP32 group accumulation, without shuffles.
__global__ void project_words(const signed char* q, const float* xs, const signed char* weights, const __half* ws,
                              const int* expert_ids, __half* out, int rows, int n, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * n) return;
  int row = index / n, channel = index % n, expert = expert_ids[row];
  const int* words = reinterpret_cast<const int*>(weights);
  const int* activations = reinterpret_cast<const int*>(q);
  float result = 0.0F;
  for (int group = 0; group < k / kBlock; ++group) {
    int dot = 0;
#pragma unroll
    for (int word = 0; word < kBlock / kDotValuesPerLane; ++word) {
      int activation = activations[(row * (k / kBlock) + group) * (kBlock / kDotValuesPerLane) + word];
      int weight = words[((expert * (k / kBlock) + group) * (kBlock / kDotValuesPerLane) + word) * n + channel];
      dot = __dp4a(activation, weight, dot);
    }
    float factor = __fmul_rn(xs[row * (k / kBlock) + group],
                             __half2float(ws[(expert * (n / kBlock) + channel / kBlock) * (k / kBlock) + group]));
    result = __fadd_rn(result, __fmul_rn(static_cast<float>(dot), factor));
  }
  out[index] = __float2half_rn(result);
}

__global__ void project_nibbles(const signed char* q, const float* xs, const signed char* weights, const __half* ws,
                                const int* expert_ids, __half* out, int rows, int n, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * n) return;
  int row = index / n, channel = index % n, expert = expert_ids[row];
  const unsigned short* words = reinterpret_cast<const unsigned short*>(weights);
  const int* activations = reinterpret_cast<const int*>(q);
  float result = 0.0F;
  for (int group = 0; group < k / kBlock; ++group) {
    int dot = 0;
#pragma unroll
    for (int word = 0; word < kBlock / kDotValuesPerLane; ++word) {
      int activation = activations[(row * (k / kBlock) + group) * (kBlock / kDotValuesPerLane) + word];
      unsigned int packed =
          words[((expert * (k / kBlock) + group) * (kBlock / kDotValuesPerLane) + word) * n + channel];
      unsigned int expanded =
          (packed & 15U) | ((packed & 240U) << 4) | ((packed & 3840U) << 8) | ((packed & 61440U) << 12);
      int weight = static_cast<int>(__vsub4(expanded, 0x08080808U));
      dot = __dp4a(activation, weight, dot);
    }
    float factor = __fmul_rn(xs[row * (k / kBlock) + group],
                             __half2float(ws[(expert * (n / kBlock) + channel / kBlock) * (k / kBlock) + group]));
    result = __fadd_rn(result, __fmul_rn(static_cast<float>(dot), factor));
  }
  out[index] = __float2half_rn(result);
}

extern "C" int sidecard_oracle_forward_packed(const void* input, const void* ids, const void* gate, const void* gs,
                                              const void* down, const void* ds, void* input_q, void* input_scales,
                                              void* gate_up, void* hidden, void* hidden_q, void* hidden_scales,
                                              void* output, int rows, int width, int intermediate, int bits) {
  if (rows < 1 || rows > kMaxRoutes || width != 4096 || intermediate != 2048 || (bits != 4 && bits != 8))
    return static_cast<int>(cudaErrorInvalidValue);
  quantize<<<(rows * width / kBlock + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(input), static_cast<signed char*>(input_q), static_cast<float*>(input_scales), rows,
      width, bits);
  project_nibbles<<<(rows * 2 * intermediate + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const signed char*>(input_q), static_cast<const float*>(input_scales),
      static_cast<const signed char*>(gate), static_cast<const __half*>(gs), static_cast<const int*>(ids),
      static_cast<__half*>(gate_up), rows, 2 * intermediate, width);
  swiglu<<<(rows * intermediate + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(gate_up), static_cast<__half*>(hidden), rows, intermediate);
  quantize<<<(rows * intermediate / kBlock + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(hidden), static_cast<signed char*>(hidden_q), static_cast<float*>(hidden_scales), rows,
      intermediate, bits);
  project_nibbles<<<(rows * width + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const signed char*>(hidden_q), static_cast<const float*>(hidden_scales),
      static_cast<const signed char*>(down), static_cast<const __half*>(ds), static_cast<const int*>(ids),
      static_cast<__half*>(output), rows, width, intermediate);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int sidecard_oracle_forward_transposed(const void* input, const void* ids, const void* gate, const void* gs,
                                                  const void* down, const void* ds, void* input_q, void* input_scales,
                                                  void* gate_up, void* hidden, void* hidden_q, void* hidden_scales,
                                                  void* output, int rows, int width, int intermediate, int bits) {
  if (rows < 1 || rows > kMaxRoutes || width != 4096 || intermediate != 2048 || (bits != 4 && bits != 8))
    return static_cast<int>(cudaErrorInvalidValue);
  quantize<<<(rows * width / kBlock + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(input), static_cast<signed char*>(input_q), static_cast<float*>(input_scales), rows,
      width, bits);
  project_words<<<(rows * 2 * intermediate + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const signed char*>(input_q), static_cast<const float*>(input_scales),
      static_cast<const signed char*>(gate), static_cast<const __half*>(gs), static_cast<const int*>(ids),
      static_cast<__half*>(gate_up), rows, 2 * intermediate, width);
  swiglu<<<(rows * intermediate + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(gate_up), static_cast<__half*>(hidden), rows, intermediate);
  quantize<<<(rows * intermediate / kBlock + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(hidden), static_cast<signed char*>(hidden_q), static_cast<float*>(hidden_scales), rows,
      intermediate, bits);
  project_words<<<(rows * width + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const signed char*>(hidden_q), static_cast<const float*>(hidden_scales),
      static_cast<const signed char*>(down), static_cast<const __half*>(ds), static_cast<const int*>(ids),
      static_cast<__half*>(output), rows, width, intermediate);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int sidecard_oracle_forward(const void* input, const void* ids, const void* gate, const void* gs,
                                       const void* down, const void* ds, void* input_q, void* input_scales,
                                       void* gate_up, void* hidden, void* hidden_q, void* hidden_scales, void* output,
                                       int rows, int width, int intermediate, int bits) {
  if (rows < 1 || rows > kMaxRoutes || width != 4096 || intermediate != 2048 || (bits != 4 && bits != 8))
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

__global__ void reduce_local(const __half* raw, const int* row_map, const float* weights, float* output, int rows,
                             int width, int routes) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * width) return;
  int row = index / width, channel = index % width;
  float total = 0.0F;
  for (int slot = 0; slot < 8; ++slot) {
    int route = row_map[row * 8 + slot];
    float value = route >= 0 && route < routes ? __half2float(raw[route * width + channel]) : 0.0F;
    total = __fadd_rn(total, __fmul_rn(weights[row * 8 + slot], value));
  }
  output[index] = total;
}

extern "C" int sidecard_oracle_reduce_local(const void* raw, const void* row_map, const void* weights, void* output,
                                            int rows, int width, int routes) {
  if (rows < 1 || rows > 640 || width != 4096 || routes < 1 || routes > kMaxRoutes)
    return static_cast<int>(cudaErrorInvalidValue);
  reduce_local<<<(rows * width + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const __half*>(raw), static_cast<const int*>(row_map), static_cast<const float*>(weights),
      static_cast<float*>(output), rows, width, routes);
  return static_cast<int>(cudaGetLastError());
}
