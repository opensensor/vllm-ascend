// SPDX-License-Identifier: Apache-2.0
// Standalone candidate: retain qualified W4 arithmetic and compact W2/W3.
#include "cuda_oracle.cu"

constexpr int kCompactValues = 16;
constexpr int kMaxCompactExperts = 288;

template <int Bits>
__global__ void pack_cube_compact(const unsigned char* input, unsigned short* output, int experts, int n, int k) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<size_t>(experts) * n * k / kCompactValues) return;
  int channel = index % n;
  size_t remainder = index / n;
  int octet = remainder % (kBlock / kCompactValues);
  remainder /= kBlock / kCompactValues;
  int group = remainder % (k / kBlock);
  int expert = remainder / (k / kBlock);
  int physical_channel = (channel % kCubeN) / 2 + (channel % 2) * (kCubeN / 2);
  unsigned long long packed = 0;
#pragma unroll
  for (int lane = 0; lane < kCompactValues; ++lane) {
    int column = group * kBlock + octet * kCompactValues + lane;
    size_t tile = (static_cast<size_t>(expert) * (n / kCubeN) + channel / kCubeN) * (k / kCubeK) + column / kCubeK;
    size_t base = tile * kCubeN * kCubeK * Bits / 8;
    int position = ((column % kCubeK) / kCubeInnerK * kCubeN + physical_channel) * kCubeInnerK + column % kCubeInnerK;
    constexpr int plane_values = kCubeN * kCubeK / (Bits == 3 ? 8 : 4);
    int field = position / plane_values;
    size_t offset = base + position % plane_values;
    unsigned int value = input[offset];
    if constexpr (Bits == 3)
      value |= static_cast<unsigned int>(input[offset + plane_values]) << 8 |
               static_cast<unsigned int>(input[offset + 2 * plane_values]) << 16;
    unsigned int code = (value >> (Bits * field)) & ((1U << Bits) - 1);
    packed |= static_cast<unsigned long long>(code ^ (1U << (Bits - 1))) << (Bits * lane);
  }
  if constexpr (Bits == 2) {
    reinterpret_cast<unsigned int*>(output)[index] = static_cast<unsigned int>(packed);
  } else {
#pragma unroll
    for (int word = 0; word < Bits; ++word)
      output[((index / n) * Bits + word) * n + channel] = static_cast<unsigned short>(packed >> (16 * word));
  }
}

extern "C" int sidecard_oracle_pack_cube_compact(const void* input, void* output, int experts, int n, int k, int bits) {
  if (experts < 1 || experts > kMaxCompactExperts || n < kCubeN || n % kCubeN || k < kCubeK || k % kCubeK ||
      (bits != 2 && bits != 3 && bits != 4))
    return static_cast<int>(cudaErrorInvalidValue);
  if (bits == 4) {
    size_t words = static_cast<size_t>(experts) * n * k / kDotValuesPerLane;
    pack_cube_nibbles<<<(words + kThreads - 1) / kThreads, kThreads>>>(
        static_cast<const unsigned char*>(input), static_cast<unsigned short*>(output), experts, n, k, bits);
    return static_cast<int>(cudaGetLastError());
  }
  size_t octets = static_cast<size_t>(experts) * n * k / kCompactValues;
  if (bits == 2)
    pack_cube_compact<2><<<(octets + kThreads - 1) / kThreads, kThreads>>>(
        static_cast<const unsigned char*>(input), static_cast<unsigned short*>(output), experts, n, k);
  else
    pack_cube_compact<3><<<(octets + kThreads - 1) / kThreads, kThreads>>>(
        static_cast<const unsigned char*>(input), static_cast<unsigned short*>(output), experts, n, k);
  return static_cast<int>(cudaGetLastError());
}

// Sixteen codes are stored in `Bits` aligned word planes across N channels.
// Signed codes are biased by 2^(bits-1), without changing quantization.
template <int Bits>
__device__ uint2 compact_pair(const unsigned short* weights, size_t pair, int channel, int n) {
  if constexpr (Bits == 2) {
    unsigned int packed = reinterpret_cast<const unsigned int*>(weights)[pair * n + channel];
    return make_uint2(packed & 65535U, packed >> 16);
  } else {
    unsigned int first = weights[(pair * Bits) * n + channel];
    unsigned int middle = weights[(pair * Bits + 1) * n + channel];
    unsigned int last = weights[(pair * Bits + 2) * n + channel];
    return make_uint2(first | ((middle & 255U) << 16), (middle >> 8) | (last << 8));
  }
}

template <int Bits>
__device__ unsigned int expand_compact_four(unsigned int packed) {
  // Spread two adjacent code pairs, then spread each pair into byte lanes.
  // This replaces eight dynamic shifts/masks with fixed bit permutations.
  if constexpr (Bits == 2) {
    packed &= 0xffU;
    packed = (packed | (packed << 12)) & 0x000f000fU;
    return (packed | (packed << 6)) & 0x03030303U;
  } else {
    packed &= 0xfffU;
    packed = (packed | (packed << 10)) & 0x003f003fU;
    return (packed | (packed << 5)) & 0x07070707U;
  }
}

template <int Bits>
__global__ void project_compact(const signed char* q, const float* xs, const unsigned short* weights, const __half* ws,
                                const int* ids, __half* out, int rows, int n, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * n) return;
  int row = index / n, channel = index % n, expert = ids[row];
  const int* activations = reinterpret_cast<const int*>(q);
  const unsigned int bias = (1U << (Bits - 1)) * 0x01010101U;
  float result = 0;
  for (int group = 0; group < k / kBlock; ++group) {
    int dot = 0;
#pragma unroll
    for (int pair = 0; pair < kBlock / kCompactValues; ++pair) {
      size_t position = (static_cast<size_t>(expert) * (k / kBlock) + group) * (kBlock / kCompactValues) + pair;
      uint2 codes = compact_pair<Bits>(weights, position, channel, n);
#pragma unroll
      for (int octet = 0; octet < 2; ++octet) {
        unsigned int packed = octet == 0 ? codes.x : codes.y;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          unsigned int expanded = expand_compact_four<Bits>(packed >> (Bits * half * kDotValuesPerLane));
          int activation =
              activations[(row * (k / kBlock) + group) * (kBlock / kDotValuesPerLane) + pair * 4 + octet * 2 + half];
          dot = __dp4a(activation, static_cast<int>(__vsub4(expanded, bias)), dot);
        }
      }
    }
    float factor = __fmul_rn(xs[row * (k / kBlock) + group],
                             __half2float(ws[(expert * (n / kBlock) + channel / kBlock) * (k / kBlock) + group]));
    result = __fadd_rn(result, __fmul_rn(static_cast<float>(dot), factor));
  }
  out[index] = __float2half_rn(result);
}

void dispatch_compact_project(const signed char* q, const float* xs, const unsigned short* weights, const __half* ws,
                              const int* ids, __half* out, int rows, int n, int k, int bits, cudaStream_t stream) {
  if (bits == 2)
    project_compact<2><<<(rows * n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(q, xs, weights, ws, ids, out, rows, n, k);
  else
    project_compact<3><<<(rows * n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(q, xs, weights, ws, ids, out, rows, n, k);
}

extern "C" int sidecard_oracle_forward_compact_stream(const void* input, const void* ids, const void* gate, const void* gs,
                                               const void* down, const void* ds, void* input_q, void* input_scales,
                                               void* gate_up, void* hidden, void* hidden_q, void* hidden_scales,
                                               void* output, int rows, int width, int intermediate, int activation_bits,
                                               int weight_bits, void* stream_handle) {
  if (rows < 1 || rows > kMaxRoutes || width != 4096 || intermediate != 2048 ||
      (activation_bits != 4 && activation_bits != 8) || (weight_bits != 2 && weight_bits != 3))
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
  quantize<<<(rows * width / kBlock + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const __half*>(input), static_cast<signed char*>(input_q), static_cast<float*>(input_scales), rows,
      width, activation_bits);
  dispatch_compact_project(static_cast<const signed char*>(input_q), static_cast<const float*>(input_scales),
                           static_cast<const unsigned short*>(gate), static_cast<const __half*>(gs),
                           static_cast<const int*>(ids), static_cast<__half*>(gate_up), rows, 2 * intermediate, width,
                           weight_bits, stream);
  swiglu<<<(rows * intermediate + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const __half*>(gate_up), static_cast<__half*>(hidden), rows, intermediate);
  quantize<<<(rows * intermediate / kBlock + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      static_cast<const __half*>(hidden), static_cast<signed char*>(hidden_q), static_cast<float*>(hidden_scales), rows,
      intermediate, activation_bits);
  dispatch_compact_project(static_cast<const signed char*>(hidden_q), static_cast<const float*>(hidden_scales),
                           static_cast<const unsigned short*>(down), static_cast<const __half*>(ds),
                           static_cast<const int*>(ids), static_cast<__half*>(output), rows, width, intermediate,
                           weight_bits, stream);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int sidecard_oracle_forward_compact(const void* input, const void* ids, const void* gate, const void* gs,
                                               const void* down, const void* ds, void* input_q, void* input_scales,
                                               void* gate_up, void* hidden, void* hidden_q, void* hidden_scales,
                                               void* output, int rows, int width, int intermediate, int activation_bits,
                                               int weight_bits) {
  return sidecard_oracle_forward_compact_stream(input, ids, gate, gs, down, ds, input_q, input_scales,
                                                gate_up, hidden, hidden_q, hidden_scales, output,
                                                rows, width, intermediate, activation_bits, weight_bits, nullptr);
}
