// SPDX-License-Identifier: Apache-2.0
// Synchronous CPU arithmetic/layout fixture. This does not model device timing.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#define __aicore__
#define __gm__
#define __global__
#define KERNEL_TASK_TYPE_DEFAULT(x)
#define ASCENDC_ASSERT(condition, detail)                           \
  do {                                                              \
    if (!(condition)) throw std::runtime_error("native assertion"); \
  } while (0)
using GM_ADDR = uint8_t*;
using half = _Float16;
namespace AscendC {
struct int4b_t {
  uint8_t packed;
};
enum class HardEvent {
  M_MTE1,
  V_MTE2,
  MTE2_V,
  MTE2_MTE3,
  V_MTE3,
  MTE3_MTE1,
  MTE3_MTE2,
  MTE1_MTE3,
  MTE1_M,
  M_V,
  V_M,
  MTE1_MTE2,
  MTE2_MTE1,
  MTE3_V,
  S_MTE2,
  MTE2_S
};
enum class TPosition { VECCALC, A1, B1, A2, B2, CO1 };
enum class RoundMode { CAST_NONE };
enum class BlockMode { BLOCK_MODE_MATRIX };
constexpr int PIPE_V = 1;
inline uint32_t blockIndex = 0;
inline std::array<std::array<bool, 8>, 16> pending{};
inline std::array<uint64_t, 6> used{};
inline std::array<uint64_t, 6> peaks{};
struct GlobalRange {
  uint8_t* begin;
  size_t bytes;
};
inline std::vector<GlobalRange> globals;
inline uint32_t GetBlockIdx() { return blockIndex; }
inline uint32_t GetBlockNum() { return 8; }
inline void InitSocState() {}
template <int>
void PipeBarrier() {}
template <HardEvent E>
void SetFlag(uint32_t slot) {
  if (slot >= 8 || pending[static_cast<size_t>(E)][slot]) throw std::runtime_error("event reuse/out of range");
  pending[static_cast<size_t>(E)][slot] = true;
}
template <HardEvent E>
void WaitFlag(uint32_t slot) {
  if (slot >= 8 || !pending[static_cast<size_t>(E)][slot]) throw std::runtime_error("unmatched event");
  pending[static_cast<size_t>(E)][slot] = false;
}
template <class T>
struct Tensor {
  T* data{};
  uint8_t* begin{};
  size_t bytes{};
  int space{-1};
  void Check(size_t count) const {
    auto p = reinterpret_cast<uint8_t*>(data);
    if (!begin || p < begin || p > begin + bytes || count * sizeof(T) > static_cast<size_t>(begin + bytes - p))
      throw std::runtime_error("tensor byte overrun");
  }
  Tensor operator[](int64_t offset) const {
    if (offset < 0) throw std::runtime_error("negative tensor offset");
    auto result = *this;
    result.data += offset;
    result.Check(0);
    return result;
  }
  template <class U>
  Tensor<U> ReinterpretCast() const {
    return {reinterpret_cast<U*>(data), begin, bytes, space};
  }
  void SetGlobalBuffer(T* pointer) {
    data = pointer;
    begin = reinterpret_cast<uint8_t*>(pointer);
    space = -1;
    auto found = std::find_if(globals.begin(), globals.end(), [&](auto range) { return range.begin == begin; });
    if (found == globals.end()) throw std::runtime_error("unregistered GM tensor");
    bytes = found->bytes;
  }
  void SetValue(int64_t offset, T value) {
    auto view = (*this)[offset];
    view.Check(1);
    view.data[0] = value;
  }
  T GetValue(int64_t offset) const {
    auto view = (*this)[offset];
    view.Check(1);
    return view.data[0];
  }
};
template <class T>
using LocalTensor = Tensor<T>;
template <class T>
using GlobalTensor = Tensor<T>;
template <TPosition P>
struct TBuf {
  std::vector<uint8_t> memory;
  size_t capacity{};
  uint8_t* aligned{};
  template <class T>
  Tensor<T> Get() {
    return {reinterpret_cast<T*>(aligned), aligned, capacity, static_cast<int>(P)};
  }
};
struct TPipe {
  template <TPosition P>
  void InitBuffer(TBuf<P>& buffer, uint32_t bytes) {
    auto p = static_cast<size_t>(P);
    used[p] += bytes;
    const size_t totalL1 = used[1] + used[2];
    if (used[0] > 253952 || totalL1 > 1048576 || used[3] > 65536 || used[4] > 65536 || used[5] > 262144)
      throw std::runtime_error("local arena overcommit");
    peaks[p] = std::max(peaks[p], used[p]);
    buffer.memory.assign(bytes + 32, 0xA7);
    buffer.capacity = bytes;
    buffer.aligned =
        reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(buffer.memory.data()) + 31) & ~uintptr_t(31));
  }
};
struct DataCopyParams {
  uint16_t blockCount, blockLen, srcStride, dstStride;
};
struct DataCopyEnhancedParams {
  BlockMode blockMode{};
};
struct LoadData2DParams {
  uint16_t repeatTimes{}, srcStride{};
  bool ifTranspose{};
};
struct MmadParams {
  uint16_t m{}, n{}, k{};
  bool cmatrixInitVal{};
};
template <class T>
void Duplicate(Tensor<T> dst, T value, uint32_t count) {
  dst.Check(count);
  std::fill_n(dst.data, count, value);
}
template <class T>
void DataCopy(Tensor<T> dst, Tensor<T> src, uint32_t count) {
  dst.Check(count);
  src.Check(count);
  std::copy_n(src.data, count, dst.data);
}
template <class T>
void DataCopy(Tensor<T> dst, Tensor<T> src, DataCopyParams p) {
  for (uint32_t block = 0; block < p.blockCount; ++block) {
    size_t destination = block * (p.blockLen + p.dstStride) * 32 / sizeof(T);
    size_t source = block * (p.blockLen + p.srcStride) * 32 / sizeof(T);
    auto d = dst[destination], s = src[source];
    d.Check(p.blockLen * 32 / sizeof(T));
    s.Check(p.blockLen * 32 / sizeof(T));
    std::memcpy(d.data, s.data, p.blockLen * 32);
  }
}
inline void DataCopy(Tensor<int32_t> dst, Tensor<int32_t> src, DataCopyParams p, DataCopyEnhancedParams) {
  // CO1 matrix readback counts 16x16 INT32 matrices, not 32-byte UB blocks.
  if (p.srcStride || p.dstStride) throw std::runtime_error("unsupported matrix copy stride");
  DataCopy(dst, src, p.blockCount * p.blockLen * 16 * 16);
}
inline void LoadData(Tensor<int4b_t> dst, Tensor<int4b_t> src, LoadData2DParams p) {
  if (p.ifTranspose) throw std::runtime_error("unexpected transpose");
  for (uint32_t repeat = 0; repeat < p.repeatTimes; ++repeat)
    DataCopy(dst[repeat * 512], src[repeat * p.srcStride * 512], 512);
}
inline int Nibble(const int4b_t* p, uint32_t byte, uint32_t halfByte) {
  uint8_t packed = p[byte].packed;
  int value = (packed >> (halfByte * 4)) & 15;
  return value >= 8 ? value - 16 : value;
}
inline void Mmad(Tensor<int32_t> dst, Tensor<int4b_t> a, Tensor<int4b_t> b, MmadParams p) {
  if (!((p.m == 32 && p.n == 128) || (p.m == 64 && p.n == 160)) || p.k != 128 || !p.cmatrixInitVal)
    throw std::runtime_error("unsupported CPU MMAD geometry");
  dst.Check(p.m * p.n);
  a.Check(p.m * p.k / 2);
  b.Check(p.n * p.k / 2);
  for (uint32_t row = 0; row < p.m; ++row)
    for (uint32_t col = 0; col < p.n; ++col) {
      int32_t dot = 0;
      for (uint32_t k = 0; k < p.k; ++k) {
        uint32_t kb = k / 64, byte = (k % 64) / 2;
        uint32_t ai = (row / 16) * 1024 + kb * 512 + (row % 16) * 32 + byte;
        uint32_t bi = kb * (p.n * 32) + (col / 16) * 512 + (col % 16) * 32 + byte;
        dot += Nibble(a.data, ai, k % 2) * Nibble(b.data, bi, k % 2);
      }
      dst.data[(col / 16) * p.m * 16 + row * 16 + col % 16] = dot;
    }
}
template <class D, class S>
void Cast(Tensor<D> dst, Tensor<S> src, RoundMode, uint32_t count) {
  dst.Check(count);
  src.Check(count);
  for (uint32_t i = 0; i < count; ++i) dst.data[i] = static_cast<D>(src.data[i]);
}
template <class T>
void Muls(Tensor<T> dst, Tensor<T> src, T scale, uint32_t count) {
  dst.Check(count);
  src.Check(count);
  for (uint32_t i = 0; i < count; ++i) dst.data[i] = src.data[i] * scale;
}
template <class F>
void Unary(Tensor<float> dst, Tensor<float> src, uint32_t mask, uint32_t repeats,
           std::initializer_list<uint8_t> strides, F operation) {
  std::vector<uint8_t> s(strides);
  if (s.size() != 4) throw std::runtime_error("unary stride shape");
  for (uint32_t r = 0; r < repeats; ++r)
    for (uint32_t i = 0; i < mask; ++i) {
      uint32_t di = r * s[2] * 8 + (i / 8) * s[0] * 8 + i % 8, si = r * s[3] * 8 + (i / 8) * s[1] * 8 + i % 8;
      dst[di].Check(1);
      src[si].Check(1);
      dst.data[di] = operation(src.data[si]);
    }
}
inline void Muls(Tensor<float> dst, Tensor<float> src, float value, uint32_t mask, uint32_t repeats,
                 std::initializer_list<uint8_t> strides) {
  Unary(dst, src, mask, repeats, strides, [&](float x) { return x * value; });
}
template <class F>
void Binary(Tensor<float> dst, Tensor<float> a, Tensor<float> b, uint32_t mask, uint32_t repeats,
            std::initializer_list<uint8_t> strides, F operation) {
  std::vector<uint8_t> s(strides);
  if (s.size() != 6) throw std::runtime_error("binary stride shape");
  for (uint32_t r = 0; r < repeats; ++r)
    for (uint32_t i = 0; i < mask; ++i) {
      uint32_t di = r * s[3] * 8 + (i / 8) * s[0] * 8 + i % 8;
      uint32_t ai = r * s[4] * 8 + (i / 8) * s[1] * 8 + i % 8, bi = r * s[5] * 8 + (i / 8) * s[2] * 8 + i % 8;
      dst[di].Check(1);
      a[ai].Check(1);
      b[bi].Check(1);
      dst.data[di] = operation(a.data[ai], b.data[bi]);
    }
}
inline void Add(Tensor<float> d, Tensor<float> a, Tensor<float> b, uint32_t mask, uint32_t repeats,
                std::initializer_list<uint8_t> s) {
  Binary(d, a, b, mask, repeats, s, [](float x, float y) { return x + y; });
}
inline void Sub(Tensor<float> d, Tensor<float> a, Tensor<float> b, uint32_t mask, uint32_t repeats,
                std::initializer_list<uint8_t> s) {
  Binary(d, a, b, mask, repeats, s, [](float x, float y) { return x - y; });
}
inline void Mul(Tensor<float> d, Tensor<float> a, Tensor<float> b, uint32_t mask, uint32_t repeats,
                std::initializer_list<uint8_t> s) {
  Binary(d, a, b, mask, repeats, s, [](float x, float y) { return x * y; });
}
}  // namespace AscendC
