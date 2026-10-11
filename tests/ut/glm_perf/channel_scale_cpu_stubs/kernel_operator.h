// SPDX-License-Identifier: Apache-2.0
// CPU protocol interpretation; half is float, not device numerics.

#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <type_traits>
#define __aicore__
namespace AscendC {
using half = float;
using GM_ADDR = unsigned char*;
#define __gm__
#define __global__
#define GetBlockIdx() currentCore_
#define GetBlockNum() 8
inline void InitSocState() {}
enum Pipe { PIPE_V, PIPE_ALL };
template <Pipe>
void PipeBarrier() {}
template <class T>
struct LocalTensor {
  T* data;
  T GetValue(unsigned n) const { return data[n]; }
  void SetValue(unsigned n, T value) { data[n] = value; }
  LocalTensor operator[](unsigned n) const { return {data + n}; }
};
template <class T>
void Adds(LocalTensor<T> dst, LocalTensor<T> src, T value, unsigned count) {
  for (unsigned i = 0; i < count; ++i) dst.data[i] = src.data[i] + value;
}
struct BinaryRepeatParams {
  unsigned d, a, b, dr, ar, br;
};
inline void Add(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b, unsigned n) {
  for (unsigned i = 0; i < n; ++i) d.data[i] = a.data[i] + b.data[i];
}
inline void Mul(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b, unsigned n) {
  for (unsigned i = 0; i < n; ++i) d.data[i] = a.data[i] * b.data[i];
}
template <bool multiply>
void Binary(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b, uint64_t mask, unsigned count,
            BinaryRepeatParams p) {
  for (unsigned r = 0; r < count; ++r)
    for (unsigned i = 0; i < mask; ++i) {
      unsigned di = r * p.dr * 8 + (i / 8) * p.d * 8 + i % 8;
      unsigned ai = r * p.ar * 8 + (i / 8) * p.a * 8 + i % 8;
      unsigned bi = r * p.br * 8 + (i / 8) * p.b * 8 + i % 8;
      d.data[di] = multiply ? a.data[ai] * b.data[bi] : a.data[ai] + b.data[bi];
    }
}
inline void Add(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b, uint64_t mask, unsigned count,
                BinaryRepeatParams p) {
  Binary<false>(d, a, b, mask, count, p);
}
inline void Mul(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b, uint64_t mask, unsigned count,
                BinaryRepeatParams p) {
  Binary<true>(d, a, b, mask, count, p);
}
inline const int32_t* auditRanks = nullptr;
inline unsigned long gmRankReads = 0;
template <class T>
struct GlobalTensor {
  T* data = nullptr;
  void SetGlobalBuffer(T* pointer) { data = pointer; }
  GlobalTensor operator[](unsigned long n) const { return {data + n}; }
  T GetValue(unsigned long n) const {
    if constexpr (std::is_same_v<T, int32_t>)
      if (data == auditRanks) ++gmRankReads;
    return data[n];
  }
};
enum class TPosition { VECCALC };
template <TPosition>
struct TBuf {
  std::vector<uint64_t> storage;
  template <class T>
  LocalTensor<T> Get() {
    return {reinterpret_cast<T*>(storage.data())};
  }
};
struct TPipe {
  template <TPosition P>
  void InitBuffer(TBuf<P>& buffer, unsigned bytes) {
    buffer.storage.resize((bytes + 7) / 8);
  }
};
template <class T>
void Duplicate(LocalTensor<T> dst, T value, unsigned count) {
  for (unsigned i = 0; i < count; ++i) dst.data[i] = value;
}
template <class T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, unsigned count) {
  std::memcpy(dst.data, src.data, count * sizeof(T));
}
template <class T>
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, unsigned count) {
  std::memcpy(dst.data, src.data, count * sizeof(T));
}
enum class RoundMode { CAST_NONE };
template <class D, class S>
void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode, unsigned count) {
  for (unsigned i = 0; i < count; ++i) dst.data[i] = src.data[i];
}
struct BrcbParams {
  unsigned a, b;
};
inline void Brcb(LocalTensor<float> dst, LocalTensor<float> src, unsigned, BrcbParams) {
  for (unsigned i = 0; i < 8; ++i)
    for (unsigned lane = 0; lane < 8; ++lane) dst.data[i * 8 + lane] = src.data[i];
}

}  // namespace AscendC
