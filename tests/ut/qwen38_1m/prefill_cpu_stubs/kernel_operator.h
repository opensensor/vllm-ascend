// SPDX-License-Identifier: Apache-2.0
// Bounds-aware CPU model of the small vector/DMA subset used by these probes.
// Events are deliberately no-ops: passing here does not qualify device order.
#ifndef QWEN_PREFILL_CPU_KERNEL_OPERATOR_H
#define QWEN_PREFILL_CPU_KERNEL_OPERATOR_H
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
#define __global__
#define __gm__
using GM_ADDR = void*;
using half = _Float16;
namespace AscendC {
inline int64_t blockIndex = 0, blockCount = 8;
inline uint32_t allocatedUB = 0;
inline void InitSocState() { allocatedUB = 0; }
inline int64_t GetBlockIdx() { return blockIndex; }
inline int64_t GetBlockNum() { return blockCount; }
enum class TPosition { VECCALC };
enum class HardEvent { MTE2_V, MTE2_S, V_MTE2, S_V, V_S, V_MTE3, MTE3_V, MTE3_MTE2, MTE2_MTE3 };
enum class RoundMode { CAST_NONE };
enum Pipe { PIPE_V, PIPE_ALL };
constexpr int EVENT_ID0 = 0;
template <HardEvent>
void SetFlag(int) {}
template <HardEvent>
void WaitFlag(int) {}
template <Pipe>
void PipeBarrier() {}
template <typename T>
struct LocalTensor {
  T* p;
  size_t capacity;
  LocalTensor operator[](size_t index) const {
    assert(index <= capacity);
    return {p + index, capacity - index};
  }
  T GetValue(size_t index) const {
    assert(index < capacity);
    return p[index];
  }
  void SetValue(size_t index, T value) {
    assert(index < capacity);
    p[index] = value;
  }
};
template <typename T>
struct GlobalTensor {
  T* p;
  void SetGlobalBuffer(T* pointer) { p = pointer; }
  GlobalTensor operator[](size_t index) const { return {p + index}; }
  T GetValue(size_t index) const { return p[index]; }
};
template <TPosition>
struct TBuf {
  std::vector<uint64_t> data;
  template <typename T>
  LocalTensor<T> Get() {
    return {reinterpret_cast<T*>(data.data()), data.size() * 8 / sizeof(T)};
  }
};
struct TPipe {
  template <TPosition P>
  void InitBuffer(TBuf<P>& b, size_t bytes) {
    allocatedUB += bytes;
    assert(allocatedUB <= 96 * 1024);
    b.data.resize((bytes + 7) / 8);
  }
};
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, size_t count) {
  assert(count <= dst.capacity && count * sizeof(T) % 32 == 0);
  std::memcpy(dst.p, src.p, count * sizeof(T));
}
template <typename T>
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, size_t count) {
  assert(count <= src.capacity && count * sizeof(T) % 32 == 0);
  std::memcpy(dst.p, src.p, count * sizeof(T));
}
template <typename T, typename U>
void Cast(LocalTensor<T> dst, LocalTensor<U> src, RoundMode, size_t count) {
  assert(count <= dst.capacity && count <= src.capacity);
  for (size_t i = 0; i < count; ++i) dst.SetValue(i, static_cast<T>(src.GetValue(i)));
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> src, float value, size_t count) {
  assert(count <= dst.capacity && count <= src.capacity);
  for (size_t i = 0; i < count; ++i) dst.SetValue(i, src.GetValue(i) * value);
}
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, size_t count) {
  assert(count <= dst.capacity && count <= a.capacity && count <= b.capacity);
  for (size_t i = 0; i < count; ++i) dst.SetValue(i, a.GetValue(i) * b.GetValue(i));
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, size_t count) {
  assert(count <= dst.capacity && count <= a.capacity && count <= b.capacity);
  for (size_t i = 0; i < count; ++i) dst.SetValue(i, a.GetValue(i) + b.GetValue(i));
}
inline void Exp(LocalTensor<float> dst, LocalTensor<float> src, size_t count) {
  assert(count <= dst.capacity && count <= src.capacity);
  for (size_t i = 0; i < count; ++i) dst.SetValue(i, std::exp(src.GetValue(i)));
}
}  // namespace AscendC
#endif
