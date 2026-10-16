// SPDX-License-Identifier: Apache-2.0
// Isolated instruction/address/mask model, not an SDK or timing simulator.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#define __aicore__
#define __ubuf__
namespace AscendC {
constexpr unsigned PIPE_V = 1, PIPE_ALL = 2;
struct Region {
  uintptr_t begin, end;
  bool writable;
};
struct CastRecord {
  float* dst;
  int32_t* src;
  unsigned rows, db, sb, dr, sr;
};
struct ArithmeticRecord {
  bool add;
  float *dst, *a, *b;
  unsigned repeats, db, ab, bb, dr, ar, br;
};
struct Trace {
  unsigned addresses = 0, slices = 0, masks = 0, normalizations = 0, resets = 0;
  unsigned gathers = 0, preparations = 0, barriers = 0, mask = 64;
  bool normal = true, highFull = true, pending = false;
  int32_t* activeRaw = nullptr;
  unsigned rawAdds = 0;
  std::vector<Region> regions;
  std::vector<CastRecord> casts;
  std::vector<ArithmeticRecord> arithmetic;
  std::unordered_map<int32_t*, bool> rawBusy;
};
inline Trace trace;
inline void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline void Register(void* p, size_t bytes, bool writable) {
  const auto begin = reinterpret_cast<uintptr_t>(p);
  trace.regions.push_back({begin, begin + bytes, writable});
}
inline void Access(const void* p, bool write) {
  const auto address = reinterpret_cast<uintptr_t>(p);
  Check(address % 4 == 0, "unaligned pointer access");
  for (const auto& region : trace.regions)
    if (address >= region.begin && address + 4 <= region.end && (!write || region.writable)) return;
  throw std::runtime_error(write ? "pointer write exceeds permitted region" : "pointer read exceeds region");
}
inline void ClearInstructions() {
  trace.addresses = trace.slices = trace.masks = trace.normalizations = trace.resets = trace.gathers = trace.barriers =
      0;
  trace.casts.clear();
  trace.arithmetic.clear();
}
inline void SetMaskNorm() {
  trace.normal = true;
  ++trace.normalizations;
}
template <typename T>
void SetVectorMask(int32_t count) {
  static_assert(sizeof(T) == 4);
  Check(trace.normal && count >= 1 && count <= 64, "invalid normal vector mask");
  trace.mask = count;
  trace.highFull = false;
  ++trace.masks;
}
inline void ResetMask() {
  trace.mask = 64;
  trace.highFull = true;
  ++trace.resets;
}
template <unsigned Pipe>
void PipeBarrier() {
  static_assert(Pipe == PIPE_V || Pipe == PIPE_ALL);
  ++trace.barriers;
  trace.pending = false;
  if (trace.activeRaw && trace.rawAdds == 2) {
    trace.rawBusy[trace.activeRaw] = false;
    trace.activeRaw = nullptr;
    trace.rawAdds = 0;
  }
}
template <typename T>
struct LocalTensor {
  T* data;
  size_t count;
  uint64_t GetPhyAddr() const {
    ++trace.addresses;
    return reinterpret_cast<uintptr_t>(data);
  }
  LocalTensor operator[](size_t offset) const {
    ++trace.slices;
    Check(offset <= count, "descriptor slicing bounds");
    return {data + offset, count - offset};
  }
};
struct BrcbRepeatParams {
  unsigned dstBlkStride, dstRepStride;
};
inline void Brcb(const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& src, unsigned repeat,
                 BrcbRepeatParams p) {
  Check(!trace.pending && repeat == 4 && p.dstBlkStride == 1 && p.dstRepStride == 8, "Brcb dependency/geometry");
  Check(src.count >= 32 && dst.count >= 256, "Brcb view bounds");
  for (unsigned r = 0; r < 32; ++r)
    for (unsigned lane = 0; lane < 8; ++lane) {
      Access(src.data + r, false);
      Access(dst.data + r * 8 + lane, true);
      dst.data[r * 8 + lane] = src.data[r];
    }
  ++trace.preparations;
  trace.mask = 64;
  trace.highFull = true;
  trace.pending = true;
}
inline void Gather(const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<uint32_t>& indices,
                   uint32_t base, unsigned count) {
  Check(!trace.pending && base == 0 && count == 256, "Gather dependency/geometry");
  Check(dst.count >= count && indices.count >= count, "Gather view bounds");
  for (unsigned i = 0; i < count; ++i) {
    Access(indices.data + i, false);
    Check(indices.data[i] % 4 == 0 && indices.data[i] / 4 < src.count, "Gather byte offset");
    Access(src.data + indices.data[i] / 4, false);
    Access(dst.data + i, true);
    std::memcpy(dst.data + i, src.data + indices.data[i] / 4, 4);
  }
  ++trace.gathers;
  trace.mask = 64;
  trace.highFull = false;
  trace.pending = true;
}
}  // namespace AscendC
inline void vconv_s322f32(float* dst, int32_t* src, uint8_t repeats, uint8_t db, uint8_t sb, uint8_t dr, uint8_t sr) {
  using namespace AscendC;
  Check(trace.normal && trace.mask == 16, "pointer cast lacks 16-lane mask");
  trace.casts.push_back({dst, src, repeats, db, sb, dr, sr});
  for (auto& item : trace.rawBusy) {
    const auto source = reinterpret_cast<uintptr_t>(src), begin = reinterpret_cast<uintptr_t>(item.first);
    if (source >= begin && source < begin + 32768) {
      Check(item.second, "raw product read before readback/reuse");
      Check(!trace.activeRaw || trace.activeRaw == item.first, "cross-slot consumer overlap");
      trace.activeRaw = item.first;
    }
  }
  for (unsigned r = 0; r < repeats; ++r)
    for (unsigned lane = 0; lane < 16; ++lane) {
      auto* d = dst + r * dr * 8 + lane / 8 * db * 8 + lane % 8;
      auto* s = src + r * sr * 8 + lane / 8 * sb * 8 + lane % 8;
      Access(d, true);
      Access(s, false);
      *d = static_cast<float>(*s);
    }
  trace.pending = true;
}
inline void Arithmetic(bool add, float* dst, float* a, float* b, uint8_t repeats, uint8_t db, uint8_t ab, uint8_t bb,
                       uint8_t dr, uint8_t ar, uint8_t br) {
  using namespace AscendC;
  Check(trace.normal && trace.mask == 64, "pointer arithmetic lacks 64-lane mask");
  Check(!trace.pending || (!add && dr == 16), "arithmetic dependency lacks fence");
  trace.arithmetic.push_back({add, dst, a, b, repeats, db, ab, bb, dr, ar, br});
  for (unsigned r = 0; r < repeats; ++r)
    for (unsigned lane = 0; lane < 64; ++lane) {
      auto* d = dst + r * dr * 8 + lane / 8 * db * 8 + lane % 8;
      auto* x = a + r * ar * 8 + lane / 8 * ab * 8 + lane % 8;
      auto* y = b + r * br * 8 + lane / 8 * bb * 8 + lane % 8;
      Access(d, true);
      Access(x, false);
      Access(y, false);
      volatile float rounded = add ? *x + *y : *x * *y;
      *d = rounded;
    }
  if (add && trace.activeRaw) ++trace.rawAdds;
  trace.pending = true;
}
inline void vmul(float* dst, float* a, float* b, uint8_t repeats, uint8_t db, uint8_t ab, uint8_t bb, uint8_t dr,
                 uint8_t ar, uint8_t br) {
  Arithmetic(false, dst, a, b, repeats, db, ab, bb, dr, ar, br);
}
inline void vadd(float* dst, float* a, float* b, uint8_t repeats, uint8_t db, uint8_t ab, uint8_t bb, uint8_t dr,
                 uint8_t ar, uint8_t br) {
  Arithmetic(true, dst, a, b, repeats, db, ab, bb, dr, ar, br);
}
