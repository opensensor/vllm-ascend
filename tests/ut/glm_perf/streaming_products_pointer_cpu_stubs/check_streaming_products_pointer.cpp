// SPDX-License-Identifier: Apache-2.0
// Actual pointer helper; independent scalar reference and bounded owner schedule.
#include "glm_streaming_products_pointer.h"
#include <array>
#include <cmath>
#include <iostream>
#include <string>
using namespace AscendC;
using namespace GlmStreamingContract;
constexpr uint32_t CANARY = 0x7fcabcde;
uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}
float Value(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}
float Product(float a, float b) {
  volatile float value = a * b;
  return value;
}
float Sum(float a, float b) {
  volatile float value = a + b;
  return value;
}
void Invoke(LocalTensor<int32_t> raw, LocalTensor<float> converted, LocalTensor<float> factor,
            LocalTensor<float> broadcast, LocalTensor<float> accumulator, LocalTensor<float> sx0,
            LocalTensor<float> sx1, LocalTensor<float> weight0, LocalTensor<float> weight1,
            LocalTensor<uint32_t> indices, unsigned count, unsigned physical, bool groupMajor) {
  GlmStreamingProductsPointer::ConsumeDirect(raw, converted, factor, broadcast, accumulator, sx0, sx1, weight0, weight1,
                                             indices, count, physical, groupMajor);
}
void Run(unsigned firstCount, unsigned stride, unsigned k) {
  Check(
      firstCount >= 1 && firstCount <= 31 && (stride == 1 || stride == 4 || stride == 128) && (k == 2048 || k == 4096),
      "arguments");
  trace = {};
  std::vector<int32_t> raw(PRODUCT_STORAGE_BYTES / 4 + 32, 0x12345678);
  std::vector<float> converted(ELEMENTS + 32, Value(CANARY)), mask(MASK_BYTES / 4 + 32, Value(CANARY));
  std::vector<float> accumulator(ELEMENTS + 32, Value(CANARY)), expected(accumulator);
  std::vector<float> sx0(ROWS * 128 + 32, Value(CANARY)), sx1(sx0), weight0(COLUMNS), weight1(COLUMNS);
  Register(raw.data(), raw.size() * 4, false);
  Register(converted.data(), ELEMENTS * 4, true);
  Register(accumulator.data(), ELEMENTS * 4, true);
  Register(mask.data(), mask.size() * 4, false);
  Register(mask.data(), ELEMENTS * 4, true);
  Register(mask.data() + BROADCAST_OFFSET / 4, ROWS * 8 * 4, true);
  Register(mask.data() + SCALE_GATHER_INDEX_OFFSET / 4, SCALE_GATHER_INDEX_BYTES, true);
  Register(sx0.data(), sx0.size() * 4, false);
  Register(sx1.data(), sx1.size() * 4, false);
  Register(weight0.data(), weight0.size() * 4, false);
  Register(weight1.data(), weight1.size() * 4, false);
  for (unsigned slot = 0; slot < 2; ++slot) trace.rawBusy[raw.data() + slot * PRODUCT_SLOT_BYTES / 4] = false;
  auto* originalIndices = reinterpret_cast<uint32_t*>(mask.data() + RETAINED_INDEX_OFFSET / 4);
  auto* expanded = reinterpret_cast<uint32_t*>(mask.data() + SCALE_GATHER_INDEX_OFFSET / 4);
  bool release[2]{};
  unsigned pairs = 0, batches = 0;
  const unsigned densities[] = {firstCount, 31, 1, 4, 16, 17, 0, firstCount};
  for (unsigned count : densities) {
    if (!count) {
      Check(!trace.activeRaw, "empty batch retained raw reader");
      continue;
    }
    for (unsigned row = 0; row < ROWS; ++row)
      originalIndices[row] = (stride == 1 || row < count ? row : 0) * stride * 4;
    GlmStreamingScaleGather::Prepare({expanded, 256}, {originalIndices, ROWS});
    ++batches;
    const auto retained = mask;
    std::fill_n(accumulator.data(), count * COLUMNS, 0.0f);
    expected = accumulator;
    const unsigned physical = count <= 16 ? 32 : 64;
    for (unsigned pair = 0; pair < k / 64; ++pair) {
      const unsigned slot = pair % 2, offset = slot * PRODUCT_SLOT_BYTES / 4;
      if (release[slot]) release[slot] = false;  // Owner consumes V_M before another Cube issue.
      Check(!trace.rawBusy[raw.data() + offset], "CO1 release permitted live raw overwrite");
      std::vector<int32_t> co1(physical * COLUMNS, 0x23456789);
      const auto beforeRaw = raw;
      for (unsigned part = 0; part < 2; ++part) {
        auto& sx = part ? sx1 : sx0;
        auto& weights = part ? weight1 : weight0;
        for (unsigned row = 0; row < ROWS; ++row) {
          const float values[] = {1.0f, -0.0f, 0.0f, 0.333333343f, 1.000000119f, -0.375f, 0.000000059604645f, 1e12f};
          sx[row * stride] = row < count ? values[(row + pair + part * 3) % 8] : Value(0x7fa12345 + row);
        }
        for (unsigned col = 0; col < COLUMNS; ++col) {
          const float values[] = {1.0f, -1.0f, 1.000000119f, 0.142857149f, -0.0f, 0.25f};
          weights[col] = values[(col / 32 + part + pair) % 6];
        }
        for (unsigned row = 0; row < count; ++row)
          for (unsigned col = 0; col < COLUMNS; ++col) {
            int32_t dot = static_cast<int32_t>((row * 41 + col * 29 + pair * 101) % 4097) - 2048;
            if (part) dot = -dot;
            co1[(col / 16 * physical + part * count + row) * 16 + col % 16] = dot;
            expected[row * COLUMNS + col] =
                Sum(expected[row * COLUMNS + col],
                    Product(static_cast<float>(dot), Product(weights[col], sx[row * stride])));
          }
      }
      std::copy(co1.begin(), co1.end(), raw.begin() + offset);
      trace.rawBusy[raw.data() + offset] = true;
      release[slot] = true;
      const auto afterReadback = raw;
      const auto savedSx0 = sx0, savedSx1 = sx1;
      ClearInstructions();
      trace.mask = 1;
      trace.normal = false;
      trace.highFull = true;
      const auto prepCount = trace.preparations;
      Invoke({raw.data() + offset, PRODUCT_SLOT_BYTES / 4}, {converted.data(), ELEMENTS}, {mask.data(), ELEMENTS},
             {mask.data() + BROADCAST_OFFSET / 4, ROWS * 8}, {accumulator.data(), ELEMENTS},
             {sx0.data(), ROWS * stride}, {sx1.data(), ROWS * stride}, {weight0.data(), COLUMNS},
             {weight1.data(), COLUMNS}, {expanded, 256}, count, physical, stride == 1);
      Check(trace.addresses == 7 && trace.slices == 0, "UB bases not queried once or descriptor slicing remained");
      Check(trace.masks == 4 && trace.normalizations == 4 && trace.resets == 1 && trace.normal && trace.mask == 64 &&
                trace.highFull,
            "mask setup/restoration changed");
      Check(trace.barriers == 10 && trace.gathers == 2 && trace.preparations == prepCount,
            "fences/Gather/preparation count changed");
      Check(trace.casts.size() == 16 && trace.arithmetic.size() == 8, "instruction count changed");
      Check(!trace.pending && !trace.rawBusy[raw.data() + offset] && !trace.activeRaw,
            "raw reader survived final consumer fence");
      for (unsigned part = 0; part < 2; ++part) {
        for (unsigned strip = 0; strip < 8; ++strip) {
          const auto& c = trace.casts[part * 8 + strip];
          Check(c.src == raw.data() + offset + (strip * physical + part * count) * 16 &&
                    c.dst == converted.data() + strip * 16 && c.rows == count && c.db == 1 && c.sb == 1 && c.dr == 16 &&
                    c.sr == 2,
                "pointer cast geometry changed");
        }
        for (unsigned op = 0; op < 4; ++op) {
          const auto& a = trace.arithmetic[part * 4 + op];
          Check(a.add == (op == 3) && a.db == 1 && a.ab == 1, "arithmetic order/block strides changed");
          Check(a.repeats == (op < 2 ? count : 2 * count) && a.bb == (op < 2 ? 0u : 1u) &&
                    a.dr == (op < 2 ? 16u : 8u) && a.ar == (op < 2 ? 0u : 8u) && a.br == (op < 2 ? 1u : 8u),
                "arithmetic repeat/broadcast strides changed");
        }
      }
      for (unsigned i = 0; i < accumulator.size(); ++i)
        Check(Bits(accumulator[i]) == Bits(expected[i]), "ordered FP32 output or accumulator guard mismatch");
      Check(raw == afterReadback && std::memcmp(sx0.data(), savedSx0.data(), sx0.size() * 4) == 0 &&
                std::memcmp(sx1.data(), savedSx1.data(), sx1.size() * 4) == 0,
            "raw/scales modified");
      for (unsigned i = 0; i < raw.size(); ++i)
        if (i < offset || i >= offset + physical * COLUMNS)
          Check(raw[i] == beforeRaw[i], "adjacent raw slot/tail overwritten");
      for (unsigned i = ELEMENTS; i < converted.size(); ++i)
        Check(Bits(converted[i]) == CANARY, "conversion guard changed");
      for (unsigned i = ELEMENTS + ROWS * 8; i < mask.size(); ++i)
        Check(Bits(mask[i]) == Bits(retained[i]), "retained rows/routes/expanded indices/source/tail overwritten");
      for (unsigned row = 0; row < ROWS; ++row)
        for (unsigned lane = 0; lane < 8; ++lane)
          Check(Bits(mask[BROADCAST_OFFSET / 4 + row * 8 + lane]) ==
                    Bits(sx1[(stride == 1 || row < count ? row : 0) * stride]),
                "literal or clamped padded scale bits changed");
      ++pairs;
    }
    release[0] = release[1] = false;
  }
  Check(trace.preparations == batches, "indices prepared inside product consumer");
  std::cout << "pointer casts/math, exact K32 sums, masks, raw slots and retained guards passed: " << pairs
            << " pairs\n";
}
void Witnesses() {
  trace = {};
  std::vector<int32_t> raw(PRODUCT_SLOT_BYTES / 4);
  std::vector<float> converted(ELEMENTS), factor(ELEMENTS), broadcast(ROWS * 8), acc(ELEMENTS, -3.0f);
  std::vector<float> sx(ROWS, 1.0f), w0(COLUMNS, Value(0x3f800001)), w1(COLUMNS, 0.0f);
  std::vector<uint32_t> indices(256);
  for (unsigned row = 0; row < ROWS; ++row)
    for (unsigned lane = 0; lane < 8; ++lane) indices[row * 8 + lane] = row * 4;
  for (auto* v : {&converted, &factor, &broadcast, &acc}) Register(v->data(), v->size() * 4, true);
  for (auto* v : {&sx, &w0, &w1}) Register(v->data(), v->size() * 4, false);
  Register(raw.data(), raw.size() * 4, false);
  Register(indices.data(), indices.size() * 4, false);
  for (unsigned col = 0; col < COLUMNS; ++col) raw[col / 16 * 32 * 16 + col % 16] = 3;
  auto invoke = [&] {
    Invoke({raw.data(), raw.size()}, {converted.data(), ELEMENTS}, {factor.data(), ELEMENTS},
           {broadcast.data(), ROWS * 8}, {acc.data(), ELEMENTS}, {sx.data(), ROWS}, {sx.data(), ROWS},
           {w0.data(), COLUMNS}, {w1.data(), COLUMNS}, {indices.data(), 256}, 1, 32, true);
  };
  invoke();
  const float separate = Sum(-3.0f, Product(3.0f, w0[0]));
  Check(Bits(separate) != Bits(std::fma(3.0f, w0[0], -3.0f)) && Bits(acc[0]) == Bits(separate),
        "FMA/reassociation changed");
  std::fill(raw.begin(), raw.end(), 0);
  std::fill(acc.begin(), acc.end(), -0.0f);
  std::fill(w0.begin(), w0.end(), -0.0f);
  std::fill(w1.begin(), w1.end(), -0.0f);
  invoke();
  Check(Bits(acc[0]) == 0x80000000, "signed zero changed");
  std::cout << "FMA and signed-zero witnesses passed\n";
}
void BothLiveSlots() {
  trace = {};
  std::vector<int32_t> raw(PRODUCT_STORAGE_BYTES / 4);
  std::vector<float> converted(ELEMENTS), factor(ELEMENTS), broadcast(ROWS * 8), acc(ELEMENTS);
  std::vector<float> scales(ROWS, 1.0f), weights(COLUMNS, 1.0f);
  std::vector<uint32_t> indices(256);
  for (unsigned row = 0; row < ROWS; ++row)
    for (unsigned lane = 0; lane < 8; ++lane) indices[row * 8 + lane] = row * 4;
  for (auto* v : {&converted, &factor, &broadcast, &acc}) Register(v->data(), v->size() * 4, true);
  for (auto* v : {&scales, &weights}) Register(v->data(), v->size() * 4, false);
  Register(raw.data(), raw.size() * 4, false);
  Register(indices.data(), indices.size() * 4, false);
  auto* first = raw.data();
  auto* second = raw.data() + PRODUCT_SLOT_BYTES / 4;
  trace.rawBusy[first] = trace.rawBusy[second] = true;
  for (auto* slot : {first, second}) {
    Invoke({slot, PRODUCT_SLOT_BYTES / 4}, {converted.data(), ELEMENTS}, {factor.data(), ELEMENTS},
           {broadcast.data(), ROWS * 8}, {acc.data(), ELEMENTS}, {scales.data(), ROWS}, {scales.data(), ROWS},
           {weights.data(), COLUMNS}, {weights.data(), COLUMNS}, {indices.data(), 256}, MAX_ROWS, 64, true);
    Check(!trace.rawBusy[slot] && !trace.activeRaw, "live-slot consumption did not release its reader");
    if (slot == first) Check(trace.rawBusy[second], "first slot prematurely released peer reader");
  }
  std::cout << "both raw slots live together: independent consumer fences passed\n";
}
void Faults() {
  trace = {};
  std::vector<int32_t> raw(PRODUCT_SLOT_BYTES / 4);
  std::vector<float> dst(64), a(64), b(64);
  Register(raw.data(), raw.size() * 4, false);
  Register(dst.data(), dst.size() * 4, true);
  Register(a.data(), a.size() * 4, false);
  Register(b.data(), b.size() * 4, false);
  unsigned rejected = 0;
  auto reject = [&](auto action) {
    try {
      action();
    } catch (const std::runtime_error&) {
      ++rejected;
      trace.pending = false;
      return;
    }
    throw std::runtime_error("invalid intrinsic admission accepted");
  };
  trace.normal = false;
  reject([&] { vconv_s322f32(dst.data(), raw.data(), 1, 1, 1, 16, 2); });
  SetMaskNorm();
  SetVectorMask<float>(64);
  reject([&] { vconv_s322f32(dst.data(), raw.data(), 1, 1, 1, 16, 2); });
  SetVectorMask<float>(16);
  reject([&] { vconv_s322f32(dst.data() + 64, raw.data(), 1, 1, 1, 16, 2); });
  reject([&] { vconv_s322f32(dst.data(), raw.data() + raw.size(), 1, 1, 1, 16, 2); });
  trace.rawBusy[raw.data()] = false;
  reject([&] { vconv_s322f32(dst.data(), raw.data(), 1, 1, 1, 16, 2); });
  SetVectorMask<float>(64);
  trace.pending = true;
  reject([&] { vadd(dst.data(), a.data(), b.data(), 1, 1, 1, 1, 8, 8, 8); });
  reject([&] { vmul(dst.data(), a.data(), b.data(), 2, 1, 1, 1, 8, 8, 8); });
  Check(rejected == 7, "fault count mismatch");
  std::cout << "7 mask, pointer-bound, raw-lifetime and dependency faults rejected\n";
}
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "faults")
      Faults();
    else {
      Check(argc == 4, "rows stride K");
      Run(std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3]));
      Witnesses();
      BothLiveSlots();
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
