// SPDX-License-Identifier: Apache-2.0
// Runs the actual helper with bounded CPU tensor/event semantics.
#include "glm_streaming_products.h"
#include "glm_streaming_scale_gather.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace AscendC;
using namespace GlmStreamingContract;
using namespace GlmStreamingProducts;
uint32_t Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float Value(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float Multiply(float a, float b) {
  volatile float f = a * b;
  return f;
}
float Sum(float a, float b) {
  volatile float f = a + b;
  return f;
}
void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void ConsumeGather(const ProductViews& v, unsigned count, unsigned physical, bool contiguous) {
  ConsumeDirect<true>(v.raw, v.converted, v.factors, v.rowBroadcast, v.accumulator, v.sx0, v.sx1, v.weight0, v.weight1,
                      v.rowIndices, count, physical, contiguous);
}
void PrepareScaleIndices(std::vector<float>& mask, unsigned count, unsigned stride) {
  using namespace GlmStreamingScaleGather;
  const unsigned sourceOffset = stride == 1 ? GROUP_SOURCE_OFFSET : RETAINED_INDEX_OFFSET;
  auto* source = reinterpret_cast<uint32_t*>(mask.data() + sourceOffset / 4);
  for (unsigned row = 0; row < GlmStreamingContract::ROWS; ++row)
    source[row] = (stride == 1 || row < count ? row : 0) * stride * 4;
  Prepare({reinterpret_cast<uint32_t*>(mask.data() + EXPANDED_OFFSET / 4), GlmStreamingScaleGather::ELEMENTS},
          {source, GlmStreamingContract::ROWS});
}

void TestPair(unsigned count, bool contiguous, unsigned pair, std::vector<int32_t>& raw, std::vector<float>& mask,
              std::vector<float>& converted, std::vector<float>& accumulator, std::vector<float>& expected,
              unsigned slot, bool gatherScales = false, unsigned rowStride = MAX_K / GROUP) {
  const unsigned physical = count <= 16 ? 32 : 64;
  std::vector<int32_t> co1(2 * ELEMENTS, 0x13572468);
  std::vector<float> sx0(ROWS * MAX_K / GROUP, Value(0x7fc12345));
  std::vector<float> sx1(sx0), w0(COLUMNS), w1(COLUMNS);
  auto indices = reinterpret_cast<uint32_t*>(mask.data() + RETAINED_INDEX_OFFSET / 4);
  if (!gatherScales)
    for (unsigned r = 0; r < ROWS; ++r) indices[r] = (r < count ? r : 0) * MAX_K / GROUP * 4;
  const auto savedIndices = std::vector<uint32_t>(indices, indices + ROWS);
  const auto savedMask = mask;
  for (unsigned part = 0; part < 2; ++part) {
    auto& sx = part == 0 ? sx0 : sx1;
    auto& weights = part == 0 ? w0 : w1;
    for (unsigned r = 0; r < count; ++r) {
      const float scales[] = {1.0f, 0.333333343f, -0.0f, 0.0f, -0.375f, 1.000000119f, 0.000000059604645f, 1.0e12f};
      sx[contiguous ? r : r * rowStride] = scales[(r + pair * 3 + part) % 8];
    }
    for (unsigned c = 0; c < COLUMNS; ++c) {
      const float scales[] = {1.0f, -1.0f, 1.000000119f, 0.142857149f, -0.0f, 0.25f};
      weights[c] = scales[(c / GROUP + pair + part) % 6];
    }
    for (unsigned r = 0; r < count; ++r)
      for (unsigned c = 0; c < COLUMNS; ++c) {
        const int32_t dot = static_cast<int32_t>((r * 41 + c * 29 + pair * 101) % 4097) - 2048;
        // Cancel adjacent independent groups while varying their factors.
        const int32_t value = part == 0 ? dot : -dot;
        co1[(c / 16 * physical + part * count + r) * 16 + c % 16] = value;
        const float factor = Multiply(weights[c], sx[contiguous ? r : r * rowStride]);
        const float scaled = Multiply(static_cast<float>(value), factor);
        expected[r * COLUMNS + c] = Sum(expected[r * COLUMNS + c], scaled);
      }
  }
  const auto rawBefore = raw;
  const unsigned offset = slot * PRODUCT_SLOT_BYTES / 4;
  SetFlag<HardEvent::M_V>(slot);
  ReadBack({raw.data() + offset, PRODUCT_SLOT_BYTES / 4}, {co1.data(), co1.size()}, physical, slot);
  Check(events[static_cast<unsigned>(HardEvent::V_M)][slot], "readback did not release CO1");
  const auto readBackRaw = raw;
  ProductViews v{
      {raw.data() + offset, PRODUCT_SLOT_BYTES / 4},
      {converted.data(), ELEMENTS},
      {mask.data(), ELEMENTS},
      {mask.data() + BROADCAST_OFFSET / 4, ROWS * 8},
      {accumulator.data(), ELEMENTS},
      {sx0.data(), sx0.size()},
      {sx1.data(), sx1.size()},
      {w0.data(), COLUMNS},
      {w1.data(), COLUMNS},
      {gatherScales ? reinterpret_cast<uint32_t*>(mask.data() + GlmStreamingScaleGather::EXPANDED_OFFSET / 4) : indices,
       gatherScales ? GlmStreamingScaleGather::ELEMENTS : ROWS}};
  ResetCounters();
  normalMaskMode = false;
  vectorMask = 1;
  const unsigned preparationsBefore = indexPreparations;
  if (gatherScales)
    ConsumeGather(v, count, physical, contiguous);
  else
    Consume(v, count, physical, contiguous);
  Check(indexPreparations == preparationsBefore, "consumer rebuilt scale indices per K pair");
  Check(maskSetups == 2 && maskNormalizations == 2, "cast mask setup was not amortized per independent group");
  Check(casts == 2 * COLUMNS / FRACTAL_COLUMNS && copies == 0, "direct cast must eliminate UB extraction copies");
  Check(castElements == 2 * count * COLUMNS, "cast converted padding or omitted useful products");
  for (unsigned part = 0; part < 2; ++part)
    for (unsigned strip = 0; strip < COLUMNS / FRACTAL_COLUMNS; ++strip) {
      const auto& cast = castRanges[part * COLUMNS / FRACTAL_COLUMNS + strip];
      Check(cast.source == raw.data() + offset + (strip * physical + part * count) * FRACTAL_COLUMNS,
            "cast read wrong independent group or N16 strip");
      Check(cast.destination == converted.data() + strip * FRACTAL_COLUMNS, "cast wrote wrong output strip");
      Check(cast.mask == FRACTAL_COLUMNS && cast.repeats == count && cast.dstRepStride == COLUMNS / 8 &&
                cast.srcRepStride == FRACTAL_COLUMNS / 8,
            "cast row/fractal strides changed");
    }
  Check(raw == readBackRaw, "direct cast modified raw products/padding/adjacent slots");
  Check(multiplies == 6 && adds == 2, "arithmetic issues changed");
  Check(gathers == (gatherScales ? 2u : contiguous ? 0u : 2u), "scale layout gather count wrong");
  Check(floatBroadcasts == (gatherScales ? 0u : 2u), "per-pair Brcb was not eliminated");
  for (unsigned width : gatherWidths) Check(width == (gatherScales ? 256u : 32u), "wrong scale Gather width");
  Check(barriers == (gatherScales || contiguous ? 10u : 12u), "vector dependency fences changed");
  if (gatherScales) {
    for (unsigned row = 0; row < ROWS; ++row)
      for (unsigned lane = 0; lane < 8; ++lane) {
        const unsigned sourceRow = contiguous || row < count ? row : 0;
        Check(Bits(mask[BROADCAST_OFFSET / 4 + row * 8 + lane]) ==
                  Bits(sx1[contiguous ? sourceRow : sourceRow * rowStride]),
              "Gather changed literal group-major/clamped row-major scale bits");
      }
    for (unsigned i = RETAINED_INDEX_OFFSET / 4; i < mask.size(); ++i)
      Check(Bits(mask[i]) == Bits(savedMask[i]), "consumer overwrote row indices, route ends, expanded table or tail");
  }
  for (unsigned i = 0; i < ELEMENTS; ++i)
    Check(Bits(expected[i]) == Bits(accumulator[i]), "FP32 mismatch or accumulator padding write");
  for (unsigned i = 0; i < raw.size(); ++i)
    if (i < offset || i >= offset + physical * COLUMNS)
      Check(raw[i] == rawBefore[i], "raw consumer changed adjacent slot/tail/guard");
  for (unsigned i = 0; i < ROWS; ++i) Check(indices[i] == savedIndices[i], "retained row index overwritten");
  for (unsigned i = ELEMENTS + ROWS * 8; i < RETAINED_INDEX_OFFSET / 4; ++i)
    Check(Bits(mask[i]) == 0x7fcabcde, "factor/broadcast scratch overrun");
  for (unsigned i = ELEMENTS; i < accumulator.size(); ++i)
    Check(Bits(accumulator[i]) == 0x7fcabcde, "accumulator guard overwritten");
  for (unsigned i = ELEMENTS; i < converted.size(); ++i)
    Check(Bits(converted[i]) == 0x7fcabcde, "conversion guard overwritten");
}

void TestRounding(bool gatherScales = false) {
  std::vector<int32_t> raw(2 * ELEMENTS, 0);
  std::vector<float> converted(ELEMENTS), factor(ELEMENTS), broadcast(ROWS * 8), acc(ELEMENTS, -3.0f);
  std::vector<float> sx(ROWS, 1.0f), w0(COLUMNS, Value(0x3f800001)), w1(COLUMNS, 0.0f);
  std::vector<uint32_t> indices(ROWS, 0);
  std::vector<uint32_t> expanded(GlmStreamingScaleGather::ELEMENTS);
  if (gatherScales) {
    for (unsigned row = 0; row < ROWS; ++row) indices[row] = row * 4;
    GlmStreamingScaleGather::Prepare({expanded.data(), expanded.size()}, {indices.data(), indices.size()});
  }
  for (unsigned c = 0; c < COLUMNS; ++c) raw[c / 16 * 32 * 16 + c % 16] = 3;
  ProductViews views{
      {raw.data(), raw.size()},
      {converted.data(), ELEMENTS},
      {factor.data(), ELEMENTS},
      {broadcast.data(), broadcast.size()},
      {acc.data(), ELEMENTS},
      {sx.data(), ROWS},
      {sx.data(), ROWS},
      {w0.data(), COLUMNS},
      {w1.data(), COLUMNS},
      {gatherScales ? expanded.data() : indices.data(), gatherScales ? expanded.size() : indices.size()}};
  if (gatherScales)
    ConsumeGather(views, 1, 32, true);
  else
    Consume(views, 1, 32, true);
  const float separate = Sum(-3.0f, Multiply(3.0f, w0[0]));
  Check(Bits(separate) != Bits(std::fma(3.0f, w0[0], -3.0f)), "rounding witness did not distinguish FMA");
  Check(Bits(acc[0]) == Bits(separate), "consumer fused multiplication/addition");
  std::fill(raw.begin(), raw.end(), 0);
  std::fill(acc.begin(), acc.end(), -0.0f);
  std::fill(w0.begin(), w0.end(), -0.0f);
  std::fill(w1.begin(), w1.end(), -0.0f);
  if (gatherScales)
    ConsumeGather(views, 1, 32, true);
  else
    Consume(views, 1, 32, true);
  Check(Bits(acc[0]) == 0x80000000, "signed-zero arithmetic changed");
}

void TestBothLiveSlots(bool gatherScales = false) {
  std::vector<int32_t> raw(PRODUCT_STORAGE_BYTES / 4), co1(ELEMENTS);
  std::vector<float> converted(ELEMENTS), factors(ELEMENTS), broadcast(ROWS * 8), accumulator(ELEMENTS);
  std::vector<float> scales(ROWS, 1.0f), weights(COLUMNS, 1.0f);
  std::vector<uint32_t> indices(ROWS);
  std::vector<uint32_t> expanded(GlmStreamingScaleGather::ELEMENTS);
  if (gatherScales) {
    for (unsigned row = 0; row < ROWS; ++row) indices[row] = row * 4;
    GlmStreamingScaleGather::Prepare({expanded.data(), expanded.size()}, {indices.data(), indices.size()});
  }
  for (unsigned slot = 0; slot < SLOTS; ++slot) {
    SetFlag<HardEvent::M_V>(slot);
    ReadBack({raw.data() + slot * PRODUCT_SLOT_BYTES / 4, PRODUCT_SLOT_BYTES / 4}, {co1.data(), co1.size()}, 32, slot);
  }
  for (unsigned slot = 0; slot < SLOTS; ++slot) {
    ProductViews views{
        {raw.data() + slot * PRODUCT_SLOT_BYTES / 4, PRODUCT_SLOT_BYTES / 4},
        {converted.data(), ELEMENTS},
        {factors.data(), ELEMENTS},
        {broadcast.data(), broadcast.size()},
        {accumulator.data(), ELEMENTS},
        {scales.data(), ROWS},
        {scales.data(), ROWS},
        {weights.data(), COLUMNS},
        {weights.data(), COLUMNS},
        {gatherScales ? expanded.data() : indices.data(), gatherScales ? expanded.size() : indices.size()}};
    if (gatherScales)
      ConsumeGather(views, 16, 32, true);
    else
      Consume(views, 16, 32, true);
  }
  Drain(true, true);
  for (const auto& pending : rawReaders) Check(!pending.second, "live raw reader after both-slot drain");
}

void TestGatherBits() {
  // Exercise arbitrary representations through the actual true consumer branch;
  // only scale replication bits are compared for nonfinite arithmetic inputs.
  constexpr uint32_t patterns[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
                                   0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
                                   0x7fc12345, 0x7fa12345, 0xffa12345, 0xffffffff};
  std::vector<int32_t> raw(PRODUCT_SLOT_BYTES / sizeof(int32_t));
  std::vector<float> converted(ELEMENTS), factors(ELEMENTS), accumulator(ELEMENTS), broadcast(ROWS * 8);
  std::vector<float> scales(ROWS), weights(COLUMNS, 1.0f);
  std::vector<uint32_t> indices(ROWS), expanded(GlmStreamingScaleGather::ELEMENTS);
  for (unsigned row = 0; row < ROWS; ++row) {
    indices[row] = row * 4;
    std::memcpy(&scales[row], &patterns[row % 16], sizeof(float));
  }
  GlmStreamingScaleGather::Prepare({expanded.data(), expanded.size()}, {indices.data(), indices.size()});
  ProductViews views{{raw.data(), raw.size()},       {converted.data(), ELEMENTS},
                     {factors.data(), ELEMENTS},     {broadcast.data(), broadcast.size()},
                     {accumulator.data(), ELEMENTS}, {scales.data(), ROWS},
                     {scales.data(), ROWS},          {weights.data(), COLUMNS},
                     {weights.data(), COLUMNS},      {expanded.data(), expanded.size()}};
  ConsumeGather(views, MAX_ROWS, 64, true);
  for (unsigned row = 0; row < ROWS; ++row)
    for (unsigned lane = 0; lane < 8; ++lane)
      Check(Bits(broadcast[row * 8 + lane]) == patterns[row % 16], "true consumer changed arbitrary scale bits");
}

int main(int argc, char** argv) {
  Check(argc == 4 || argc == 5, "provide row count, scale layout, K and optional Gather stride");
  const unsigned requested = std::stoul(argv[1]);
  const bool contiguous = std::stoul(argv[2]) != 0;
  const unsigned k = std::stoul(argv[3]);
  const bool gatherScales = argc == 5;
  const unsigned rowStride = gatherScales ? std::stoul(argv[4]) : MAX_K / GROUP;
  Check(!gatherScales || (rowStride == 1 && contiguous) || ((rowStride == 4 || rowStride == 128) && !contiguous),
        "invalid Gather scale stride/layout");
  Check(k == 2048 || k == MAX_K, "invalid test K");
  Check(requested >= 1 && requested <= MAX_ROWS, "invalid test row count");
  std::vector<int32_t> raw(PRODUCT_STORAGE_BYTES / 4 + 32, 0x24681357);
  std::vector<float> mask(COLUMNS * WEIGHT_TILE_K / 4, Value(0x7fcabcde));
  std::vector<float> converted(ELEMENTS + 32, Value(0x7fcabcde));
  std::vector<float> accumulator(ELEMENTS + 32, Value(0x7fcabcde));
  std::vector<float> expected(accumulator);
  // The same two raw slots are reused for complete K4096, then density changes.
  const unsigned replays[] = {requested, 31, 5, 4, 16, 17, 0, requested};
  unsigned launches = 0;
  const unsigned preparationsBefore = indexPreparations;
  unsigned batches = 0;
  for (unsigned count : replays) {
    if (count == 0) {
      Drain(false, false);
      continue;
    }
    std::fill_n(accumulator.data(), count * COLUMNS, 0.0f);
    expected = accumulator;
    if (gatherScales) PrepareScaleIndices(mask, count, rowStride);
    ++batches;
    bool used[2]{};
    for (unsigned pair = 0; pair < k / (2 * GROUP); ++pair) {
      const unsigned slot = pair % SLOTS;
      WaitReusable(slot, used[slot]);
      TestPair(count, contiguous, pair, raw, mask, converted, accumulator, expected, slot, gatherScales, rowStride);
      used[slot] = true;
      ++launches;
    }
    Drain(used[0], used[1]);
    Check(!events[0][0] && !events[0][1] && !events[1][0] && !events[1][1], "undrained events");
  }
  Check(indexPreparations - preparationsBefore == (gatherScales ? batches : 0u),
        "expanded indices were not prepared exactly once per changed batch");
  bool rejected = false;
  try {
    WaitReusable(0, true);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Check(rejected, "unmatched release was accepted");
  SetFlag<HardEvent::M_V>(0);
  rejected = false;
  try {
    SetFlag<HardEvent::M_V>(0);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Check(rejected, "double-issued result event accepted");
  WaitFlag<HardEvent::M_V>(0);
  // A V_M wait alone does not release the UB raw slot. Detect premature
  // next readback even though CO1 has legitimately been released.
  std::vector<int32_t> co1(ELEMENTS);
  SetFlag<HardEvent::M_V>(0);
  ReadBack({raw.data(), PRODUCT_SLOT_BYTES / 4}, {co1.data(), co1.size()}, 32, 0);
  WaitReusable(0, true);
  SetFlag<HardEvent::M_V>(0);
  rejected = false;
  try {
    ReadBack({raw.data(), PRODUCT_SLOT_BYTES / 4}, {co1.data(), co1.size()}, 32, 0);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Check(rejected, "CO1 release incorrectly allowed live raw overwrite");
  rawReaders.clear();
  TestBothLiveSlots(gatherScales);
  rawReaders.clear();
  TestRounding(gatherScales);
  if (gatherScales) TestGatherBits();
  std::cout << "direct strided casts, ordered K32 factors, slot reuse and guards passed: " << launches << " pairs\n";
  if (gatherScales)
    std::cout << "expanded scale Gather, exact bits, route-cache independence and two cast masks passed\n";
}
