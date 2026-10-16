// SPDX-License-Identifier: Apache-2.0
#include "glm_streaming_operands.h"
#include <iostream>
#include <string>
using namespace AscendC;
using namespace GlmStreamingOperands;
constexpr uint8_t POISON = 0xcd;
LocalTensor<int8_t> Local(std::vector<uint8_t>& storage, size_t offset = 0) {
  return {storage.data() + offset, storage.size() - offset};
}
GlobalTensor<int8_t> Global(std::vector<uint8_t>& storage) { return {storage.data(), storage.size()}; }
void Run(unsigned count, unsigned pairs, bool reuse, unsigned bits, bool bounded = false) {
  const unsigned rows = count <= 16 ? 32 : 64;
  // Distinct offsets and poisoned guards catch a helper ignoring selected views.
  constexpr unsigned GUARD = 64, A_SLOT = 2048, B_SLOT = 4096;
  std::vector<uint8_t> gm(GUARD + pairs * A_SLOT + GUARD, POISON);
  std::vector<uint8_t> l1a(GUARD + (bounded ? 2 : pairs) * A_SLOT + GUARD, POISON);
  std::vector<uint8_t> l1b(GUARD + pairs * B_SLOT + GUARD, POISON);
  std::vector<uint8_t> l0a(GUARD + 2 * A_SLOT + GUARD, POISON);
  std::vector<uint8_t> l0b(GUARD + 2 * B_SLOT + GUARD, POISON);
  const int bound = 1 << (bits - 1);
  for (unsigned pair = 0; pair < pairs; ++pair) {
    std::fill_n(gm.data() + GUARD + pair * A_SLOT, A_SLOT, 0);
    for (unsigned part = 0; part < 2; ++part)
      for (unsigned row = 0; row < count; ++row)
        for (unsigned col = 0; col < 32; ++col) {
          const int lo = static_cast<int>((pair + row + col + part) % 16) - 8;
          const int hi = static_cast<int>((pair * 3 + row + col + part) % 16) - 8;
          gm[GUARD + pair * A_SLOT + (part * count + row) * 32 + col] = (lo & 15) | ((hi & 15) << 4);
        }
    for (unsigned byte = 0; byte < B_SLOT; ++byte) {
      const int lo = static_cast<int>((pair + byte) % (2 * bound)) - bound;
      const int hi = static_cast<int>((pair * 5 + byte) % (2 * bound)) - bound;
      // These are the prepared W4 / reconstructed W2/W3 signed INT4 codes.
      l1b[GUARD + pair * B_SLOT + byte] = (lo & 15) | ((hi & 15) << 4);
    }
  }
  Model state;
  model = &state;
  for (unsigned slot = 0; slot < 2; ++slot) {
    state.a[slot] = l0a.data() + GUARD + slot * A_SLOT;
    state.b[slot] = l0b.data() + GUARD + slot * B_SLOT;
  }
  for (unsigned projection = 0; projection < (reuse ? 2u : 1u); ++projection) {
    bool used[2] = {false, false};
    for (unsigned pair = 0; pair < pairs; ++pair) {
      const unsigned slot = pair % 2;
      WaitReusable(slot, used[slot]);
      const unsigned aOffset = GUARD + (bounded ? slot : pair) * A_SLOT;
      const bool copied = bounded || projection == 0;
      if (copied) StageActivation(Local(l1a, aOffset), Global(gm), GUARD + pair * A_SLOT, slot);
      GlmStreamingContract::OperandViews views{Local(l1a, aOffset), Local(l0a, GUARD + slot * A_SLOT),
                                               Local(l1b, GUARD + pair * B_SLOT), Local(l0b, GUARD + slot * B_SLOT)};
      // With immutable resident L1 caches, no old MTE1 load can be overwritten.
      Load(views, rows, slot, copied);
      CubeIssue(slot);
      Require(std::memcmp(views.l0a.data, gm.data() + GUARD + pair * A_SLOT, rows * 32) == 0,
              "activation paired rows differ");
      Require(std::memcmp(views.l0b.data, views.l1b.data, B_SLOT) == 0, "weight codes differ");
      if (rows == 32)
        for (unsigned i = A_SLOT / 2; i < A_SLOT; ++i)
          Require(views.l0a.data[i] == POISON, "sparse upper block written");
      ReleaseOperands(slot);
      used[slot] = true;
    }
    Drain(used[0], used[1]);
    state.CheckEmpty();
  }
  const unsigned projections = reuse ? 2 : 1;
  Require(state.stages == pairs * (bounded ? projections : 1), "activation copy count differs");
  Require(state.activationLoads == pairs * projections * rows / 32, "M32 block loads differ");
  Require(state.weightLoads == pairs * projections, "weights missing or duplicated");
  for (auto* storage : {&gm, &l1a, &l1b, &l0a, &l0b}) {
    for (unsigned i = 0; i < GUARD; ++i) {
      Require((*storage)[i] == POISON, "leading guard overwritten");
      Require((*storage)[storage->size() - 1 - i] == POISON, "trailing guard overwritten");
    }
  }
  std::cout << "operand bytes, event ownership, cache reuse and drain passed\n";
}
void Negative(const std::string& mode) {
  Model state;
  model = &state;
  std::vector<uint8_t> a(2048), b(4096), c(2048), d(4096);
  state.a[0] = c.data();
  state.b[0] = d.data();
  GlmStreamingContract::OperandViews views{Local(a), Local(c), Local(b), Local(d)};
  if (mode == "unmatched") WaitReusable(0, true);
  if (mode == "unready") Load(views, 64, 0, true);
  if (mode == "duplicate") {
    StageActivation(Local(a), Global(c), 0, 0);
    StageActivation(Local(a), Global(c), 0, 0);
  }
  if (mode == "release") ReleaseOperands(0);
  if (mode == "slot") WaitReusable(2, true);
  if (mode == "overwrite") {
    Load(views, 64, 0, false);
    CubeIssue(0);
    ReleaseOperands(0);
    Load(views, 64, 0, false);
  }
  if (mode == "l1-reader") {
    Load(views, 64, 0, false);
    StageActivation(Local(a), Global(c), 0, 1);
  }
  if (mode == "l1-backward") {
    Load(views, 64, 0, false);
    CubeIssue(0);
    ReleaseOperands(0);
    WaitFlag<HardEvent::M_MTE1>(0);
    StageActivation(Local(a), Global(c), 0, 0);
  }
  if (mode == "source-bounds") StageActivation(Local(a), Global(c), 64, 0);
  if (mode == "destination-bounds") StageActivation(Local(a, 64), Global(c), 0, 0);
  throw std::logic_error("negative check unexpectedly succeeded");
}
int main(int argc, char** argv) {
  try {
    if (argc == 2) {
      try {
        Negative(argv[1]);
      } catch (const std::runtime_error& error) {
        std::cout << "rejected: " << error.what() << '\n';
        return 0;
      }
      return 1;
    }
    Require(argc == 5 || argc == 6, "expected rows, pairs, reuse, bits and optional bounded staging");
    Run(std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3]) != 0, std::stoul(argv[4]), argc == 6);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
