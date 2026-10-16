# SPDX-License-Identifier: Apache-2.0
"""Compile the actual issue loop against a combined ownership/event model.

The operand/product helpers have their own instruction-semantics tests. This
harness executes Projection::StreamKTile itself with checked helper boundaries;
it proves coordinator ordering, not device timing or intrinsic legality.
"""

import subprocess
from pathlib import Path

import pytest
import regex as re

from tools.glm_perf.fused_offset_tables import projection_descriptor

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/glm_perf/glm_fused_moe.cpp"


def _method(source, name):
    declaration = re.search(rf"__aicore__ inline [\w:]+ {name}\(", source)
    assert declaration is not None
    start = declaration.start()
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _dispatch_expressions(source):
    call = source.index("StreamKTile(kTile, count, sx, weightVectors, groupMajorScales);")
    start = source.rfind("if (", 0, call) + len("if (")
    expression = source[start : source.index(") {", start)]
    handoff = re.search(r"const bool streamingHandoff = ([^;]+);", source)
    assert handoff is not None
    return expression, handoff.group(1)


MODEL = r"""
#include "glm_streaming_contract.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define __aicore__
using event_t = uint32_t;
constexpr uint32_t EVENT_ID0 = 0, EVENT_ID1 = 1, PIPE_ALL = 0;
constexpr uint32_t M = 32, N = 128, NZ_K = 256, GROUP = 32, K0 = 64;
constexpr uint32_t B_BYTES = 4096, RAW_BYTES = 16384;
constexpr uint32_t ACTIVATION_CACHE_SLOT_BYTES = 2048, CACHED_PROJECTION_BYTES = 262144;
constexpr uint32_t BULK_TOKENS = 16, MAX_CACHED_SCALE_ROWS = 4;
void Check(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}
struct int4b_t {};
template<class T> struct Tensor {
  uint8_t* data;
  size_t bytes;
  Tensor operator[](size_t index) const {
    Check(index * sizeof(T) <= bytes, "tensor offset overflow");
    return {data + index * sizeof(T), bytes - index * sizeof(T)};
  }
  template<class U> Tensor<U> ReinterpretCast() const { return {data, bytes}; }
};
template<class T> using LocalTensor = Tensor<T>;
struct Buffer {
  std::vector<uint8_t> data;
  explicit Buffer(size_t bytes): data(bytes) {}
  template<class T> Tensor<T> Get() { return {data.data(), data.size()}; }
};
enum class HardEvent { MTE2_MTE1, MTE1_M, MTE1_MTE2, M_MTE1, M_V, V_M };
struct State {
  bool flags[6][2]{};
  bool staged[2]{}, loaded[2]{}, co1Busy[2]{}, rawUnread[2]{};
  uint32_t groups[2]{}, tile = 0, issued = 0, consumed = 0;
  uint32_t count = 0, rows = 0, stageCount = 0, barrierCount = 0;
  uint32_t projection = 0, k = 0, batch = 3, bits = 0;
  bool cache = false, reuse = false;
  uint8_t* c = nullptr;
  uint8_t* raw = nullptr;
  uint8_t* a1 = nullptr;
  uint8_t* b1 = nullptr;
  uint8_t* a2 = nullptr;
  uint8_t* b2 = nullptr;
  uint8_t* loadedA[2]{};
  uint8_t* loadedB[2]{};
  size_t a1Bytes = 0, b1Bytes = 0;
  uint8_t expectedWeight = 0;
  void Empty() {
    for (auto& kind : flags) for (bool value : kind) Check(!value, "unconsumed event at boundary");
    for (unsigned slot = 0; slot < 2; ++slot) {
      Check(!co1Busy[slot], "CO1 not released at boundary");
      Check(!rawUnread[slot], "product not consumed at boundary");
      Check(!staged[slot], "staged activation not loaded");
    }
  }
} state;
template<HardEvent Kind> void SetFlag(event_t slot) {
  Check(slot < 2, "event slot overflow");
  Check(!state.flags[static_cast<unsigned>(Kind)][slot], "double-set event");
  state.flags[static_cast<unsigned>(Kind)][slot] = true;
}
template<HardEvent Kind> void WaitFlag(event_t slot) {
  Check(slot < 2, "event slot overflow");
  Check(state.flags[static_cast<unsigned>(Kind)][slot], "unmatched event wait");
  state.flags[static_cast<unsigned>(Kind)][slot] = false;
  if constexpr (Kind == HardEvent::V_M) state.co1Busy[slot] = false;
}
template<unsigned Pipe> void PipeBarrier() {
  static_assert(Pipe == PIPE_ALL);
  Check(state.issued == 0 || (state.issued == 4 && state.consumed == 8), "steady-state PIPE_ALL");
  state.Empty();
  ++state.barrierCount;
}
template<class T> void DataCopy(Tensor<T> dst, Tensor<T> src, uint32_t count) {
  Check(count * sizeof(T) <= dst.bytes && count * sizeof(T) <= src.bytes, "copy overflow");
  std::memcpy(dst.data, src.data, count * sizeof(T));
}
namespace GlmRouteInput { constexpr uint32_t PAIR_BYTES = 2048; }
namespace GlmStreamingContract {
struct OperandViews { Tensor<int8_t> l1a, l0a, l1b, l0b; };
struct ProductViews { uint32_t slot, group; };
}
namespace GlmStreamingOperands {
void WaitReusable(uint32_t slot, bool used) {
  if (used) {
    WaitFlag<HardEvent::M_MTE1>(slot);
    WaitFlag<HardEvent::MTE1_MTE2>(slot);
  }
}
void StageActivation(Tensor<int8_t> dst, Tensor<int8_t> src, uint64_t offset, uint32_t slot) {
  const uint32_t group = state.tile * 8 + state.issued * 2;
  const uint64_t expected = (state.batch * (state.k / 64) + group / 2) * 2048;
  Check(offset == expected, "packed activation GM offset changed");
  Check(!state.flags[static_cast<unsigned>(HardEvent::MTE1_MTE2)][slot], "L1 backward fence missing");
  #ifdef GLM_FUSED_GATE_UP
  const size_t aOffset = group * ACTIVATION_CACHE_SLOT_BYTES;
  #else
  const size_t aOffset = slot * 2048;
  #endif
  Check(dst.data == state.a1 + aOffset, "L1 activation view mismatch");
  DataCopy(dst, src[offset], 2048);
  state.staged[slot] = true;
  ++state.stageCount;
  SetFlag<HardEvent::MTE2_MTE1>(slot);
}
void Load(const GlmStreamingContract::OperandViews& views, uint32_t rows, uint32_t slot, bool copied) {
  Check(rows == state.rows, "physical paired rows changed");
  const uint32_t group = state.tile * 8 + state.issued * 2;
  #ifdef GLM_FUSED_GATE_UP
  const size_t aOffset = group * ACTIVATION_CACHE_SLOT_BYTES;
  #else
  const size_t aOffset = slot * 2048;
  #endif
  Check(views.l1a.data == state.a1 + aOffset, "loaded L1 activation view mismatch");
  Check(views.l1a.bytes == state.a1Bytes - aOffset, "loaded L1 activation capacity mismatch");
  Check(views.l0a.data == state.a2 + slot * 2048, "loaded L0 activation slot mismatch");
  Check(views.l0a.bytes == 4096 - slot * 2048, "loaded L0 activation capacity mismatch");
  Check(views.l0b.data == state.b2 + slot * 4096, "loaded L0 weight slot mismatch");
  Check(views.l0b.bytes == 8192 - slot * 4096, "loaded L0 weight capacity mismatch");
  if (copied) {
    WaitFlag<HardEvent::MTE2_MTE1>(slot);
    Check(state.staged[slot], "activation not staged");
    state.staged[slot] = false;
  } else {
    Check(state.reuse, "uncopied activation without gate cache reuse");
  }
  const size_t expectedOffset = (state.cache ? state.projection * CACHED_PROJECTION_BYTES + state.tile * RAW_BYTES : 0)
                               + state.issued * B_BYTES;
  Check(views.l1b.data == state.b1 + expectedOffset, "resident weight view mismatch");
  Check(views.l1b.bytes == state.b1Bytes - expectedOffset, "resident weight capacity mismatch");
  Check(views.l1b.data[0] == state.expectedWeight, "direct W4 L1 overwritten by stale packedB");
  DataCopy(views.l0a, views.l1a, rows * 32);
  DataCopy(views.l0b, views.l1b, 4096);
  state.loaded[slot] = true;
  state.loadedA[slot] = views.l0a.data;
  state.loadedB[slot] = views.l0b.data;
  SetFlag<HardEvent::MTE1_MTE2>(slot);
  SetFlag<HardEvent::MTE1_M>(slot);
}
void LoadDirect(const Tensor<int8_t>& l1a,const Tensor<int8_t>& l0a,const Tensor<int8_t>& l1b,
                const Tensor<int8_t>& l0b,uint32_t rows,uint32_t slot,bool copied) {
  Load({l1a,l0a,l1b,l0b},rows,slot,copied);
}
void ReleaseOperands(uint32_t slot) { SetFlag<HardEvent::M_MTE1>(slot); }
void Drain(bool a, bool b) { WaitReusable(0,a); WaitReusable(1,b); }
}
struct MmadParams { uint32_t m, n, k; bool cmatrixInitVal; };
void Mmad(Tensor<int32_t> co1, Tensor<int4b_t> a, Tensor<int4b_t> b, MmadParams mm) {
  const uint32_t slot = (co1.data - state.c) / 32768;
  Check(state.loaded[slot] && !state.flags[static_cast<unsigned>(HardEvent::MTE1_M)][slot], "Cube before operand wait");
  Check(!state.co1Busy[slot], "CO1 overwritten before V_M release");
  Check(a.data == state.loadedA[slot] && b.data == state.loadedB[slot], "Cube uses a different L0 operand view");
  Check(mm.m == state.rows && mm.n == 128 && mm.k == 64 && mm.cmatrixInitVal, "Cube geometry changed");
  state.loaded[slot] = false;
  state.co1Busy[slot] = true;
  state.groups[slot] = state.tile * 8 + state.issued * 2;
  ++state.issued;
}
namespace GlmStreamingProducts {
void WaitReusable(uint32_t slot, bool used) { if (used) WaitFlag<HardEvent::V_M>(slot); }
void ReadBack(Tensor<int32_t> raw, Tensor<int32_t>, uint32_t rows, uint32_t slot) {
  WaitFlag<HardEvent::M_V>(slot);
  Check(rows == state.rows && raw.data == state.raw + slot * 32768, "raw product slot mismatch");
  Check(!state.rawUnread[slot], "raw product overwritten before Consume");
  state.rawUnread[slot] = true;
  SetFlag<HardEvent::V_M>(slot);
}
void Consume(GlmStreamingContract::ProductViews views, uint32_t count, uint32_t rows, bool) {
  Check(count == state.count && rows == state.rows, "consumer geometry changed");
  Check(state.rawUnread[views.slot], "Consume before readback or duplicate Consume");
  Check(views.group == state.groups[views.slot] && views.group == state.tile * 8 + state.consumed,
        "K32 group addition order changed");
  // The next pair must be issued before consuming the previous pair, except drain.
  Check(state.issued == std::min(4u, state.consumed / 2 + 2), "next Cube was not issued before previous Consume");
  state.consumed += 2;
  state.rawUnread[views.slot] = false;
}
void Drain(bool a, bool b) { WaitReusable(0,a); WaitReusable(1,b); }
}
struct Projection {
  #ifdef GLM_FUSED_GATE_UP
  Buffer a1_{262144}, b1_{524288};
  #else
  Buffer a1_{4096}, b1_{262144};
  #endif
  Buffer a2_{4096}, b2_{8192}, c_{65536}, decoded_{65536}, packedB_{16384};
  Buffer input_{2097152};
  Tensor<int8_t> inputHigh_ = input_.Get<int8_t>(), low_ = input_.Get<int8_t>();
  int64_t k_ = 4096, prepackedBatch_ = 3;
  uint32_t cacheProjection_ = 0, bits_ = 4;
  bool cacheActive_ = false, reuseActivations_ = false;
  void ConsumeStreamingProducts(uint32_t slot, uint32_t inner, int64_t tile, uint32_t count, uint32_t rows,
                                 const Tensor<float>&, const Tensor<float>&, bool groupMajor) {
    GlmStreamingProducts::Consume({slot, static_cast<uint32_t>(tile) * 8 + inner},count,rows,groupMajor);
  }
  __METHOD__
};
using StreamSignature = void (Projection::*)(int64_t,uint32_t,const Tensor<float>&,const Tensor<float>&,bool);
static_assert(std::is_same_v<decltype(&Projection::StreamKTile),StreamSignature>,
              "stream input descriptors must borrow");
bool Dispatch(uint32_t tokens_, uint32_t activationBits_, uint32_t count) {
  const bool cachedFactors = count <= MAX_CACHED_SCALE_ROWS;
  return __DISPATCH__;
}
bool Handoff(uint32_t tokens_, uint32_t activationBits_, uint32_t count) { return __HANDOFF__; }
int main(int argc, char** argv) {
  try {
    Check(argc == 7, "expected rows K cache bits reuse tokens");
    const uint32_t count = std::stoul(argv[1]), k = std::stoul(argv[2]), bits = std::stoul(argv[4]);
    const bool cache = std::stoul(argv[3]), reuse = std::stoul(argv[5]);
    const uint32_t tokens = std::stoul(argv[6]);
    const bool expectDispatch = tokens > 16 && bits == 4 && count > 4;
    Check(Dispatch(tokens,bits,count) == expectDispatch, "sparse/A8/decode dispatch changed");
    Check(Handoff(tokens,bits,count) == expectDispatch, "compact handoff dispatch changed");
    if (expectDispatch) {
      Projection projection;
      projection.cacheActive_ = cache;
      projection.bits_ = bits;
      projection.k_ = k;
      projection.reuseActivations_ = reuse;
      projection.cacheProjection_ = reuse ? 1 : 0;
      state.cache = cache;
      state.reuse = reuse;
      state.projection = projection.cacheProjection_;
      state.k = k;
      state.count = count;
      state.rows = count > 16 ? 64 : 32;
      state.bits = bits;
      state.c = projection.c_.data.data();
      state.raw = projection.decoded_.data.data();
      state.a1 = projection.a1_.data.data();
      state.b1 = projection.b1_.data.data();
      state.a2 = projection.a2_.data.data();
      state.b2 = projection.b2_.data.data();
      state.a1Bytes = projection.a1_.data.size();
      state.b1Bytes = projection.b1_.data.size();
      #ifdef GLM_DIRECT_W4_L1
      const bool direct = true;
      #else
      const bool direct = false;
      #endif
      state.expectedWeight = cache || direct ? 7 : 3;
      std::fill(projection.b1_.data.begin(), projection.b1_.data.end(), state.expectedWeight);
      std::fill(projection.packedB_.data.begin(), projection.packedB_.data.end(), direct ? 99 : 3);
      Buffer scales{65536};
      for (uint32_t tile = 0; tile < k / 256; ++tile) {
        state.tile = tile;
        state.issued = state.consumed = 0;
        projection.StreamKTile(tile,count,scales.Get<float>(),scales.Get<float>(),false);
        state.Empty();
        Check(state.issued == 4 && state.consumed == 8, "incomplete K256 tile");
      }
      #ifdef GLM_FUSED_GATE_UP
      const uint32_t expectedStages = reuse ? 0 : k / 64;
      #else
      const uint32_t expectedStages = k / 64;
      #endif
      Check(state.stageCount == expectedStages, "gate/up activation cache copy count changed");
    }
    std::cout << "actual StreamKTile schedule and qualified dispatch passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def _compile(tmp_path, *, gate, direct, mutation=None):
    source = SOURCE.read_text()
    method = _method(source, "StreamKTile")
    if mutation:
        old, new = mutation
        assert old in method
        method = method.replace(old, new, 1)
    dispatch, handoff = _dispatch_expressions(source)
    program = MODEL.replace("__METHOD__", method).replace("__DISPATCH__", dispatch).replace("__HANDOFF__", handoff)
    path = tmp_path / "dispatch.cpp"
    path.write_text(program)
    executable = tmp_path / "dispatch"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            f"-I{ROOT / 'tools/glm_perf'}",
            *(["-DGLM_FUSED_GATE_UP"] if gate else []),
            *(["-DGLM_DIRECT_W4_L1"] if direct else []),
            str(path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return executable


@pytest.fixture(scope="module", params=[(False, False), (False, True), (True, False), (True, True)])
def checker(request, tmp_path_factory):
    gate, direct = request.param
    return _compile(tmp_path_factory.mktemp("streaming-dispatch"), gate=gate, direct=direct), gate


@pytest.mark.parametrize("rows", [5, 15, 16, 17, 30, 31])
@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_actual_complete_loop_preserves_ownership_and_offsets(checker, rows, k, cache, reuse):
    executable, gate = checker
    if reuse and not gate:
        pytest.skip("down has no gate activation cache")
    result = subprocess.run(
        [str(executable), str(rows), str(k), str(int(cache)), "4", str(int(reuse)), "1280"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "actual StreamKTile schedule and qualified dispatch passed" in result.stdout


@pytest.mark.parametrize("rows", [1, 4, 5, 15, 16, 17, 30, 31])
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 17, 640, 1280])
@pytest.mark.parametrize("activation_bits", [4, 8])
def test_actual_dispatch_retains_sparse_a8_and_decode(checker, rows, tokens, activation_bits):
    executable, _ = checker
    subprocess.run(
        [str(executable), str(rows), "2048", "1", str(activation_bits), "0", str(tokens)],
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        ("GlmStreamingOperands::WaitReusable(slot, used[slot]);", ""),
        ("GlmStreamingProducts::WaitReusable(slot, used[slot]);", ""),
        ("WaitFlag<HardEvent::MTE1_M>(event);", ""),
        ("if (inner)", "if (false)"),
        ("GlmStreamingOperands::Drain(used[0], used[1]);", ""),
        ("GlmStreamingProducts::Drain(used[0], used[1]);", ""),
        ("previousInner = inner;", "previousInner = 0;"),
        ("const uint32_t physicalRows = count > M / 2 ? 2 * M : M;", "const uint32_t physicalRows = M;"),
    ],
)
def test_combined_model_detects_ordering_faults(tmp_path, mutation):
    executable = _compile(tmp_path, gate=False, direct=False, mutation=mutation)
    result = subprocess.run([str(executable), "31", "4096", "1", "4", "0", "1280"], text=True, capture_output=True)
    assert result.returncode != 0
    assert result.stderr.strip()


@pytest.mark.parametrize(
    "mutation",
    [
        ("a2_.Get<int8_t>()[slot * GlmStreamingContract::OPERAND_A_SLOT_BYTES]", "a2_.Get<int8_t>()[0]"),
        ("b2_.Get<int8_t>()[slot * GlmStreamingContract::OPERAND_B_SLOT_BYTES]", "b2_.Get<int8_t>()[0]"),
        ("LoadDirect(a1_.Get<int8_t>()[aOffset]", "LoadDirect(a1_.Get<int8_t>()[0]"),
        ("l0a, b1_.Get<int8_t>()[bOffset]", "l0a, b1_.Get<int8_t>()[0]"),
        ("l0a, b1_.Get<int8_t>()[bOffset], l0b", "l0b, b1_.Get<int8_t>()[bOffset], l0a"),
        ("Mmad(co1, l0a.ReinterpretCast<int4b_t>()", "Mmad(co1, a2_.Get<int8_t>().ReinterpretCast<int4b_t>()"),
    ],
)
def test_actual_direct_operand_call_and_cube_alias_mutations_are_detected(tmp_path, mutation):
    executable = _compile(tmp_path, gate=True, direct=True, mutation=mutation)
    result = subprocess.run([str(executable), "31", "4096", "1", "4", "1", "1280"], capture_output=True, text=True)
    assert result.returncode != 0
    assert result.stderr.strip()


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("tokens", [1, 2, 8, 17, 640, 1280])
@pytest.mark.parametrize("activation_bits", [4, 8])
@pytest.mark.parametrize("weight_bits", [2, 3, 4])
def test_streaming_retains_external_descriptor_abi(gate_up, tokens, activation_bits, weight_bits):
    header = (tokens * 8, 72, 4096, 2048, weight_bits, activation_bits, tokens, 8)
    options = {
        "prefill_rows_32": True,
        "route_compact_down_scales": True,
        "raw_hidden_scales": True,
        "native_route_columns": not gate_up,
        "quad_hidden_quant": gate_up,
        "group_major_down_scales": True,
    }
    expected = projection_descriptor(header, gate_up=gate_up, options=options)
    actual = projection_descriptor(header, gate_up=gate_up, options={**options, "streaming_pipeline": True})
    assert actual[:9] == expected[:9]
    assert actual[9] == expected[9] + 32
    assert actual[10 : len(expected)] == expected[10:]
    assert actual[len(expected) :] == tuple(row * 4 | (row + 1) * 4 << 32 for row in range(0, 32, 2))


VIEWS_MODEL = r"""
#include "glm_streaming_contract.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define __aicore__
namespace AscendC {
template<class T> struct LocalTensor {
  uint8_t* data;
  size_t bytes;
  bool watched = false;
  inline static unsigned watchedCopies = 0;
  LocalTensor(): data(nullptr), bytes(0) {}
  LocalTensor(uint8_t* ptr, size_t size): data(ptr), bytes(size) {}
  LocalTensor(const LocalTensor& other): data(other.data), bytes(other.bytes), watched(other.watched) {
    if (watched) ++watchedCopies;
  }
  LocalTensor operator[](size_t index) const {
    const size_t offset = index * sizeof(T);
    if (offset > bytes) throw std::runtime_error("view exceeds backing storage");
    return {data + offset, bytes - offset};
  }
};
}
using namespace AscendC;
namespace GlmStreamingContract { __PRODUCT_VIEWS__ }
constexpr uint32_t M = 32, N = 128, NZ_K = 256, GROUP = 32, ELEMENTS = M * N;
constexpr uint32_t SCALE_ROW_INDEX_OFFSET = GlmStreamingContract::RETAINED_INDEX_OFFSET;
struct Buffer {
  std::vector<uint8_t> storage;
  unsigned gets = 0;
  explicit Buffer(size_t bytes): storage(bytes) {}
  template<class T> LocalTensor<T> Get() {
    ++gets;
    return {storage.data(), storage.size()};
  }
};
void Check(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
GlmStreamingContract::ProductViews observed;
uint32_t observedCount = 0, observedRows = 0, observedCalls = 0;
bool observedLayout = false;
namespace GlmStreamingScaleGather { constexpr uint32_t EXPANDED_OFFSET = 27008; }
namespace __CALLBACK_NAMESPACE__ {
__CALLBACK_SIGNATURE__ {
  observed = {raw,converted,factors,rowBroadcast,accumulator,sx0,sx1,weight0,weight1,rowIndices};
  observedCount = count;
  observedRows = physicalRows;
  observedLayout = groupMajorScales;
  ++observedCalls;
}
}
struct Projection {
  Buffer decoded_{65536}, streamConverted_{16384}, mask_{32768}, results_{16384};
  LocalTensor<float> Accumulator() { return results_.Get<float>(); }
  __VIEW_METHOD__
};
using ConsumerSignature = void (Projection::*)(uint32_t,uint32_t,int64_t,uint32_t,uint32_t,
                                               const LocalTensor<float>&,const LocalTensor<float>&,bool);
static_assert(std::is_same_v<decltype(&Projection::ConsumeStreamingProducts),ConsumerSignature>,
              "consumer input descriptors must borrow");
int main(int argc, char** argv) {
  try {
    Check(argc == 3, "expected K and scale layout");
    const unsigned k = std::stoul(argv[1]);
    const bool groupMajor = std::stoul(argv[2]);
    Projection projection;
    Buffer scales(32 * 128 * 4), otherScales(32 * 128 * 4);
    Buffer weights(8 * 128 * 4), otherWeights(8 * 128 * 4);
    for (unsigned tile = 0; tile < k / 256; ++tile) {
      for (unsigned inner : {0u,2u,4u,6u})
        for (unsigned slot : {0u,1u}) {
        for (unsigned count : {1u,4u,5u,15u,16u,17u,30u,31u}) {
          // Alternate live input descriptor storage to detect stale retention.
          auto sx = slot ? otherScales.Get<float>() : scales.Get<float>();
          auto w = slot ? otherWeights.Get<float>() : weights.Get<float>();
          sx.watched = w.watched = true;
          const unsigned copies = LocalTensor<float>::watchedCopies;
          const unsigned calls = observedCalls;
          const unsigned physicalRows = count <= 16 ? 32 : 64;
          projection.ConsumeStreamingProducts(slot,inner,tile,count,physicalRows,sx,w,groupMajor);
          Check(observedCalls == calls + 1, "consumer callback not called exactly once");
          Check(observedCount == count && observedRows == physicalRows && observedLayout == groupMajor,
                "direct consumer scalar geometry changed");
          Check(LocalTensor<float>::watchedCopies == copies, "consumer copies a borrowed input descriptor");
          const auto& actual = observed;
          const unsigned scaleOffset = (tile * 8 + inner) * (groupMajor ? 32 : 1);
          Check(actual.raw.data == projection.decoded_.storage.data() + slot * 32768, "wrong raw slot offset");
          Check(actual.raw.bytes == 65536 - slot * 32768, "wrong raw slot capacity");
          Check(actual.converted.data == projection.streamConverted_.storage.data(), "wrong conversion plane");
          Check(actual.converted.bytes == 16384, "wrong conversion capacity");
          Check(actual.accumulator.data == projection.results_.storage.data(), "wrong accumulator lifetime");
          Check(actual.accumulator.bytes == 16384, "wrong accumulator capacity");
          Check(actual.factors.data == projection.mask_.storage.data(), "wrong factor scratch");
          Check(actual.factors.bytes == 32768, "wrong factor capacity");
          Check(actual.rowBroadcast.data == projection.mask_.storage.data() + 16384, "wrong row broadcast alias");
          Check(actual.rowBroadcast.bytes == 16384, "wrong row broadcast capacity");
          Check(actual.rowIndices.data == projection.mask_.storage.data() + 27008, "retained scale indices aliased");
          Check(actual.rowIndices.bytes == 5760, "wrong row index capacity");
          Check(actual.sx0.data == sx.data + scaleOffset * 4, "wrong first scale column");
          Check(actual.sx0.bytes == sx.bytes - scaleOffset * 4, "wrong first scale capacity");
          Check(actual.sx1.data == sx.data + (scaleOffset + (groupMajor ? 32 : 1)) * 4, "wrong second scale column");
          Check(actual.sx1.bytes == sx.bytes - (scaleOffset + (groupMajor ? 32 : 1)) * 4,
                "wrong second scale capacity");
          Check(actual.weight0.data == w.data + inner * 128 * 4, "wrong first weight vector");
          Check(actual.weight0.bytes == w.bytes - inner * 128 * 4, "wrong first weight capacity");
          Check(actual.weight1.data == w.data + (inner + 1) * 128 * 4, "wrong second weight vector");
          Check(actual.weight1.bytes == w.bytes - (inner + 1) * 128 * 4, "wrong second weight capacity");
        }
        }
    }
    std::cout << "all ten direct-consumer fields, borrowed inputs, tail geometry and fixed aliases passed\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def _compile_views(tmp_path, mutation=None):
    source = SOURCE.read_text()
    method = _method(source, "ConsumeStreamingProducts")
    if mutation:
        old, new = mutation
        assert old in method
        method = method.replace(old, new, 1)
    contract = (ROOT / "tools/glm_perf/glm_streaming_contract.h").read_text()
    declaration = re.search(r"struct ProductViews\s*\{[^}]+\};", contract)
    assert declaration is not None
    products = (ROOT / "tools/glm_perf/glm_streaming_products_pointer.h").read_text()
    signature = _method(products, "ConsumeDirect").split("{", 1)[0]
    namespace = re.search(r"namespace\s+(\w+)\s*\{", products)
    assert namespace is not None
    program = (
        VIEWS_MODEL.replace("__VIEW_METHOD__", method)
        .replace("__PRODUCT_VIEWS__", declaration.group(0))
        .replace("__CALLBACK_SIGNATURE__", signature)
        .replace("__CALLBACK_NAMESPACE__", namespace.group(1))
    )
    path = tmp_path / "borrowed-views.cpp"
    path.write_text(program)
    executable = tmp_path / "borrowed-views"
    subprocess.run(
        ["c++", "-std=c++17", "-O2", f"-I{ROOT / 'tools/glm_perf'}", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


@pytest.fixture(scope="module")
def views_checker(tmp_path_factory):
    return _compile_views(tmp_path_factory.mktemp("streaming-borrowed-views"))


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("group_major", [False, True])
def test_actual_direct_consumer_preserves_all_ten_fields_borrowing_and_tail_geometry(views_checker, k, group_major):
    result = subprocess.run(
        [str(views_checker), str(k), str(int(group_major))], check=True, capture_output=True, text=True
    )
    assert "all ten direct-consumer fields, borrowed inputs, tail geometry and fixed aliases passed" in result.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        ("decoded_.Get<int32_t>()[rawOffset]", "decoded_.Get<int32_t>()"),
        ("mask_.Get<float>(),", "mask_.Get<float>()[ELEMENTS],"),
        ("GlmStreamingScaleGather::EXPANDED_OFFSET / sizeof(uint32_t)", "0"),
        ("mask_.Get<float>()[ELEMENTS]", "mask_.Get<float>()"),
        ("Accumulator()", "streamConverted_.Get<float>()"),
        ("count, physicalRows,", "physicalRows, count,"),
        ("groupMajorScales);", "!groupMajorScales);"),
    ],
)
def test_borrowed_descriptor_alias_mutations_are_detected(tmp_path, mutation):
    executable = _compile_views(tmp_path, mutation)
    result = subprocess.run([str(executable), "4096", "1"], capture_output=True, text=True)
    assert result.returncode != 0
    assert result.stderr.strip()


@pytest.mark.parametrize("argument", ["sx", "weightVectors"])
def test_product_input_by_value_mutation_is_rejected(tmp_path, argument):
    mutation = (f"const LocalTensor<float>& {argument}", f"LocalTensor<float> {argument}")
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero") as error:
        _compile_views(tmp_path, mutation)
    assert "consumer input descriptors must borrow" in error.value.stderr


@pytest.mark.parametrize("argument", ["sx", "weightVectors"])
def test_issue_loop_input_by_value_mutation_is_rejected(tmp_path, argument):
    mutation = (f"const LocalTensor<float>& {argument}", f"LocalTensor<float> {argument}")
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero") as error:
        _compile(tmp_path, gate=False, direct=False, mutation=mutation)
    assert "stream input descriptors must borrow" in error.value.stderr
