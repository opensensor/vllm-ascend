# SPDX-License-Identifier: Apache-2.0
"""Compile the actual expert epilogue and compare DMA placement and pipe order."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.glm_fused_moe import NativeFusedMoE

SOURCE = Path(__file__).resolve().parents[3] / "tools/glm_perf/glm_fused_moe.cpp"


def method(source, name):
    start = source.index("  __aicore__ inline void " + name + "(")
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


STUB = r"""

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
using half = _Float16;
using int4b_t = int8_t;
#define __aicore__
constexpr uint32_t N = 128, NZ_N = 16, M = 32, K0 = 64, GROUP = 32, BULK_TOKENS = 16;
constexpr bool PAIR_PREFILL_SCALE_GROUPS = true;
constexpr int EVENT_ID0 = 0;
enum Pipe { PIPE_ALL, PIPE_V };
enum class HardEvent { M_V, V_M };
enum class RoundMode { CAST_NONE };
std::vector<int> operations;
template <Pipe p>
void PipeBarrier() {
  operations.push_back(p == PIPE_ALL ? 10 : 11);
}
template <HardEvent p>
void SetFlag(int) {
  operations.push_back(p == HardEvent::M_V ? 20 : 30);
}
template <HardEvent p>
void WaitFlag(int) {
  operations.push_back(p == HardEvent::M_V ? 21 : 31);
}
template <typename T>
struct LocalTensor {
  T* pointer;
  LocalTensor operator[](uint64_t index) const { return {pointer + index}; }
  T GetValue(uint64_t index) const { return pointer[index]; }
  template <typename U>
  LocalTensor<U> ReinterpretCast() const {
    return {reinterpret_cast<U*>(pointer)};
  }
};
struct Buffer {
  std::vector<uint64_t> storage = std::vector<uint64_t>(8192);
  template <typename T>
  LocalTensor<T> Get() {
    return {reinterpret_cast<T*>(storage.data())};
  }
};
struct DataCopyParams {
  uint16_t blockCount, blockLen, srcStride, dstStride;
};
enum class BlockMode { BLOCK_MODE_MATRIX };
struct DataCopyEnhancedParams {
  BlockMode blockMode;
};
struct MmadParams {
  uint32_t m, n, k;
  bool cmatrixInitVal;
};
void Mmad(LocalTensor<int32_t> c, LocalTensor<int4b_t>, LocalTensor<int4b_t>, MmadParams p) {
  operations.push_back(1);
  for (uint32_t i = 0; i < p.m * p.n; i++) c.pointer[i] = i;
}
void DataCopy(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, DataCopyParams p, DataCopyEnhancedParams) {
  operations.push_back(2);
  std::copy_n(src.pointer, p.blockCount * p.blockLen * 256, dst.pointer);
}
void Cast(LocalTensor<half> dst, LocalTensor<float> src, RoundMode, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) dst.pointer[i] = static_cast<half>(src.pointer[i]);
}
void Cast(LocalTensor<float> dst, LocalTensor<half> src, RoundMode, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) dst.pointer[i] = static_cast<float>(src.pointer[i]);
}
void Muls(LocalTensor<float> dst, LocalTensor<float> src, float scale, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) dst.pointer[i] = src.pointer[i] * scale;
}
void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) dst.pointer[i] = a.pointer[i] + b.pointer[i];
}
void DataCopy(LocalTensor<half> dst, LocalTensor<half> src, DataCopyParams p) {
  for (uint32_t r = 0; r < p.blockCount; r++)
    std::copy_n(src.pointer + r * (p.blockLen + p.srcStride) * 16, p.blockLen * 16,
                dst.pointer + r * (p.blockLen + p.dstStride) * 16);
}
template <typename T>
void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, uint32_t n) {
  std::copy_n(src.pointer, n, dst.pointer);
}
struct State {
  Buffer output_, outputRow_, reduction_, c_, a2_, b2_, products_, accum_, low_;
  int64_t tokens_ = 640, n_ = 4096, topK_ = 2, firstToken_ = 0;
  uint32_t activationBits_ = 4;
  std::vector<half> routes = std::vector<half>(48 * 4096, half(-77));
  std::vector<float> ys = std::vector<float>(48 * 4096, -19);
  std::vector<int64_t> order = std::vector<int64_t>(48);
  std::vector<float> weights = std::vector<float>(48);
  LocalTensor<half> routedHalf_{routes.data()};
  LocalTensor<float> y_{ys.data()}, weights_{weights.data()};
  LocalTensor<int64_t> order_{order.data()};
  template <typename T>
  LocalTensor<T> Products() {
    return products_.Get<T>();
  }
  LocalTensor<float> LowProducts() { return low_.Get<float>(); }
  LocalTensor<float> Accumulator() { return accum_.Get<float>(); }
  State() {
    for (int i = 0; i < 48; i++) {
      order[i] = i;
      weights[i] = 0.125f * (i % 7);
    }
    for (int i = 0; i < 32 * 128; i++) Accumulator().pointer[i] = (i % 503 - 251) * 0.037f;
  }
};
"""


def test_real_combine_bytes_and_product_dependencies(tmp_path):
    source = SOURCE.read_text()
    methods = method(source, "Product") + "\n" + method(source, "Combine")
    code = STUB + "\n#define GLM_FP16_ROUTE_WORKSPACE\n#define GLM_NATIVE_ROUTE_COLUMNS\n"
    code += "struct Parent:State {\n" + methods + "\n};\n"
    code += (
        "#define GLM_PRODUCT_PIPE_EVENTS\n#define GLM_BULK_ROUTE_STORE\nstruct Candidate:State {\n" + methods + "\n};\n"
    )
    code += r"""

int main() {
  for (int tokens : {2, 640})
    for (int rows : {1, 8, 16, 17, 31})
      for (int width : {256, 2048, 4096}) {
        Parent old;
        Candidate next;
        old.tokens_ = next.tokens_ = tokens;
        old.n_ = next.n_ = width;
        old.Combine(1, 8, rows);
        next.Combine(1, 8, rows);
        assert(old.routes == next.routes);
        assert(old.ys == next.ys);
        assert(old.reduction_.storage == next.reduction_.storage);
      }
  Parent old;
  Candidate next;
  operations.clear();
  old.Product(31, true);
  assert(operations == std::vector<int>({1, 10, 2, 10}));
  operations.clear();
  next.Product(31, true);
  assert(operations == std::vector<int>({1, 20, 21, 2, 11, 30, 31}));
  assert(old.products_.storage == next.products_.storage);
}
"""
    cpp = tmp_path / "actual.cpp"
    cpp.write_text(code)
    binary = tmp_path / "actual"
    subprocess.run(["c++", "-std=c++17", "-O0", str(cpp), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "options",
    [{"product_pipe_events": True}, {"bulk_route_store": True}, {"product_pipe_events": 1}, {"bulk_route_store": "1"}],
)
def test_invalid_schedule_rejected_before_directory_creation(tmp_path, options):
    with pytest.raises(ValueError):
        builder.build(tmp_path / "build", tmp_path, tmp_path, **options)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("compact", (False, True))
def test_builder_compiles_both_flags_into_all_projection_entries(tmp_path, monkeypatch, compact):
    commands = []

    def fake(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1] if "-o" in command else command[2]).write_bytes(b"offline-stub")

    monkeypatch.setattr(builder.subprocess, "run", fake)
    monkeypatch.setattr(builder, "include_paths", lambda: [])
    monkeypatch.setattr(builder, "library_paths", lambda: [])
    monkeypatch.setattr(
        builder.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    result = builder.build(
        tmp_path / "build",
        tmp_path,
        tmp_path,
        version=992,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        fp16_route_workspace=True,
        native_route_columns=True,
        product_pipe_events=True,
        bulk_route_store=True,
        prepared_weight_layout=compact,
        compact_w4_scratch=compact,
    )
    provenance = json.loads((result / "provenance.json").read_text())
    assert provenance["_build"]["product_pipe_events"] and provenance["_build"]["bulk_route_store"]
    entries = [command for command in commands if len(command) > 1 and Path(command[1]).name == "glm_fused_moe.cpp"]
    assert len(entries) == (4 if compact else 2)
    assert all("-DGLM_PRODUCT_PIPE_EVENTS" in command for command in entries)
    for command in entries:
        is_down = "down" in Path(command[2]).stem
        assert ("-DGLM_BULK_ROUTE_STORE" in command) is is_down


@pytest.mark.parametrize(
    "options", [{"product_pipe_events": 1}, {"bulk_route_store": "true"}, {"bulk_route_store": True}]
)
def test_runtime_rejects_invalid_schedule_before_loading_any_kernel(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="scheduling flags|native-column"):
        NativeFusedMoE(
            tmp_path,
            namespace="unused",
            activation_bits=4,
            launch=lambda *args: None,
            kernel_factory=lambda *args: pytest.fail("invalid bundle loaded"),
        )
