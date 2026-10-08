# SPDX-License-Identifier: Apache-2.0
"""Compile both QSA softmax bodies with per-repeat reduction/stride semantics."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_qsa_shared as builder
from tools.glm_perf.qsa_shared_probe import fixture
from tools.glm_perf.stage_qsa_accumulate_rows import transform as accumulation
from tools.glm_perf.stage_qsa_output_rows import transform as output
from tools.glm_perf.stage_qsa_shared_cache import transform as shared
from tools.glm_perf.stage_qsa_softmax_heads import EXTRA_UB_BYTES, transform

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "csrc/attention/qsa_sparse_attention_v310/op_kernel/qsa_cube_sparse_attention_v310.h"
)


def method(source, name):
    start = source.index("    __aicore__ inline", source.index(name) - 40)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


STUB = r"""

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
using half = _Float16;
#define __aicore__
constexpr int64_t MAX_QUERY_HEADS = 16, TOKEN_TILE = 64, NZ_INNER = 16, INT32_ALIGNMENT = 8;
constexpr int EVENT_ID7 = 7;
enum Pipe { PIPE_V };
enum class HardEvent { V_S, S_V };
enum class RoundMode { CAST_NONE };
enum class ReduceOrder { ORDER_VALUE_INDEX };
template <Pipe>
void PipeBarrier() {}
template <HardEvent>
void SetFlag(int) {}
template <HardEvent>
void WaitFlag(int) {}
template <typename T>
struct LocalTensor {
  T* pointer;
  size_t count;
  LocalTensor operator[](size_t n) const {
    assert(n <= count);
    return {pointer + n, count - n};
  }
  T GetValue(size_t n) const {
    assert(n < count);
    return pointer[n];
  }
  void SetValue(size_t n, T value) {
    assert(n < count);
    pointer[n] = value;
  }
};
struct Buffer {
  std::vector<uint64_t> storage;
  Buffer(size_t bytes = 4096) : storage(bytes / 8) {}
  template <typename T>
  LocalTensor<T> Get() {
    return {reinterpret_cast<T*>(storage.data()), storage.size() * 8 / sizeof(T)};
  }
};
struct UnaryRepeatParams {
  uint8_t dstBlkStride, srcBlkStride, dstRepStride, srcRepStride;
};
template <typename T>
void Adds(LocalTensor<T> dst, LocalTensor<T> src, T value, size_t count) {
  for (size_t i = 0; i < count; i++) dst.SetValue(i, T(src.GetValue(i) + value));
}
template <typename T>
void Adds(LocalTensor<T> dst, LocalTensor<T> src, T value, size_t mask, int repeats, UnaryRepeatParams p) {
  const int block = 32 / sizeof(T);
  for (int r = 0; r < repeats; r++)
    for (size_t i = 0; i < mask; i++) {
      auto d = r * p.dstRepStride * block + i / block * p.dstBlkStride * block + i % block;
      auto s = r * p.srcRepStride * block + i / block * p.srcBlkStride * block + i % block;
      dst.SetValue(d, T(src.GetValue(s) + value));
    }
}
void Duplicate(LocalTensor<float> dst, float value, size_t n) {
  for (size_t i = 0; i < n; i++) dst.SetValue(i, value);
}
void Exp(LocalTensor<float> dst, LocalTensor<float> src, size_t n) {
  for (size_t i = 0; i < n; i++) dst.SetValue(i, std::exp(src.GetValue(i)));
}
void Cast(LocalTensor<half> dst, LocalTensor<float> src, RoundMode, size_t n) {
  for (size_t i = 0; i < n; i++) dst.SetValue(i, half(src.GetValue(i)));
}
void WholeReduceMax(LocalTensor<float> dst, LocalTensor<float> src, int mask, int repeats, int dstStride, int srcBlock,
                    int srcRepeat, ReduceOrder) {
  for (int r = 0; r < repeats; r++) {
    float largest = src.GetValue(r * srcRepeat * 8);
    int index = 0;
    for (int i = 1; i < mask; i++) {
      float value = src.GetValue(r * srcRepeat * 8 + i / 8 * srcBlock * 8 + i % 8);
      if (value > largest) {
        largest = value;
        index = i;
      }
    }
    dst.SetValue(r * dstStride * 2, largest);
    float encoded;
    std::memcpy(&encoded, &index, 4);
    dst.SetValue(r * dstStride * 2 + 1, encoded);
  }
}
void WholeReduceSum(LocalTensor<float> dst, LocalTensor<float> src, int mask, int repeats, int dstStride, int srcBlock,
                    int srcRepeat) {
  assert(mask == 64);
  for (int r = 0; r < repeats; r++) {
    float lanes[64];
    for (int i = 0; i < 64; i++) lanes[i] = src.GetValue(r * srcRepeat * 8 + i / 8 * srcBlock * 8 + i % 8);
    for (int width = 1; width < 64; width *= 2)
      for (int i = 0; i < 64; i += 2 * width) lanes[i] = lanes[i] + lanes[i + width];
    dst.SetValue(r * dstStride, lanes[0]);
  }
}
struct State {
  Buffer scoreBuf_, softmaxBuf_, probabilityBuf_, reduceBuf_{64}, softmaxBatchBuf_{256};
  State() {
    auto scores = scoreBuf_.Get<float>();
    for (int i = 0; i < 1024; i++) scores.SetValue(i, (i % 131 - 65) * 0.071f);
  }
};
"""


def test_complete_softmax_host_bits_for_partial_heads_tiles_and_carry(tmp_path):
    source = SOURCE.read_text()
    candidate = transform(accumulation(output(shared(source))))
    names = ("ReduceMaximum(", "ReduceSum(", "ScalarExp(", "SoftmaxTile(")
    bodies = "\n".join(method(candidate, name) for name in names)
    code = STUB + "\nstruct Parent:State {\n" + bodies + "\n};\n"
    code += "#define GLM_QSA_SOFTMAX_HEAD_BATCH\nstruct Candidate:State {\n" + bodies + "\n};\n"
    code += r"""

int main() {
  for (int heads : {1, 2, 3, 4, 12, 16})
    for (int tokens : {1, 2, 31, 63, 64}) {
      Parent old;
      Candidate next;
      float om[16], nm[16], os[16], ns[16], ow[16], nw[16];
      for (int h = 0; h < 16; h++) {
        om[h] = nm[h] = -0.125f * (h + 1);
        os[h] = ns[h] = h % 2 ? 0.0f : 1.75f;
        ow[h] = nw[h] = -99.0f;
      }
      for (int phase = 0; phase < 3; phase++) {
        auto a = old.scoreBuf_.Get<float>();
        auto b = next.scoreBuf_.Get<float>();
        for (int i = 0; i < 1024; i++)
          a.SetValue(i, (i % 131 - 65) * 0.071f + phase * 0.031f), b.SetValue(i, a.GetValue(i));
        old.SoftmaxTile(heads, tokens, 64, om, os, ow);
        next.SoftmaxTile(heads, tokens, 64, nm, ns, nw);
        assert(std::memcmp(om, nm, sizeof(om)) == 0);
        assert(std::memcmp(os, ns, sizeof(os)) == 0);
        assert(std::memcmp(ow, nw, sizeof(ow)) == 0);
        assert(old.probabilityBuf_.storage == next.probabilityBuf_.storage);
        assert(old.softmaxBuf_.storage == next.softmaxBuf_.storage);
        assert(old.scoreBuf_.storage == next.scoreBuf_.storage);
      }
    }
}
"""
    cpp = tmp_path / "softmax.cpp"
    cpp.write_text(code)
    binary = tmp_path / "softmax"
    subprocess.run(
        ["c++", "-std=c++17", "-O0", "-ffp-contract=off", str(cpp), "-o", str(binary)], check=True, capture_output=True
    )
    subprocess.run([str(binary)], check=True, capture_output=True)


def test_disabled_softmax_preserves_parent_and_staging_rejects_drift():
    parent = accumulation(output(shared(SOURCE.read_text())))
    candidate = transform(parent)

    def preprocess(source):
        stripped = "\n".join(line for line in source.splitlines() if not line.startswith("#include"))
        return subprocess.run(
            ["c++", "-E", "-P", "-x", "c++", "-"], input=stripped, text=True, capture_output=True, check=True
        ).stdout

    assert preprocess(parent) == preprocess(candidate)
    assert EXTRA_UB_BYTES == 256 and candidate.count("EVENT_ID7") > 0
    with pytest.raises(ValueError, match="already staged"):
        transform(candidate)
    with pytest.raises(ValueError, match="arithmetic changed"):
        transform(parent.replace("ScalarExp(rowMax[head]", "DifferentExp(rowMax[head]"))


@pytest.mark.parametrize("invalid", (1, "true", None))
def test_builder_rejects_non_boolean_softmax_flag_before_creating_output(tmp_path, invalid):
    with pytest.raises(ValueError):
        builder.build(
            tmp_path / "build", SOURCE.parents[4], tmp_path, tmp_path / "compiler", 992, softmax_head_batch=invalid
        )
    assert not (tmp_path / "build").exists()


def test_builder_freezes_softmax_helper_and_compiles_only_the_selected_variant(tmp_path, monkeypatch):
    commands = []

    def fake(command, **kwargs):
        commands.append(command)
        target = command[command.index("-o") + 1] if "-o" in command else command[2]
        Path(target).write_bytes(b"offline-stub")

    monkeypatch.setattr(builder.subprocess, "run", fake)
    monkeypatch.setattr(builder, "include_paths", lambda: [])
    monkeypatch.setattr(builder, "library_paths", lambda: [])
    monkeypatch.setattr(
        builder.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    result = builder.build(
        tmp_path / "build",
        SOURCE.parents[4],
        tmp_path,
        tmp_path / "compiler",
        992,
        vector_output=True,
        vector_accumulate=True,
        softmax_head_batch=True,
    )
    assert result["softmax_head_batch"] and result["extra_ub_bytes"] == 256
    assert not result["full_operator_evaluated"] and not result["serving_evaluated"]
    assert "-DGLM_QSA_SOFTMAX_HEAD_BATCH" not in commands[0]
    assert "-DGLM_QSA_SOFTMAX_HEAD_BATCH" in commands[1]
    provenance = json.loads((tmp_path / "build/provenance.json").read_text())
    assert any(key.endswith("stage_qsa_softmax_heads.py") for key in provenance["files"])


@pytest.mark.parametrize("heads", (1, 3, 12))
def test_nightly_partial_head_fixture_matches_declared_geometry(heads):
    query, key, value, _, query_cpu, _, _ = fixture("cpu", 512, 2, True, query_heads_per_kv=heads)
    assert query.shape == query_cpu.shape == (2, heads, 512)
    assert key is value


@pytest.mark.parametrize("heads", (0, 17, 1.5, True))
def test_partial_head_fixture_rejects_invalid_head_count(heads):
    with pytest.raises(ValueError, match="query heads per KV"):
        fixture("cpu", 512, 2, True, query_heads_per_kv=heads)
