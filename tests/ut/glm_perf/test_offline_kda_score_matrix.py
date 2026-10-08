# SPDX-License-Identifier: Apache-2.0
"""Run the actual cached full-row branch with bounded host vector instructions."""

import hashlib
import subprocess
import tarfile
from pathlib import Path

import pytest

from tools.glm_perf.stage_kda_gated_key_reuse import transform as reuse
from tools.glm_perf.stage_kda_score_matrix import FLAG, stage, transform
from tools.glm_perf.stage_kda_score_row_batch import transform as row_batch
from tools.glm_perf.stage_kda_score_vector_reduce import HELPER
from tools.glm_perf.stage_kda_score_vector_reduce import transform as vector_reduce

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"
ARCHIVE = (
    ROOT / "artifacts/glm-perf-310p/prefill-score-batching-20261008/measurements/qualified-kda-score-row-batch.tar.gz"
)
MEMBER = (
    "opp-score-row-batch/vendors/custom_transformer/op_impl/ai_core/tbe/"
    "custom_transformer_impl/ascendc/chunk_kda_fwd/chunk_kda_fwd_prepare.h"
)
DEFINES = [
    "GLM_KDA_SCORE_CACHE_COLUMNS",
    "GLM_KDA_SCORE_VECTOR_REDUCE",
    "GLM_KDA_GATED_KEY_REUSE",
    "GLM_KDA_SCORE_ROW_BATCH",
]


def parent():
    return row_batch(reuse(vector_reduce(HEADER.read_text())))


def branch(source):
    start = source.index("#if defined(GLM_KDA_GATED_KEY_REUSE) && defined(GLM_KDA_SCORE_CACHE_COLUMNS)")
    ending = "            return;\n        }\n#endif"
    end = source.index(ending, start) + len(ending)
    return source[start:end]


STUB = r"""

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>
using half = _Float16;
using T = half;
#define __aicore__
constexpr bool SAFE_GATE = true;
constexpr float LN2 = 0.6931471805599453094f;
template <typename A, typename B>
using IsSameType = std::is_same<A, B>;
enum Pipe { PIPE_V };
enum class HardEvent { MTE2_V, V_MTE3, MTE3_MTE2, MTE3_V };
enum class RoundMode { CAST_NONE };
template <Pipe>
void PipeBarrier() {}
template <HardEvent>
void SetFlag(int) {}
template <HardEvent>
void WaitFlag(int) {}
template <typename V>
struct LocalTensor {
  V* p;
  size_t n;
  LocalTensor operator[](size_t i) const {
    assert(i <= n);
    return {p + i, n - i};
  }
  V GetValue(size_t i) const {
    assert(i < n);
    return p[i];
  }
  void SetValue(size_t i, V x) {
    assert(i < n);
    p[i] = x;
  }
};
struct Buffer {
  std::vector<uint64_t> bytes;
  Buffer(size_t n = 131072) : bytes(n / 8) {}
  template <typename V>
  LocalTensor<V> Get() {
    return {reinterpret_cast<V*>(bytes.data()), bytes.size() * 8 / sizeof(V)};
  }
};
struct BinaryRepeatParams {
  uint8_t dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride;
};
template <typename V>
void Sub(LocalTensor<V> d, LocalTensor<V> a, LocalTensor<V> b, uint32_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, V(float(a.GetValue(i)) - float(b.GetValue(i))));
}
template <typename V>
void Mul(LocalTensor<V> d, LocalTensor<V> a, LocalTensor<V> b, uint32_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, V(float(a.GetValue(i)) * float(b.GetValue(i))));
}
template <typename V, bool sub>
void Binary(LocalTensor<V> d, LocalTensor<V> a, LocalTensor<V> b, uint64_t mask, int repeat, BinaryRepeatParams p) {
  const int block = 32 / sizeof(V);
  for (int r = 0; r < repeat; r++)
    for (size_t i = 0; i < mask; i++) {
      auto di = r * p.dstRepStride * block + i / block * p.dstBlkStride * block + i % block;
      auto ai = r * p.src0RepStride * block + i / block * p.src0BlkStride * block + i % block;
      auto bi = r * p.src1RepStride * block + i / block * p.src1BlkStride * block + i % block;
      float av = float(a.GetValue(ai)), bv = float(b.GetValue(bi));
      d.SetValue(di, V(sub ? av - bv : av * bv));
    }
}
template <typename V>
void Sub(LocalTensor<V> d, LocalTensor<V> a, LocalTensor<V> b, uint64_t m, int r, BinaryRepeatParams p) {
  Binary<V, true>(d, a, b, m, r, p);
}
template <typename V>
void Mul(LocalTensor<V> d, LocalTensor<V> a, LocalTensor<V> b, uint64_t m, int r, BinaryRepeatParams p) {
  Binary<V, false>(d, a, b, m, r, p);
}
template <typename V>
void Muls(LocalTensor<V> d, LocalTensor<V> s, V x, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, V(float(s.GetValue(i)) * float(x)));
}
void ClampFp16ExpInput(LocalTensor<half> a, size_t n) {
  for (size_t i = 0; i < n; i++) a.SetValue(i, half(std::clamp(float(a.GetValue(i)), -16.0f, 10.0f)));
}
void Exp(LocalTensor<half> d, LocalTensor<half> s, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, half(std::exp(float(s.GetValue(i)))));
}
void Cast(LocalTensor<float> d, LocalTensor<half> s, RoundMode, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, float(s.GetValue(i)));
}
void Duplicate(LocalTensor<float> d, float x, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, x);
}
void Adds(LocalTensor<float> d, LocalTensor<float> s, float x, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, s.GetValue(i) + x);
}
void Add(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, a.GetValue(i) + b.GetValue(i));
}
void WholeReduceSum(LocalTensor<float> d, LocalTensor<float> s, int mask, int repeat, int ds, int bs, int rs) {
  assert(mask == 64);
  for (int r = 0; r < repeat; r++) {
    float a[64];
    for (int i = 0; i < 64; i++) a[i] = s.GetValue(r * rs * 8 + i / 8 * bs * 8 + i % 8);
    for (int w = 1; w < 64; w *= 2)
      for (int i = 0; i < 64; i += w * 2) a[i] = a[i] + a[i + w];
    d.SetValue(r * ds, a[0]);
  }
}
void Gather(LocalTensor<float> d, LocalTensor<float> s, LocalTensor<uint32_t> idx, uint32_t base, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, s.GetValue((base + idx.GetValue(i)) / 4));
}
template <typename V>
void CopyVectorIn(LocalTensor<V> d, LocalTensor<V> s, size_t offset, size_t n) {
  for (size_t i = 0; i < n; i++) d.SetValue(i, s.GetValue(offset + i));
}
template <typename V>
void DataCopy(LocalTensor<V> d, LocalTensor<V> s, size_t n) {
  CopyVectorIn(d, s, 0, n);
}
struct State {
  const uint64_t K_ = 128, BT_ = 64;
  int mte2ToVEvent_ = 0, vToMte3Event_ = 1, mte3ToVEvent_ = 2, mte3ToMte2Events_[1] = {3};
  Buffer vecBuf_, gatedKeyReuseBuf_{16384}, scoreReduceFirstBuf_{2048}, scoreReduceSecondBuf_{2048},
      scoreReduceIndicesBuf_{256};
  std::vector<half> qv = std::vector<half>(8192), kv = qv, gv = qv;
  std::vector<float> ak = std::vector<float>(4096, -99.0f), aq = ak;
  LocalTensor<half> q_{qv.data(), 8192}, k_{kv.data(), 8192}, gk_{gv.data(), 8192};
  LocalTensor<float> akk_{ak.data(), 4096}, aqk_{aq.data(), 4096};
  uint64_t QOffset(uint64_t, uint64_t, uint64_t row, uint64_t col) { return row * 128 + col; }
  uint64_t KVOffset(uint64_t, uint64_t, uint64_t row, uint64_t col, uint64_t) { return row * 128 + col; }
  uint64_t AOffset(uint64_t, uint64_t, uint64_t row, uint64_t col) { return row * 64 + col; }
  void ReduceDotProduct310P(LocalTensor<float>, uint64_t, LocalTensor<float>, LocalTensor<float>) { assert(false); }
  State(int seed) {
    for (int i = 0; i < 8192; i++) {
      qv[i] = half(((i * 17 + seed) % 251 - 125) * 0.013f);
      kv[i] = half(((i * 31 + seed) % 197 - 98) * 0.017f);
      gv[i] = half(-((i + seed) % 64) * 0.021f);
    }
    auto arena = vecBuf_.Get<half>();
    CopyVectorIn(arena[80 * 1024 / 2], k_, 0, 8192);
    CopyVectorIn(arena[96 * 1024 / 2], gk_, 0, 8192);
    auto idx = scoreReduceIndicesBuf_.Get<uint32_t>();
    for (int i = 0; i < 64; i++) idx.SetValue(i, i * 32);
  }
};
"""
PRELUDE = r"""

void Run() {
  uint64_t b = 0, h = 0, hv = 0, start = 0, curT = 64;
  bool cacheColumns = true, vectorScoreReduce = true;
  auto typedArena = vecBuf_.Get<half>();
  auto rowVector = typedArena[0], gRow = typedArena[128], kCol = typedArena[256], gCol = typedArena[384],
       gate = typedArena[512], kGated = typedArena[640], product = typedArena[768];
  auto productFp32 = vecBuf_.Get<float>()[1024], scoreRow = vecBuf_.Get<float>()[1152],
       partials = vecBuf_.Get<float>()[1280];
  auto cachedKeys = typedArena[80 * 1024 / 2], cachedGates = typedArena[96 * 1024 / 2];
  auto scoreFirst = scoreReduceFirstBuf_.Get<float>(), scoreSecond = scoreReduceSecondBuf_.Get<float>();
  auto scoreIndices = scoreReduceIndicesBuf_.Get<uint32_t>();
"""


def test_actual_full_row_branch_preserves_both_score_matrices_and_column_cache(tmp_path):
    candidate = transform(parent())
    code = STUB + "\n" + "\n".join("#define " + name for name in DEFINES) + "\n"
    code += "struct Parent:State {using State::State;\n" + HELPER + PRELUDE + branch(candidate) + "\n}};\n"
    code += "#define " + FLAG + "\nstruct Candidate:State {using State::State;\n"
    code += HELPER + PRELUDE + branch(candidate) + "\n}};\n"
    code += r"""

int main() {
  for (int seed : {0, 1, 17, 71}) {
    Parent old(seed);
    Candidate next(seed);
    auto original = old.vecBuf_.bytes;
    old.Run();
    next.Run();
    assert(old.ak == next.ak);
    assert(old.aq == next.aq);
    assert(old.gatedKeyReuseBuf_.bytes == next.gatedKeyReuseBuf_.bytes);
    for (size_t i = 80 * 1024 / 8; i < 128 * 1024 / 8; i++) {
      assert(old.vecBuf_.bytes[i] == original[i]);
      assert(next.vecBuf_.bytes[i] == original[i]);
    }
    for (int r = 0; r < 64; r++)
      for (int c = r + 1; c < 64; c++) assert(next.ak[r * 64 + c] == 0.0f && next.aq[r * 64 + c] == 0.0f);
  }
}
"""
    source = tmp_path / "branch.cpp"
    source.write_text(code)
    binary = tmp_path / "branch"
    result = subprocess.run(
        ["c++", "-std=c++17", "-O0", "-ffp-contract=off", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_disabled_matrix_preserves_entire_parent_and_fallback_tokens():
    outputs = []
    for source in (parent(), transform(parent())):
        source = "\n".join(line for line in source.splitlines() if not line.startswith("#include"))
        outputs.append(
            subprocess.run(
                ["c++", "-E", "-P", "-x", "c++", *["-D" + flag for flag in DEFINES], "-"],
                input=source,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.split()
        )
    assert outputs[0] == outputs[1]


def test_stage_actual_deployed_header_admission_and_exclusive_output(tmp_path):
    with tarfile.open(ARCHIVE) as archive, archive.extractfile(MEMBER) as handle:
        original = handle.read()
    source = tmp_path / "deployed.h"
    source.write_bytes(original)
    destination = tmp_path / "candidate.h"
    report = stage(source, destination, hashlib.sha256(original).hexdigest())
    assert not report["full_kda_evaluated"] and not report["serving_evaluated"]
    assert report["extra_ub_bytes"] == 0 and source.read_bytes() == original
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == report["candidate_sha256"]
    with pytest.raises(FileExistsError):
        stage(source, destination, report["source_sha256"])
    with pytest.raises(ValueError, match="qualified parent"):
        stage(source, tmp_path / "bad.h", "0" * 64)
    assert not (tmp_path / "bad.h").exists()
    with pytest.raises(ValueError, match="already staged"):
        transform(destination.read_text())
    with pytest.raises(ValueError, match="anchors changed"):
        transform(source.read_text().replace("curT == 64 && cacheColumns", "curT == 63 && cacheColumns"))


@pytest.mark.parametrize("missing", DEFINES)
def test_matrix_compile_rejects_unqualified_feature_combinations(missing):
    source = transform(parent()).split("#endif", 1)[0] + "#endif\n"
    result = subprocess.run(
        ["c++", "-E", "-P", "-x", "c++", "-D" + FLAG, *["-D" + f for f in DEFINES if f != missing], "-"],
        input=source,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "score matrix requires" in result.stderr
