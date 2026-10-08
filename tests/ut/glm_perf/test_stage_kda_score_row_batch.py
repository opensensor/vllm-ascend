# SPDX-License-Identifier: Apache-2.0
"""Protect inactive arithmetic, source admission and the scratch lifetime bounds."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.glm_perf.stage_kda_gated_key_reuse import transform as reuse
from tools.glm_perf.stage_kda_score_row_batch import (
    COLUMN_CACHE_BYTE_OFFSET,
    EXTRA_UB_BYTES,
    FLAG,
    FLOAT_PRODUCT_BYTE_OFFSET,
    GATE_BYTE_OFFSET,
    MAX_PRODUCT_END_BYTE,
    REDUCE,
    stage,
    transform,
)
from tools.glm_perf.stage_kda_score_vector_reduce import transform as score

HEADER = Path(__file__).resolve().parents[3] / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"


def parent():
    return reuse(score(HEADER.read_text()))


@pytest.mark.parametrize("reuse_enabled", (False, True))
def test_disabled_batch_keeps_reference_preprocessed_tokens(tmp_path, reuse_enabled):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host preprocessor")
    definitions = ["-DGLM_KDA_SCORE_CACHE_COLUMNS", "-DGLM_KDA_SCORE_VECTOR_REDUCE"]
    if reuse_enabled:
        definitions.append("-DGLM_KDA_GATED_KEY_REUSE")
    outputs = []
    for source in (parent(), transform(parent())):
        begin = source.index("    __aicore__ inline void ComputeRawAqkAkkVector310P(")
        end = source.index("\n    }\n#endif", begin)
        path = tmp_path / "method.cpp"
        path.write_text(source[begin:end])
        outputs.append(
            subprocess.run(
                [compiler, "-E", "-P", *definitions, str(path)], check=True, capture_output=True
            ).stdout.split()
        )
    assert outputs[0] == outputs[1]


def test_new_storage_is_disjoint_from_live_rows_and_immutable_cache():
    assert EXTRA_UB_BYTES == 0
    live_vector_end = 1024 * 4 + 128 * 4 + 4 * 8 * 4 + 64 * 4
    assert live_vector_end <= GATE_BYTE_OFFSET
    for rows in range(1, 65):
        elements = rows * 128
        assert GATE_BYTE_OFFSET + elements * 2 <= FLOAT_PRODUCT_BYTE_OFFSET
        assert FLOAT_PRODUCT_BYTE_OFFSET + elements * 4 <= MAX_PRODUCT_END_BYTE
    assert MAX_PRODUCT_END_BYTE <= COLUMN_CACHE_BYTE_OFFSET


def test_batch_staging_rejects_wrong_parent_and_overwrite(tmp_path):
    source = tmp_path / "source.h"
    source.write_text(parent())
    original = source.read_bytes()
    candidate = tmp_path / "candidate.h"
    record = stage(source, candidate, hashlib.sha256(original).hexdigest())
    assert source.read_bytes() == original
    assert record["compiler_define"] == FLAG
    assert record["candidate_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert not record["full_kda_evaluated"] and not record["serving_evaluated"]
    with pytest.raises(FileExistsError):
        stage(source, candidate, record["source_sha256"])
    with pytest.raises(ValueError, match="qualified parent"):
        stage(source, tmp_path / "wrong.h", "0" * 64)
    assert not (tmp_path / "wrong.h").exists()
    with pytest.raises(ValueError, match="already staged"):
        transform(candidate.read_text())
    with pytest.raises(ValueError, match="full-row gated-key"):
        transform(parent().replace("curT == 64 && cacheColumns", "curT == 63 && cacheColumns"))


@pytest.mark.parametrize(
    "changed", ("KDA_VEC_ARENA_ELEMENTS = 32768", "fp32ArenaOffset = 1024", "scoreCacheByteOffset = 80")
)
def test_changed_arena_bounds_are_rejected(changed):
    with pytest.raises(ValueError, match="arena bounds"):
        transform(parent().replace(changed, changed + "0"))


@pytest.mark.parametrize("vector_reduce", (False, True))
def test_column_product_alias_binds_the_scalar_reference_parameter(tmp_path, vector_reduce):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    # Match LocalTensor slicing's by-value result and the original scalar
    # reducer's non-const reference. This catches the rejected first build.
    source = (
        """
#include <cstdint>
using half = uint16_t;
template<typename T> struct LocalTensor {
    LocalTensor<T> operator[](uint64_t) { return {}; }
};
enum Pipe { PIPE_V };
template<Pipe> void PipeBarrier() {}
enum class RoundMode { CAST_NONE };
void Cast(LocalTensor<float>&, LocalTensor<half>&, RoundMode, uint32_t) {}
void ReduceScoreToVector310P(uint64_t, LocalTensor<float>) {}
void ReduceDotProduct310P(LocalTensor<float>&, uint64_t, LocalTensor<float>&, LocalTensor<float>&) {}
void run(uint64_t row, uint64_t K_) {
    LocalTensor<float> batchFloat, scoreRow, partials;
    LocalTensor<half> batchHalf;
    uint32_t batchElements = (row + 1) * K_;
    bool vectorScoreReduce = true;
"""
        + REDUCE
        + "\n}\n"
    )
    path = tmp_path / "reduction.cpp"
    path.write_text(source)
    definitions = ["-DGLM_KDA_SCORE_ROW_BATCH"]
    if vector_reduce:
        definitions.append("-DGLM_KDA_SCORE_VECTOR_REDUCE")
    subprocess.run([compiler, "-std=c++17", "-fsyntax-only", *definitions, str(path)], check=True, capture_output=True)
