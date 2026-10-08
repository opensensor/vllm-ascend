# SPDX-License-Identifier: Apache-2.0
"""Preserve partial-cache fallback, bounds, scalar math and frozen parents."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.glm_perf.stage_kda_score_cache_tails import FLAG, stage, transform

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"


def test_disabled_tail_flag_preserves_existing_cached_and_uncached_tokens(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host preprocessor")
    original = HEADER.read_text()
    results = []
    for source in (original, transform(original)):
        begin = source.index("    __aicore__ inline void ComputeRawAqkAkkVector310P(")
        end = source.index("\n#endif", source.index("\n    }", begin))
        path = tmp_path / "cache.cpp"
        path.write_text(source[begin:end])
        results.append(
            subprocess.run(
                [compiler, "-E", "-P", "-DGLM_KDA_SCORE_CACHE_COLUMNS", str(path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
        )
    assert results[0] == results[1]
    candidate = transform(original)
    assert "scoreCacheByteOffset + 2 * cacheElements * sizeof(T) <=" in candidate
    assert "KDA_VEC_ARENA_ELEMENTS * sizeof(float)" in candidate
    assert "curT > 0 && curT <= scoreCacheMaxRows" in candidate
    assert candidate.count("ReduceDotProduct310P(\n                    scoreRow") == 2
    assert original == HEADER.read_text()


def test_partial_cache_staging_never_overwrites_or_accepts_changed_parent(tmp_path):
    source = tmp_path / "parent.h"
    source.write_bytes(HEADER.read_bytes())
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "candidate.h"
    record = stage(source, destination, expected)
    assert record["compiler_define"] == FLAG and record["extra_ub_bytes"] == 0
    assert not record["serving_eligible"]
    assert "recurrent-state mismatch" in record["known_gate_failure"]
    assert source.read_bytes() == HEADER.read_bytes()
    assert record["candidate_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        stage(source, destination, expected)
    with pytest.raises(ValueError, match="qualified parent"):
        stage(source, tmp_path / "wrong.h", "0" * 64)
    assert not (tmp_path / "wrong.h").exists()
    with pytest.raises(ValueError, match="already staged"):
        transform(destination.read_text())
    with pytest.raises(ValueError, match="source changed"):
        transform(HEADER.read_text().replace("curT == scoreCacheMaxRows", "changed"))
