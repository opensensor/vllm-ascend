# SPDX-License-Identifier: Apache-2.0
"""Keep staging separate and retain the complete disabled scalar score path."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.glm_perf.stage_kda_beta_scale_vector import transform as beta_transform
from tools.glm_perf.stage_kda_score_vector_reduce import EXTRA_UB_BYTES, FLAG, stage, transform

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"


def test_disabled_score_flag_preserves_reference_preprocessed_tokens(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host preprocessor")
    original = HEADER.read_text()
    results = []
    for source in (original, transform(original)):
        begin = source.index("    __aicore__ inline void ComputeRawAqkAkkVector310P(")
        end = source.index("\n#endif", source.index("\n    }", begin))
        path = tmp_path / "scalar.cpp"
        path.write_text(source[begin:end])
        results.append(
            subprocess.run([compiler, "-E", "-P", str(path)], check=True, capture_output=True, text=True).stdout.split()
        )
    assert results[0] == results[1]
    assert original == HEADER.read_text()


def test_staging_composes_after_beta_without_overwriting_parent(tmp_path):
    parent = tmp_path / "beta.h"
    parent.write_text(beta_transform(HEADER.read_text()))
    before = parent.read_bytes()
    target = tmp_path / "score.h"
    record = stage(parent, target, hashlib.sha256(before).hexdigest())
    assert parent.read_bytes() == before
    assert record["compiler_define"] == FLAG and record["extra_ub_bytes"] == EXTRA_UB_BYTES == 4352
    assert record["candidate_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not record["full_kda_evaluated"] and not record["serving_evaluated"]
    with pytest.raises(FileExistsError):
        stage(parent, target, record["source_sha256"])
    with pytest.raises(ValueError, match="qualified parent"):
        stage(parent, tmp_path / "wrong.h", "0" * 64)
    assert not (tmp_path / "wrong.h").exists()
    with pytest.raises(ValueError, match="already staged"):
        transform(target.read_text())
    with pytest.raises(ValueError, match="anchor changed"):
        transform(
            HEADER.read_text().replace(
                "ReduceDotProduct310P(\n                    scoreRow", "other(\n                    scoreRow"
            )
        )


def test_vector_reduction_requires_qualified_cached_full_chunks():
    source = transform(HEADER.read_text())
    assert "curT == 64 && cacheColumns" in source
    assert "const bool vectorScoreReduce = false;" in source
    assert "SAFE_GATE && IsSameType<T, half>::value" in source
