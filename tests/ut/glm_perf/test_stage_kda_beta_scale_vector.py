# SPDX-License-Identifier: Apache-2.0
"""Frozen parent checks, disabled-path preservation and isolated staging."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.glm_perf.stage_kda_beta_scale_vector import EXTRA_UB_BYTES, FLAG, stage, transform

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"


def test_disabled_flag_keeps_the_complete_scalar_method_tokens(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ preprocessor")
    original = HEADER.read_text()
    candidate = transform(original)
    results = []
    for source in (original, candidate):
        start = source.index("    __aicore__ inline void ScaleTailRowsByBeta310P(")
        end = source.index("    __aicore__ inline void ScaleRowsByBeta310P(", start)
        path = tmp_path / "scalar.cpp"
        path.write_text(source[start:end])
        result = subprocess.run([compiler, "-E", "-P", str(path)], check=True, capture_output=True, text=True)
        results.append(result.stdout.split())
    assert results[0] == results[1]
    assert original == HEADER.read_text(), "staging must not alter the production header"


def test_stage_owns_new_file_and_pins_qualified_parent(tmp_path):
    source = tmp_path / "parent.h"
    source.write_bytes(HEADER.read_bytes())
    before = source.read_bytes()
    expected = hashlib.sha256(before).hexdigest()
    destination = tmp_path / "candidate.h"
    record = stage(source, destination, expected)
    assert source.read_bytes() == before
    assert record["compiler_define"] == FLAG and record["extra_ub_bytes"] == EXTRA_UB_BYTES == 2048
    assert record["candidate_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert record["full_kda_evaluated"] is False and record["serving_evaluated"] is False
    with pytest.raises(FileExistsError):
        stage(source, destination, expected)
    with pytest.raises(ValueError, match="qualified parent"):
        stage(source, tmp_path / "wrong.h", "0" * 64)
    assert not (tmp_path / "wrong.h").exists()


def test_changed_or_repeated_source_refused_before_mutation():
    original = HEADER.read_text()
    with pytest.raises(ValueError, match="already staged"):
        transform(transform(original))
    with pytest.raises(ValueError, match="scalar method changed"):
        transform(original.replace("FloatToType<T>(value * betaScale)", "other(value)"))
    with pytest.raises(ValueError, match="anchor changed"):
        transform(original.replace("        (void)rowLocal;\n", ""))
