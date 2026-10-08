# SPDX-License-Identifier: Apache-2.0
"""Protect the inactive path, source identity and writable-cache alias boundary."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.glm_perf.stage_kda_gated_key_reuse import EXTRA_UB_BYTES, FLAG, stage, transform
from tools.glm_perf.stage_kda_score_vector_reduce import transform as score_transform

HEADER = Path(__file__).resolve().parents[3] / "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"


def test_disabled_reuse_preserves_reference_preprocessed_tokens(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host preprocessor")
    outputs = []
    parent = score_transform(HEADER.read_text())
    for source in (parent, transform(parent)):
        begin = source.index("    __aicore__ inline void ComputeRawAqkAkkVector310P(")
        end = source.index("\n    }\n#endif", begin)
        path = tmp_path / "source.cpp"
        path.write_text(source[begin:end])
        outputs.append(
            subprocess.run([compiler, "-E", "-P", str(path)], check=True, capture_output=True).stdout.split()
        )
    assert outputs[0] == outputs[1]


def test_gated_key_staging_preserves_parent_and_rejects_reuse(tmp_path):
    parent = tmp_path / "parent.h"
    parent.write_text(score_transform(HEADER.read_text()))
    original = parent.read_bytes()
    destination = tmp_path / "candidate.h"
    record = stage(parent, destination, hashlib.sha256(original).hexdigest())
    assert parent.read_bytes() == original
    assert record["compiler_define"] == FLAG
    assert record["extra_ub_bytes"] == EXTRA_UB_BYTES == 16384
    assert record["candidate_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert not record["full_kda_evaluated"] and not record["serving_evaluated"]
    with pytest.raises(FileExistsError):
        stage(parent, destination, record["source_sha256"])
    with pytest.raises(ValueError, match="qualified parent"):
        stage(parent, tmp_path / "wrong.h", "0" * 64)
    assert not (tmp_path / "wrong.h").exists()
    with pytest.raises(ValueError, match="already staged"):
        transform(destination.read_text())
    with pytest.raises(ValueError, match="separate score passes"):
        transform(parent.read_text().replace("row < curT", "row < changed"))


def test_each_first_pass_resets_writable_scratch_before_cached_alias_reads():
    source = transform(score_transform(HEADER.read_text()))
    begin = source.index("            LocalTensor<T> gatedKeys =")
    end = source.index("            return;", begin)
    branch = source[begin:end]
    assert branch.index("kGated = typedArena[5 * K_]") < branch.index("kGated = gatedKeys[col * K_]")
    assert branch.count("Sub(gate, gRow, gCol") == 1
    assert branch.count("Mul(product, rowVector, kGated") == 2
    assert branch.count("SetFlag<HardEvent::MTE3_V>") == 2
    assert branch.count("WaitFlag<HardEvent::MTE3_V>") == 2
    assert "SAFE_GATE && IsSameType<T, half>::value && K_ == 128" in source
    assert "BT_ == 64 && curT == 64 && cacheColumns" in source
