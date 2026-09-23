# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for the Ascend 310P custom-op package manifest."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO_ROOT / "csrc" / "build_aclnn.sh"
QSA_SPARSE_ATTENTION_DEF = (
    REPO_ROOT
    / "csrc"
    / "attention"
    / "qsa_sparse_attention_v310"
    / "op_host"
    / "qsa_sparse_attention_v310_def.cpp"
)

QWEN4EXP_RUNTIME_OPS = {
    "causal_conv1d_v310",
    "gdn_gating_v310",
    "recurrent_gated_delta_rule_v310",
    "chunk_fwd_o",
    "chunk_gated_delta_rule_fwd_h",
    "qsa_index_cache_update_v310",
    "qsa_indexer_score_v310",
    "qsa_sparse_attention_v310",
}


def _ascend310_ops(script: str) -> set[str]:
    branch = script.split('if [[ "$SOC_VERSION" =~ ^ascend310 ]]', 1)[1]
    branch = branch.split('elif [[ "$SOC_VERSION"', 1)[0]
    array = branch.split("CUSTOM_OPS_ARRAY=(", 1)[1].split(")", 1)[0]
    return set(re.findall(r'^\s*"([a-z0-9_]+)"\s*$', array, re.MULTILINE))


def test_qwen4exp_runtime_ops_are_packaged_together() -> None:
    script = BUILD_SCRIPT.read_text()
    missing_ops = QWEN4EXP_RUNTIME_OPS - _ascend310_ops(script)
    assert not missing_ops, f"310P custom-op package is missing: {sorted(missing_ops)}"


def test_custom_op_build_does_not_reuse_stale_cmake_manifest() -> None:
    script = BUILD_SCRIPT.read_text()
    assert "rm -rf -- build output build_out" in script


def test_qsa_sparse_attention_accepts_runtime_nz_kv_cache() -> None:
    op_def = QSA_SPARSE_ATTENTION_DEF.read_text()
    for cache_name in ("keyCache", "valueCache"):
        cache_input = op_def.split(f'this->Input("{cache_name}")', 1)[1].split(";", 1)[0]
        assert "FormatList({ge::FORMAT_FRACTAL_NZ})" in cache_input
