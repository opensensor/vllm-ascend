# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for incremental 310P custom-op builds."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_kernel_changes_invalidate_copy_and_compile_stamps():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert 'find "${op_path}/op_kernel" -type f -newer "${source_stamp}"' in build_script
    assert 'rm -f -- "${source_stamp}"' in build_script
    assert '-name "${op_name}_${SOC_ARG}_*.done"' in build_script


def test_w2_shared_kernel_changes_invalidate_compile_stamps():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert '"${op_name}" == "w2_blocked_dequant_matmul_v310"' in build_script
    assert 'find "${ROOT_DIR}/csrc/moe/common/kernel_utils" -type f' in build_script


def test_host_changes_invalidate_generated_operator_metadata():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert 'find "${op_path}/op_host" -type f -newer "${generated_proto}"' in build_script
    assert 'rm -f -- "${generated_proto}"' in build_script


def test_gcc15_protobuf_build_includes_cstdint():
    protobuf_cmake = (REPO_ROOT / "csrc" / "cmake" / "third_party" / "ascend_protobuf.cmake").read_text()

    assert "CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15" in protobuf_cmake
    assert 'string(APPEND protobuf_CXXFLAGS " -include cstdint")' in protobuf_cmake


def test_chunk_kda_is_registered_and_built_for_310p():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()
    op_definition = (
        REPO_ROOT
        / "csrc"
        / "attention"
        / "chunk_kda_fwd"
        / "op_host"
        / "chunk_kda_fwd_def.cpp"
    ).read_text()

    assert '"chunk_kda_fwd"' in build_script
    assert 'AddConfig("ascend310p", config)' in op_definition


def test_chunk_kda_uses_only_default_task_type_on_310p():
    kernel = (
        REPO_ROOT
        / "csrc"
        / "attention"
        / "chunk_kda_fwd"
        / "op_kernel"
        / "chunk_kda_fwd.cpp"
    ).read_text()

    guard = "#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)"
    assert kernel.index(guard) < kernel.index("KERNEL_TASK_TYPE(1")
    assert kernel.index(guard) < kernel.index("KERNEL_TASK_TYPE(2")
