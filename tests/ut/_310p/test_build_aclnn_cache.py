# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for incremental 310P custom-op builds."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_kernel_changes_invalidate_copy_and_compile_stamps():
    build_script = (REPO_ROOT / "csrc" / "build_aclnn.sh").read_text()

    assert 'find "${op_path}/op_kernel" -type f -newer "${source_stamp}"' in build_script
    assert '"${op_path}/op_host/CMakeLists.txt" -nt "${source_stamp}"' in build_script
    assert 'rm -f -- "${source_stamp}"' in build_script
    assert '"${binary_root}/src/${op_name}" -maxdepth 1 -type f' in build_script
    assert "-name '*.py' -delete" in build_script
    assert 'cmp -s "${generated_launcher}" "${copied_launcher}"' in build_script
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
        REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "chunk_kda_fwd_def.cpp"
    ).read_text()

    assert '"chunk_kda_fwd"' in build_script
    assert 'AddConfig("ascend310p", config)' in op_definition


def test_chunk_kda_uses_unified_default_task_type_on_310p():
    kernel = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd.cpp").read_text()
    cmake = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "CMakeLists.txt").read_text()
    entry_signature = 'extern "C" __global__ __aicore__ void chunk_kda_fwd('
    assert kernel.count(entry_signature) == 1

    arch20_start = kernel.rindex("#if defined(KDA_310P_DEFAULT_TASK)")
    arch20_end = kernel.index("#else", arch20_start)
    task_selection_end = kernel.index("#endif", arch20_end)
    arch20_entry = kernel[arch20_start:arch20_end]

    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AICORE);" in arch20_entry
    assert "KERNEL_TASK_TYPE(1" not in arch20_entry
    assert "KERNEL_TASK_TYPE(2" not in arch20_entry
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);" in kernel[arch20_end:task_selection_end]
    assert "KdaForward::RunKernel(" in kernel[task_selection_end:]
    assert "KERNEL_TASK_TYPE(1" not in kernel
    assert "KERNEL_TASK_TYPE(2" not in kernel
    assert 'if("ascend310p" IN_LIST ASCEND_COMPUTE_UNIT)' in cmake
    assert "OPTIONS -DKDA_310P_DEFAULT_TASK=1" in cmake


def test_chunk_kda_dispatches_without_keyed_runtime_selection_on_310p():
    kernel = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel" / "chunk_kda_fwd.cpp").read_text()

    arch20_dispatch = kernel[
        kernel.index(
            "#if defined(KDA_310P_DEFAULT_TASK)", kernel.index("GET_TILING_DATA_WITH_STRUCT")
        ) : kernel.index("if (TILING_KEY_IS(1))")
    ]
    assert "tilingData.chunkSize == 64" in arch20_dispatch
    assert "tilingData.kHeadDim == 128" in arch20_dispatch
    assert "tilingData.vHeadDim == 128" in arch20_dispatch
    assert "DispatchGenericSafeGate" in arch20_dispatch
    assert arch20_dispatch.rstrip().endswith("#else")


def test_chunk_kda_uses_unified_default_tiling_key_on_310p():
    tiling = (REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "chunk_kda_fwd_tiling.cpp").read_text()

    assert "isAscend310P ? 0 : (useChunk64K128V128Template ? 2 : 1)" in tiling
    assert "SocVersion::ASCEND310P" in tiling


def test_chunk_kda_normalizes_mixed_vector_core_indices_on_310p():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    common = (kernel_dir / "chunk_kda_fwd_common.h").read_text()
    gate = (
        REPO_ROOT
        / "csrc"
        / "attention"
        / "kda_gate_cumsum"
        / "op_kernel"
        / "kda_gate_cumsum_kernel.h"
    ).read_text()
    tiling = (
        REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "chunk_kda_fwd_tiling.cpp"
    ).read_text()

    assert "return static_cast<uint64_t>(block_idx);" in common
    assert "uint64_t coreIdx = static_cast<uint64_t>(block_idx);" in gate
    assert "(isAscend310P ? 1 : 2)" in tiling

    for filename in (
        "chunk_kda_fwd_prepare.h",
        "chunk_kda_fwd_post_wu.h",
        "chunk_kda_fwd_finalize.h",
    ):
        source = (kernel_dir / filename).read_text()
        assert "KdaForward::GetPhysicalBlockIdx()" in source
        assert "GetBlockIdx()" not in source


def test_chunk_kda_emits_both_unified_core_pipelines_on_310p():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    arch20_guard = "#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)"

    common = (kernel_dir / "chunk_kda_fwd_common.h").read_text()
    gate_dispatch = common[common.index("void RunGateCumsum") : common.index("void RunFrontEnd")]
    assert arch20_guard in gate_dispatch
    assert "CompilesVectorPipeline()" in gate_dispatch
    assert "DispatchKdaGateCumsum" in gate_dispatch

    expected_pipelines = {
        "chunk_kda_fwd_prepare.h": ("op.ProcessAic();", "op.ProcessAiv();"),
        "chunk_kda_fwd_post_wu.h": ("op.ProcessAic();", "op.ProcessAiv();"),
        "chunk_kda_fwd_finalize.h": ("op.ProcessAic();", "op.ProcessAiv();"),
    }
    for filename, calls in expected_pipelines.items():
        source = (kernel_dir / filename).read_text()
        assert source.count(arch20_guard) >= 2
        assert "CompilesCubePipeline()" in source
        assert "CompilesVectorPipeline()" in source
        assert all(call in source for call in calls)


def test_chunk_kda_uses_fp16_score_workspace_on_310p():
    prepare = (
        REPO_ROOT
        / "csrc"
        / "attention"
        / "chunk_kda_fwd"
        / "op_kernel"
        / "chunk_kda_fwd_prepare.h"
    ).read_text()
    score_type = prepare[prepare.index("using AKK_T") : prepare.index("template <typename TilingData>")]

    assert "__CCE_AICORE__ == 200" in score_type
    assert "using SCORE_T = T;" in score_type


def test_chunk_kda_uses_no_fixpipe_mmad_on_310p():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    unified_policy = "MmadPingpongTlaMulti<KdaArchTag, true, false>"

    for filename in (
        "chunk_kda_fwd_prepare.h",
        "chunk_kda_fwd_post_wu.h",
        "chunk_kda_fwd_finalize.h",
    ):
        source = (kernel_dir / filename).read_text()
        policy_block = source[source.index("using KdaArchTag") : source.index("using KdaL1TileShape")]
        assert policy_block.count(unified_policy) == 2


def test_chunk_kda_keeps_fp32_solve_off_310p_cube():
    prepare = (
        REPO_ROOT
        / "csrc"
        / "attention"
        / "chunk_kda_fwd"
        / "op_kernel"
        / "chunk_kda_fwd_prepare.h"
    ).read_text()

    capability_end = prepare.index("using KdaArchTag")
    capability_start = prepare.rindex("#if defined(__CCE_AICORE__)", 0, capability_end)
    capability = prepare[capability_start:capability_end]
    assert "__CCE_AICORE__ == 200" in capability
    assert "KDA_SUPPORTS_FP32_CUBE_SOLVE = false" in capability
    assert prepare.count("if constexpr (KDA_SUPPORTS_FP32_CUBE_SOLVE)") >= 2


def test_chunk_kda_prunes_unsupported_bf16_inputs_on_310p():
    op_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host"
    cmake = (op_dir / "CMakeLists.txt").read_text()
    op_def = (op_dir / "chunk_kda_fwd_def.cpp").read_text()
    op_def_310p = (op_dir / "310p" / "chunk_kda_fwd_def.cpp").read_text()
    api = (op_dir / "op_api" / "aclnn_chunk_kda_fwd.cpp").read_text()

    assert '"ascend310p" IN_LIST ASCEND_COMPUTE_UNIT' in cmake
    assert "310p/chunk_kda_fwd_def.cpp" in cmake
    assert "ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16" in op_def_310p
    assert "ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16" not in op_def_310p
    assert 'this->AICore().AddConfig("ascend310p", config)' in op_def_310p
    assert "Ascend 310P requires float16 q, k and v." in api
    signature_pattern = r'this->(?:Input|Output|Attr)\("([^"]+)"\)'
    assert re.findall(signature_pattern, op_def_310p) == re.findall(signature_pattern, op_def)


def test_chunk_kda_uses_physical_stage_boundaries_on_310p():
    api = (
        REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_host" / "op_api" / "aclnn_chunk_kda_fwd.cpp"
    ).read_text()

    assert 'std::strstr(socName, "Ascend310P")' in api
    assert "const bool splitStages = IsAscend310P() ||" in api


def test_kda_varlen_boundaries_use_310p_scalar_global_reads():
    op_dir = REPO_ROOT / "csrc" / "attention" / "kda_gate_cumsum" / "op_kernel"
    for source_name in ("kda_gate_cumsum.cpp", "kda_gate_cumsum_kernel.h"):
        kernel = (op_dir / source_name).read_text()
        read_int64 = kernel[kernel.index("ReadInt64") : kernel.index("ExpScalar")]

        assert "__CCE_AICORE__ == 200" in read_int64
        assert "return tensor.GetValue(offset);" in read_int64


def test_chunk_kda_helpers_are_registered_for_310p():
    for op_name in ("kda_gate_cumsum", "kda_layout_swap12"):
        op_definition = (REPO_ROOT / "csrc" / "attention" / op_name / "op_host" / f"{op_name}_def.cpp").read_text()

        assert 'AddConfig("ascend310p", aicoreConfig)' in op_definition


def test_kda_layout_swap_avoids_keyed_task_types_on_310p():
    kernel = (
        REPO_ROOT / "csrc" / "attention" / "kda_layout_swap12" / "op_kernel" / "kda_layout_swap12.cpp"
    ).read_text()

    guard = "#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)"
    for tiling_key in range(3):
        task_type = f"KERNEL_TASK_TYPE({tiling_key}"
        assert kernel.rindex(guard, 0, kernel.index(task_type)) >= 0


def test_chunk_kda_helpers_exclude_bf16_on_310p():
    for op_name in ("kda_gate_cumsum", "kda_layout_swap12"):
        op_dir = REPO_ROOT / "csrc" / "attention" / op_name
        kernel = (op_dir / "op_kernel" / f"{op_name}.cpp").read_text()
        tiling = (op_dir / "op_host" / f"{op_name}_tiling.cpp").read_text()

        arch_guard = "#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)"
        assert kernel.index(arch_guard) < kernel.index("bfloat16_t")
        assert "SocVersion::ASCEND310P" in tiling
        assert "ge::DT_BF16" in tiling


def test_chunk_kda_loads_310p_compat_before_catlass_kernels():
    kernel_dir = REPO_ROOT / "csrc" / "attention" / "chunk_kda_fwd" / "op_kernel"
    common_header = (kernel_dir / "chunk_kda_fwd_common.h").read_text()
    compat_header = (kernel_dir / "arch20" / "compat_310p.h").read_text()

    compat_include = '#include "arch20/compat_310p.h"'
    first_catlass_kernel = '#include "../../kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h"'
    assert common_header.index(compat_include) < common_header.index(first_catlass_kernel)
    assert "#define CATLASS_UNIFIED_CORE 1" in compat_header
    assert "struct bfloat16_t" in compat_header
    assert "#define PIPE_FIX PIPE_MTE3" in compat_header
    assert "#define LoadDataWithSparse LoadDataWithSparseCal" in compat_header


def test_generic_kda_state_kernel_avoids_host_debug_header_on_310p():
    kernel = (
        REPO_ROOT
        / "csrc"
        / "moe"
        / "chunk_gated_delta_rule_fwd_h"
        / "op_kernel"
        / "gemm"
        / "kernel"
        / "gdn_fwd_h_kernel.hpp"
    ).read_text()

    arch_guard = "#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)"
    assert kernel.index(arch_guard) < kernel.index('#include "catlass/debug.hpp"')
