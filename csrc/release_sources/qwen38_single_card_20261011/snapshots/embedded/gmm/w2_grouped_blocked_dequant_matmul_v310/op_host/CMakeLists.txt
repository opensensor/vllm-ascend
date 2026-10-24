# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE w2_grouped_blocked_dequant_matmul_v310_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME W2GroupedBlockedDequantMatmulV310
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
    -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
    -I${CMAKE_SOURCE_DIR}/moe/common
    -I${CMAKE_SOURCE_DIR}/gmm/w2_blocked_dequant_matmul_v310/op_kernel
)
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE w2_grouped_blocked_dequant_matmul_v310 ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
