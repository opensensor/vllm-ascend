# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
add_op_to_compiled_list()
# The grouped kernel includes the standalone W2 kernel header. Register it as
# a source dependency so edits to the shared unpack/dequant code invalidate
# both compiled objects, including the local build-cache identity.
set(w2_grouped_blocked_dequant_matmul_v310_depends
    gmm/w2_blocked_dequant_matmul_v310
    CACHE STRING "Kernel source dependencies for grouped W2/W4" FORCE)
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE w2_grouped_blocked_dequant_matmul_v310_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME W2GroupedBlockedDequantMatmulV310
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
    -DGLM_W2_SCALE_PAIR
    -DGLM_W2_GROUPED_RINT_UNPACK
    -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
    -I${CMAKE_SOURCE_DIR}/moe/common
    -I${CMAKE_SOURCE_DIR}/gmm/w2_blocked_dequant_matmul_v310/op_kernel
)
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE w2_grouped_blocked_dequant_matmul_v310 ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
