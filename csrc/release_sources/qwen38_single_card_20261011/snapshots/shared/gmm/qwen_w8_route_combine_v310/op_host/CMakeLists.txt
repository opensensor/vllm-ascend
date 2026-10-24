# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE qwen_w8_route_combine_v310_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME QwenW8RouteCombineV310
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
    -I${CMAKE_SOURCE_DIR}/third_party/catlass/include
    -I${CMAKE_SOURCE_DIR}/moe/common
)
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE qwen_w8_route_combine_v310 ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
