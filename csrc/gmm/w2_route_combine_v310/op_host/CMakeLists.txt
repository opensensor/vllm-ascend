# SPDX-License-Identifier: Apache-2.0
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE w2_route_combine_v310_def.cpp)
endif()
add_ops_compile_options(OP_NAME W2RouteCombineV310 OPTIONS --cce-auto-sync=off
    -I${CMAKE_CURRENT_SOURCE_DIR}/..)
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE w2_route_combine_v310 ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
