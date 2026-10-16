# The root/custom GMM discovery paths can add op_host directly, bypassing
# the package-level CMakeLists.txt. Keep admission guarded at this entry.
option(SIDECARD_BUILD_MHC_POST_V310 "Build private sidecard mHC post candidate" OFF)
if (NOT SIDECARD_BUILD_MHC_POST_V310)
    return()
endif()
# Root and GMM traversals can reach this file in the same configuration.
include_guard(GLOBAL)

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE sidecard_mhc_post_v310_def.cpp)
endif()
add_ops_compile_options(OP_NAME SidecardMhcPostV310 OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sidecard_mhc_post_v310 ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
