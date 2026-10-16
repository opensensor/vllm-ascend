# Automatic directory discovery must not enable an unqualified candidate,
# including when ASCEND_OP_NAME is all. This is a CMake cache option only.
option(SIDECARD_BUILD_MHC_POST_V310 "Build private sidecard mHC post candidate" OFF)
if (SIDECARD_BUILD_MHC_POST_V310)
    add_subdirectory(op_host)
endif()
