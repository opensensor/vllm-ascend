# Authenticated private profile, explicitly opted into by the execution builder.
# The same cache is loaded by outer CMake and its nested prepare_build project.
set(SIDECARD_BUILD_KDA_V310 ON CACHE BOOL "Private KDA candidate" FORCE)
set(SIDECARD_BUILD_MHC_PRE_V310 ON CACHE BOOL "Private mHC pre candidate" FORCE)
set(SIDECARD_BUILD_MHC_POST_V310 ON CACHE BOOL "Private mHC post candidate" FORCE)
get_filename_component(_sidecard_prepared_third_party "${CMAKE_CURRENT_LIST_DIR}/../third_party" ABSOLUTE)
set(ASCEND_PROTOBUF_SHARED_INCLUDE "${_sidecard_prepared_third_party}/ascend_protobuf/src"
    CACHE PATH "Authenticated prepared protobuf headers" FORCE)
set(ASCEND_PROTOC "${_sidecard_prepared_third_party}/ascend_protobuf/bin"
    CACHE PATH "Authenticated prepared protoc directory" FORCE)
set(FORCE_REBUILD_CANN_3RD OFF CACHE BOOL "Reuse authenticated prepared dependencies" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "No dependency downloads" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Private execution kernel build type" FORCE)
