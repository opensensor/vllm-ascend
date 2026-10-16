# The parent attention build discovers directories automatically. Keep this
# unqualified private operator opt-in, including when ASCEND_OP_NAME is all.
option(SIDECARD_BUILD_KDA_V310 "Build private sidecard KDA decode candidate" OFF)
if (SIDECARD_BUILD_KDA_V310)
    add_subdirectory(op_host)
endif()
