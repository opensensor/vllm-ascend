# SPDX-License-Identifier: Apache-2.0

# Detect a package whose selected operators have AI CPU kernels exclusively.
# AI Core metadata generators cannot run for such a package because opbuild
# does not emit any aic-*-ops-info.ini file.
function(detect_aicpu_only_package result source_dir enabled)
    set(${result} OFF PARENT_SCOPE)
    if(NOT enabled OR ARGC LESS 4)
        return()
    endif()

    foreach(op_name IN LISTS ARGN)
        if(op_name STREQUAL "all" OR op_name STREQUAL "ALL")
            return()
        endif()

        file(GLOB op_dirs LIST_DIRECTORIES true
            "${source_dir}/*/${op_name}"
            "${source_dir}/experimental/*/${op_name}")
        if(NOT op_dirs)
            return()
        endif()
        foreach(op_dir IN LISTS op_dirs)
            if(NOT IS_DIRECTORY "${op_dir}" OR
               NOT EXISTS "${op_dir}/op_kernel_aicpu" OR
               EXISTS "${op_dir}/op_kernel")
                return()
            endif()
        endforeach()
    endforeach()

    set(${result} ON PARENT_SCOPE)
endfunction()
