# ClunkOpt.cmake — clunk_opt() function for drop-in superoptimisation
#
# Usage:
#   find_package(Clunk)
#   clunk_opt(my_target)
#
# This runs the Clunk superoptimiser on the given target's source files
# after compilation. Clunk operates on LLVM IR, so this function:
#   1. Emits the target's LLVM IR using Clang's -emit-llvm flag
#   2. Runs the Clunk optimiser on each IR file
#   3. Compiles the optimised IR back into object files
#   4. Relinks the target with the optimised objects

function(clunk_opt target)
    if(NOT Clunk_FOUND)
        message(WARNING "Clunk not found — clunk_opt(${target}) is a no-op. "
                        "Install Clunk or set Clunk_DIR to enable superoptimisation.")
        return()
    endif()

    # Find the clunk executable
    find_program(CLUNK_EXECUTABLE
        NAMES clunk
        PATHS "${Clunk_LIB_DIR}/../bin" "${Clunk_LIB_DIR}/bin"
              /usr/local/bin /usr/bin
        NO_DEFAULT_PATH
    )
    if(NOT CLUNK_EXECUTABLE)
        find_program(CLUNK_EXECUTABLE NAMES clunk)
    endif()
    if(NOT CLUNK_EXECUTABLE)
        message(WARNING "clunk_opt(${target}): 'clunk' executable not found — skipping")
        return()
    endif()

    # Find clang (needed to emit LLVM IR)
    find_program(CLUNK_CLANG
        NAMES clang clang-19 clang-18 clang-17 clang-16 clang-15
        PATHS /usr/local/bin /usr/bin
    )
    if(NOT CLUNK_CLANG)
        message(WARNING "clunk_opt(${target}): clang not found — cannot emit LLVM IR. "
                        "Install clang to enable clunk_opt().")
        return()
    endif()

    # Get target source files
    get_target_property(SOURCES ${target} SOURCES)
    if(NOT SOURCES)
        message(WARNING "clunk_opt(${target}): target has no sources — skipping")
        return()
    endif()

    # Filter to C/C++ sources only (skip .h, .rc, .manifest, etc.)
    set(IR_SOURCES "")
    foreach(src ${SOURCES})
        string(TOLOWER "${src}" src_lower)
        if(src_lower MATCHES "\\.(c|cc|cpp|cxx|c\\+\\+)$")
            list(APPEND IR_SOURCES "${src}")
        endif()
    endforeach()

    if(NOT IR_SOURCES)
        message(WARNING "clunk_opt(${target}): no C/C++ sources found — skipping")
        return()
    endif()

    list(LENGTH IR_SOURCES NUM_SOURCES)
    math(EXPR ESTIMATE_SECONDS "${NUM_SOURCES} * 30")

    message(STATUS "Clunk: clunk_opt(${target}) — ${NUM_SOURCES} source file(s), "
                   "estimated ${ESTIMATE_SECONDS}s optimisation time")

    # Get compile flags from the target
    get_target_property(INCLUDE_DIRS ${target} INCLUDE_DIRECTORIES)
    get_target_property(COMPILE_DEFS ${target} COMPILE_DEFINITIONS)
    get_target_property(CXX_STANDARD ${target} CXX_STANDARD)

    # Build compile flags string
    set(COMPILE_FLAGS "")
    if(CXX_STANDARD)
        string(APPEND COMPILE_FLAGS " -std=c++${CXX_STANDARD}")
    endif()
    if(INCLUDE_DIRS)
        foreach(dir ${INCLUDE_DIRS})
            string(APPEND COMPILE_FLAGS " -I${dir}")
        endforeach()
    endif()
    if(COMPILE_DEFS)
        foreach(def ${COMPILE_DEFS})
            string(APPEND COMPILE_FLAGS " -D${def}")
        endforeach()
    endif()

    # Directory for IR files and optimised objects
    set(CLUNK_WORK_DIR "${CMAKE_CURRENT_BINARY_DIR}/clunk_${target}")
    file(MAKE_DIRECTORY ${CLUNK_WORK_DIR})

    # Opt-level can be overridden by setting CLUNK_OPT_LEVEL
    if(NOT DEFINED CLUNK_OPT_LEVEL)
        set(CLUNK_OPT_LEVEL 2)
    endif()

    # Generate commands: for each source, emit IR → optimise → compile back
    set(OPTIMISED_OBJS "")
    set(ALL_LL_FILES "")
    set(ALL_OPT_LL_FILES "")

    set(src_idx 0)
    foreach(src ${IR_SOURCES})
        math(EXPR src_idx "${src_idx} + 1")

        # Get absolute source path
        if(IS_ABSOLUTE "${src}")
            set(src_abs "${src}")
        else()
            set(src_abs "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        endif()

        # Derive base name
        get_filename_component(src_base "${src}" NAME_WE)
        set(ll_file    "${CLUNK_WORK_DIR}/${src_base}.ll")
        set(opt_ll     "${CLUNK_WORK_DIR}/${src_base}_opt.ll")
        set(obj_file   "${CLUNK_WORK_DIR}/${src_base}_opt.o")

        # Step 1: Emit LLVM IR from source using Clang
        add_custom_command(
            OUTPUT "${ll_file}"
            COMMAND ${CLUNK_CLANG} -S -emit-llvm ${COMPILE_FLAGS} -o "${ll_file}" "${src_abs}"
            DEPENDS "${src_abs}"
            COMMENT "Clunk: emitting LLVM IR for ${src}"
            VERBATIM
        )

        # Step 2: Run Clunk optimiser on the IR
        add_custom_command(
            OUTPUT "${opt_ll}"
            COMMAND ${CLUNK_EXECUTABLE} --opt-level ${CLUNK_OPT_LEVEL} --output "${opt_ll}" "${ll_file}"
            DEPENDS "${ll_file}" ${CLUNK_EXECUTABLE}
            COMMENT "Clunk: optimising ${src}"
            VERBATIM
        )

        # Step 3: Compile optimised IR back to object file
        add_custom_command(
            OUTPUT "${obj_file}"
            COMMAND ${CLUNK_CLANG} -c "${opt_ll}" -o "${obj_file}"
            DEPENDS "${opt_ll}"
            COMMENT "Clunk: compiling optimised ${src}"
            VERBATIM
        )

        list(APPEND OPTIMISED_OBJS "${obj_file}")
        list(APPEND ALL_LL_FILES "${ll_file}")
        list(APPEND ALL_OPT_LL_FILES "${opt_ll}")
    endforeach()

    # Create a custom target that depends on all optimised objects
    set(clunk_target "clunk_opt_${target}")
    add_custom_target(${clunk_target} ALL
        DEPENDS ${OPTIMISED_OBJS}
        COMMENT "Clunk: all optimised objects for ${target}"
    )

    # Make the actual target depend on the clunk target
    # and link the optimised objects
    add_dependencies(${target} ${clunk_target})
    target_link_libraries(${target} PRIVATE ${OPTIMISED_OBJS})

    message(STATUS "Clunk: clunk_opt(${target}) configured — "
                   "${NUM_SOURCES} file(s) will be superoptimised at build time")
endfunction()
