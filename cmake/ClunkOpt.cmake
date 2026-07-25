# ClunkOpt.cmake — clunk_opt() function for drop-in superoptimisation
#
# Usage:
#   find_package(Clunk REQUIRED)
#   clunk_opt(my_target)
#
# This replaces the target's C/C++ source compilations with a
# source → LLVM IR → clunk superoptimiser → object file pipeline.
# Non-C/C++ sources (headers, assembly, .rc, etc.) are left untouched.
#
# The original C/C++ sources are removed from the target's source list
# and replaced with the optimised object files, avoiding duplicate symbols.
#
# Options (set before calling clunk_opt):
#   CLUNK_TIME_BUDGET     — seconds per source file (default: 10)
#   CLUNK_OPT_LEVEL       — clunk optimisation level (default: 2)
#   CLUNK_ENABLE_CACHE    — enable content-hash caching (default: ON)
#   CLUNK_CACHE_DIR       — override cache directory
#                           (default: ${CMAKE_BINARY_DIR}/.clunk_cache)
#   CLUNK_CLANG           — override clang executable path
#
# Limitations:
#   - Compile flags are reconstructed from CMake target properties.
#     Generator expressions containing $<...> are filtered out.
#     For complex projects, consider integrating with compile_commands.json.
#   - Config-specific flags (e.g. CMAKE_CXX_FLAGS_RELEASE) are included
#     for the active CMAKE_BUILD_TYPE but not for multi-config generators
#     (Visual Studio, Xcode, Ninja Multi-Config).
#   - Caching tracks source content + flags + clunk version; header changes
#     are not tracked. Use CLUNK_ENABLE_CACHE=OFF during active development.

# ---------------------------------------------------------------------------
# Helper: collect INTERFACE compile flags from a target's linked libraries.
#
# Appends -I, -isystem, -D, and raw option flags from each linked library's
# INTERFACE_* properties to the _accumulated_flags string.
# ---------------------------------------------------------------------------
function(_clunk_collect_interface_flags accumulated_flags_var target)
    set(_flags "${${accumulated_flags_var}}")

    get_target_property(_link_libs "${target}" LINK_LIBRARIES)
    if(NOT _link_libs)
        set(${accumulated_flags_var} "${_flags}" PARENT_SCOPE)
        return()
    endif()

    foreach(_lib ${_link_libs})
        # Only process CMake targets (skip plain paths, linker flags, etc.)
        if(NOT TARGET "${_lib}")
            continue()
        endif()

        # Resolve ALIAS targets to their underlying target
        get_target_property(_alias "${_lib}" ALIAS_TARGET)
        if(_alias AND TARGET "${_alias}")
            set(_lib "${_alias}")
        endif()

        # INTERFACE_INCLUDE_DIRECTORIES
        get_target_property(_iface_inc "${_lib}" INTERFACE_INCLUDE_DIRECTORIES)
        if(_iface_inc)
            foreach(_d ${_iface_inc})
                if(NOT _d MATCHES "[\\$<]")
                    string(APPEND _flags " -I${_d}")
                endif()
            endforeach()
        endif()

        # INTERFACE_SYSTEM_INCLUDE_DIRECTORIES → -isystem
        get_target_property(_iface_sys_inc "${_lib}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
        if(_iface_sys_inc)
            foreach(_d ${_iface_sys_inc})
                if(NOT _d MATCHES "[\\$<]")
                    string(APPEND _flags " -isystem${_d}")
                endif()
            endforeach()
        endif()

        # INTERFACE_COMPILE_DEFINITIONS
        get_target_property(_iface_defs "${_lib}" INTERFACE_COMPILE_DEFINITIONS)
        if(_iface_defs)
            foreach(_d ${_iface_defs})
                if(NOT _d MATCHES "[\\$<]")
                    string(APPEND _flags " -D${_d}")
                endif()
            endforeach()
        endif()

        # INTERFACE_COMPILE_OPTIONS
        get_target_property(_iface_opts "${_lib}" INTERFACE_COMPILE_OPTIONS)
        if(_iface_opts)
            foreach(_o ${_iface_opts})
                if(NOT _o MATCHES "[\\$<]")
                    string(APPEND _flags " ${_o}")
                endif()
            endforeach()
        endif()
    endforeach()

    set(${accumulated_flags_var} "${_flags}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Helper: extract code-generation flags relevant for .ll → .o compilation.
#
# When compiling LLVM IR back to an object file, flags like -I, -D, -std
# are unnecessary (they're already encoded in the IR).  Only codegen flags
# such as -fPIC, -march, -mcpu, etc. are needed.
# ---------------------------------------------------------------------------
function(_clunk_extract_codegen_flags cg_flags_var common_opts_var pic_flag)
    set(_cg "${${cg_flags_var}}")

    if(pic_flag)
        string(APPEND _cg " -fPIC")
    endif()

    # Extract -m* flags (architecture / CPU tuning) from compile options
    foreach(_o ${${common_opts_var}})
        if(_o MATCHES "^-m")
            string(APPEND _cg " ${_o}")
        endif()
    endforeach()

    set(${cg_flags_var} "${_cg}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
function(clunk_opt target)

    # ===================================================================
    # 1. Validate prerequisites
    # ===================================================================
    if(NOT Clunk_FOUND)
        message(WARNING "Clunk not found — clunk_opt(${target}) is a no-op. "
                        "Install Clunk or set Clunk_DIR to enable superoptimisation.")
        return()
    endif()

    # ===================================================================
    # 2. Find the clunk executable
    # ===================================================================
    if(NOT DEFINED CLUNK_EXECUTABLE OR NOT CLUNK_EXECUTABLE)
        find_program(CLUNK_EXECUTABLE
            NAMES clunk
            PATHS "${Clunk_LIB_DIR}/../bin" "${Clunk_LIB_DIR}/bin"
                  /usr/local/bin /usr/bin
            NO_DEFAULT_PATH
        )
        if(NOT CLUNK_EXECUTABLE)
            find_program(CLUNK_EXECUTABLE NAMES clunk)
        endif()
    endif()

    if(NOT CLUNK_EXECUTABLE OR NOT EXISTS "${CLUNK_EXECUTABLE}")
        message(WARNING "clunk_opt(${target}): 'clunk' executable not found — skipping")
        return()
    endif()

    # ===================================================================
    # 3. Find clang — prefer the project's own compiler (avoids version
    #    mismatch), fall back to searching PATH for a clang binary.
    # ===================================================================
    if(NOT DEFINED CLUNK_CLANG OR NOT CLUNK_CLANG)
        set(CLUNK_CLANG "${CMAKE_CXX_COMPILER}")
        # If the project compiler is not clang-based (e.g. GCC), search PATH
        if(NOT CLUNK_CLANG OR NOT EXISTS "${CLUNK_CLANG}")
            find_program(CLUNK_CLANG
                NAMES clang clang-19 clang-18 clang-17 clang-16 clang-15
            )
        endif()
    endif()

    if(NOT CLUNK_CLANG OR NOT EXISTS "${CLUNK_CLANG}")
        message(WARNING "clunk_opt(${target}): clang not found — cannot emit LLVM IR. "
                        "Install clang or set CLUNK_CLANG to enable clunk_opt().")
        return()
    endif()

    # ===================================================================
    # 4. Set defaults for configurable options
    # ===================================================================
    if(NOT DEFINED CLUNK_TIME_BUDGET)
        set(CLUNK_TIME_BUDGET 10)    # seconds per source file
    endif()
    if(NOT DEFINED CLUNK_OPT_LEVEL)
        set(CLUNK_OPT_LEVEL 2)
    endif()
    if(NOT DEFINED CLUNK_ENABLE_CACHE)
        set(CLUNK_ENABLE_CACHE ON)
    endif()
    if(NOT DEFINED CLUNK_CACHE_DIR)
        set(CLUNK_CACHE_DIR "${CMAKE_BINARY_DIR}/.clunk_cache")
    endif()

    # Obtain clunk version string for cache key invalidation
    execute_process(
        COMMAND "${CLUNK_EXECUTABLE}" --version
        OUTPUT_VARIABLE _clunk_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _clunk_version_rc
    )
    if(NOT _clunk_version_rc EQUAL 0)
        set(_clunk_version "unknown")
    endif()

    # ===================================================================
    # 5. Get target sources — separate C/C++ (to be replaced) from
    #    everything else (headers, assembly, .rc, .manifest, etc.)
    # ===================================================================
    get_target_property(_all_sources "${target}" SOURCES)
    if(NOT _all_sources)
        message(STATUS "Clunk: clunk_opt(${target}) — target has no sources, skipping")
        return()
    endif()

    set(_c_sources "")       # Sources that go through the clunk pipeline
    set(_non_c_sources "")   # Everything else — stays in the target

    foreach(_src ${_all_sources})
        string(TOLOWER "${_src}" _src_lower)
        if(_src_lower MATCHES "\\.(c|cc|cpp|cxx|c\\+\\+)$")
            list(APPEND _c_sources "${_src}")
        else()
            list(APPEND _non_c_sources "${_src}")
        endif()
    endforeach()

    if(NOT _c_sources)
        message(STATUS "Clunk: clunk_opt(${target}) — no C/C++ sources found, skipping")
        return()
    endif()

    list(LENGTH _c_sources _num_sources)
    math(EXPR _estimate_seconds "${_num_sources} * ${CLUNK_TIME_BUDGET}")

    message(STATUS "Clunk: clunk_opt(${target}) — ${_num_sources} source file(s), "
                   "estimated ${_estimate_seconds}s optimisation time")

    # ===================================================================
    # 6. Collect compile flags from the target and its linked libraries
    #
    #    We build two categories:
    #      _common_flags  — flags shared across all languages (includes,
    #                       definitions, options, PIC)
    #      _c_lang_flags  — C-specific flags (CMAKE_C_FLAGS, C_STANDARD)
    #      _cxx_lang_flags — C++ specific flags (CMAKE_CXX_FLAGS, CXX_STANDARD)
    # ===================================================================

    # --- 6a. Target's own properties ---

    set(_common_flags "")
    set(_raw_compile_opts "")   # kept for codegen flag extraction later

    # INCLUDE_DIRECTORIES
    get_target_property(_inc_dirs "${target}" INCLUDE_DIRECTORIES)
    if(_inc_dirs)
        foreach(_d ${_inc_dirs})
            if(NOT _d MATCHES "[\\$<]")
                string(APPEND _common_flags " -I${_d}")
            endif()
        endforeach()
    endif()

    # COMPILE_DEFINITIONS
    get_target_property(_defs "${target}" COMPILE_DEFINITIONS)
    if(_defs)
        foreach(_d ${_defs})
            if(NOT _d MATCHES "[\\$<]")
                string(APPEND _common_flags " -D${_d}")
            endif()
        endforeach()
    endif()

    # COMPILE_OPTIONS
    get_target_property(_opts "${target}" COMPILE_OPTIONS)
    if(_opts)
        foreach(_o ${_opts})
            if(NOT _o MATCHES "[\\$<]")
                string(APPEND _common_flags " ${_o}")
                list(APPEND _raw_compile_opts "${_o}")
            endif()
        endforeach()
    endif()

    # POSITION_INDEPENDENT_CODE
    get_target_property(_pic "${target}" POSITION_INDEPENDENT_CODE)
    set(_need_pic FALSE)
    if(_pic)
        set(_need_pic TRUE)
        string(APPEND _common_flags " -fPIC")
    endif()

    # CMAKE_<LANG>_FLAGS — base flags + build-type–specific flags
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)

    set(_c_base_flags "${CMAKE_C_FLAGS}")
    if(_build_type_upper)
        set(_c_bt "${CMAKE_C_FLAGS_${_build_type_upper}}")
        if(_c_bt)
            string(APPEND _c_base_flags " ${_c_bt}")
        endif()
    endif()

    set(_cxx_base_flags "${CMAKE_CXX_FLAGS}")
    if(_build_type_upper)
        set(_cxx_bt "${CMAKE_CXX_FLAGS_${_build_type_upper}}")
        if(_cxx_bt)
            string(APPEND _cxx_base_flags " ${_cxx_bt}")
        endif()
    endif()

    # C / C++ language standard from target properties
    # C_STANDARD
    get_target_property(_c_std "${target}" C_STANDARD)
    set(_c_std_flag "")
    if(_c_std)
        get_target_property(_c_ext "${target}" C_EXTENSIONS)
        if(_c_ext)
            set(_c_std_flag "-std=gnu${_c_std}")
        else()
            set(_c_std_flag "-std=c${_c_std}")
        endif()
    endif()

    # CXX_STANDARD
    get_target_property(_cxx_std "${target}" CXX_STANDARD)
    set(_cxx_std_flag "")
    if(_cxx_std)
        get_target_property(_cxx_ext "${target}" CXX_EXTENSIONS)
        if(_cxx_ext)
            set(_cxx_std_flag "-std=gnu++${_cxx_std}")
        else()
            set(_cxx_std_flag "-std=c++${_cxx_std}")
        endif()
    endif()

    # --- 6b. Interface properties from linked libraries ---
    _clunk_collect_interface_flags(_common_flags "${target}")

    # --- 6c. Code-generation flags for .ll → .o step ---
    _clunk_extract_codegen_flags(_cg_flags _raw_compile_opts _need_pic)

    # ===================================================================
    # 7. Create working directory and (optionally) cache directory
    # ===================================================================
    set(_work_dir "${CMAKE_CURRENT_BINARY_DIR}/clunk_${target}")
    file(MAKE_DIRECTORY "${_work_dir}")

    if(CLUNK_ENABLE_CACHE)
        file(MAKE_DIRECTORY "${CLUNK_CACHE_DIR}")
    endif()

    # ===================================================================
    # 8. Per-source: source → LLVM IR → clunk → object file
    # ===================================================================
    set(_optimised_objs "")

    foreach(_src ${_c_sources})

        # --- Absolute source path ---
        if(IS_ABSOLUTE "${_src}")
            set(_src_abs "${_src}")
        else()
            set(_src_abs "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        endif()

        # --- Check GENERATED property ---
        get_source_file_property(_is_generated "${_src}" GENERATED)
        if(_is_generated AND NOT EXISTS "${_src_abs}")
            message(STATUS "Clunk: skipping GENERATED source ${_src} "
                           "(does not exist at configure time)")
            continue()
        endif()

        # --- Determine language from file extension ---
        string(TOLOWER "${_src}" _src_lower)
        if(_src_lower MATCHES "\\.c$")
            set(_lang "C")
            set(_lang_base_flags "${_c_base_flags}")
            set(_lang_std_flag   "${_c_std_flag}")
        else()
            set(_lang "CXX")
            set(_lang_base_flags "${_cxx_base_flags}")
            set(_lang_std_flag   "${_cxx_std_flag}")
        endif()

        # --- Assemble full flags for source → IR step ---
        set(_ir_flags "${_common_flags} ${_lang_base_flags}")
        if(_lang_std_flag)
            string(APPEND _ir_flags " ${_lang_std_flag}")
        endif()

        # --- Derive unique output file paths ---
        # Replace path separators with _ to avoid collisions between
        # sources with the same filename in different directories.
        get_filename_component(_src_name "${_src}" NAME_WE)
        get_filename_component(_src_dir  "${_src}" DIRECTORY)
        if(_src_dir)
            string(REGEX REPLACE "[/\\]" "_" _dir_part "${_src_dir}")
            set(_safe_base "${_dir_part}_${_src_name}")
        else()
            set(_safe_base "${_src_name}")
        endif()

        set(_ll_file   "${_work_dir}/${_safe_base}.ll")
        set(_opt_ll    "${_work_dir}/${_safe_base}_opt.ll")
        set(_obj_file  "${_work_dir}/${_safe_base}_opt.o")

        # --- Caching ---
        set(_use_cache FALSE)
        set(_cached_obj "")

        if(CLUNK_ENABLE_CACHE AND NOT _is_generated AND EXISTS "${_src_abs}")
            file(SHA256 "${_src_abs}"  _src_hash)
            string(SHA256 _flags_hash   "${_ir_flags}")
            # Include clunk version + opt level to invalidate on toolchain update
            string(SHA256 _cache_key
                "${_src_hash}${_flags_hash}${_clunk_version}${CLUNK_OPT_LEVEL}")
            set(_cached_obj "${CLUNK_CACHE_DIR}/${_cache_key}.o")

            if(EXISTS "${_cached_obj}")
                set(_use_cache TRUE)
            endif()
        endif()

        if(_use_cache)
            # ---- Cache HIT: copy pre-built object ----
            add_custom_command(
                OUTPUT  "${_obj_file}"
                COMMAND "${CMAKE_COMMAND}" -E copy "${_cached_obj}" "${_obj_file}"
                DEPENDS "${_cached_obj}"
                COMMENT "Clunk: using cached object for ${_src}"
                VERBATIM
            )
        else()
            # ---- Full pipeline: source → IR → clunk → object ----

            # Step 1: source → LLVM IR
            add_custom_command(
                OUTPUT  "${_ll_file}"
                COMMAND "${CLUNK_CLANG}" -S -emit-llvm ${_ir_flags}
                        -o "${_ll_file}" "${_src_abs}"
                DEPENDS "${_src_abs}"
                COMMENT "Clunk: emitting LLVM IR for ${_src}"
                VERBATIM
            )

            # Step 2: Run clunk superoptimiser
            add_custom_command(
                OUTPUT  "${_opt_ll}"
                COMMAND "${CLUNK_EXECUTABLE}"
                        --opt-level ${CLUNK_OPT_LEVEL}
                        --time-budget ${CLUNK_TIME_BUDGET}
                        --output "${_opt_ll}" "${_ll_file}"
                DEPENDS "${_ll_file}" "${CLUNK_EXECUTABLE}"
                COMMENT "Clunk: optimising ${_src}"
                VERBATIM
            )

            # Step 3: optimised IR → object file (+ optional cache store)
            if(CLUNK_ENABLE_CACHE AND _cached_obj)
                add_custom_command(
                    OUTPUT  "${_obj_file}"
                    COMMAND "${CLUNK_CLANG}" -c ${_cg_flags}
                            "${_opt_ll}" -o "${_obj_file}"
                    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                            "${_obj_file}" "${_cached_obj}"
                    DEPENDS "${_opt_ll}"
                    COMMENT "Clunk: compiling optimised ${_src}"
                    VERBATIM
                )
            else()
                add_custom_command(
                    OUTPUT  "${_obj_file}"
                    COMMAND "${CLUNK_CLANG}" -c ${_cg_flags}
                            "${_opt_ll}" -o "${_obj_file}"
                    DEPENDS "${_opt_ll}"
                    COMMENT "Clunk: compiling optimised ${_src}"
                    VERBATIM
                )
            endif()
        endif()

        # Mark intermediate / output files as GENERATED so IDEs and
        # CMake know they are build artifacts, not source files.
        set_source_files_properties(
            "${_ll_file}" "${_opt_ll}" "${_obj_file}"
            PROPERTIES GENERATED TRUE
        )

        list(APPEND _optimised_objs "${_obj_file}")
    endforeach()

    # Bail out if every source was skipped (e.g. all GENERATED + absent)
    if(NOT _optimised_objs)
        message(STATUS "Clunk: clunk_opt(${target}) — no sources to optimise, skipping")
        return()
    endif()

    # ===================================================================
    # 9. Replace target sources — fix for duplicate symbol errors
    #
    #    PROBLEM: the old approach used target_link_libraries to add
    #    optimised objects ON TOP of the original objects, causing every
    #    symbol to be defined twice.
    #
    #    FIX: remove all C/C++ sources from the target's source list
    #    and replace them with the optimised object files.
    #
    #    set_target_properties(SOURCES ...) replaces the entire source
    #    list, so we must include ALL non-C/C++ sources (headers, .rc,
    #    .manifest, .asm, etc.) to avoid dropping them accidentally.
    # ===================================================================
    set_target_properties("${target}" PROPERTIES SOURCES "${_non_c_sources}")
    target_sources("${target}" PRIVATE ${_optimised_objs})

    # ===================================================================
    # 10. Summary
    # ===================================================================
    list(LENGTH _optimised_objs _num_optimised)
    message(STATUS "Clunk: clunk_opt(${target}) configured — "
                   "${_num_optimised} file(s) will be superoptimised at build time")
endfunction()
