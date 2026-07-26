# ClunkOpt.cmake — clunk_opt() function for drop-in superoptimisation
#
# Usage:
#   find_package(Clunk REQUIRED)
#   include(${Clunk_DIR}/ClunkOpt.cmake)   # if not auto-included
#   clunk_opt(my_target)
#
# This replaces the target's C/C++/Fortran source compilations with a
# source → LLVM IR → clunk superoptimiser → object file pipeline.
# Non-optimised sources (headers, assembly, .rc, etc.) are left untouched.
#
# The original C/C++/Fortran sources are removed from the target's source
# list and replaced with the optimised object files, avoiding duplicate
# symbols.
#
# Options (set before calling clunk_opt):
#   CLUNK_TIME_BUDGET         — seconds per source file (default: 10)
#   CLUNK_TOTAL_TIME_BUDGET   — total seconds across all sources in this
#                               target (default: 0 = no cap).  When the
#                               per-file estimate exceeds this, the
#                               per-file budget is scaled down.
#   CLUNK_OPT_LEVEL           — clunk optimisation level (default: 2)
#   CLUNK_ENABLE_CACHE        — enable content-hash caching (default: ON)
#   CLUNK_CACHE_DIR           — override cache directory
#                               (default: ${CMAKE_BINARY_DIR}/.clunk_cache)
#   CLUNK_CLANG               — override clang executable (C/C++ sources)
#   CLUNK_FLANG               — override flang executable (Fortran sources)
#   CLUNK_USE_COMPILE_COMMANDS — when ON and compile_commands.json exists,
#                               read per-source compile flags from it
#                               instead of reconstructing from target
#                               properties (default: ON).  More accurate
#                               but requires CMAKE_EXPORT_COMPILE_COMMANDS=ON.
#
# Language support:
#   - C      : .c                          → uses clang
#   - C++    : .cc .cpp .cxx .c++          → uses clang
#   - Fortran: .f .f90 .f95 .f03 .f08 .for
#              .F .F90 .F95 .F03 .F08      → uses flang-new / flang
#
# Limitations:
#   - gfortran is NOT supported (it doesn't emit LLVM IR).  If the
#     project's CMAKE_Fortran_COMPILER is gfortran, clunk_opt() will
#     search PATH for flang-new and fall back to leaving Fortran
#     sources unoptimised if flang isn't found.
#   - compile_commands.json lookup uses python3 (CMake has no built-in
#     JSON parser).  Falls back to property-walk flags if python3 is absent.
#   - Caching tracks source content + flags + clunk version; header changes
#     are not tracked.  Use CLUNK_ENABLE_CACHE=OFF during active development.

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
    # 3. Find clang AND flang — we need clang for C/C++ sources and flang
    #    for Fortran sources.  Both produce LLVM IR via -emit-llvm, which
    #    is what clunk consumes.
    #
    #    Discovery order:
    #      a) CLUNK_CLANG / CLUNK_FLANG cache variables (user override)
    #      b) The project's own C/CXX/Fortran compiler (if it's flang/clang)
    #      c) PATH search for clang / flang-new / flang binaries
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
        # clang is only required if the target has C/C++ sources.  We'll
        # warn-and-skip later if so.  Don't hard-fail here — the target
        # might be Fortran-only.
        set(CLUNK_CLANG "")
    endif()

    # Fortran compiler discovery.  Modern flang is shipped as `flang-new`
    # (LLVM 13+) or `flang` (LLVM 5+, classic).  We prefer `flang-new`
    # because classic flang does NOT support -emit-llvm in the way clunk
    # needs (it goes through a different middle-end).
    if(NOT DEFINED CLUNK_FLANG OR NOT CLUNK_FLANG)
        # Try the project's Fortran compiler first.
        set(CLUNK_FLANG "${CMAKE_Fortran_COMPILER}")
        if(CLUNK_FLANG AND EXISTS "${CLUNK_FLANG}")
            # Verify it actually supports -emit-llvm.  gfortran does NOT
            # (it uses GCC's middle-end, not LLVM).  If the user pointed
            # CMAKE_Fortran_COMPILER at gfortran, we have to fall back to
            # searching PATH for flang-new.
            execute_process(
                COMMAND "${CLUNK_FLANG}" --version
                OUTPUT_VARIABLE _flang_version
                ERROR_QUIET
                RESULT_VARIABLE _flang_rc
            )
            if(_flang_rc EQUAL 0 AND _flang_version MATCHES "flang|LLVM")
                # Looks like flang — keep it.
            elseif(_flang_rc EQUAL 0 AND _flang_version MATCHES "GNU Fortran|gfortran")
                # gfortran — can't emit LLVM IR.  Fall through to PATH search.
                set(CLUNK_FLANG "")
            endif()
        endif()

        if(NOT CLUNK_FLANG OR NOT EXISTS "${CLUNK_FLANG}")
            find_program(CLUNK_FLANG
                NAMES flang-new flang flang-19 flang-18 flang-17
                DOC "Path to flang (LLVM Fortran frontend) — required for clunk_opt() on Fortran sources"
            )
        endif()
    endif()

    if(NOT CLUNK_FLANG OR NOT EXISTS "${CLUNK_FLANG}")
        # flang is only required if the target has Fortran sources.
        set(CLUNK_FLANG "")
    endif()

    # ===================================================================
    # 4. Set defaults for configurable options
    # ===================================================================
    if(NOT DEFINED CLUNK_TIME_BUDGET)
        set(CLUNK_TIME_BUDGET 10)    # seconds per source file
    endif()
    if(NOT DEFINED CLUNK_TOTAL_TIME_BUDGET)
        # Total wall-clock budget across ALL sources in this target.
        # Default: no cap (per-file CLUNK_TIME_BUDGET is the only limit).
        # Set this to avoid runaway CI times on large projects — e.g.
        #   set(CLUNK_TOTAL_TIME_BUDGET 600)   # 10 minutes max per target
        set(CLUNK_TOTAL_TIME_BUDGET 0)
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
    # When ON, clunk_opt() reads per-source compile flags from
    # compile_commands.json (if it exists) instead of reconstructing
    # them from CMake target properties.  This is more accurate (it
    # captures generator expressions, transitive dependencies, etc.)
    # but requires CMAKE_EXPORT_COMPILE_COMMANDS=ON.
    if(NOT DEFINED CLUNK_USE_COMPILE_COMMANDS)
        set(CLUNK_USE_COMPILE_COMMANDS ON)
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
    # 5. Get target sources — separate C/C++/Fortran (to be replaced)
    #    from everything else (headers, assembly, .rc, .manifest, etc.)
    # ===================================================================
    get_target_property(_all_sources "${target}" SOURCES)
    if(NOT _all_sources)
        message(STATUS "Clunk: clunk_opt(${target}) — target has no sources, skipping")
        return()
    endif()

    set(_opt_sources "")       # Sources that go through the clunk pipeline
    set(_non_opt_sources "")   # Everything else — stays in the target

    # Recognised source extensions:
    #   C       : .c
    #   C++     : .cc .cpp .cxx .c++
    #   Fortran : .f .f90 .f95 .f03 .f08 .for .fpp .F .F90 .F95 .F03 .F08
    # Capital .F* means run the preprocessor (flang handles this automatically).
    foreach(_src ${_all_sources})
        string(TOLOWER "${_src}" _src_lower)
        if(_src_lower MATCHES "\\.(c|cc|cpp|cxx|c\\+\\+)$" OR
           _src_lower MATCHES "\\.(f|f90|f95|f03|f08|for|fpp|f15|f18)$" OR
           _src      MATCHES "\\.[Ff][Pp]?[Pp]?[0958]*$")
            list(APPEND _opt_sources "${_src}")
        else()
            list(APPEND _non_opt_sources "${_src}")
        endif()
    endforeach()

    if(NOT _opt_sources)
        message(STATUS "Clunk: clunk_opt(${target}) — no C/C++/Fortran sources found, skipping")
        return()
    endif()

    list(LENGTH _opt_sources _num_sources)

    # ── Per-file time budget with total cap ──────────────────────────────
    # If CLUNK_TOTAL_TIME_BUDGET is set and the naive per-file estimate
    # would exceed it, scale down the per-file budget.  This prevents a
    # 200-file target from running for 200 × 30s = 100 minutes when the
    # user only budgeted 10 minutes for the whole target.
    set(_per_file_budget "${CLUNK_TIME_BUDGET}")
    if(CLUNK_TOTAL_TIME_BUDGET GREATER 0)
        math(EXPR _naive_total "${_num_sources} * ${CLUNK_TIME_BUDGET}")
        if(_naive_total GREATER CLUNK_TOTAL_TIME_BUDGET)
            # Scale down per-file budget so total fits.
            math(EXPR _per_file_budget "${CLUNK_TOTAL_TIME_BUDGET} / ${_num_sources}")
            if(_per_file_budget LESS 1)
                set(_per_file_budget 1)
            endif()
            message(STATUS "Clunk: clunk_opt(${target}) — total budget ${CLUNK_TOTAL_TIME_BUDGET}s "
                           "across ${_num_sources} files; scaling per-file budget to ${_per_file_budget}s")
        endif()
    endif()

    math(EXPR _estimate_seconds "${_num_sources} * ${_per_file_budget}")

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

    # Fortran base flags — same pattern as C/C++ above.
    # Note: flang-new accepts the same -std=f2008 / -std=f2018 syntax as
    # gfortran, so we don't need separate dialect handling.
    set(_fortran_base_flags "${CMAKE_Fortran_FLAGS}")
    if(_build_type_upper)
        set(_fortran_bt "${CMAKE_Fortran_FLAGS_${_build_type_upper}}")
        if(_fortran_bt)
            string(APPEND _fortran_base_flags " ${_fortran_bt}")
        endif()
    endif()

    # Fortran_STANDARD (CMake 3.25+; older CMakes don't set this and we
    # fall back to whatever -std= flag is in CMAKE_Fortran_FLAGS).
    set(_fortran_std_flag "")
    get_target_property(_fortran_std "${target}" Fortran_STANDARD)
    if(_fortran_std)
        set(_fortran_std_flag "-std=f${_fortran_std}")
    endif()

    # --- 6b. Interface properties from linked libraries ---
    _clunk_collect_interface_flags(_common_flags "${target}")

    # --- 6c. Code-generation flags for .ll → .o step ---
    _clunk_extract_codegen_flags(_cg_flags _raw_compile_opts _need_pic)

    # --- 6d. compile_commands.json (optional but more accurate) --------
    # When CMAKE_EXPORT_COMPILE_COMMANDS is ON and the file exists, we
    # query it for the exact compile command CMake recorded for each
    # source file.  This captures generator expressions, transitive
    # INTERFACE_* flags, and per-source overrides that the property-walk
    # above can miss.  The lookup happens per-source inside the loop.
    set(_compile_commands_json "")
    if(CLUNK_USE_COMPILE_COMMANDS)
        set(_compile_commands_json "${CMAKE_BINARY_DIR}/compile_commands.json")
        if(NOT EXISTS "${_compile_commands_json}")
            set(_compile_commands_json "")
        endif()
    endif()

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

    foreach(_src ${_opt_sources})

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
        # Sets _lang, _lang_base_flags, _lang_std_flag, _lang_compiler.
        string(TOLOWER "${_src}" _src_lower)
        set(_lang_compiler "")
        if(_src_lower MATCHES "\\.c$")
            set(_lang "C")
            set(_lang_base_flags "${_c_base_flags}")
            set(_lang_std_flag   "${_c_std_flag}")
            set(_lang_compiler   "${CLUNK_CLANG}")
        elseif(_src_lower MATCHES "\\.(cc|cpp|cxx|c\\+\\+)$")
            set(_lang "CXX")
            set(_lang_base_flags "${_cxx_base_flags}")
            set(_lang_std_flag   "${_cxx_std_flag}")
            set(_lang_compiler   "${CLUNK_CLANG}")
        elseif(_src_lower MATCHES "\\.(f|f90|f95|f03|f08|for|fpp|f15|f18)$" OR
               _src      MATCHES "\\.[Ff][Pp]?[Pp]?[0958]*$")
            set(_lang "Fortran")
            set(_lang_base_flags "${_fortran_base_flags}")
            set(_lang_std_flag   "${_fortran_std_flag}")
            set(_lang_compiler   "${CLUNK_FLANG}")
        else()
            # Shouldn't happen — classifier above already filtered.
            continue()
        endif()

        # --- Skip if we don't have a compiler for this language ---
        if(NOT _lang_compiler OR NOT EXISTS "${_lang_compiler}")
            message(WARNING "Clunk: clunk_opt(${target}) — no ${_lang} compiler found "
                            "for ${_src}; passing it through unoptimised.")
            list(APPEND _non_opt_sources "${_src}")
            continue()
        endif()

        # --- Assemble full flags for source → IR step ---
        # Default: reconstruct from target properties.
        set(_ir_flags "${_common_flags} ${_lang_base_flags}")
        if(_lang_std_flag)
            string(APPEND _ir_flags " ${_lang_std_flag}")
        endif()

        # ── compile_commands.json override ─────────────────────────────
        # If the user enabled CLUNK_USE_COMPILE_COMMANDS and the database
        # exists, look up the exact compile command CMake recorded for
        # this source.  This is more accurate than the property walk
        # above — it captures generator expressions, transitive deps,
        # and per-source overrides.
        #
        # We use a Python one-liner (CMake has no built-in JSON parser
        # before 3.19, and even then it's experimental).  If Python
        # isn't available, fall back to the property-walk flags.
        if(_compile_commands_json)
            execute_process(
                COMMAND python3 -c
                    "import json,sys; \
                     db=json.load(open(sys.argv[1])); \
                     target=sys.argv[2]; src=sys.argv[3]; \
                     for e in db: \
                         if e.get('file','').endswith(src.split('/')[-1]) and target in e.get('command',''): \
                             print(e['command']); break"
                    "${_compile_commands_json}" "${target}" "${_src_abs}"
                OUTPUT_VARIABLE _cc_cmd
                RESULT_VARIABLE _cc_rc
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(_cc_rc EQUAL 0 AND _cc_cmd)
                # Strip the leading compiler invocation and the source
                # file argument; keep everything else as flags.
                # _cc_cmd looks like:  /usr/bin/clang -c -DFOO=1 ... /path/to/src.c -o src.o
                # We want: -DFOO=1 ... (everything between the compiler and the source path)
                string(REPLACE " " ";" _cc_tokens "${_cc_cmd}")
                set(_cc_flags "")
                set(_skip_next FALSE)
                set(_seen_compiler FALSE)
                foreach(_tok ${_cc_tokens})
                    if(NOT _seen_compiler)
                        set(_seen_compiler TRUE)
                        continue()
                    endif()
                    if(_tok STREQUAL "-o")
                        set(_skip_next TRUE)
                        continue()
                    endif()
                    if(_skip_next)
                        set(_skip_next FALSE)
                        continue()
                    endif()
                    # Stop at the source file path (absolute or matching _src)
                    if(_tok STREQUAL "${_src_abs}" OR _tok STREQUAL "${_src}")
                        break()
                    endif()
                    # Skip -c (we're using -S -emit-llvm instead)
                    if(_tok STREQUAL "-c")
                        continue()
                    endif()
                    string(APPEND _cc_flags " ${_tok}")
                endforeach()
                if(_cc_flags)
                    set(_ir_flags "${_cc_flags}")
                endif()
            endif()
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
            # Use the language-appropriate compiler for both the
            # source → IR step AND the IR → object step.  clang can
            # recompile flang-emitted IR (and vice versa) because LLVM
            # IR is language-agnostic, but using the same compiler
            # avoids subtle ABI mismatches in Fortran runtime support.

            # Step 1: source → LLVM IR
            add_custom_command(
                OUTPUT  "${_ll_file}"
                COMMAND "${_lang_compiler}" -S -emit-llvm ${_ir_flags}
                        -o "${_ll_file}" "${_src_abs}"
                DEPENDS "${_src_abs}"
                COMMENT "Clunk: emitting LLVM IR for ${_src}"
                VERBATIM
            )

            # Step 2: Run clunk superoptimiser
            # Uses _per_file_budget (may be lower than CLUNK_TIME_BUDGET
            # if CLUNK_TOTAL_TIME_BUDGET forced scaling).
            add_custom_command(
                OUTPUT  "${_opt_ll}"
                COMMAND "${CLUNK_EXECUTABLE}"
                        --opt-level ${CLUNK_OPT_LEVEL}
                        --time-budget ${_per_file_budget}
                        --output "${_opt_ll}" "${_ll_file}"
                DEPENDS "${_ll_file}" "${CLUNK_EXECUTABLE}"
                COMMENT "Clunk: optimising ${_src}"
                VERBATIM
            )

            # Step 3: optimised IR → object file (+ optional cache store)
            # We use _lang_compiler (NOT CLUNK_CLANG) here so flang IR
            # is recompiled by flang, preserving Fortran runtime linkage.
            if(CLUNK_ENABLE_CACHE AND _cached_obj)
                add_custom_command(
                    OUTPUT  "${_obj_file}"
                    COMMAND "${_lang_compiler}" -c ${_cg_flags}
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
                    COMMAND "${_lang_compiler}" -c ${_cg_flags}
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
    #    FIX: remove all C/C++/Fortran sources from the target's source
    #    list and replace them with the optimised object files.
    #
    #    set_target_properties(SOURCES ...) replaces the entire source
    #    list, so we must include ALL non-optimised sources (headers,
    #    .rc, .manifest, .asm, skipped-due-to-no-compiler, etc.) to
    #    avoid dropping them accidentally.
    # ===================================================================
    set_target_properties("${target}" PROPERTIES SOURCES "${_non_opt_sources}")
    target_sources("${target}" PRIVATE ${_optimised_objs})

    # ===================================================================
    # 10. Summary
    # ===================================================================
    list(LENGTH _optimised_objs _num_optimised)
    message(STATUS "Clunk: clunk_opt(${target}) configured — "
                   "${_num_optimised} file(s) will be superoptimised at build time")
endfunction()
