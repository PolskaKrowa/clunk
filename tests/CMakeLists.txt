# Each test file has its own main(), so build as separate executables

set(TEST_SOURCES
    test_ir
    test_parser
    test_evaluator
    test_search
    test_search_threadpool
    test_stoke_moves
    test_peephole_miner
    test_peephole_harvest
    test_peephole_cegis
    test_pattern
    test_egraph
    test_pipeline
    test_smt
    test_smt_soundness
    test_smt_cegis
    test_alive_verifier
    test_rewrite_cache
    test_cost_model_tti
    test_vector_synth
    test_loop_opt
    test_mem_opt
    test_fp_opt
    test_mca
    test_known_bits
    test_dataflow_prune
    test_powers_of_two
    test_strength_reduce
    test_cross_function
    test_smt_bounded_unrolling
    test_opt_level_max
)

if(CLUNK_ENABLE_GPU)
    list(APPEND TEST_SOURCES
        test_gpu_emitter
        test_gpu_occupancy
        test_gpu_divergence
    )
endif()

foreach(test_name ${TEST_SOURCES})
    add_executable(clunk_test_${test_name} ${test_name}.cpp)
    target_link_libraries(clunk_test_${test_name} PRIVATE clunk_core)
    target_compile_options(clunk_test_${test_name} PRIVATE -Wall -Wextra)
    add_test(NAME ${test_name} COMMAND clunk_test_${test_name})
endforeach()