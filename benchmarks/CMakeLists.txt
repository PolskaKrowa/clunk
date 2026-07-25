add_executable(clunk_bench bench_search.cpp)
target_link_libraries(clunk_bench PRIVATE clunk_core)
target_compile_options(clunk_bench PRIVATE -Wall -Wextra)
