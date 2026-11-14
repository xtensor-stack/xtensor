############################################################################
# Comparison benchmarks: xtensor vs raw C++
############################################################################

set(COMPARISON_BENCHMARK
    blas1_comparison.cpp
    comparison_main.cpp
)

add_executable(benchmark_comparison ${COMPARISON_BENCHMARK})
target_link_libraries(benchmark_comparison PUBLIC xtensor ${GBENCHMARK_LIBRARIES})
target_include_directories(benchmark_comparison PRIVATE ${XTENSOR_INCLUDE_DIR} ${GBENCHMARK_INCLUDE_DIRS})

add_custom_target(xcomparison
    COMMAND benchmark_comparison
    DEPENDS benchmark_comparison)
