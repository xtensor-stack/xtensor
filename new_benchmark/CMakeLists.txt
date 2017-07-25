# dependencies for running the benchmarks 
# (we use google benchmarks)
find_package(Benchmark REQUIRED)
include_directories(${Benchmark_INCLUDE_DIR})

# include xtensor
include_directories(${XTENSOR_INCLUDE_DIR})


# other array libraries to compare with
# TODO make them optional
find_package(BLITZ REQUIRED)
include_directories(${BLITZ_INCLUDE_DIR})


# Set libs
find_package(Threads REQUIRED)
SET(BM_LIBS     
    ${Benchmark_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT}  
    pthread 
    ${BLITZ_LIBRARIES}
)


# add a single benchmark
add_executable(exp_benchmark exp_benchmark.cxx)
target_link_libraries(exp_benchmark ${BM_LIBS})
