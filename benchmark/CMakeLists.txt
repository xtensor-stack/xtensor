############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.22)
include(FetchContent)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-benchmark)

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG(-march=native arch_native_supported)
    if(arch_native_supported AND NOT CMAKE_CXX_FLAGS MATCHES "-march")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wunused-parameter -Wextra -Wreorder")

    if(NOT "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
        CHECK_CXX_COMPILER_FLAG("-std=c++20" HAS_CPP20_FLAG)
        if (HAS_CPP20_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
        else()
            message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++17 support!")
        endif()
    endif()

    # Enable link time optimization and set the default symbol
    # visibility to hidden (very important to obtain small binaries)
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        # Default symbol visibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

        # Check for Link Time Optimization support
        # (GCC/Clang)
        # LTO had to be removed as google benchmark doesn't build with it
        # CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        # if (HAS_LTO_FLAG)
        #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        # endif()

        # Intel equivalent to LTO is called IPO
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif()
        endif()
    endif()
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
    endforeach()
endif()


if(DOWNLOAD_GBENCHMARK OR GBENCHMARK_SRC_DIR)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG main)

    FetchContent_Declare(googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG main) # need main for benchmark::benchmark

    FetchContent_MakeAvailable(
            googletest
            googlebenchmark)
    set(GBENCHMARK_INCLUDE_DIRS "${googlebenchmark_SOURCE_DIR}/include")
    set(GBENCHMARK_LIBRARIES benchmark)
else()
    find_package(benchmark REQUIRED)
endif()

find_package(xsimd)
if (xsimd_FOUND)
    include_directories(${xsimd_INCLUDE_DIRS})
    add_definitions("-DXTENSOR_USE_XSIMD=1")
endif()

include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${GBENCHMARK_INCLUDE_DIRS})

set(XTENSOR_BENCHMARK
    benchmark_assign.cpp
    benchmark_builder.cpp
    benchmark_container.cpp
    benchmark_creation.cpp
    benchmark_increment_stepper.cpp
    benchmark_lambda_expressions.cpp
    benchmark_math.cpp
    benchmark_random.cpp
    benchmark_reducer.cpp
    benchmark_views.cpp
    benchmark_xshape.cpp
    benchmark_view_access.cpp
    benchmark_view_assignment.cpp
    benchmark_view_adapt.cpp
    benchmark_stl.cpp
    main.cpp
)


set(XTENSOR_BENCHMARK_TARGET benchmark_xtensor)
add_executable(${XTENSOR_BENCHMARK_TARGET} EXCLUDE_FROM_ALL ${XTENSOR_BENCHMARK} ${XTENSOR_HEADERS})
target_link_libraries(${XTENSOR_BENCHMARK_TARGET} PUBLIC xtensor ${GBENCHMARK_LIBRARIES})

if(XTENSOR_USE_TBB)
    target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PUBLIC XTENSOR_USE_TBB)
    target_include_directories(${XTENSOR_BENCHMARK_TARGET} PUBLIC ${TBB_INCLUDE_DIRS})
    target_link_libraries(${XTENSOR_BENCHMARK_TARGET} PUBLIC ${TBB_LIBRARIES})
endif()
if(XTENSOR_USE_OPENMP)
    target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PUBLIC XTENSOR_USE_OPENMP)
endif()

add_custom_target(xbenchmark
    COMMAND benchmark_xtensor
    DEPENDS ${XTENSOR_BENCHMARK_TARGET})

add_custom_target(xpowerbench
    COMMAND echo "sudo needed to set cpu power governor to performance"
    COMMAND sudo cpupower frequency-set --governor performance
    COMMAND benchmark_xtensor --benchmark_out=results.csv --benchmark_out_format=csv
    COMMAND sudo cpupower frequency-set --governor powersave
    DEPENDS ${XTENSOR_BENCHMARK_TARGET})
