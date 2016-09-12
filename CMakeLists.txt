cmake_minimum_required(VERSION 3.1)
project(ndarray)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "Unsupported compiler -- ndarray requires C++14 support!")
    endif()

    # Enable link time optimization and set the default symbol
    # visibility to hidden (very important to obtain small binaries)
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        # Default symbol visibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

        # Check for Link Time Optimization support
        # (GCC/Clang)
        CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        if (HAS_LTO_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        endif()

        # Intel equivalent to LTO is called IPO
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif()
        endif()
    endif()
endif()

find_package(GTest REQUIRED)
find_package(Threads)

include_directories(include)
include_directories(${GTEST_INCLUDE_DIRS})

set(NDARRAY_TARGET test_ndarray)

set(NDARRAY_HEADERS
    include/ndarray/ndarray.hpp
    include/ndarray/ndarray_base.hpp
    include/ndarray/ndbroadcast.hpp
    include/ndarray/ndexpression.hpp
    include/ndarray/ndindex.hpp
    include/ndarray/ndoperation.hpp
    include/ndarray/ndoperation_expression.hpp
    include/ndarray/ndutils.hpp
)

set(NDARRAY_TESTS
    test/main.cpp
    test/test_ndarray.cpp
    test/test_ndarray_base.hpp
    test/test_ndutils.cpp
)

add_executable(${NDARRAY_TARGET} ${NDARRAY_TESTS} ${NDARRAY_HEADERS})
target_link_libraries(${NDARRAY_TARGET} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

