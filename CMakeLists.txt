cmake_minimum_required(VERSION 3.1)
project(xtensor)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

    if (HAS_CPP14_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++14 support!")
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

set(XTENSOR_TARGET test_xtensor)

set(XTENSOR_HEADERS
    include/xtensor/xarray.hpp
    include/xtensor/xarray_base.hpp
    include/xtensor/xexpression.hpp
    include/xtensor/xindex.hpp
    include/xtensor/xio.hpp
    include/xtensor/xiterator.hpp
    include/xtensor/xmath.hpp
    include/xtensor/xoperation.hpp
    include/xtensor/xfunction.hpp
    include/xtensor/xslice.hpp
    include/xtensor/xview.hpp
    include/xtensor/xutils.hpp
)

set(XTENSOR_TESTS
    test/main.cpp
    test/test_common.hpp
    test/test_xadaptor_semantic.cpp
    test/test_xarray.cpp
    test/test_xarray_adaptor.cpp
    test/test_xarray_semantic.cpp
    test/test_xfunction.cpp
    test/test_xiterator.cpp
    test/test_xio.cpp
    test/test_xmath.cpp
    test/test_xnoalias.cpp
    test/test_xoperation.cpp
    test/test_xscalar.cpp
    test/test_xscalar_semantic.cpp
    test/test_xsemantic.hpp
    test/test_xvectorize.cpp
    test/test_xview.cpp
    test/test_xutils.cpp
)

add_executable(${XTENSOR_TARGET} ${XTENSOR_TESTS} ${XTENSOR_HEADERS})
target_link_libraries(${XTENSOR_TARGET} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

