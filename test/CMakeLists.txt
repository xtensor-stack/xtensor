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

include_directories(../include)
include_directories(${GTEST_INCLUDE_DIRS})

set(XTENSOR_TARGET test_xtensor)
set(XTENSOR_INCLUDE ../include)

set(XTENSOR_HEADERS
    ${XTENSOR_INCLUDE}/xtensor/xarray.hpp
    ${XTENSOR_INCLUDE}/xtensor/xarray_base.hpp
    ${XTENSOR_INCLUDE}/xtensor/xassign.hpp
    ${XTENSOR_INCLUDE}/xtensor/xexception.hpp
    ${XTENSOR_INCLUDE}/xtensor/xexpression.hpp
    ${XTENSOR_INCLUDE}/xtensor/xfunction.hpp
    ${XTENSOR_INCLUDE}/xtensor/xindex.hpp
    ${XTENSOR_INCLUDE}/xtensor/xio.hpp
    ${XTENSOR_INCLUDE}/xtensor/xiterator.hpp
    ${XTENSOR_INCLUDE}/xtensor/xmath.hpp
    ${XTENSOR_INCLUDE}/xtensor/xnoalias.hpp
    ${XTENSOR_INCLUDE}/xtensor/xoperation.hpp
    ${XTENSOR_INCLUDE}/xtensor/xscalar.hpp
    ${XTENSOR_INCLUDE}/xtensor/xsemantic.hpp
    ${XTENSOR_INCLUDE}/xtensor/xslice.hpp
    ${XTENSOR_INCLUDE}/xtensor/xutils.hpp
    ${XTENSOR_INCLUDE}/xtensor/xvectorize.hpp
    ${XTENSOR_INCLUDE}/xtensor/xview.hpp
)

set(XTENSOR_TESTS
    main.cpp
    test_common.hpp
    test_xadaptor_semantic.cpp
    test_xarray.cpp
    test_xarray_adaptor.cpp
    test_xarray_semantic.cpp
    test_xfunction.cpp
    test_xiterator.cpp
    test_xio.cpp
    test_xmath.cpp
    test_xnoalias.cpp
    test_xoperation.cpp
    test_xscalar.cpp
    test_xscalar_semantic.cpp
    test_xsemantic.hpp
    test_xvectorize.cpp
    test_xview.cpp
    test_xview_semantic.cpp
    test_xutils.cpp
)

add_executable(${XTENSOR_TARGET} ${XTENSOR_TESTS} ${XTENSOR_HEADERS})
target_link_libraries(${XTENSOR_TARGET} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

