############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)

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

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
    endforeach()
endif()

find_package(GTest REQUIRED)
find_package(Threads)

include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${GTEST_INCLUDE_DIRS})

set(XTENSOR_TESTS
    main.cpp
    test_common.hpp
    test_xadaptor_semantic.cpp
    test_xarray.cpp
    test_xarray_adaptor.cpp
    test_xbroadcast.cpp
    test_xbuilder.cpp
    test_xcontainer_semantic.cpp
    test_xfunction.cpp
    test_xiterator.cpp
    test_xio.cpp
    test_xmath.cpp
    test_xnoalias.cpp
    test_xoperation.cpp
    test_xscalar.cpp
    test_xscalar_semantic.cpp
    test_xsemantic.hpp
    test_xtensor.cpp
    test_xtensor_adaptor.cpp
    test_xtensor_semantic.cpp
    test_xvectorize.cpp
    test_xview.cpp
    test_xview_semantic.cpp
    test_xutils.cpp
)

set(XTENSOR_TARGET test_xtensor)
add_executable(${XTENSOR_TARGET} ${XTENSOR_TESTS} ${XTENSOR_HEADERS})
target_link_libraries(${XTENSOR_TARGET} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

