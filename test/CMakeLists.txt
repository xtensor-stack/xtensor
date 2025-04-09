############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.29)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-test CXX)

    enable_testing()

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
endif ()

find_package(doctest            REQUIRED)
find_package(Threads)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

include(set_compiler_flag.cmake)

if(CPP23)
  # User requested C++23, but compiler might not oblige.
  set_compiler_flag(
    _cxx_std_flag CXX
    "-std=c++23"  # this should work with GNU, Intel, PGI
    "/std:c++23"  # this should work with MSVC
  )
  if(_cxx_std_flag)
    message(STATUS "Building with C++23")
  endif()
else()
  set_compiler_flag(
    _cxx_std_flag CXX REQUIRED
    "-std=c++20"  # this should work with GNU, Intel, PGI
    "/std:c++20"  # this should work with MSVC
  )
  message(STATUS "Building with C++20")
endif()

if(NOT _cxx_std_flag)
  message(FATAL_ERROR "xtensor needs a C++20-compliant compiler.")
endif()

OPTION(XTENSOR_ENABLE_WERROR "Turn on -Werror" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DXSIMD_ENABLE_XTL_COMPLEX=1")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
  CHECK_CXX_COMPILER_FLAG(-march=native arch_native_supported)
  if(arch_native_supported AND NOT CMAKE_CXX_FLAGS MATCHES "-march")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -Wunused-parameter -Wextra -Wreorder -Wconversion -Wno-sign-conversion ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable -ftemplate-backtrace-limit=0")
  if (XTENSOR_DISABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
  endif()
  if (XTENSOR_ENABLE_WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -DSKIP_ON_WERROR")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /Zc:__cplusplus /MP /bigobj")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
  if (XTENSOR_DISABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c-")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    CHECK_CXX_COMPILER_FLAG(-march=native arch_native_supported)
    if(arch_native_supported AND NOT CMAKE_CXX_FLAGS MATCHES "-march")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
    if (XTENSOR_DISABLE_EXCEPTIONS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
    endif()
    if (XTENSOR_ENABLE_WERROR)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -DSKIP_ON_WERROR")
    endif()
  elseif (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
    if (XTENSOR_DISABLE_EXCEPTIONS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c-")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    endif()
  endif()
else()
  message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()


# For unit test and coverity scan.
# The Coverity scanner takes time and it could cause build timeout(10mins) in Travis CI.
# Therefore, we need keep this small but complete for analysis.
set(TEST_HEADERS
    test_common.hpp
    test_common_macros.hpp
    test_utils.hpp
    test_xsemantic.hpp
)
set(COMMON_BASE
    test_xadaptor_semantic.cpp
    test_xarray_adaptor.cpp
    test_xarray.cpp
    test_xblockwise_reducer.cpp
    test_xbroadcast.cpp
    test_xbuilder.cpp
    test_xcontainer_semantic.cpp
    test_xeval.cpp
    test_xexception.cpp
    test_xexpression.cpp
    test_xexpression_traits.cpp
    test_xfunction.cpp
    test_xfunc_on_xexpression.cpp
    test_xmultiindex_iterator.cpp
    test_xiterator.cpp
    test_xmath.cpp
    test_xoperation.cpp
    test_xoptional_assembly.cpp
    test_xreducer.cpp
    test_xscalar.cpp
    test_xscalar_semantic.cpp
    test_xshape.cpp
    test_xstorage.cpp
    test_xstrided_view.cpp
    test_xstrides.cpp
    test_xtensor.cpp
    test_xtensor_adaptor.cpp
    test_xtensor_semantic.cpp
    test_xview.cpp
    test_xview_semantic.cpp
    test_xutils.cpp
    test_xsimd8.cpp
)

set(XTENSOR_TESTS
    main.cpp
    test_xaccumulator.cpp
    test_xadapt.cpp
    test_strided_assign.cpp
    test_xassign.cpp
    test_xaxis_iterator.cpp
    test_xaxis_slice_iterator.cpp
    test_xbuffer_adaptor.cpp
    test_xchunked_array.cpp
    test_xchunked_view.cpp
    test_xcomplex.cpp
    test_xcsv.cpp
    test_xdatesupport.cpp
    test_xdynamic_view.cpp
    test_xfunctor_adaptor.cpp
    test_xfixed.cpp
    test_xfft.cpp
    test_xhistogram.cpp
    test_xpad.cpp
    test_xindex_view.cpp
    test_xinfo.cpp
    test_xio.cpp
    test_xlayout.cpp
    test_xmanipulation.cpp
    test_xmasked_view.cpp
    test_xmath_result_type.cpp
    test_xnan_functions.cpp
    test_xnoalias.cpp
    test_xnorm.cpp
    test_xnpy.cpp
    test_xoptional.cpp
    test_xoptional_assembly_adaptor.cpp
    test_xoptional_assembly_storage.cpp
    test_xset_operation.cpp
    test_xrandom.cpp
    test_xrepeat.cpp
    test_xsort.cpp
    test_xsimd.cpp
    test_xvectorize.cpp
    test_extended_xmath_interp.cpp
    test_extended_broadcast_view.cpp
    test_extended_xmath_reducers.cpp
    test_extended_xhistogram.cpp
    test_extended_xsort.cpp
    test_sfinae.cpp
)

if(nlohmann_json_FOUND)
    list(APPEND XTENSOR_TESTS test_xjson.cpp)
    list(APPEND XTENSOR_TESTS test_xmime.cpp)
    list(APPEND XTENSOR_TESTS test_xexpression_holder.cpp)
endif()

# Add files for npy tests
set(XNPY_FILES
    bool
    bool_fortran
    double
    double_fortran
    int
    unsignedlong
    unsignedlong_fortran
)

foreach(filename IN LISTS XNPY_FILES)
    foreach(suffix .be.npy .le.npy)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/files/xnpy_files/${filename}${suffix}
            ${CMAKE_CURRENT_BINARY_DIR}/files/xnpy_files/${filename}${suffix} COPYONLY)
    endforeach()
endforeach()

file(GLOB XTENSOR_PREPROCESS_FILES files/cppy_source/*.cppy)

# This target should only be run when the test source files have been changed.
add_custom_target(
    preprocess_cppy
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/files/preprocess.py
    DEPENDS ${XTENSOR_PREPROCESS_FILES}
)

foreach(filename IN LISTS COMMON_BASE XTENSOR_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} main.cpp ${filename} ${TEST_HEADERS} ${XTENSOR_HEADERS})
    if(XTENSOR_USE_XSIMD)
        target_compile_definitions(${targetname}
                                   PRIVATE
                                   XTENSOR_USE_XSIMD
                                   XSIMD_ENABLE_XTL_COMPLEX)
        target_link_libraries(${targetname} PRIVATE xsimd)
    endif()
    if(XTENSOR_USE_TBB)
        target_compile_definitions(${targetname} PRIVATE XTENSOR_USE_TBB)
        target_include_directories(${targetname} PRIVATE ${TBB_INCLUDE_DIRS})
        target_link_libraries(${targetname} PRIVATE ${TBB_LIBRARIES})
    endif()
    if(XTENSOR_USE_OPENMP)
        target_compile_definitions(${targetname} PRIVATE XTENSOR_USE_OPENMP)
    endif()
    target_include_directories(${targetname} PRIVATE ${XTENSOR_INCLUDE_DIR})
    target_link_libraries(${targetname} PRIVATE xtensor doctest::doctest ${CMAKE_THREAD_LIBS_INIT})
    add_custom_target(
        x${targetname}
        COMMAND ${targetname}
        DEPENDS ${targetname} ${filename} ${XTENSOR_HEADERS})
    add_test(NAME ${targetname} COMMAND ${targetname})
endforeach()

add_executable(test_xtensor_lib  main.cpp ${COMMON_BASE} ${XTENSOR_TESTS} ${TEST_HEADERS} ${XTENSOR_HEADERS})
if(XTENSOR_USE_XSIMD)
    target_compile_definitions(test_xtensor_lib
                               PRIVATE
                               XTENSOR_USE_XSIMD
                               XSIMD_ENABLE_XTL_COMPLEX)
    target_link_libraries(test_xtensor_lib PRIVATE xsimd)
endif()
if(XTENSOR_USE_TBB)
    target_compile_definitions(test_xtensor_lib PRIVATE XTENSOR_USE_TBB)
    target_include_directories(test_xtensor_lib PRIVATE ${TBB_INCLUDE_DIRS})
    target_link_libraries(test_xtensor_lib PRIVATE ${TBB_LIBRARIES})
endif()
if(XTENSOR_USE_OPENMP)
    target_compile_definitions(test_xtensor_lib PRIVATE XTENSOR_USE_OPENMP)
endif()

target_include_directories(test_xtensor_lib PRIVATE ${XTENSOR_INCLUDE_DIR})
target_link_libraries(test_xtensor_lib PRIVATE xtensor  doctest::doctest ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(xtest COMMAND test_xtensor_lib DEPENDS test_xtensor_lib)
add_test(NAME xtest COMMAND test_xtensor_lib)

# Some files will be compiled twice, however compiling common files in a static
# library and linking test_xtensor_lib with it removes half of the tests at
# runtime.
add_library(test_xtensor_core_lib STATIC ${COMMON_BASE} ${TEST_HEADERS} ${XTENSOR_HEADERS})
target_include_directories(test_xtensor_core_lib PRIVATE ${XTENSOR_INCLUDE_DIR})

target_link_libraries(test_xtensor_core_lib PRIVATE xtensor doctest::doctest ${CMAKE_THREAD_LIBS_INIT})
target_precompile_headers(test_xtensor_core_lib PRIVATE ${XTENSOR_HEADERS})
add_custom_target(coverity COMMAND coverity_scan DEPENDS test_xtensor_core_lib)
