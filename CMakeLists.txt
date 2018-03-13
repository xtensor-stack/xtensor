############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor)

set(XTENSOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_config.hpp" xtensor_version_defines
     REGEX "#define XTENSOR_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor_version_defines})
    if(ver MATCHES "#define XTENSOR_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
message(STATUS "Building xtensor v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

find_package(xtl 0.4.1 REQUIRED)
message(STATUS "Found xtl: ${xtl_INCLUDE_DIRS}/xtl")

# Build
# =====

set(XTENSOR_HEADERS
    ${XTENSOR_INCLUDE_DIR}/xtensor/xaccumulator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xadapt.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xarray.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xassign.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xaxis_iterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbroadcast.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbuffer_adaptor.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbuilder.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcomplex.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xconcepts.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcontainer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcsv.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xeval.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexception.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexpression.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfixed.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfunction.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfunctor_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xgenerator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xindex_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xinfo.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xio.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xiterable.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xiterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xlayout.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xmath.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xnoalias.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xnorm.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xnpy.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoffset_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoperation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional_assembly.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional_assembly_base.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xrandom.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xreducer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xscalar.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xsemantic.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xshape.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xslice.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xsort.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstorage.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstrided_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstrides.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_config.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_forward.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_simd.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xutils.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xvectorize.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xview.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xview_utils.hpp
)

add_library(xtensor INTERFACE)
target_include_directories(xtensor INTERFACE $<BUILD_INTERFACE:${XTENSOR_INCLUDE_DIR}>
                                             $<INSTALL_INTERFACE:include>)
target_link_libraries(xtensor INTERFACE xtl)

OPTION(XTENSOR_ENABLE_ASSERT "xtensor bound check" OFF)
OPTION(XTENSOR_CHECK_DIMENSION "xtensor dimension check" OFF)
OPTION(XTENSOR_USE_XSIMD "simd acceleration for xtensor" OFF)
OPTION(BUILD_TESTS "xtensor test suite" OFF)
OPTION(BUILD_BENCHMARK "xtensor benchmark" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)
OPTION(DOWNLOAD_GBENCHMARK "download google benchmark and build from source" ON)
OPTION(DEFAULT_COLUMN_MAJOR "set default layout to column major" OFF)
OPTION(DISABLE_VS2017 "disables the compilation of some test with Visual Studio 2017" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(XTENSOR_ENABLE_ASSERT OR XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_ASSERT)
endif()

if(XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_CHECK_DIMENSION)
endif()

if(XTENSOR_USE_XSIMD)
    add_definitions(-DXTENSOR_USE_XSIMD)
    find_package(xsimd 4 REQUIRED)
    message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
    target_link_libraries(xtensor INTERFACE xsimd)
endif()

message(STATUS "${XTENSOR_DEPENDENCIES}")

if(DEFAULT_COLUMN_MAJOR)
    add_definitions(-DDEFAULT_LAYOUT=layout_type::column_major)
endif()

if(DISABLE_VS2017)
    add_definitions(-DDISABLE_VS2017)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xtensor
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${XTENSOR_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor)

set(XTENSOR_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xtensorConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XTENSOR_CMAKECONFIG_INSTALL_DIR})

# xtensor is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensorConfigVersion.cmake so that an xtensorConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XTENSOR_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
