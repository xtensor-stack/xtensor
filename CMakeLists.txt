############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
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
message(STATUS "xtensor v${${PROJECT_NAME}_VERSION}")


# Build
# =====

set(XTENSOR_HEADERS
    ${XTENSOR_INCLUDE_DIR}/xtensor/xarray.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xassign.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbroadcast.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xbuilder.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcontainer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xeval.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexception.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xexpression.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfunction.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xgenerator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xindexview.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xio.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xiterable.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xiterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xmath.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xnoalias.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoperation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xrandom.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xreducer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xscalar.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xsemantic.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xslice.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstorage.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xstrides.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_config.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xtensor_forward.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xutils.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xvectorize.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xview.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoptional.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xmissing.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xfunctorview.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xoffsetview.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xcomplex.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/xview_utils.hpp
)

OPTION(XTENSOR_ENABLE_ASSERT "xtensor bound check" OFF)
OPTION(BUILD_TESTS "xtensor test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(XTENSOR_ENABLE_ASSERT)
    add_definitions(-DXTENSOR_ENABLE_ASSERT)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
    add_subdirectory(benchmark)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES ${XTENSOR_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor)

# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(XTENSOR_CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}" CACHE STRING "install path for xtensorConfig.cmake")

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

