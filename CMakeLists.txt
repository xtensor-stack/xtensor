############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor)

# Versionning
# ===========

set(XTENSOR_MAJOR_VERSION 0)
set(XTENSOR_MINOR_VERSION 3)
set(XTENSOR_PATCH_VERSION 0)
set(${PROJECT_NAME}_VERSION 
    ${XTENSOR_MAJOR_VERSION}.${XTENSOR_MINOR_VERSION}.${XTENSOR_PATCH_VERSION})
message(STATUS "xtensor v${${PROJECT_NAME}_VERSION}")


# Build
# =====

set(XTENSOR_HEADERS
    include/xtensor/xarray.hpp
    include/xtensor/xassign.hpp
    include/xtensor/xbroadcast.hpp
    include/xtensor/xcontainer.hpp
    include/xtensor/xexception.hpp
    include/xtensor/xexpression.hpp
    include/xtensor/xfunction.hpp
    include/xtensor/xio.hpp
    include/xtensor/xiterator.hpp
    include/xtensor/xmath.hpp
    include/xtensor/xnoalias.hpp
    include/xtensor/xoperation.hpp
    include/xtensor/xscalar.hpp
    include/xtensor/xsemantic.hpp
    include/xtensor/xslice.hpp
    include/xtensor/xutils.hpp
    include/xtensor/xvectorize.hpp
    include/xtensor/xview.hpp
)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       XTENSOR_HEADERS "${XTENSOR_HEADERS}")

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
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_CMAKECONFIG_INSTALL_DIR})

