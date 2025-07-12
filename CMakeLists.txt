############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.15..3.29)
project(xtensor CXX)

set(XTENSOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${XTENSOR_INCLUDE_DIR}/xtensor/core/xtensor_config.hpp" xtensor_version_defines
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

set(xtl_REQUIRED_VERSION 0.8.0)
if(TARGET xtl)
    set(xtl_VERSION ${XTL_VERSION_MAJOR}.${XTL_VERSION_MINOR}.${XTL_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if(${xtl_VERSION} VERSION_LESS ${xtl_REQUIRED_VERSION})
        message(ERROR "Mismatch xtl versions. Found '${xtl_VERSION}' but requires: '${xtl_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtl v${xtl_VERSION}")
    endif()
else()
    find_package(xtl ${xtl_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtl: ${xtl_INCLUDE_DIRS}/xtl")
endif()

find_package(nlohmann_json 3.1.1 QUIET)

# Optional dependencies
# =====================

OPTION(XTENSOR_USE_XSIMD "simd acceleration for xtensor" OFF)
OPTION(XTENSOR_USE_TBB "enable parallelization using intel TBB" OFF)
OPTION(XTENSOR_USE_OPENMP "enable parallelization using OpenMP" OFF)
if(XTENSOR_USE_TBB AND XTENSOR_USE_OPENMP)
    message(
        FATAL
        "XTENSOR_USE_TBB and XTENSOR_USE_OPENMP cannot both be active at once"
    )
endif()

if(XTENSOR_USE_XSIMD)
    set(xsimd_REQUIRED_VERSION 13.2.0)
    if(TARGET xsimd)
        set(xsimd_VERSION ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
        # Note: This is not SEMVER compatible comparison
        if(${xsimd_VERSION} VERSION_LESS ${xsimd_REQUIRED_VERSION})
            message(ERROR "Mismatch xsimd versions. Found '${xsimd_VERSION}' but requires: '${xsimd_REQUIRED_VERSION}'")
        else()
            message(STATUS "Found xsimd v${xsimd_VERSION}")
        endif()
    else()
        find_package(xsimd ${xsimd_REQUIRED_VERSION} REQUIRED)
        message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
    endif()
endif()

if(XTENSOR_USE_TBB)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
    find_package(TBB REQUIRED)
    message(STATUS "Found intel TBB: ${TBB_INCLUDE_DIRS}")
endif()

if(XTENSOR_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        # Set openmp variables now

        # Create private target just for this lib
        # https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
        # Probably not safe for cmake < 3.4 ..
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX_xtensor IMPORTED INTERFACE)
        set_property(
            TARGET
            OpenMP::OpenMP_CXX_xtensor
            PROPERTY
            INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS}
        )
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(
            TARGET
            OpenMP::OpenMP_CXX_xtensor
            PROPERTY
            INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)

        message(STATUS "OpenMP Found")
    else()
        message(FATAL "Failed to locate OpenMP")
    endif()
endif()

# Build
# =====

set(XTENSOR_HEADERS
    ${XTENSOR_INCLUDE_DIR}/xtensor/chunk/xchunked_array.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/chunk/xchunked_assign.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/chunk/xchunked_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xadapt.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xarray.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xbuffer_adaptor.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xcontainer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xfixed.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xscalar.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xstorage.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/containers/xtensor.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xaccessible.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xassign.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xeval.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xexpression.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xexpression_traits.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xfunction.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xiterable.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xiterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xlayout.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xmath.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xmultiindex_iterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xnoalias.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xoperation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xsemantic.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xshape.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xstrides.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xtensor_config.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xtensor_forward.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/core/xvectorize.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/generators/xbuilder.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/generators/xgenerator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/generators/xrandom.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/io/xcsv.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/io/xinfo.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/io/xio.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/io/xjson.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/io/xmime.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/io/xnpy.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xcomplex.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xexpression_holder.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xfft.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xhistogram.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xmanipulation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xpad.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xset_operation.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/misc/xsort.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/optional/xoptional.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/optional/xoptional_assembly.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/optional/xoptional_assembly_base.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/optional/xoptional_assembly_storage.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/reducers/xaccumulator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/reducers/xblockwise_reducer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/reducers/xblockwise_reducer_functors.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/reducers/xnorm.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/reducers/xreducer.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/utils/xexception.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/utils/xtensor_simd.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/utils/xutils.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xaxis_iterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xaxis_slice_iterator.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xbroadcast.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xdynamic_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xfunctor_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xindex_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xmasked_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xoffset_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xrepeat.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xslice.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xstrided_view.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xstrided_view_base.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xview.hpp
    ${XTENSOR_INCLUDE_DIR}/xtensor/views/xview_utils.hpp
)

add_library(xtensor INTERFACE)

target_include_directories(xtensor INTERFACE
    $<BUILD_INTERFACE:${XTENSOR_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(xtensor INTERFACE cxx_std_20)

target_link_libraries(xtensor INTERFACE xtl)

OPTION(XTENSOR_ENABLE_ASSERT "xtensor bound check" OFF)
OPTION(XTENSOR_CHECK_DIMENSION "xtensor dimension check" OFF)
OPTION(XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS "xtensor force the use of temporary memory when assigning instead of an automatic overlap check" ON)
OPTION(BUILD_TESTS "xtensor test suite" OFF)
OPTION(BUILD_BENCHMARK "xtensor benchmark" OFF)
OPTION(DOWNLOAD_GBENCHMARK "download google benchmark and build from source" ON)
OPTION(DEFAULT_COLUMN_MAJOR "set default layout to column major" OFF)
OPTION(CPP23 "enables C++23 (experimental)" OFF)
OPTION(XTENSOR_DISABLE_EXCEPTIONS "Disable C++ exceptions" OFF)
OPTION(DISABLE_MSVC_ITERATOR_CHECK "Disable the MVSC iterator check" ON)


if(XTENSOR_ENABLE_ASSERT OR XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_ASSERT)
endif()

if(XTENSOR_CHECK_DIMENSION)
    add_definitions(-DXTENSOR_ENABLE_CHECK_DIMENSION)
endif()

if(XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS)
    add_definitions(-DXTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS)
endif()

if(DEFAULT_COLUMN_MAJOR)
    add_definitions(-DXTENSOR_DEFAULT_LAYOUT=layout_type::column_major)
endif()

if(MSVC AND DISABLE_MSVC_ITERATOR_CHECK)
    add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

if(XTENSOR_USE_OPENMP)
    # Link xtensor itself to OpenMP to propagate to user projects
    target_link_libraries(xtensor INTERFACE OpenMP::OpenMP_CXX_xtensor)
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

install(DIRECTORY ${XTENSOR_INCLUDE_DIR}/xtensor
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(XTENSOR_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE
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
        DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig/")

# Write single include
# ====================

function(PREPEND var prefix)
   set(listVar "")
   foreach(f ${ARGN})
      list(APPEND listVar "${prefix}${f}")
   endforeach(f)
   set(${var} "${listVar}" PARENT_SCOPE)
endfunction()

function(POSTFIX var postfix)
   set(listVar "")
   foreach(f ${ARGN})
      list(APPEND listVar "${f}${postfix}")
   endforeach(f)
   set(${var} "${listVar}" PARENT_SCOPE)
endfunction()

set(XTENSOR_SINGLE_INCLUDE ${XTENSOR_HEADERS})
string(REPLACE "${XTENSOR_INCLUDE_DIR}/" "" XTENSOR_SINGLE_INCLUDE "${XTENSOR_SINGLE_INCLUDE}")
list(REMOVE_ITEM XTENSOR_SINGLE_INCLUDE
    xtensor/misc/xexpression_holder.hpp
    xtensor/io/xjson.hpp
    xtensor/io/xmime.hpp
    xtensor/io/xnpy.hpp)

PREPEND(XTENSOR_SINGLE_INCLUDE "#include <" ${XTENSOR_SINGLE_INCLUDE})
POSTFIX(XTENSOR_SINGLE_INCLUDE ">" ${XTENSOR_SINGLE_INCLUDE})
string(REPLACE ";" "\n" XTENSOR_SINGLE_INCLUDE "${XTENSOR_SINGLE_INCLUDE}")
string(CONCAT XTENSOR_SINGLE_INCLUDE "#ifndef XTENSOR\n" "#define XTENSOR\n\n" "${XTENSOR_SINGLE_INCLUDE}" "\n\n#endif\n")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/xtensor.hpp" "${XTENSOR_SINGLE_INCLUDE}")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/xtensor.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
